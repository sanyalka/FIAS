"""GARI v1: memory mapped regional shards, exact lookup and graph traversal."""
import hashlib
import json
import mmap
import struct
from pathlib import Path
from .xmlio import digest, normalize

MAGIC = b'GARI0001'
HEADER = struct.Struct('<8sQ')
RECORD = struct.Struct('<QIIQIBBH')  # oid, adm, mun, payload offset/size, level, kind, flags
LOOKUP = struct.Struct('<QI')
ROOT = 0xFFFFFFFE
MISSING = 0xFFFFFFFF


def key(node):
    if node['kind'] == 'house':
        return json.dumps([node.get(x, '') for x in ('HOUSENUM', 'HOUSETYPE', 'ADDNUM1', 'ADDTYPE1', 'ADDNUM2', 'ADDTYPE2')], ensure_ascii=False, separators=(',', ':'))
    return json.dumps([node['name'], node['type']], ensure_ascii=False, separators=(',', ':'))


class Index:
    def __init__(self, path, verify=True):
        self.file = open(path, 'rb')
        self.mm = None
        self._children = {}
        try:
            self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
            magic, size = HEADER.unpack_from(self.mm)
            if magic != MAGIC or size > len(self.mm) - HEADER.size:
                raise ValueError('Invalid index header')
            self.meta = json.loads(self.mm[HEADER.size:HEADER.size + size])
            if self.meta['format'] != 1:
                raise ValueError('Unsupported index version')
            self.base = HEADER.size + size
            for region, s in self.meta['regions'].items():
                start, end = self.base + s['offset'], self.base + s['offset'] + s['bytes']
                if not self.base <= start <= end <= len(self.mm):
                    raise ValueError('Invalid shard bounds')
                if self.pool_relative(s) > s['bytes']:
                    raise ValueError('Truncated shard')
                if verify:
                    h = hashlib.sha256()
                    for pos in range(start, end, 1024 * 1024):
                        h.update(self.mm[pos:min(end, pos + 1024 * 1024)])
                    if h.hexdigest() != s['sha256']:
                        raise ValueError(f'Corrupt region {region}')
        except Exception:
            self.close()
            raise

    def close(self):
        if self.mm is not None:
            self.mm.close()
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def raw(self, region, i):
        s = self.meta['regions'][region]
        if not 0 <= i < s['count']:
            raise IndexError(i)
        return RECORD.unpack_from(self.mm, self.base + s['offset'] + i * RECORD.size)

    @staticmethod
    def pool_relative(s):
        return s['count'] * (RECORD.size + LOOKUP.size * (1 + len(s.get('scoped_modes', []))))

    def node(self, region, i):
        s = self.meta['regions'][region]
        oid, adm, mun, offset, size, level, kind, flags = self.raw(region, i)
        pool = self.base + s['offset'] + self.pool_relative(s)
        end = pool + offset + size
        if end > self.base + s['offset'] + s['bytes']:
            raise ValueError('Invalid payload bounds')
        result = json.loads(self.mm[pool + offset:end])
        result.update(oid=oid, level=level, kind='house' if kind else 'object', flags=flags)
        return result

    def find_oid(self, region, oid):
        lo, hi = 0, self.meta['regions'][region]['count']
        while lo < hi:
            mid = (lo + hi) // 2
            value = self.raw(region, mid)[0]
            if value < oid:
                lo = mid + 1
            else:
                hi = mid
        return lo if lo < self.meta['regions'][region]['count'] and self.raw(region, lo)[0] == oid else None

    def find(self, region, node, parent=None, mode='adm', limit=None):
        s = self.meta['regions'][region]
        base = self.base + s['offset'] + s['count'] * RECORD.size
        wanted = normalize(key(node))
        scoped = parent is not None and mode in s.get('scoped_modes', [])
        if scoped:
            base += s['count'] * LOOKUP.size * (1 + s['scoped_modes'].index(mode))
        hashed = digest(f'{parent}:{wanted}' if scoped else wanted)
        lo, hi = 0, s['count']
        while lo < hi:
            mid = (lo + hi) // 2
            value, _ = LOOKUP.unpack_from(self.mm, base + mid * LOOKUP.size)
            if value < hashed:
                lo = mid + 1
            else:
                hi = mid
        result = []
        while lo < s['count']:
            value, i = LOOKUP.unpack_from(self.mm, base + lo * LOOKUP.size)
            if value != hashed:
                break
            candidate = self.node(region, i)
            parent_ok = parent is None or self.raw(region, i)[1 if mode == 'adm' else 2] == parent
            if parent_ok and candidate['kind'] == node['kind'] and candidate['level'] == node.get('level', candidate['level']) and normalize(key(candidate)) == wanted:
                result.append(i)
                if limit and len(result) >= limit:
                    break
            lo += 1
        return result

    def object_descendants(self, region, parent, mode):
        cache_key = (region, mode)
        if cache_key not in self._children:
            children = {}
            column = 1 if mode == 'adm' else 2
            for i in range(self.meta['regions'][region]['count']):
                raw = self.raw(region, i)
                if raw[6] == 0 and not raw[-1] & (1 if mode == 'adm' else 2):
                    children.setdefault(raw[column], []).append(i)
            self._children[cache_key] = children
        pending, seen = [parent], set()
        while pending:
            item = pending.pop()
            if item in seen:
                continue
            seen.add(item)
            yield item
            pending.extend(self._children[cache_key].get(item, []))

    def chain(self, region, i, mode='adm'):
        if mode not in ('adm', 'mun') or not self.meta['regions'][region]['hierarchies'].get(mode):
            raise ValueError(f'Unavailable hierarchy: {mode}')
        result, visited = [], set()
        while i != ROOT:
            if i == MISSING or i in visited:
                raise ValueError('Incomplete or cyclic chain')
            visited.add(i)
            raw = self.raw(region, i)
            if raw[-1] & (1 if mode == 'adm' else 2):
                raise ValueError('Quarantined hierarchy record')
            result.append(i)
            i = raw[1 if mode == 'adm' else 2]
        return result[::-1]

    def match(self, region, components, mode='adm'):
        """Exact ordered ancestor subsequence; no arbitrary choice on ambiguity."""
        if region not in self.meta['regions']:
            raise ValueError('Region outside index coverage')
        if mode not in ('adm', 'mun') or not self.meta['regions'][region]['hierarchies'].get(mode):
            raise ValueError(f'Unavailable hierarchy: {mode}')
        if not components:
            return []
        sets = [set(self.find(region, c)) for c in components[:-1]]
        if sets and (not all(sets)):
            return []
        if components[-1]['kind'] == 'house' and sets and mode in self.meta['regions'][region].get('scoped_modes', []):
            candidates = set()
            for parent in sets[-1]:
                for child in self.object_descendants(region, parent, mode):
                    candidates.update(self.find(region, components[-1], parent=child, mode=mode))
            sets.append(candidates)
        else:
            sets.append(set(self.find(region, components[-1])))
        matches = []
        for leaf in sorted(sets[-1]):
            try:
                chain = self.chain(region, leaf, mode)
            except ValueError:
                continue
            pos = 0
            for allowed in sets:
                while pos < len(chain) and chain[pos] not in allowed:
                    pos += 1
                if pos == len(chain):
                    break
                pos += 1
            else:
                matches.append(leaf)
        return matches
