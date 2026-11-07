"""Build a database-free index, one region in memory at a time."""
import argparse
from collections import Counter
from datetime import date
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
from .index import HEADER, MAGIC, RECORD, LOOKUP, ROOT, MISSING, key
from .xmlio import Source, NAME, current, digest, normalize


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode('utf-8')


def build_region(source, region, snapshot, folder, hierarchy='both'):
    report = {'region': region, 'files': {}, 'issues': {}, 'samples': {}, 'levels': {}, 'hierarchies': {}}
    nodes, duplicates = {}, set()

    def issue(name, value):
        report['issues'][name] = report['issues'].get(name, 0) + 1
        examples = report['samples'].setdefault(name, [])
        if len(examples) < 10:
            examples.append(value)

    def records(kind, required=True):
        name = source.one(region, kind, required)
        if name is None:
            return
        stamp = NAME.match(Path(name).name)[2]
        if stamp != snapshot.replace('-', ''):
            raise ValueError(f'Mixed snapshot: {name}, expected {snapshot}')
        stats = {'name': name, 'bytes': source.size(name), 'rows': 0, 'selected': 0}
        report['files'][kind] = stats
        print(f'[{region}] {kind}', flush=True)
        for row in source.iter(name):
            stats['rows'] += 1
            if current(row, snapshot, actual=kind in ('AS_ADDR_OBJ', 'AS_HOUSES')):
                stats['selected'] += 1
                yield row
            elif row.get('ISACTIVE') == '1' and row.get('ISACTUAL', '1') == '1':
                issue('active_outside_interval_' + kind, row.get('OBJECTID'))

    levels, house_fields = Counter(), Counter()
    for kind in ('AS_ADDR_OBJ', 'AS_HOUSES'):
        for r in records(kind):
            oid = int(r['OBJECTID'])
            if oid in nodes:
                duplicates.add(oid)
                issue('duplicate_objectid', oid)
                continue  # quarantine; do not silently use latest version
            if kind == 'AS_ADDR_OBJ':
                payload = {'name': r['NAME'], 'type': r['TYPENAME']}
                level, is_house = int(r['LEVEL']), 0
            else:
                payload = {f: r[f] for f in ('HOUSENUM', 'HOUSETYPE', 'ADDNUM1', 'ADDTYPE1', 'ADDNUM2', 'ADDTYPE2') if r.get(f)}
                level, is_house = 10, 1
                house_fields.update(payload.keys())
                if not any(payload.get(f) for f in ('HOUSENUM', 'ADDNUM1', 'ADDNUM2')):
                    duplicates.add(oid)
                    issue('house_without_number', oid)
            nodes[oid] = (compact(payload), level, is_house)
            levels[str(level)] += 1
    report['levels'] = dict(levels)
    report['house_fields'] = dict(house_fields)
    ids = sorted(nodes)
    if len(ids) >= ROOT:
        raise ValueError('Region exceeds 32-bit local index capacity')
    local = {oid: i for i, oid in enumerate(ids)}
    parents = {}
    bad_modes = {}
    for mode, kind in (('adm', 'AS_ADM_HIERARCHY'), ('mun', 'AS_MUN_HIERARCHY')):
        available = (hierarchy == 'both' or hierarchy == mode) and source.one(region, kind, required=False) is not None
        report['hierarchies'][mode] = available
        edge, bad, seen = {}, set(duplicates), set()
        if available:
            for r in records(kind):
                oid = int(r['OBJECTID'])
                if oid not in nodes:
                    continue
                if oid in seen:
                    bad.add(oid)
                    issue(mode + '_duplicate_edge', oid)
                    continue
                seen.add(oid)
                parent = int(r.get('PARENTOBJID', '0'))
                edge[oid] = parent
                if mode == 'adm' and r.get('REGIONCODE') and int(r['REGIONCODE']) != int(region):
                    bad.add(oid)
                    issue('adm_regioncode_mismatch', oid)
                path = r.get('PATH', '')
                if path:
                    pieces = path.split('.')
                    if pieces[-1] != str(oid) or (parent and (len(pieces) < 2 or pieces[-2] != str(parent))):
                        bad.add(oid)
                        issue(mode + '_path_tail_mismatch', oid)
                if parent and parent not in nodes:
                    bad.add(oid)
                    issue(mode + '_missing_parent', {'oid': oid, 'parent': parent})
                if parent == 0 and nodes[oid][1] != 1:
                    bad.add(oid)
                    issue(mode + '_unexpected_root', oid)
            for oid in ids:
                if oid not in edge:
                    bad.add(oid)
                    issue(mode + '_missing_edge', oid)
            # Linear graph walk; propagate errors to descendants, detect cycles.
            state = {oid: False for oid in bad}
            for oid in ids:
                trail, active = [], set()
                cur = oid
                while cur and cur not in state:
                    if cur in active:
                        issue(mode + '_cycle', cur)
                        state[cur] = False
                        break
                    if cur not in nodes:
                        break
                    active.add(cur)
                    trail.append(cur)
                    cur = edge.get(cur, -1)
                good = cur == 0 or state.get(cur, False)
                for item in trail:
                    state[item] = good
            bad = {oid for oid in ids if not state.get(oid, False)}
            report[mode + '_quarantined'] = len(bad)
            report[mode + '_house_parent_levels'] = dict(Counter(str(nodes[edge[oid]][1]) if edge.get(oid) in nodes else 'MISSING' for oid in ids if nodes[oid][2]))
        parents[mode], bad_modes[mode] = edge, bad

    report['objects'] = len(ids)
    # A missing AS_MUN_HIERARCHY is reported, never fabricated from ADM.
    pool, pool_map = bytearray(), {}
    hashes = []
    scoped_modes = [m for m in ('adm', 'mun') if report['hierarchies'][m]]
    scoped_hashes = {m: [] for m in scoped_modes}
    report['scoped_modes'] = scoped_modes
    shard = folder / (region + '.shard')
    with shard.open('wb') as out:
        for i, oid in enumerate(ids):
            payload, level, kind = nodes[oid]
            if payload not in pool_map:
                pool_map[payload] = len(pool)
                pool.extend(payload)
            pointers = []
            flags = 0
            for bit, mode in ((1, 'adm'), (2, 'mun')):
                p = parents[mode].get(oid)
                pointers.append(ROOT if p == 0 else local.get(p, MISSING))
                if oid in bad_modes[mode]:
                    flags |= bit
            out.write(RECORD.pack(oid, *pointers, pool_map[payload], len(payload), level, kind, flags))
            node = json.loads(payload)
            node['kind'] = 'house' if kind else 'object'
            wanted = normalize(key(node))
            hashes.append((digest(wanted), i))
            for mode in scoped_modes:
                scoped_hashes[mode].append((digest(f'{pointers[0 if mode == "adm" else 1]}:{wanted}'), i))
        for hashed, i in sorted(hashes):
            out.write(LOOKUP.pack(hashed, i))
        for mode in scoped_modes:
            for hashed, i in sorted(scoped_hashes[mode]):
                out.write(LOOKUP.pack(hashed, i))
        out.write(pool)
    report['pool_unique_strings'] = len(pool_map)
    report['index_bytes'] = shard.stat().st_size
    h = hashlib.sha256()
    with shard.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return shard, report, h.hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='xml_data', help='Extracted root or GAR ZIP')
    p.add_argument('--regions', nargs='+', default=['20', '23', '77', '99'], help='Region codes or all')
    p.add_argument('--hierarchy', choices=['adm', 'both'], default='adm')
    p.add_argument('--cache', help='Persistent regional build cache for restart after interruption')
    p.add_argument('--date', required=True, help='Current export date YYYY-MM-DD; historical reconstruction is not supported')
    p.add_argument('--output', default='artifacts/gar.gari')
    args = p.parse_args(argv)
    date.fromisoformat(args.date)
    if len(set(args.regions)) != len(args.regions):
        p.error('Duplicate regions')
    started = time.perf_counter()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    source = Source(args.source)
    if args.regions == ['all']:
        args.regions = sorted({name.replace('\\', '/').split('/')[0] for name in source.names if '/' in name.replace('\\', '/') and name.replace('\\', '/').split('/')[0].isdigit()})
    # Validate the complete input before starting a long run.
    for region in args.regions:
        for kind in ('AS_ADDR_OBJ', 'AS_HOUSES', 'AS_ADM_HIERARCHY'):
            source.one(region, kind)
    try:
        with tempfile.TemporaryDirectory(prefix='gar-build-', dir=output.parent) as tmp:
            folder = Path(tmp)
            metadata = {'format': 1, 'snapshot': args.date, 'normalization': 'casefold_whitespace_v1', 'regions': {}, 'types': {}}
            reports, shards, offset = [], [], 0
            for kind in ('AS_HOUSE_TYPES', 'AS_ADDHOUSE_TYPES', 'AS_OBJECT_LEVELS', 'AS_ADDR_OBJ_TYPES'):
                name = source.one('', kind)
                if NAME.match(Path(name).name)[2] != args.date.replace('-', ''):
                    raise ValueError(f'Mixed dictionary snapshot: {name}')
                metadata['types'][kind] = list(source.iter(name))
            for region in sorted(args.regions):
                target = Path(args.cache) if args.cache else folder
                target.mkdir(parents=True, exist_ok=True)
                checkpoint = target / (region + '.json')
                signature = {'snapshot': args.date, 'hierarchy': args.hierarchy, 'build_schema': 2, 'source': str(source.path.resolve()), 'size': source.path.stat().st_size, 'mtime_ns': source.path.stat().st_mtime_ns}
                cached = json.loads(checkpoint.read_text(encoding='utf-8')) if checkpoint.exists() else None
                shard = target / (region + '.shard')
                if cached and cached['signature'] == signature and shard.exists() and shard.stat().st_size == cached['report']['index_bytes']:
                    h = hashlib.sha256()
                    with shard.open('rb') as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b''):
                            h.update(block)
                    if h.hexdigest() != cached['checksum']:
                        raise ValueError(f'Corrupt cached shard {region}')
                    report, checksum = cached['report'], cached['checksum']
                    print(f'[{region}] resume completed shard', flush=True)
                else:
                    shard, report, checksum = build_region(source, region, args.date, target, args.hierarchy)
                    checkpoint.write_text(json.dumps({'signature': signature, 'report': report, 'checksum': checksum}, ensure_ascii=False), encoding='utf-8')
                print(f'[{region}] complete: {report["objects"]} records, {report["index_bytes"]} bytes', flush=True)
                reports.append(report)
                shards.append(shard)
                size = shard.stat().st_size
                metadata['regions'][region] = {'offset': offset, 'bytes': size, 'count': report['objects'], 'sha256': checksum, 'hierarchies': report['hierarchies'], 'scoped_modes': report['scoped_modes'], 'quarantined': {m: report.get(m + '_quarantined', 0) for m in ('adm', 'mun')}}
                offset += size
            header = compact(metadata)
            staging = folder / 'combined.gari'
            with staging.open('wb') as out:
                out.write(HEADER.pack(MAGIC, len(header)))
                out.write(header)
                for shard in shards:
                    with shard.open('rb') as f:
                        shutil.copyfileobj(f, out, 1024 * 1024)
            from .index import Index
            with Index(staging):
                pass
            staging.replace(output)
            result = {'snapshot': args.date, 'elapsed_seconds': round(time.perf_counter() - started, 2), 'index_bytes': output.stat().st_size, 'regions': reports}
            output.with_suffix('.audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps({k: v for k, v in result.items() if k != 'regions'}), flush=True)
    finally:
        source.close()


if __name__ == '__main__':
    main()
