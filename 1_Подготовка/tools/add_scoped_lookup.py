"""Upgrade an existing GARI index with parent-context lookup, without rereading XML."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gar.index import Index, HEADER, MAGIC, RECORD, LOOKUP, key
from gar.xmlio import normalize, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', default='artifacts/gar.gari')
    p.add_argument('--output', default='artifacts/gar_scoped.gari')
    args = p.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        temp = Path(temp)
        with Index(args.index) as index:
            meta = json.loads(json.dumps(index.meta))
            offset, paths = 0, []
            for region, shard in index.meta['regions'].items():
                path = temp / region
                modes = [m for m in ('adm', 'mun') if shard['hierarchies'][m]]
                start = index.base + shard['offset']
                end_primary = start + shard['count'] * (RECORD.size + LOOKUP.size)
                with path.open('wb') as out:
                    for pos in range(start, end_primary, 1024 * 1024):
                        out.write(index.mm[pos:min(end_primary, pos + 1024 * 1024)])
                    for mode in modes:
                        values = []
                        for i in range(shard['count']):
                            node = index.node(region, i)
                            parent = index.raw(region, i)[1 if mode == 'adm' else 2]
                            values.append((digest(f'{parent}:{normalize(key(node))}'), i))
                        for hashed, i in sorted(values):
                            out.write(LOOKUP.pack(hashed, i))
                    pool = start + index.pool_relative(shard)
                    end = start + shard['bytes']
                    for pos in range(pool, end, 1024 * 1024):
                        out.write(index.mm[pos:min(end, pos + 1024 * 1024)])
                h = hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        h.update(block)
                meta['regions'][region].update(offset=offset, bytes=path.stat().st_size, sha256=h.hexdigest(), scoped_modes=modes)
                offset += path.stat().st_size
                paths.append(path)
                print(region, path.stat().st_size, flush=True)
        content = json.dumps(meta, ensure_ascii=False, separators=(',', ':')).encode()
        staging = temp / 'combined'
        with staging.open('wb') as out:
            out.write(HEADER.pack(MAGIC, len(content)))
            out.write(content)
            for path in paths:
                with path.open('rb') as stream:
                    shutil.copyfileobj(stream, out)
        with Index(staging):
            pass
        staging.replace(output)


if __name__ == '__main__':
    main()
