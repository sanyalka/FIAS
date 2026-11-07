"""Benchmark exact structured lookup, excluding Excel and neural inference."""
import argparse
import json
from pathlib import Path
import statistics
import time
from .index import Index
from .xmlio import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', default='artifacts/gar.gari')
    p.add_argument('--output', default='artifacts/benchmark.json')
    args = p.parse_args()
    started = time.perf_counter()
    with Index(args.index) as index:
        load = time.perf_counter() - started
        rows = {}
        for region, meta in index.meta['regions'].items():
            samples = []
            for i in range(meta['count']):
                row = index.raw(region, i)
                if row[6] == 1 and not row[-1] & 1 and (digest(str(row[0])) % 1000 == 0 or meta['count'] < 1000):
                    samples.append(i)
                    if len(samples) == 100:
                        break
            durations, counts = [], []
            for i in samples:
                components = [index.node(region, j) for j in index.chain(region, i)]
                if not durations:
                    index.match(region, components)  # warm caches
                start = time.perf_counter()
                result = index.match(region, components)
                durations.append((time.perf_counter() - start) * 1000)
                counts.append(len(result))
            rows[region] = {'samples': len(durations), 'median_ms': statistics.median(durations), 'p95_ms': sorted(durations)[int(0.95 * (len(durations) - 1))], 'ambiguous': sum(n != 1 for n in counts)}
    report = {'checksum_verified_open_seconds': load, 'regions': rows, 'scope': 'Warm exact structured matching only; no Excel, neural parser, or disk-cold OS cache guarantee.'}
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
