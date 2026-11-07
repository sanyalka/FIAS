"""Validate full graph traversability and sampled lookup round trips; audit datasets."""
import argparse
from collections import Counter
import json
from pathlib import Path
from .index import Index, ROOT, MISSING
from .xmlio import digest, normalize


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', default='artifacts/gar.gari')
    p.add_argument('--dataset')
    p.add_argument('--output', default='artifacts/verification.json')
    args = p.parse_args(argv)
    report = {'regions': {}, 'dataset': {}}
    with Index(args.index) as index:
        for region, meta in index.meta['regions'].items():
            counts = Counter()
            previous = -1
            for i in range(meta['count']):
                row = index.raw(region, i)
                if row[0] <= previous:
                    raise ValueError('Unsorted or duplicate OBJECTID')
                previous = row[0]
                for mode, bit, column in (('adm', 1, 1), ('mun', 2, 2)):
                    if not meta['hierarchies'][mode]:
                        continue
                    if row[-1] & bit:
                        counts[mode + '_quarantined'] += 1
                        continue
                    pointer = row[column]
                    if pointer != ROOT and not 0 <= pointer < meta['count']:
                        raise ValueError('Unflagged invalid parent')
                    # Validate every non-quarantined chain (depth is small).
                    index.chain(region, i, mode)
                    counts[mode + '_valid_chains'] += 1
                # Deterministic lookup sample, roughly 1/1000 plus first 100 records.
                if i < 100 or digest(str(row[0])) % 1000 == 0:
                    node = index.node(region, i)
                    scope = {'parent': row[1], 'mode': 'adm'} if node['kind'] == 'house' and 'adm' in meta.get('scoped_modes', []) else {}
                    if i not in index.find(region, node, **scope):
                        raise ValueError('Lookup lost node')
                    if index.find_oid(region, row[0]) != i:
                        raise ValueError('OBJECTID lookup lost node')
                    counts['roundtrip_lookups'] += 1
            report['regions'][region] = dict(counts)
        if args.dataset:
            path = Path(args.dataset)
            groups, texts = {}, {}
            counts = Counter()
            for split in ('train', 'validation', 'test'):
                with (path / f'checks_{split}.jsonl').open(encoding='utf-8') as stream:
                    for line in stream:
                        row = json.loads(line)
                        group = row['group']
                        if groups.get(group, split) != split:
                            raise ValueError('Group leakage')
                        groups[group] = split
                        key = normalize(row['text'])
                        if key in texts:
                            raise ValueError('Duplicate normalized text')
                        texts[key] = split
                        found = index.match(row['region'], row['components'], row['hierarchy'])
                        if (len(found) == 1) != (row['verdict'] == 'ВАЛИДНЫЙ') or (row['verdict'] == 'НЕВАЛИДНЫЙ' and found):
                            raise ValueError('Incorrect synthetic label')
                        counts[split] += 1
            report['dataset'] = dict(counts, disjoint_groups=len(groups), unique_texts=len(texts))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
