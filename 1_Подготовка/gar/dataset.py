"""Generate reproducible BIO component labels and index-checked negative examples."""
import argparse
from collections import Counter
import hashlib
import heapq
import json
from pathlib import Path
import random
import re
from .index import Index
from .xmlio import digest, normalize

TOKEN = re.compile(r'\w+|[^\w\s]', re.UNICODE)
FIELDS = {1: 'REGION', 2: 'ADM_DISTRICT', 3: 'MUN_DISTRICT', 4: 'SETTLEMENT', 5: 'CITY', 6: 'LOCALITY', 7: 'PLANNING', 8: 'STREET', 10: 'HOUSE', 13: 'LEGACY13', 14: 'LEGACY14', 15: 'LEGACY15', 16: 'LEGACY16'}


def tokenize(text):
    return [(m.group(), m.start(), m.end()) for m in TOKEN.finditer(text)]


def render(node, types):
    if node['kind'] == 'object':
        return node['type'] + ' ' + node['name']
    parts = []
    for number, typ, catalog in (('HOUSENUM', 'HOUSETYPE', 'AS_HOUSE_TYPES'), ('ADDNUM1', 'ADDTYPE1', 'AS_ADDHOUSE_TYPES'), ('ADDNUM2', 'ADDTYPE2', 'AS_ADDHOUSE_TYPES')):
        if node.get(number):
            type_id = node.get(typ, '')
            if type_id and type_id not in types[catalog]:
                raise ValueError(f'Unknown {catalog} code {type_id}')
            parts.append((types[catalog].get(type_id, '') + ' ' + node[number]).strip())
    return ' '.join(parts)


def tagged(components, types):
    text, spans = '', []
    for node in components:
        if text:
            text += ', '
        start = len(text)
        text += render(node, types)
        spans.append({'start': start, 'end': len(text), 'field': FIELDS[node['level']]})
    tokens, labels = [], []
    begun = set()
    for token, start, end in tokenize(text):
        tokens.append(token)
        label = 'O'
        for i, span in enumerate(spans):
            if span['start'] <= start and end <= span['end']:
                label = ('I-' if i in begun else 'B-') + span['field']
                begun.add(i)
                break
        labels.append(label)
    return {'text': text, 'spans': spans, 'tokens': tokens, 'labels': labels}


def split_for(group, seed):
    value = digest(f'{seed}:{group}') % 100
    return 'train' if value < 80 else 'validation' if value < 90 else 'test'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', default='artifacts/gar.gari')
    p.add_argument('--output', default='artifacts/dataset')
    p.add_argument('--max-houses-per-region', type=int, default=10000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--hierarchy', choices=['adm', 'mun'], default='adm')
    args = p.parse_args(argv)
    if args.max_houses_per_region < 1:
        p.error('max-houses-per-region must be positive')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    counts, seen, coverage = Counter(), {}, {}
    streams = {f'{task}_{split}': (output / f'{task}_{split}.jsonl').open('w', encoding='utf-8') for task in ('tagger', 'checks') for split in ('train', 'validation', 'test')}
    rng = random.Random(args.seed)
    try:
        with Index(args.index) as index:
            types = {catalog: {r['ID']: r['SHORTNAME'] for r in index.meta['types'][catalog]} for catalog in ('AS_HOUSE_TYPES', 'AS_ADDHOUSE_TYPES')}
            for region, meta in index.meta['regions'].items():
                before = counts.copy()
                if not meta['hierarchies'][args.hierarchy]:
                    counts['unavailable_regions'] += 1
                    coverage[region] = {'status': 'hierarchy_unavailable', 'sampled_houses': 0}
                    continue
                # Stable min-hash reservoir: representative over the entire region.
                reservoir = []
                for i in range(meta['count']):
                    raw = index.raw(region, i)
                    if raw[6] != 1 or raw[7] & (1 if args.hierarchy == 'adm' else 2):
                        continue
                    rank = digest(f'{args.seed}:{region}:{raw[0]}')
                    heapq.heappush(reservoir, (-rank, i))
                    if len(reservoir) > args.max_houses_per_region:
                        heapq.heappop(reservoir)
                selected = [i for _, i in sorted(reservoir)]
                for num, i in enumerate(selected):
                    chain = index.chain(region, i, args.hierarchy)
                    components = [index.node(region, j) for j in chain]
                    if any(n['level'] not in FIELDS for n in components):
                        counts['unsupported_level'] += 1
                        continue
                    try:
                        for node in components:
                            render(node, types)
                    except ValueError:
                        counts['unknown_type_skipped'] += 1
                        continue
                    if len(index.match(region, components, args.hierarchy)) != 1:
                        counts['ambiguous_positive_skipped'] += 1
                        continue
                    # Hold out whole immediate-parent groups, including all house variants.
                    group = f'{region}:{args.hierarchy}:{components[-2]["oid"] if len(components) > 1 else components[-1]["oid"]}'
                    split = split_for(group, args.seed)
                    reference = {'region': region, 'hierarchy': args.hierarchy, 'group': group, 'objectid': components[-1]['oid']}
                    examples = [(components, 'VALID', None)]
                    # Synthetic typo; accept as negative only after exact oracle check.
                    position = rng.randrange(len(components))
                    mutated = [dict(n) for n in components]
                    field = 'HOUSENUM' if mutated[position]['kind'] == 'house' else 'name'
                    value = mutated[position].get(field, '')
                    if value:
                        at = rng.randrange(len(value))
                        mutated[position][field] = value[:at] + rng.choice('абвгдежзиклмнопрстуфхцчшэюя') + value[at + 1:]
                        if not index.match(region, mutated, args.hierarchy):
                            reason = 'WRONG_HIERARCHY' if index.find(region, mutated[position], limit=1) else 'COMPONENT_NOT_FOUND'
                            examples.append((mutated, reason, FIELDS[mutated[position]['level']]))
                        else:
                            counts['mutation_still_valid_skipped'] += 1
                    # Replace complete house designation with a REAL house from another path.
                    if len(selected) > 1:
                        replacement = index.node(region, selected[(num + 1) % len(selected)])
                        wrong = components[:-1] + [replacement]
                        if not index.match(region, wrong, args.hierarchy):
                            examples.append((wrong, 'WRONG_HIERARCHY', 'HOUSE'))
                        else:
                            counts['replacement_still_valid_skipped'] += 1
                    for example, reason, error_field in examples:
                        try:
                            tagging = tagged(example, types)
                        except ValueError:
                            counts['unknown_type_skipped'] += 1
                            continue
                        # Dedupe across tasks' shared examples and across split groups.
                        text_key = hashlib.sha256(normalize(tagging['text']).encode()).hexdigest()
                        if text_key in seen:
                            counts['duplicate_text_skipped'] += 1
                            continue
                        seen[text_key] = split
                        tag_row = dict(tagging, **reference, augmentation=reason)
                        check_row = dict(reference, text=tagging['text'], components=example, verdict='ВАЛИДНЫЙ' if reason == 'VALID' else 'НЕВАЛИДНЫЙ', reason=reason, error_field=error_field)
                        for task, row in (('tagger', tag_row), ('checks', check_row)):
                            streams[f'{task}_{split}'].write(json.dumps(row, ensure_ascii=False) + '\n')
                            counts[f'{task}_{split}'] += 1
                        counts[reason] += 1
                print(f'[{region}] sampled houses: {len(selected)}', flush=True)
                coverage[region] = {'status': 'sampled' if selected else 'no_eligible_houses', 'sampled_houses': len(selected), 'examples': {split: counts[f'tagger_{split}'] - before[f'tagger_{split}'] for split in ('train', 'validation', 'test')}, 'skipped_ambiguous': counts['ambiguous_positive_skipped'] - before['ambiguous_positive_skipped']}
                index._children.clear()  # Don't retain every region's object adjacency in RAM.
            manifest = {'schema': 1, 'snapshot': index.meta['snapshot'], 'index_regions': index.meta['regions'], 'region_coverage': coverage, 'seed': args.seed, 'hierarchy': args.hierarchy, 'max_houses_per_region': args.max_houses_per_region, 'split_policy': 'immediate-parent hash 80/10/10; exact normalized-text global dedupe', 'counts': dict(counts), 'limitations': ['Synthetic examples only; evaluate on manually labelled real Excel addresses.', 'No training target for mandatory fields: a business completeness profile is not yet supplied.', 'No claim of generalization to unseen regions.']}
            (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(dict(counts)), flush=True)
    finally:
        for stream in streams.values():
            stream.close()


if __name__ == '__main__':
    main()
