"""Read registry coverage and identify special house designations in a built index."""
import argparse
from collections import Counter
import json
from pathlib import Path
from .index import Index
from .xmlio import Source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='xml_data')
    p.add_argument('--index', default='artifacts/gar.gari')
    p.add_argument('--output', default='artifacts/source_inspection.json')
    args = p.parse_args()
    source, reports = Source(args.source), {}
    try:
        with Index(args.index) as index:
            for region, meta in index.meta['regions'].items():
                stats, examples = Counter(), {}
                def note(name, node):
                    stats[name] += 1
                    values = examples.setdefault(name, [])
                    if len(values) < 5:
                        values.append(node)
                for i in range(meta['count']):
                    n = index.node(region, i)
                    if n['kind'] != 'house':
                        continue
                    if not n.get('HOUSENUM'):
                        note('no_primary_number', n)
                    if n.get('ADDNUM2'):
                        note('second_additional_number', n)
                    if any('/' in n.get(f, '') for f in ('HOUSENUM', 'ADDNUM1', 'ADDNUM2')):
                        note('fractional_number', n)
                    if any('-' in n.get(f, '') for f in ('HOUSENUM', 'ADDNUM1', 'ADDNUM2')):
                        note('hyphenated_number', n)
                    for number, typ, catalog in (('HOUSENUM', 'HOUSETYPE', 'AS_HOUSE_TYPES'), ('ADDNUM1', 'ADDTYPE1', 'AS_ADDHOUSE_TYPES'), ('ADDNUM2', 'ADDTYPE2', 'AS_ADDHOUSE_TYPES')):
                        codes = {r['ID'] for r in index.meta['types'][catalog]}
                        if n.get(number) and not n.get(typ):
                            note('number_without_type_' + number, n)
                        if n.get(typ) and n[typ] not in codes:
                            note('unknown_type_' + typ, n)
                name = source.one(region, 'AS_REESTR_OBJECTS', False)
                if name:
                    print(f'[{region}] registry coverage', flush=True)
                    for row in source.iter(name):
                        stats['registry_rows'] += 1
                        if row.get('ISACTIVE') != '1':
                            continue
                        level = int(row['LEVELID'])
                        stats['registry_active_level_' + str(level)] += 1
                        if level in (1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 14, 15, 16):
                            oid = int(row['OBJECTID'])
                            i = index.find_oid(region, oid)
                            if i is None:
                                note('registry_target_missing_in_index', {'oid': oid, 'level': level})
                            elif index.raw(region, i)[5] != level:
                                note('registry_level_mismatch', {'oid': oid, 'level': level})
                reports[region] = {'counts': dict(stats), 'samples': examples}
        Path(args.output).write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding='utf-8')
    finally:
        source.close()


if __name__ == '__main__':
    main()
