import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import xml.etree.ElementTree as ET

from gar.build import main as build
from gar.index import Index
from gar.xmlio import current, normalize


def xml(path, data):
    root = ET.Element('ROOT')
    for row in data:
        ET.SubElement(root, 'ROW', {k: str(v) for k, v in row.items()})
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'xml'
        (self.source / '01').mkdir(parents=True)
        self.common = dict(ISACTIVE='1', ISACTUAL='1', STARTDATE='2020-01-01', ENDDATE='2079-06-06')
        for kind in ('AS_HOUSE_TYPES', 'AS_ADDHOUSE_TYPES', 'AS_OBJECT_LEVELS', 'AS_ADDR_OBJ_TYPES'):
            self.write(kind, [], '')
        self.write('AS_ADDR_OBJ', [dict(self.common, OBJECTID=1, NAME='Регион', TYPENAME='обл.', LEVEL=1), dict(self.common, OBJECTID=2, NAME='Ленина', TYPENAME='ул.', LEVEL=8), dict(self.common, OBJECTID=3, NAME='Первый', TYPENAME='мкр.', LEVEL=7), dict(self.common, OBJECTID=4, NAME='Ленина', TYPENAME='ул.', LEVEL=8)])
        self.write('AS_HOUSES', [dict(self.common, OBJECTID=10, HOUSENUM='1', HOUSETYPE=2), dict(self.common, OBJECTID=11, HOUSENUM='1', HOUSETYPE=2, ADDNUM1='А', ADDTYPE1=1), dict(self.common, OBJECTID=12, HOUSENUM='1', HOUSETYPE=2)])
        self.edges = [dict(self.common, OBJECTID=o, PARENTOBJID=p) for o, p in [(1, 0), (2, 1), (3, 1), (4, 3), (10, 2), (11, 3), (12, 4)]]
        self.write('AS_ADM_HIERARCHY', self.edges)
        self.out = self.root / 'test.gari'

    def tearDown(self):
        self.temp.cleanup()

    def write(self, kind, data, region='01'):
        xml(self.source / region / (kind + '_20261029_test.XML'), data)

    def build(self):
        with contextlib.redirect_stdout(io.StringIO()):
            build(['--source', str(self.source), '--regions', '01', '--date', '2026-10-29', '--hierarchy', 'both', '--output', str(self.out)])

    def test_exact_search_homonyms_house_suffix_and_skipped_levels(self):
        self.build()
        with Index(self.out) as idx:
            street = dict(kind='object', name='Ленина', type='ул.')
            self.assertEqual(len(idx.find('01', street)), 2)
            house = dict(kind='house', HOUSENUM='1', HOUSETYPE='2')
            self.assertEqual(len(idx.match('01', [street, house])), 2)
            plan = idx.node('01', idx.find_oid('01', 3))
            corpus = dict(house, ADDNUM1='А', ADDTYPE1='1')
            self.assertEqual(len(idx.match('01', [plan, corpus])), 1)
            self.assertEqual(idx.match('01', [street, corpus]), [])
            self.assertEqual(len(idx.match('01', [plan, house])), 1)
            with self.assertRaises(ValueError):
                idx.chain('01', 0, 'mun')
            with self.assertRaises(ValueError):
                idx.match('01', [house], 'mun')

    def test_two_independent_hierarchies(self):
        mun = [dict(r) for r in self.edges]
        for r in mun:
            if r['OBJECTID'] == 10:
                r['PARENTOBJID'] = 3
        self.write('AS_MUN_HIERARCHY', mun)
        self.build()
        with Index(self.out) as idx:
            house = idx.find_oid('01', 10)
            self.assertEqual([idx.raw('01', i)[0] for i in idx.chain('01', house, 'adm')], [1, 2, 10])
            self.assertEqual([idx.raw('01', i)[0] for i in idx.chain('01', house, 'mun')], [1, 3, 10])

    def test_cycles_missing_parents_duplicates_quarantined(self):
        for r in self.edges:
            if r['OBJECTID'] == 2:
                r['PARENTOBJID'] = 10
            if r['OBJECTID'] == 3:
                r['PARENTOBJID'] = 999
        self.edges.append(dict(self.edges[-1]))
        self.write('AS_ADM_HIERARCHY', self.edges)
        self.build()
        with Index(self.out) as idx:
            for oid in (2, 3, 4, 10, 11, 12):
                with self.assertRaises(ValueError):
                    idx.chain('01', idx.find_oid('01', oid))

    def test_checksum_corruption(self):
        self.build()
        content = bytearray(self.out.read_bytes())
        content[-1] ^= 1
        self.out.write_bytes(content)
        with self.assertRaisesRegex(ValueError, 'Corrupt'):
            Index(self.out)

    def test_temporal_half_open_and_inactive(self):
        self.assertTrue(current(self.common, '2026-10-29'))
        self.assertFalse(current(dict(self.common, ENDDATE='2026-10-29'), '2026-10-29'))
        self.assertFalse(current(dict(self.common, ISACTIVE='0'), '2026-10-29'))
        self.assertEqual(normalize('  Ёлка  А '), 'ёлка а')
        self.assertNotEqual(normalize('ёлка'), normalize('елка'))

    def test_mixed_export_dates_rejected(self):
        path = next((self.source / '01').glob('AS_HOUSES*'))
        path.rename(path.with_name(path.name.replace('20261029', '20261028')))
        with self.assertRaisesRegex(ValueError, 'Mixed snapshot'):
            self.build()

    def test_zip_all_regions_and_checkpoint_resume(self):
        self.build()
        expected = self.out.read_bytes()
        archive = self.root / 'source.zip'
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            for file in self.source.rglob('*.XML'):
                z.write(file, file.relative_to(self.source).as_posix())
        command = ['--source', str(archive), '--regions', 'all', '--date', '2026-10-29', '--cache', str(self.root / 'cache'), '--output', str(self.out)]
        with contextlib.redirect_stdout(io.StringIO()):
            build(command)
        self.assertEqual(expected, self.out.read_bytes())
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            build(command)
        self.assertIn('resume completed shard', log.getvalue())
        self.assertEqual(expected, self.out.read_bytes())

    def test_administrative_mode_ignores_municipal_file(self):
        self.write('AS_MUN_HIERARCHY', self.edges)
        with contextlib.redirect_stdout(io.StringIO()):
            build(['--source', str(self.source), '--regions', '01', '--date', '2026-10-29', '--hierarchy', 'adm', '--output', str(self.out)])
        with Index(self.out) as index:
            self.assertEqual(index.meta['regions']['01']['hierarchies'], {'adm': True, 'mun': False})


if __name__ == '__main__':
    unittest.main()
