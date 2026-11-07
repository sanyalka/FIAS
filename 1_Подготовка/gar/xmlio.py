import contextlib
import hashlib
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

NAME = re.compile(r"^(AS_[A-Z_]+)_(\d{8})_.*\.XML$", re.I)


def normalize(value):
    # Deliberately keep ё, punctuation, numbers and house suffixes distinct.
    return " ".join(value.casefold().split())


def digest(value):
    return int.from_bytes(hashlib.blake2b(value.encode('utf-8'), digest_size=8).digest(), 'little')


def rows(stream):
    """Release children from the root as well as clearing their contents."""
    parser = ET.iterparse(stream, events=('start', 'end'))
    _, root = next(parser)
    for event, element in parser:
        if event == 'end' and element is not root:
            if element.attrib:
                yield dict(element.attrib)
            element.clear()
            root.clear()


class Source:
    def __init__(self, path):
        self.path = Path(path)
        self.zip = zipfile.ZipFile(path) if self.path.is_file() else None
        self.names = self.zip.namelist() if self.zip else [p.relative_to(self.path).as_posix() for p in self.path.rglob('*') if p.is_file() and p.suffix.lower() == '.xml']

    def close(self):
        if self.zip:
            self.zip.close()

    def files(self, region, kind):
        result = []
        for name in self.names:
            p = Path(name.replace('\\', '/'))
            match = NAME.match(p.name)
            if match and match[1].upper() == kind and p.parent.as_posix() == (region or '.'):
                result.append(name)
        return sorted(result)

    def one(self, region, kind, required=True):
        found = self.files(region, kind)
        if len(found) > 1 or (required and not found):
            raise ValueError(f'Expected one {region}/{kind}, got {found}')
        return found[0] if found else None

    @contextlib.contextmanager
    def open(self, name):
        with (self.zip.open(name) if self.zip else open(self.path / name, 'rb')) as f:
            yield f

    def iter(self, name):
        with self.open(name) as stream:
            yield from rows(stream)

    def size(self, name):
        return self.zip.getinfo(name).file_size if self.zip else (self.path / name).stat().st_size


def current(row, date, actual=True):
    return (row.get('ISACTIVE') == '1' and
            (not actual or row.get('ISACTUAL') == '1') and
            row.get('STARTDATE', '0000-00-00') <= date < row.get('ENDDATE', '9999-12-31'))
