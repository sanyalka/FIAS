"""Small dependency-free XLSX reader/writer for address input and results."""
import posixpath
import zipfile
import xml.etree.ElementTree as E

NS='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
def tag(x): return '{'+NS+'}'+x
def read(path):
    with zipfile.ZipFile(path) as z:
        strings=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            strings=[''.join(x.itertext()) for x in E.fromstring(z.read('xl/sharedStrings.xml'))]
        book=E.fromstring(z.read('xl/workbook.xml'))
        sheet=next((s for s in book.find(tag('sheets')) if s.get('name')=='Адреса'),None)
        if sheet is None: raise ValueError('Нет листа Адреса')
        rel=sheet.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
        target=next(r.get('Target') for r in E.fromstring(z.read('xl/_rels/workbook.xml.rels')) if r.get('Id')==rel)
        target=target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/'+target)
        result=[]
        for row in E.fromstring(z.read(target)).find(tag('sheetData')):
            cells={}
            for c in row:
                col=''.join(v for v in c.get('r') if v.isalpha()); i=0
                for v in col: i=i*26+ord(v)-64
                if c.find(tag('f')) is not None: raise ValueError('Формулы во входных адресах не поддерживаются')
                v=c.findtext(tag('v'),'')
                if c.get('t')=='s': v=strings[int(v)]
                elif c.get('t')=='inlineStr': v=''.join(c.find(tag('is')).itertext())
                cells[i-1]=v
            result.append((int(row.get('r')),cells))
        return result

def write(path, headers, rows):
    sheet=E.Element(tag('worksheet')); data=E.SubElement(sheet,tag('sheetData'))
    for ri,values in enumerate([headers]+rows,1):
        row=E.SubElement(data,tag('row'),r=str(ri))
        for ci,value in enumerate(values):
            num=ci+1; col=''
            while num: num,rem=divmod(num-1,26); col=chr(65+rem)+col
            cell=E.SubElement(row,tag('c'),r=f'{col}{ri}',t='inlineStr')
            E.SubElement(E.SubElement(cell,tag('is')),tag('t')).text=str(value)
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml','<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>')
        z.writestr('_rels/.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        z.writestr('xl/workbook.xml','<workbook xmlns="'+NS+'" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Результаты" sheetId="1" r:id="rId1"/></sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>')
        z.writestr('xl/worksheets/sheet1.xml',E.tostring(sheet,encoding='utf-8',xml_declaration=True))
