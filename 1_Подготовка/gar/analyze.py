import argparse
from pathlib import Path
import re
from .excel import read,write
from .index import Index
from .model import Tagger
from .xmlio import normalize

FIELDS=['region','district','city','locality','planning','street','house']
TITLES=['Регион','Район','Город','Населённый пункт','Планировочная структура','Улица','Дом']
LEVELS=[1,2,5,6,7,8]

def house(text):
    parts=re.split(r'\s+(?:к\.?|корпус|стр\.?|строение|соор\.?|сооружение)\s*',text,flags=re.I)
    markers=re.findall(r'\s+(к\.?|корпус|стр\.?|строение|соор\.?|сооружение)\s*',text,flags=re.I)
    if len(markers)>2: raise ValueError('Слишком много дополнений дома')
    main=parts[0].strip(' ,'); typ='2'
    match=re.match(r'^(д\.?|дом|зд\.?|здание|влд\.?|владение)\s+',main,re.I)
    if match:
        typ='5' if match[1].lower().startswith('з') else '1' if match[1].lower().startswith('в') else '2'
        main=main[match.end():]
    if not re.fullmatch(r'[\w/\-]+',main): raise ValueError('Не удалось однозначно разобрать номер дома')
    result={'kind':'house','HOUSENUM':main,'HOUSETYPE':typ}
    for i,(marker,value) in enumerate(zip(markers,parts[1:]),1):
        value=value.strip(' ,')
        if not re.fullmatch(r'[\w/\-]+',value): raise ValueError('Не удалось разобрать корпус или строение')
        result['ADDNUM'+str(i)]=value
        result['ADDTYPE'+str(i)]='1' if marker.lower().startswith('к') else '2' if marker.lower().startswith('ст') else '3'
    return result

class Analyzer:
    def __init__(self,index,model):
        self.index=index; self.model=model; self.cache={}
        self.type_aliases = {}
        for t in index.meta['types']['AS_ADDR_OBJ_TYPES']:
            key = (t['LEVEL'], normalize(t['SHORTNAME']).rstrip('.'))
            self.type_aliases.setdefault(key, set()).update((t['SHORTNAME'], t['NAME']))
        self.regions_by_name = {}
        for region,s in index.meta['regions'].items():
            for i in range(s['count']):
                if index.raw(region,i)[5]==1:
                    for name in self.names(index.node(region, i)):
                        self.regions_by_name.setdefault(name, []).append(region)
                    break
    def names(self,node):
        values={normalize(node['name']),normalize(node['type']+' '+node['name']),normalize(node['name']+' '+node['type'])}
        key = (str(node['level']), normalize(node['type']).rstrip('.'))
        for typ in self.type_aliases.get(key, ()):
            values.update((normalize(typ+' '+node['name']),normalize(node['name']+' '+typ)))
        return values
    def catalog(self,region):
        if region not in self.cache:
            self.cache.clear(); d={}
            for i in range(self.index.meta['regions'][region]['count']):
                raw=self.index.raw(region,i)
                if raw[6] or raw[7]&1: continue
                n=self.index.node(region,i)
                for name in self.names(n): d.setdefault((n['level'],name),set()).add(i)
            self.cache[region]=d
        return self.cache[region]
    def check(self,values):
        prediction=self.model.predict(', '.join(v for v in values if v))
        model_fields=', '.join(dict.fromkeys(p['label'][2:] for p in prediction if p['label']!='O'))
        def result(verdict,field,reason): return [verdict,field,reason,model_fields,self.index.meta['snapshot']]
        if not values[0]: return result('НЕВАЛИДНЫЙ','Регион','Не указан регион')
        regions = self.regions_by_name.get(normalize(values[0]), ())
        if len(regions)!=1: return result('НЕВАЛИДНЫЙ','Регион','Регион не найден либо неоднозначен в выбранном срезе')
        region=regions[0]; catalog=self.catalog(region); previous=None
        for col,level in enumerate(LEVELS):
            if not values[col]: continue
            found=catalog.get((level,normalize(values[col])),set())
            if not found: return result('НЕВАЛИДНЫЙ',TITLES[col],'Название или тип не найдены на указанном уровне в регионе')
            compatible={i for i in found if previous is None or set(self.index.chain(region,i)[:-1]) & previous}
            if not compatible: return result('НЕВАЛИДНЫЙ',TITLES[col],'Компонент не связан с ранее указанными компонентами')
            previous=compatible
        if not values[6]: return result('НЕВАЛИДНЫЙ','Дом','Для проверки полного адреса требуется дом')
        try: parsed=house(values[6])
        except ValueError as exc: return result('НЕВАЛИДНЫЙ','Дом',str(exc))
        found=set()
        for parent in previous or []:
            for child in self.index.object_descendants(region,parent,'adm'):
                found.update(i for i in self.index.find(region,parsed,parent=child) if not self.index.raw(region,i)[7]&1)
        if not found: return result('НЕВАЛИДНЫЙ','Дом','Дом с таким обозначением не найден в указанной иерархии')
        if len(found)>1: return result('НЕВАЛИДНЫЙ','Адрес','Недостаточно данных для однозначного подтверждения')
        missing_fields = {level: TITLES[col] for col, level in enumerate(LEVELS) if not values[col]}
        for ancestor in self.index.chain(region, next(iter(found)))[:-1]:
            level = self.index.raw(region, ancestor)[5]
            if level in missing_fields:
                node = self.index.node(region, ancestor)
                if level == 5:
                    reason = f"Пропущен город {node['name']}"
                else:
                    reason = f"В адресе пропущено звено ГАР: {node['type']} {node['name']}"
                return result('НЕВАЛИДНЫЙ', missing_fields[level], reason)
        return result('ВАЛИДНЫЙ','','Адрес подтверждён индексом')

def analyze_file(input_path, output, index_path, model_path, progress=None, cancelled=None):
    output=Path(output)
    if output.resolve()==Path(input_path).resolve(): raise ValueError('Выходной файл должен отличаться от входного')
    data=read(input_path); ids=next((cells for row,cells in data if row==1),{})
    if set(ids.values())!=set(FIELDS) or len(ids)!=len(FIELDS): raise ValueError('Нужен новый шаблон с идентификаторами в первой строке')
    columns={v:k for k,v in ids.items()}; rows=[]
    records=[(row,[cells.get(columns[f],'').strip() for f in FIELDS]) for row,cells in data if row>=3]
    records=[r for r in records if any(r[1])]
    if not records: raise ValueError('В шаблоне нет заполненных адресов')
    if progress: progress(0,len(records),'Загрузка индекса и модели…')
    with Index(index_path) as index:
        analyzer=Analyzer(index,Tagger.load(model_path))
        for done,(row,values) in enumerate(records,1):
            if cancelled and cancelled(): raise InterruptedError('Проверка отменена')
            rows.append([str(row)]+values+analyzer.check(values))
            if progress: progress(done,len(records),f'Строка {row}: {rows[-1][8]}')
    if cancelled and cancelled(): raise InterruptedError('Проверка отменена')
    temporary=output.with_name(output.name+'.tmp')
    try:
        write(temporary,['Строка Excel']+TITLES+['Вердикт','Поле','Причина','Компоненты по модели','Срез ГАР'],rows)
        temporary.replace(output)
    finally:
        if temporary.exists(): temporary.unlink()
    return {'total':len(rows),'valid':sum(r[8]=='ВАЛИДНЫЙ' for r in rows),'invalid':sum(r[8]=='НЕВАЛИДНЫЙ' for r in rows)}

def main():
    base=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(); p.add_argument('input'); p.add_argument('--output'); p.add_argument('--index',default=str(base/'data/gar.gari')); p.add_argument('--model',default=str(base/'data/best.npz')); a=p.parse_args()
    output=Path(a.output) if a.output else Path(a.input).with_name(Path(a.input).stem+'_результат.xlsx')
    analyze_file(a.input,output,a.index,a.model,lambda d,t,s:print(s,flush=True))
    print(output.resolve())
if __name__=='__main__': main()
