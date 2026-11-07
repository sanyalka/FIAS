from pathlib import Path
import shutil
import sys
import json

root=Path(__file__).resolve().parents[1]
prep=root/'1_Подготовка'; app=root/'2_Программа'
prep.mkdir(exist_ok=True);app.mkdir(exist_ok=True)
(app/'gar').mkdir(exist_ok=True);(app/'data').mkdir(exist_ok=True)
for name in ('__init__.py','analyze.py','excel.py','gui.py','index.py','model.py','dataset.py','xmlio.py'):
    shutil.copy2(root/'gar'/name,app/'gar'/name)
shutil.copy2(root/'outputs/address_template/Шаблон_адресов.xlsx',app/'Шаблон_адресов.xlsx')
for source,name in [('artifacts/all_regions/gar.gari','gar.gari'),('artifacts/all_regions/model/best.npz','best.npz')]:
    shutil.copy2(root/source,app/'data'/name)
runtime=app/'runtime';runtime.mkdir(exist_ok=True)
python=Path(sys.base_prefix)
for pattern in ('python*.exe','python*.dll','vcruntime*.dll','LICENSE*'):
    for file in python.glob(pattern):shutil.copy2(file,runtime/file.name)
for name in ('DLLs','tcl'):
    if (python/name).exists():shutil.copytree(python/name,runtime/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
shutil.copytree(python/'Lib',runtime/'Lib',dirs_exist_ok=True,ignore=shutil.ignore_patterns('site-packages','__pycache__','test','tests','idlelib','ensurepip'))
site=runtime/'Lib/site-packages';site.mkdir(exist_ok=True)
for file in (python/'Lib/site-packages').glob('numpy*'):
    if file.is_dir():shutil.copytree(file,site/file.name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','tests'))
    else:shutil.copy2(file,site/file.name)
(app/'Запустить.cmd').write_text('@echo off\r\ncd /d "%~dp0"\r\n"%~dp0runtime\\python.exe" -m gar.gui\r\nif errorlevel 1 pause\r\n',encoding='ascii')
(app/'README.txt').write_text('ПРОВЕРКА АДРЕСОВ ГАР\n\n1. Откройте Запустить.cmd.\n2. Нажмите «Сохранить шаблон», заполните его с третьей строки.\n3. Выберите заполненный Excel и путь результата.\n4. Нажмите «Проверить адреса».\n\nМожно переносить всю папку 2_Программа. Установка Python, СУБД и интернет не нужны. Не удаляйте runtime, gar и data.\n\nОтмена выполняется между адресами; загрузка индекса может занять время. Результат содержит первую обнаруженную ошибку. Модель выводит распознанные компоненты; вердикт подтверждается индексом ГАР. Поддерживается административная иерархия.\n\nПроверено в текущей Windows-среде. На отдельной Windows 10 и другом ПК пока не испытывалось.\n',encoding='utf-8')
names=['artifacts','gar','notebooks','outputs','tests','tools','xml_data','audit_gar.py','GAR_ANALYSIS.md','gar_audit.json','README.md','requirements-train.txt','TECHNICAL_SPEC.md','XML_RELATIONS.md','Проверить_адреса.cmd']
for name in names:
    source=root/name;target=prep/name
    if not source.exists():continue
    if not source.resolve().is_relative_to(root) or not target.resolve().is_relative_to(prep):raise ValueError('Unsafe move')
    if target.exists():raise ValueError(f'Target exists: {target}')
    shutil.move(str(source),str(target))
(root/'README.md').write_text('# Проект ГАР\n\n- **1_Подготовка** — XML/ZIP, исходный код, обучение, выборки, отчёты и материалы разработки. Команды подготовки запускать из этой папки.\n- **2_Программа** — автономная рабочая папка: GUI, Python/NumPy, модель, индекс и шаблон. Запуск: `2_Программа/Запустить.cmd`.\n\nПапка программы может переноситься целиком. Содержимое подготовки не требуется для проверки адресов.\n',encoding='utf-8')
print('Organized:',prep,app)
