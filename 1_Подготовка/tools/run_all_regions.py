"""Resumable local administrative-only pipeline; streams XML directly from ZIP."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    xml = Path(ROOT / "xml_data/gar_xml.zip")
    if not xml.is_file():
        print("Ошибка: необходимый файл gar_xml.zip не найден!")
        exit(1)
    
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wait-for-index', action='store_true', help='Use when the already-running builder publishes the new index')
    p.add_argument('--source', default='xml_data/gar_xml.zip')
    p.add_argument('--output', default='artifacts/all_regions')
    p.add_argument('--date', default='2026-10-29')
    p.add_argument('--houses', type=int, default=3000)
    p.add_argument('--epochs', type=int, default=10)
    args = p.parse_args()
    out = (ROOT / args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / 'run_status.json'
    status = {'state': 'running', 'stage': 'build', 'started_utc': datetime.now(timezone.utc).isoformat(), 'configuration': vars(args)}
    def save():
        status['updated_utc'] = datetime.now(timezone.utc).isoformat()
        staging = out / 'run_status.tmp'
        staging.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
        staging.replace(status_path)
    def run(stage, arguments):
        status.update(stage=stage, state='running')
        save()
        print("Запущенный процесс: " + stage)
        with (out / (stage + '.log')).open('a', encoding='utf-8') as log:
            result = subprocess.run([sys.executable, '-u', *arguments], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{stage} failed ({result.returncode}); see {stage}.log')
    save()
    try:
        if args.wait_for_index:
            while not (out / 'gar.gari').exists():
                time.sleep(15)
                save()
        else:
            run('build', ['-m', 'gar.build', '--source', args.source, '--regions', 'all', '--hierarchy', 'adm', '--date', args.date, '--cache', str(out / 'build_cache'), '--output', str(out / 'gar.gari')])
        manifest_path = out / 'dataset' / 'manifest.json'
        if not manifest_path.exists():
            run('dataset', ['-m', 'gar.dataset', '--index', str(out / 'gar.gari'), '--output', str(out / 'dataset'), '--max-houses-per-region', str(args.houses), '--hierarchy', 'adm'])
        run('verify', ['-m', 'gar.verify', '--index', str(out / 'gar.gari'), '--dataset', str(out / 'dataset'), '--output', str(out / 'verification.json')])
        latest = out / 'model' / 'latest.npz'
        epochs_done = 0
        if latest.exists():
            sys.path.insert(0, str(ROOT))
            from gar.model import Tagger
            epochs_done = Tagger.load(latest).config.get('epoch', 0)
        if epochs_done < args.epochs:
            arguments = ['-m', 'gar.train', '--data', str(out / 'dataset'), '--output', str(out / 'model'), '--epochs', str(args.epochs - epochs_done), '--limit', '0', '--feature-cache', str(out / 'feature_cache')]
            if epochs_done:
                arguments += ['--resume', str(latest)]
            run('train', arguments)
        status.update(state='complete', stage='complete')
        save()
    except BaseException as exc:
        status.update(state='failed', error=str(exc))
        save()
        raise


if __name__ == '__main__':
    main()
