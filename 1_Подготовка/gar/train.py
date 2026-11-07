"""Train the same CPU baseline locally and in Google Colab."""
import argparse
import json
import hashlib
import heapq
from pathlib import Path
import time
import numpy as np
from .dataset import FIELDS
from .model import Tagger, encode


def load_data(path, labels, buckets, limit, cache_dir=None):
    xs, ys, lengths = [], [], []
    label_ids = {label: i for i, label in enumerate(labels)}
    if cache_dir is not None and not limit:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        stamp = {'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns, 'buckets': buckets, 'labels': labels, 'features': 16, 'context': 2}
        marker = cache_dir / (path.stem + '.json')
        xp, yp, lp = [cache_dir / (path.stem + suffix) for suffix in ('.x.npy', '.y.npy', '.lengths.npy')]
        if marker.exists() and json.loads(marker.read_text()) == stamp and all(p.exists() for p in (xp, yp, lp)):
            return np.load(xp, mmap_mode='r'), np.load(yp, mmap_mode='r'), np.load(lp, mmap_mode='r')
        lengths = []
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                if len(row['tokens']) != len(row['labels']):
                    raise ValueError('Token/label length mismatch')
                lengths.append(len(row['tokens']))
        total = sum(lengths)
        if not total:
            raise ValueError(f'Empty split: {path}')
        x = np.lib.format.open_memmap(xp, mode='w+', dtype=np.int32, shape=(total, 5, 16))
        y = np.lib.format.open_memmap(yp, mode='w+', dtype=np.int64, shape=(total,))
        start = 0
        with path.open(encoding='utf-8') as stream:
            for i, line in enumerate(stream):
                row = json.loads(line)
                size = len(row['tokens'])
                x[start:start + size] = encode(row['tokens'], buckets)
                y[start:start + size] = [label_ids[v] for v in row['labels']]
                start += size
                if i and i % 25000 == 0:
                    print(f'Encoded {path.stem}: {i} sentences', flush=True)
        x.flush()
        y.flush()
        np.save(lp, np.array(lengths, dtype=np.int32))
        marker.write_text(json.dumps(stamp), encoding='utf-8')
        return x, y, np.array(lengths, dtype=np.int32)
    def selected_rows():
        with path.open(encoding='utf-8') as f:
            if not limit:
                for line in f:
                    yield json.loads(line)
                return
            reservoir = []
            for i, line in enumerate(f):
                rank = int.from_bytes(hashlib.blake2s(line.encode(), digest_size=8).digest(), 'little')
                heapq.heappush(reservoir, (-rank, i, line))
                if len(reservoir) > limit:
                    heapq.heappop(reservoir)
            for _, _, line in sorted(reservoir):
                yield json.loads(line)
    for row in selected_rows():
        if len(row['tokens']) != len(row['labels']):
            raise ValueError('Token/label length mismatch')
        x = encode(row['tokens'], buckets)
        y = np.array([label_ids[v] for v in row['labels']], dtype=np.int64)
        xs.append(x)
        ys.append(y)
        lengths.append(len(y))
    if not xs:
        raise ValueError(f'Empty split: {path}; generate more houses or regions')
    return np.concatenate(xs), np.concatenate(ys), lengths


def evaluate(model, data, batch):
    x, y, lengths = data
    count = len(model.config['labels'])
    matrix = np.zeros((count, count), dtype=np.int64)
    pred = np.concatenate([model.forward(x[i:i + batch])[0].argmax(axis=1) for i in range(0, len(y), batch)])
    np.add.at(matrix, (y, pred), 1)
    correct, start = 0, 0
    for size in lengths:
        correct += int(np.array_equal(y[start:start + size], pred[start:start + size]))
        start += size
    tp = np.diag(matrix)
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    supported = matrix.sum(axis=1) > 0
    f1 = 2 * tp / np.maximum(denominator, 1)
    return {'token_accuracy': float((pred == y).mean()), 'sentence_exact': correct / len(lengths), 'macro_f1_supported_labels': float(f1[supported].mean()), 'tokens': len(y), 'sentences': len(lengths), 'per_label': {label: {'support': int(matrix[i].sum()), 'f1': float(f1[i])} for i, label in enumerate(model.config['labels'])}, 'confusion_matrix': matrix.tolist()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='artifacts/dataset')
    p.add_argument('--output', default='artifacts/model')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--learning-rate', type=float, default=0.15)
    p.add_argument('--limit', type=int, default=0, help='Limit sentences per split for smoke testing')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', help='Resume latest.npz; same data/parameters required')
    p.add_argument('--feature-cache', help='Disk-backed encoded arrays for training without loading the whole dataset into RAM')
    args = p.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0 or args.limit < 0:
        p.error('Invalid training parameters')
    started = time.perf_counter()
    data_path, output = Path(args.data), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((data_path / 'manifest.json').read_text(encoding='utf-8'))
    labels = ['O'] + [prefix + f for f in sorted(set(FIELDS.values())) for prefix in ('B-', 'I-')]
    model = Tagger.load(args.resume) if args.resume else Tagger(labels, seed=args.seed)
    signature = {'sampling': 'minhash_sentences_v1', 'seed': args.seed, 'limit': args.limit, 'batch_size': args.batch_size, 'learning_rate': args.learning_rate, 'files': {}}
    for split in ('train', 'validation', 'test'):
        h = hashlib.sha256()
        with (data_path / f'tagger_{split}.jsonl').open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
        signature['files'][split] = h.hexdigest()
    if args.resume and model.config.get('training_signature') != signature:
        raise ValueError('Resume data or training parameters changed')
    if args.resume and not (output / 'best.npz').exists():
        raise ValueError('Resume requires original output folder with best.npz')
    if model.config['labels'] != labels:
        raise ValueError('Resume label mismatch')
    train = load_data(data_path / 'tagger_train.jsonl', labels, model.config['buckets'], args.limit, args.feature_cache)
    val = load_data(data_path / 'tagger_validation.jsonl', labels, model.config['buckets'], args.limit, args.feature_cache)
    best = float(model.config.get('best_score', -1))
    history = []
    first = int(model.config.get('epoch', 0))
    for epoch in range(first, first + args.epochs):
        # Epoch-specific RNG makes continuation reproducible with plain SGD.
        order = np.random.default_rng(args.seed + epoch).permutation(len(train[1]))
        losses = []
        progress_time = time.perf_counter()
        for at in range(0, len(order), args.batch_size):
            ids = order[at:at + args.batch_size]
            losses.append(model.step(train[0][ids], train[1][ids], args.learning_rate))
            if time.perf_counter() - progress_time >= 30:
                print(json.dumps({'epoch': epoch + 1, 'tokens_done': at + len(ids), 'tokens_total': len(order)}), flush=True)
                progress_time = time.perf_counter()
        metrics = evaluate(model, val, args.batch_size)
        score = metrics['macro_f1_supported_labels']
        extra = {'epoch': epoch + 1, 'snapshot': manifest['snapshot'], 'seed': args.seed, 'best_score': max(best, score), 'training_limit': args.limit, 'training_signature': signature, 'role': 'component_tagger_not_address_validator'}
        if score > best:
            best = score
            model.save(output / 'best.npz', extra)
        model.save(output / 'latest.npz', extra)
        summary = {'epoch': epoch + 1, 'loss': float(np.mean(losses)), 'validation_accuracy': metrics['token_accuracy'], 'validation_macro_f1': score, 'validation_sentence_exact': metrics['sentence_exact']}
        history.append(summary)
        print(json.dumps(summary), flush=True)
    # Test is read once, after model selection by validation only.
    best_path = output / 'best.npz'
    if not best_path.exists():
        raise ValueError('Resume into the original output folder containing best.npz')
    chosen = Tagger.load(best_path)
    test = load_data(data_path / 'tagger_test.jsonl', labels, chosen.config['buckets'], args.limit, args.feature_cache)
    report = {'history': history, 'test': evaluate(chosen, test, args.batch_size), 'seconds': round(time.perf_counter() - started, 2), 'model_bytes': best_path.stat().st_size, 'data_manifest': manifest, 'warning': 'Synthetic held-out parent groups only; metrics do not establish real-world address validity accuracy.'}
    (output / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('seconds', 'model_bytes')}), flush=True)


if __name__ == '__main__':
    main()
