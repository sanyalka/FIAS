import tempfile
import contextlib
import io
import json
from pathlib import Path
import unittest
import numpy as np
from gar.model import Tagger, encode


class ModelTests(unittest.TestCase):
    def test_disk_feature_cache_matches_in_memory_encoding(self):
        from gar.train import load_data
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'train.jsonl'
            rows = [{'tokens': ['дом', str(i)], 'labels': ['O', 'B-HOUSE']} for i in range(5)]
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
            expected = load_data(path, ['O', 'B-HOUSE'], 64, 0)
            for _ in range(2):
                actual = load_data(path, ['O', 'B-HOUSE'], 64, 0, root / 'cache')
                for a, b in zip(expected, actual):
                    np.testing.assert_array_equal(a, b)
                for value in actual:
                    if isinstance(value, np.memmap):
                        value._mmap.close()

    def test_training_resume_matches_uninterrupted_run(self):
        from gar.train import main as train
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / 'dataset'
            data.mkdir()
            (data / 'manifest.json').write_text(json.dumps({'snapshot': '2026-10-29'}), encoding='utf-8')
            rows = [{'tokens': ['обл.', name], 'labels': ['B-REGION', 'I-REGION']} for name in ('Один', 'Два', 'Три', 'Четыре')]
            for split in ('train', 'validation', 'test'):
                (data / f'tagger_{split}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                train(['--data', str(data), '--output', str(root / 'a'), '--epochs', '2'])
                train(['--data', str(data), '--output', str(root / 'b'), '--epochs', '1'])
                train(['--data', str(data), '--output', str(root / 'b'), '--epochs', '1', '--resume', str(root / 'b' / 'latest.npz')])
            a, b = Tagger.load(root / 'a' / 'latest.npz'), Tagger.load(root / 'b' / 'latest.npz')
            for key in a.weights:
                np.testing.assert_array_equal(a.weights[key], b.weights[key])
            with self.assertRaisesRegex(ValueError, 'parameters changed'):
                train(['--data', str(data), '--output', str(root / 'b'), '--epochs', '1', '--batch-size', '8', '--resume', str(root / 'b' / 'latest.npz')])

    def test_gradient_matches_finite_difference_and_export(self):
        model = Tagger(['O', 'B-HOUSE'], buckets=64, dim=4, hidden=5)
        x = encode(['дом', '12'], 64)
        y = np.array([0, 1])
        def loss():
            prob, _ = model.forward(x)
            return -np.log(prob[np.arange(len(y)), y]).mean()
        for name, at in [('w1', (0, 0)), ('w2', (0, 1)), ('embedding', (int(x[0, 2, 0]), 0))]:
            original = model.weights[name][at].copy()
            epsilon = 0.001
            model.weights[name][at] = original + epsilon
            plus = loss()
            model.weights[name][at] = original - epsilon
            minus = loss()
            model.weights[name][at] = original
            expected = (plus - minus) / (2 * epsilon)
            saved = {k: v.copy() for k, v in model.weights.items()}
            model.step(x, y, 0.1)
            actual = (original - model.weights[name][at]) / 0.1
            self.assertAlmostEqual(float(actual), float(expected), delta=0.0002)
            model.weights = saved
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.npz'
            model.save(path)
            restored = Tagger.load(path)
            np.testing.assert_array_equal(model.forward(x)[0], restored.forward(x)[0])


if __name__ == '__main__':
    unittest.main()
