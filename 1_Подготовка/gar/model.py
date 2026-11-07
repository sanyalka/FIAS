"""Small NumPy neural BIO tagger. No downloaded models or PyTorch at runtime."""
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
from .dataset import tokenize

FEATURE_VERSION = 1
CONTEXT = 2
FEATURES = 16


def token_features(token, buckets):
    value = '^' + token.casefold() + '$'
    grams = ['word:' + value]
    grams += [value[i:i + n] for n in (2, 3) for i in range(max(0, len(value) - n + 1))]
    grams += ['digit:' + str(token.isdigit()), 'punct:' + str(not any(c.isalnum() for c in token))]
    # Reserve padding ID 0. Stable hashing across Python processes and Colab.
    return [1 + int.from_bytes(hashlib.blake2s(g.encode(), digest_size=4).digest(), 'little') % (buckets - 1) for g in grams[:FEATURES]]


def encode(tokens, buckets):
    base = np.zeros((len(tokens) + 2 * CONTEXT, FEATURES), dtype=np.int32)
    for i, token in enumerate(tokens):
        features = token_features(token, buckets)
        base[i + CONTEXT, :len(features)] = features
    if not tokens:
        return np.zeros((0, 2 * CONTEXT + 1, FEATURES), dtype=np.int32)
    return np.stack([base[i:i + 2 * CONTEXT + 1] for i in range(len(tokens))])


class Tagger:
    def __init__(self, labels, buckets=8192, dim=24, hidden=64, seed=42):
        rng = np.random.default_rng(seed)
        self.config = {'feature_version': FEATURE_VERSION, 'labels': labels, 'buckets': buckets, 'dim': dim, 'hidden': hidden, 'context': CONTEXT, 'features': FEATURES}
        self.weights = {
            'embedding': rng.normal(0, 0.1, (buckets, dim)).astype(np.float32),
            'w1': rng.normal(0, (dim * (2 * CONTEXT + 1)) ** -0.5, (dim * (2 * CONTEXT + 1), hidden)).astype(np.float32),
            'b1': np.zeros(hidden, np.float32),
            'w2': rng.normal(0, hidden ** -0.5, (hidden, len(labels))).astype(np.float32),
            'b2': np.zeros(len(labels), np.float32),
        }
        self.weights['embedding'][0] = 0

    def forward(self, x):
        w = self.weights
        counts = np.maximum((x != 0).sum(axis=2, keepdims=True), 1).astype(np.float32)
        pooled = (w['embedding'][x].sum(axis=2) / counts).reshape(len(x), -1)
        hidden = np.tanh(pooled @ w['w1'] + w['b1'])
        logits = hidden @ w['w2'] + w['b2']
        logits -= logits.max(axis=1, keepdims=True)
        prob = np.exp(logits)
        prob /= prob.sum(axis=1, keepdims=True)
        return prob, (pooled, hidden, counts)

    def step(self, x, y, rate):
        w = self.weights
        prob, (pooled, hidden, counts) = self.forward(x)
        loss = float(-np.log(np.maximum(prob[np.arange(len(y)), y], 1e-9)).mean())
        grad = prob.copy()
        grad[np.arange(len(y)), y] -= 1
        grad /= len(y)
        dw2, db2 = hidden.T @ grad, grad.sum(axis=0)
        dh = (grad @ w['w2'].T) * (1 - hidden * hidden)
        dw1, db1 = pooled.T @ dh, dh.sum(axis=0)
        de = (dh @ w['w1'].T).reshape(len(x), 2 * CONTEXT + 1, -1) / counts
        # Scatter feature gradients; repeated IDs accumulate instead of overwriting.
        for slot in range(FEATURES):
            np.add.at(w['embedding'], x[:, :, slot].reshape(-1), (-rate * de).reshape(-1, self.config['dim']))
        w['embedding'][0] = 0
        w['w1'] -= rate * dw1
        w['b1'] -= rate * db1
        w['w2'] -= rate * dw2
        w['b2'] -= rate * db2
        return loss

    def save(self, path, extra=None):
        config = dict(self.config, **(extra or {}))
        path = Path(path)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, config=np.array(json.dumps(config, ensure_ascii=False)), **self.weights)
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            c = json.loads(str(data['config']))
            if c['feature_version'] != FEATURE_VERSION or c['context'] != CONTEXT or c['features'] != FEATURES:
                raise ValueError('Unsupported model feature schema')
            obj = cls(c['labels'], c['buckets'], c['dim'], c['hidden'])
            for name in obj.weights:
                if obj.weights[name].shape != data[name].shape:
                    raise ValueError('Invalid model tensor shape')
                obj.weights[name] = data[name].copy()
            obj.config = c
            return obj

    def predict(self, text):
        tokens = tokenize(text)
        if not tokens:
            return []
        x = encode([t[0] for t in tokens], self.config['buckets'])
        prob, _ = self.forward(x)
        return [{'token': token, 'start': start, 'end': end, 'label': self.config['labels'][int(p.argmax())], 'confidence': float(p.max())} for (token, start, end), p in zip(tokens, prob)]
