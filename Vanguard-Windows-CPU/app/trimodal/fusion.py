"""Probability-level fusion, with an explicit common label contract."""
import numpy as np

CLASSES = ('negative', 'neutral', 'positive')
FACE_CLASSES = ('angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised')

def softmax(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError('Expected a finite logits vector')
    exp = np.exp(x - x.max())
    return exp / exp.sum()

def validate_probability(values, size=3):
    p = np.asarray(values, dtype=np.float64)
    if p.shape != (size,) or not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1, atol=1e-6):
        raise ValueError('Expected a nonnegative normalized probability vector')
    return p

def face_to_valence(probabilities):
    p = validate_probability(probabilities, 7)
    mapped = np.array([p[0] + p[1] + p[2] + p[5], p[4], p[3]])
    if mapped.sum() < 1e-8:
        raise ValueError('Insufficient non-surprise probability for three-class mapping')
    return mapped / mapped.sum(), float(p[6])

def fuse(probabilities, weights):
    names = ('image', 'speech', 'egg')
    if set(probabilities) != set(names) or set(weights) != set(names):
        raise ValueError('All three modalities are required')
    w = np.array([weights[n] for n in names], dtype=float)
    if not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError('Weights must be finite, nonnegative and not all zero')
    rows = np.stack([validate_probability(probabilities[n]) for n in names])
    return (w / w.sum()) @ rows
