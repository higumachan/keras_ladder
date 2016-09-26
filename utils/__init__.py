import numpy as np
import toolz.dicttoolz as dz


def to_onehot(output, bins=10):
    return np.array(map(lambda x: [1.0 if i == x else 0.0 for i in range(bins)], output))


def to_zeros(output, bins=10):
    return np.zeros(output.shape + (bins,))


def split_labeled_unlabeled(X, y, labeled_count=100):
    if labeled_count % 10 != 0:
        raise Exception('labeled_count require multiples of 10')

    labeled_indices = []
    c = dict([(i, 0) for i in range(10)])
    for i, t in enumerate(y):
        if c[t] < labeled_count / 10:
            labeled_indices.append(i)
            c[t] += 1
        if min(c.values()) == labeled_count / 10:
            break
    return (X[labeled_indices], to_onehot(y)[labeled_indices]), (X, to_zeros(y))


def config_without_name(config):
    return dz.keyfilter(lambda x: x != 'name', config)

