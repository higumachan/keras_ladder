import keras
import keras.backend as K


def getwhere(x):
    ''' Calculate the "where" mask that contains switches indicating which
    index contained the max value when MaxPool2D was applied.  Using the
    gradient of the sum is a nice trick to keep everything high level.'''
    y_prepool, y_postpool = x
    return K.gradients(K.sum(y_postpool), y_prepool)


def PoolWhere():
    return lambda xs: keras.layers.merge(xs, mode=getwhere, output_shape=lambda xs: xs[0])

