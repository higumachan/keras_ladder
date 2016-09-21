from keras.datasets import mnist
from layers.batch_nomalize import OnlyBatchNormalization, ScaleAndShift
import keras.backend as K
import keras
from layers.vanilla_conbinator import VanillaConbinator
import numpy as np
from toolz.dicttoolz import merge


def batch_nomalize_other_layer(xs):
    x, z = xs

    mean = (K.mean(z, 0, keepdims=True))
    std = (K.var(z, 0, keepdims=True) ** 0.5 + 1e-5)

    return (x - mean) / std


def batch_nomalize_other_layer_shape(xs):
    x, z = xs

    return x


def noised_model(sigma=1.0):
    unit_count_list = [
        28 * 28,
        1000,
        500,
        500,
        250,
        250,
        250,
    ]

    x = keras.layers.Input((28, 28))
    flatten_x = keras.layers.Flatten()(x)
    y = keras.layers.GaussianNoise(sigma)(flatten_x)

    zs_pre = []
    zs_ti = []
    zs = []

    forward_layers = []
    forward_scale_and_shift_layers = []

    zs_ti.append(y)
    for unit_count in unit_count_list[1:]:
        forward_layers.append(keras.layers.Dense(unit_count))
        z_ti = forward_layers[-1](y)
        forward_scale_and_shift_layers.append(OnlyBatchNormalization(mode=2))
        z_ti = forward_scale_and_shift_layers[-1](z_ti)
        z_ti = keras.layers.GaussianNoise(sigma)(z_ti)
        y = ScaleAndShift(mode=2)(z_ti)
        y = keras.layers.Activation('relu')(y)
        zs_ti.append(z_ti)
    output_noised = keras.layers.Dense(10, activation='softmax', name='output_noised')(y)

    y = flatten_x
    zs.append(y)
    zs_pre.append(y)
    for forward_layer, forward_scale_and_shift_layer in zip(forward_layers, forward_scale_and_shift_layers):
        z_pre = forward_layer(y)
        z = OnlyBatchNormalization(mode=2)(z_pre)
        y = forward_scale_and_shift_layer(z)
        y = keras.layers.Activation('relu')(y)
        zs_pre.append(z_pre)
        zs.append(z)
    y = keras.layers.Dense(10)(y)
    output = keras.layers.Activation('softmax', name='output')(y)


    zs_bn_error = []
    z_hat = OnlyBatchNormalization(mode=2)(y)
    for i, unit_count in list(enumerate(unit_count_list))[::-1]:
        u = keras.layers.Dense(unit_count)(z_hat)
        u = OnlyBatchNormalization(mode=2)(u)
        z_hat = VanillaConbinator()([zs_ti[i], u])
        z_bn_hat = keras.layers.merge(
            [z_hat, zs_pre[i]],
            mode=batch_nomalize_other_layer,
            output_shape=batch_nomalize_other_layer_shape,
        )
        z_bn_hat = z_hat
        zs_bn_error.append(keras.layers.merge(
            [zs[i], z_bn_hat],
            mode=lambda xs: xs[0] - xs[1],
            output_shape=lambda xs: xs[0],
            name='denoise_error_{}'.format(i),
        ))

    return keras.models.Model(x, [output, output_noised] + zs_bn_error)


def normal_model():

    x = keras.layers.Input((28, 28))
    y = keras.layers.Flatten()(x)
    y = keras.layers.Dense(1000)(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.Dense(500)(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.Dense(250)(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.Dense(250)(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.Dense(250)(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.Dense(10)(y)
    output = keras.layers.Activation('softmax')(y)

    return keras.models.Model(x, output)



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
    unlabeled_indices = list(set(range(len(y))) - set(labeled_indices))
    return (X[labeled_indices], to_onehot(y[labeled_indices])), (X[unlabeled_indices], to_zeros(y[unlabeled_indices]))


def train():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()


    lams = [1000, 10, 0.2, 0.2, 0.2, 0.2, 0.2][::-1]
    unit_count_list = [
        28 * 28,
        1000,
        500,
        500,
        250,
        250,
        250,
    ]
    y_train_denoise_error = dict([("denoise_error_{}".format(i), np.zeros((y_train.shape[0], uc))) for i, uc in zip(range(7), unit_count_list)])
    y_test_denoise_error = dict([("denoise_error_{}".format(i), np.zeros((y_test.shape[0], uc))) for i, uc in zip(range(7), unit_count_list)])
    denoise_error_objective = dict([("denoise_error_{}".format(i), "mse") for i in range(7)])
    denoise_error_weights = dict([("denoise_error_{}".format(i), w) for i, w in zip(range(7), lams)])

    model = noised_model(0.3)
    model.summary()
    model.compile(
        'adam', merge({
            "output": 'categorical_crossentropy',
            'output_noised': 'categorical_crossentropy',
        },
            denoise_error_objective
        ),
        metrics=['accuracy'],
        loss_weights=merge({
            'output': 0.0,
            'output_noised': 1.0,
        }, denoise_error_weights)
    )

    (X_train_labeled, y_train_labeled), (X_train_unlabeled, y_train_unlabeled) = split_labeled_unlabeled(X_train, y_train)

    print y_train_labeled.shape, y_train_unlabeled.shape

    for i in range(100):
        model.fit(
            X_train_labeled,
            merge({
                "output": y_train_labeled,
                "output_noised": y_train_labeled
            }, y_train_denoise_error),
            validation_data=(X_test, merge({"output": to_onehot(y_test, 10), "output_noised": to_onehot(y_test, 10), }, y_test_denoise_error)),
            nb_epoch=1
        )
        model.fit(
            X_train_unlabeled,
            merge({
                "output": y_train_unlabeled,
                "output_noised": y_train_unlabeled
            }, y_train_denoise_error),
            validation_data=(X_test, merge({"output": to_onehot(y_test, 10), "output_noised": to_onehot(y_test, 10), }, y_test_denoise_error)),
            nb_epoch=1
        )


if __name__ == '__main__':
    train()
