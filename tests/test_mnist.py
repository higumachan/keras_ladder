from keras.datasets import mnist
from layers.batch_nomalize import OnlyBatchNormalization, ScaleAndShift
import keras.backend as K
import keras
from layers.vanilla_conbinator import VanillaConbinator, LadderConbinator
import numpy as np
from toolz.dicttoolz import merge
from keras.utils.visualize_util import plot
import random


class PrintOutputValAccOnly(keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs={}):
        print("val_output_acc:{val_output_acc}, val_output_noised_acc:{val_output_noised_acc}".format(**logs))


class PrintEvaluate(keras.callbacks.Callback):
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train


    def on_epoch_end(self, batch, logs={}):
        evaluate_result = self.model.evaluate(
                self.X_train, 
                self.y_train
                )
        print evaluate_result
        print(
            "train_output_acc:{output_acc}, train_output_noised_acc:{output_noised_acc}".format(
                **dict(zip(self.model.metrics_names, evaluate_result))
            )
        )
        print("val_output_acc:{val_output_acc}, val_output_noised_acc:{val_output_noised_acc}".format(**logs))


def batch_nomalize_other_layer(xs):
    x, z = xs

    mean = (K.mean(z, 0, keepdims=True))
    std = K.sqrt(K.var(z, 0, keepdims=True)) + 1e-10

    return (x - mean) / std


def batch_nomalize_other_layer_shape(xs):
    x, z = xs

    return x


def noised_model(sigma=1.0):
    unit_count_list = [
        28 * 28,
        1000,
        500,
        250,
        250,
        250,
        10
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
    for i, unit_count in list(enumerate(unit_count_list))[1:]:
        forward_layers.append(keras.layers.Dense(unit_count, bias=False, name='encoder_{}'.format(i)))
        z_ti = forward_layers[-1](y)
        z_ti = OnlyBatchNormalization(mode=2, name='encoder_noised_bn_{}'.format(i))(z_ti)
        z_ti = keras.layers.GaussianNoise(sigma, name='encoder_noised_noise_{}'.format(i))(z_ti)
        forward_scale_and_shift_layers.append(ScaleAndShift(mode=2, name='encoder_scale_and_shift_{}'.format(i)))
        y = forward_scale_and_shift_layers[-1](z_ti)
        if i != len(unit_count_list) - 1:
            y = keras.layers.Activation('relu', name='encoder_noised_{}'.format(i))(y)
        else:
            y = keras.layers.Activation('softmax', name='output_noised')(y)
        zs_ti.append(z_ti)
    output_noised = y

    y = flatten_x
    zs.append(y)
    zs_pre.append(y)
    for i, (forward_layer, forward_scale_and_shift_layer) in enumerate(zip(forward_layers, forward_scale_and_shift_layers)):
        i += 1
        z_pre = forward_layer(y)
        z = OnlyBatchNormalization(mode=2, name='encoder_clean_bn_{}'.format(i))(z_pre)
        y = forward_scale_and_shift_layer(z)
        if i != len(unit_count_list) - 1:
            y = keras.layers.Activation('relu', name='encoder_clean_{}'.format(i))(y)
        else:
            y = keras.layers.Activation('softmax', name='output')(y)
        zs_pre.append(z_pre)
        zs.append(z)
    output = y


    zs_bn_error = []
    z_hat = output_noised
    for i, unit_count in list(enumerate(unit_count_list))[::-1]:
        if i != len(unit_count_list) - 1:
            u = keras.layers.Dense(unit_count, bias=False, name='decoder_dense_{}'.format(i))(z_hat)
        else:
            u = z_hat
        u = OnlyBatchNormalization(mode=2, name='decoder_bn_{}'.format(i))(u)
        z_hat = VanillaConbinator(name='decoder_conbinator_{}'.format(i))([zs_ti[i], u])
        if i != 0:
            z_bn_hat = keras.layers.merge(
                [z_hat, zs_pre[i]],
                mode=batch_nomalize_other_layer,
                output_shape=batch_nomalize_other_layer_shape,
                name='decoder_bn_pre_{}'.format(i)
            )
        else:
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
    y = keras.layers.Dense(10, activation='softmax', name='output')(y)
    output = y

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


def generator(X_train_labeled, X_train_unlabeled, y_train_labeled, y_train_unlabeled, y_train_denoise_error):
    while True:
        labeled_count = X_train_labeled.shape[0]
        indices = range(len(y_train_unlabeled))
        random.shuffle(indices)
        X_train_unlabeled = X_train_unlabeled[indices]
        for i in range(X_train_unlabeled.shape[0] / labeled_count):
            yield (X_train_labeled, merge({
                "output": y_train_labeled,
                "output_noised": y_train_labeled,
            }, y_train_denoise_error))
            yield (
                X_train_unlabeled[labeled_count * i:labeled_count * (i + 1)],
                merge({
                    "output": y_train_unlabeled[labeled_count * i:labeled_count * (i + 1)],
                    "output_noised": y_train_unlabeled[labeled_count * i:labeled_count * (i + 1)],
                }, y_train_denoise_error)
            )


def train():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255


    labeled_count = 100
    lams = [1000, 10, 0.1, 0.1, 0.1, 0.1, 0.1]
    unit_count_list = [
        28 * 28,
        1000,
        500,
        250,
        250,
        250,
        10
    ]
    y_train_all_denoise_error = dict([("denoise_error_{}".format(i), np.zeros((y_train.shape[0], uc))) for i, uc in enumerate(unit_count_list)])
    y_train_denoise_error = dict([("denoise_error_{}".format(i), np.zeros((labeled_count, uc))) for i, uc in enumerate(unit_count_list)])
    y_test_denoise_error = dict([("denoise_error_{}".format(i), np.zeros((y_test.shape[0], uc))) for i, uc in enumerate(unit_count_list)])
    denoise_error_objective = dict([("denoise_error_{}".format(i), "mse") for i in range(7)])
    denoise_error_weights = dict([("denoise_error_{}".format(i), w) for i, w in zip(range(7), lams)])

    print denoise_error_weights

    model = noised_model(0.3)

    plot(model, 'model.png', show_shapes=True)

    model.summary()
    model.compile(
        keras.optimizers.Adam(0.002), merge({
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
    print np.sum(y_train_labeled, axis=0)

    print y_train_labeled.shape, y_train_unlabeled.shape

    def scheduler(index):
        return 0.002 if index < 100 else max(0.0, 0.002 * (150 - index) / 50.0)

    """
    nmodel = normal_model()
    nmodel.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
    nmodel.fit(
        X_train_labeled,
        y_train_labeled,
        validation_data=(X_test, to_onehot(y_test)),
        nb_epoch=150,
        callbacks=[keras.callbacks.LearningRateScheduler(scheduler)]
    )
    """

    model.fit_generator(
        generator(X_train_labeled, X_train_unlabeled, y_train_labeled, y_train_unlabeled, y_train_denoise_error),
        (y_train.shape[0] - labeled_count) * 2,
        nb_epoch=150,
        validation_data=(
            X_test,
            merge({
                "output": to_onehot(y_test),
                "output_noised": to_onehot(y_test),
            }, y_test_denoise_error)
        ),
        callbacks=[
            keras.callbacks.LearningRateScheduler(scheduler),
            PrintOutputValAccOnly(),
            PrintEvaluate(
                X_train, 
                merge({
                    "output": to_onehot(y_train),
                    "output_noised": to_onehot(y_train),
                    },y_train_all_denoise_error)
                ),
            ],
        verbose=1,
        )


if __name__ == '__main__':
    train()
