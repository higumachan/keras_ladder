import keras
import keras.backend as K
import random
from toolz.dicttoolz import merge
from layers.batch_nomalize import OnlyBatchNormalization, ScaleAndShift
from layers.vanilla_conbinator import VanillaConbinator, LadderConbinator
from layers.pool_where import (
    PoolWhere
)
from keras.datasets import cifar10
from keras.utils import np_utils
from utils import (
    split_labeled_unlabeled,
    config_without_name
)
import numpy as np
from callbacks import (
    PrintEvaluate,
    PrintOutputValAccOnly
)


layers = [
    keras.layers.Convolution2D(3, 3, 3, border_mode='same', name='conv'),
    keras.layers.Convolution2D(96, 3, 3, border_mode='same', name='conv'),
    keras.layers.Convolution2D(96, 3, 3, border_mode='same', name='conv'),
    keras.layers.Convolution2D(96, 3, 3, border_mode='same', name='conv'),
    keras.layers.MaxPooling2D(name='pool'),
    keras.layers.Convolution2D(192, 3, 3, border_mode='same', name='conv'),
    keras.layers.Convolution2D(192, 3, 3, border_mode='same', name='conv'),
    keras.layers.Convolution2D(192, 3, 3, border_mode='same', name='conv'),
    keras.layers.MaxPooling2D(name='pool'),
    keras.layers.Convolution2D(192, 3, 3, border_mode='same', name='conv'),
    keras.layers.Convolution2D(192, 1, 1, border_mode='same', name='conv'),
    keras.layers.Convolution2D(192, 1, 1, border_mode='same', name='conv'),
]


def batch_nomalize_other_layer(xs):
    x, z = xs

    mean = K.mean(z, 0, keepdims=True)
    std = K.sqrt(K.var(z, 0, keepdims=True)) + 1e-10

    return (x - mean) / std


def batch_nomalize_other_layer_shape(xs):
    x, z = xs

    return x


def create_model():
    x = keras.layers.Input((3, 32, 32))
    y = x
    for layer in layers[1:]:
        y = layer.__class__(**config_without_name(layer.get_config()))(y)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.advanced_activations.LeakyReLU(0.1)(y)
    y = keras.layers.Convolution2D(10, 1, 1, border_mode='same')(y)
    y = keras.layers.GlobalAveragePooling2D()(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation('softmax')(y)

    model = keras.models.Model(x, y)
    print model.summary()
    return model


def create_noised_model(sigma):
    x = keras.layers.Input((3, 32, 32))
    y = x
    zs_ti = []
    zs_pre = []
    zs = []
    pooling_wheres = []
    forward_layers = []
    forward_scale_and_shift_layers = []

    for i, (layer, next_layer) in list(enumerate(zip(layers, layers[1:] + [None])))[1:]:
        forward_layers.append(layer.__class__(name='encoder_{}_{}'.format(layer.name, i), **config_without_name(layer.get_config())))
        z_ti  = forward_layers[-1](y)
        if isinstance(layer, keras.layers.MaxPooling2D):
            pooling_wheres.append(PoolWhere()([y, z_ti]))
        else:
            pooling_wheres.append(None)
        z_ti = OnlyBatchNormalization(mode=2, name='encoder_noised_bn_{}'.format(i))(z_ti)
        z_ti = keras.layers.GaussianNoise(sigma, name='encoder_noised_noise_{}'.format(i))(z_ti)
        forward_scale_and_shift_layers.append(ScaleAndShift(mode=2, name='encoder_scale_and_shift_{}'.format(i)))
        y = forward_scale_and_shift_layers[-1](z_ti)
        y = keras.layers.advanced_activations.LeakyReLU(0.1)(y)
        zs_ti.append(z_ti)
    y = keras.layers.Flatten()(y)
    z_ti = keras.layers.Dense(10)(y)
    z_ti = OnlyBatchNormalization(mode=2, name='encoder_noised_bn_{}'.format(i))(z_ti)
    z_ti = keras.layers.GaussianNoise(sigma, name='encoder_noised_noise_{}'.format(i))(z_ti)
    forward_scale_and_shift_layers.append(ScaleAndShift(mode=2, name='encoder_scale_and_shift_{}'.format(i)))
    y = forward_scale_and_shift_layers[-1](z_ti)
    y = keras.layers.Activation('softmax')(y)
    zs_ti.append(z_ti)
    output_noised = y

    y = x
    zs.append(y)
    zs_pre.appen(y)
    for i, (forward_layer, forward_scale_and_shift_layer) in  enumerate(zip(forward_layers, forward_scale_and_shift_layers)):
        z_pre = forward_layer(y)
        z = OnlyBatchNormalization(mode=2, name='encoder_clean_bn_{}'.format(i))(z_pre)
        y = forward_scale_and_shift_layer(z)
        y = keras.layers.advanced_activations.LeakyReLU(0.1)(y)
        zs_pre.append(z_pre)
        zs.append(z)
    y_flatten = keras.layers.Flatten()(y)
    z_pre = keras.layers.Dense(10)(y_flatten)
    z = OnlyBatchNormalization(mode=2, name='encoder_noised_bn_{}'.format(i))(z_pre)
    forward_scale_and_shift_layers.append(ScaleAndShift(mode=2, name='encoder_scale_and_shift_{}'.format(i)))
    y = forward_scale_and_shift_layers[-1](z_ti)
    y = keras.layers.Activation('softmax')(y)
    zs_pre.append(z_pre)
    zs.append(z)


    output = y
    
    zs_bn_error = []
    z_hat = output_noised

    u = keras.layers.Dense(y_flatten.get_output_shape_for()[1], name='decoder_dense_{}'.format(len(layers)))(z_hat)
    u = OnlyBatchNormalization(mode=2, name='decoder_bn_{}'.format(len(layers)))(u)
    z_hat = VanillaConbinator(name='decoder_conbinator_{}'.format(len(layers)))([zs_ti[-1], u])
    z_bn_hat = keras.layers.merge(
        [z_hat, zs_pre[-1]],
        mode=batch_nomalize_other_layer,
        output_shape=batch_nomalize_other_layer_shape,
        name='decoder_bn_pre_{}'.format(len(layers))
    )
    z_bn_error = keras.layers.merge(
        [zs[-1], z_bn_hat],
        mode=lambda xs: xs[0] - xs[1],
        output_shape=lambda xs: xs[0],
        name='denoise_error_{}'.format(len(layers)),
    )
    zs_bn_error.append(z_bn_error)

    for i, layer in enumerate(layers)[::-1]:
        if i != len(layers) - 1:
            if isinstance(layer, keras.layers.MaxPooling2D):
                u = keras.layers.UpSampling2D(size=(2, 2), name='decoder_unpool_{}'.format(i))(z_hat)
                u = keras.layers.merge([u, pooling_wheres[i]], mode='mul', name='decoder_unpool_where_{}'.format(i))
            else:
                u = layer.__class__(name='decoder_{}_{}'.format(layer.name, i), **config_without_name(layer.get_config))(z_hat)
                u = OnlyBatchNormalization(mode=2, name='decoder_bn_{}'.format(i))(u)
        else:
            u = z_hat
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
        z_bn_error = keras.layers.merge(
            [zs[i], z_bn_hat],
            mode=lambda xs: xs[0] - xs[1],
            output_shape=lambda xs: xs[0],
            name='denoise_error_{}'.format(i),
        )
        zs_bn_error.append(z_bn_error)

    return keras.models.Model(x, [output, output_noised] + zs_bn_error)


def generator_labeled(X_train_labeled, y_train_labeled, batch_size=100):
    while True:
        for i in range(len(X_train_labeled) / batch_size):
            yield X_train_labeled[i*batch_size:(i+1)*batch_size], y_train_labeled[i*batch_size:(i+1)*batch_size]


def generator(X_train_labeled, X_train_unlabeled, y_train_labeled, y_train_unlabeled, y_train_denoise_error, batch_size=100):
    gen = generator_labeled(X_train_labeled, y_train_labeled)
    while True:
        labeled_count = X_train_labeled.shape[0]
        indices = range(len(y_train_unlabeled))
        random.shuffle(indices)
        X_train_unlabeled = X_train_unlabeled[indices]
        for i in range(X_train_unlabeled.shape[0] / labeled_count):
            X_train_labeled_batch, y_train_labeled_batch = gen.next()
            yield (X_train_labeled_batch, merge({
                "output": y_train_labeled_batch,
                "output_noised": y_train_labeled_batch,
            }, y_train_denoise_error))
            yield (
                X_train_unlabeled[batch_size * i:batch_size * (i + 1)],
                merge({
                    "output": y_train_unlabeled[batch_size * i:batch_size * (i + 1)],
                    "output_noised": y_train_unlabeled[batch_size * i:batch_size * (i + 1)],
                }, y_train_denoise_error)
            )




def main():
    nb_classes = 10

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.0
    X_test /= 255.0

    """
    lams = [
        1000,
        10,
    ] + [0.0 if isinstance(layer, keras.layers.MaxPooling2D) else 0.1 for layer in layers]
    labeled_count = 4000

    X_labeled, y_labeled, X_unlabeled, y_unlabeled = split_labeled_unlabeled(X_train, y_train, labeled_count)

    y_train_all_denoise_error = dict([("denoise_error_{}".format(i), np.zeros((y_train.shape[0], layers))) for i, layer in enumerate(layers)])
    y_train_denoise_error = dict([("denoise_error_{}".format(i), np.zeros((labeled_count, layers))) for i, layer in enumerate(layers)])
    y_test_denoise_error = dict([("denoise_error_{}".format(i), np.zeros((y_test.shape[0], layers))) for i, layer in enumerate(layers)])
    denoise_error_objective = dict([("denoise_error_{}".format(i), "mse") for i in range(len(layers))])
    denoise_error_weights = dict([("denoise_error_{}".format(i), w) for i, w in zip(range(len(layers)), lams)])


   po = PrintOutputValAccOnly()
    """
    print y_train
    (X_labeled, y_labeled), (X_unlabeled, y_unlabeled) = split_labeled_unlabeled(X_train, y_train[:,0], 4000)
    X_labeled = X_labeled[:4000]
    y_test = np_utils.to_categorical(y_test, nb_classes)

    print X_train.shape, y_train.shape

    model = create_model()
    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_labeled, y_labeled, validation_data=(X_test, y_test), nb_epoch=50, batch_size=128)


if __name__ == '__main__':
    main()

