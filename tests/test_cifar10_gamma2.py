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
    keras.layers.Convolution2D(24, 3, 3, border_mode='same', name='conv'),
    keras.layers.MaxPooling2D(name='pool'),
    keras.layers.Convolution2D(48, 3, 3, border_mode='same', name='conv'),
    keras.layers.MaxPooling2D(name='pool'),
    keras.layers.Convolution2D(48, 3, 3, border_mode='same', name='conv'),
    keras.layers.Convolution2D(48, 1, 1, border_mode='same', name='conv'),
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
    return model


def create_noised_model(sigma):
    x = keras.layers.Input((3, 32, 32))
    zs_ti = []
    zs_pre = []
    zs = []
    pooling_wheres = []
    forward_layers = []
    forward_scale_and_shift_layers = []

    # noised encoder
    y = keras.layers.GaussianNoise(sigma, name='input_noised')(x)
    zs_ti.append(y)
    for i, layer in list(enumerate(layers))[1:]:
        forward_layers.append(layer.__class__(name='encoder_{}_{}'.format(layer.name, i), **config_without_name(layer.get_config())))
        z_ti  = forward_layers[-1](y)
        if isinstance(layer, keras.layers.MaxPooling2D):
            pooling_wheres.append(PoolWhere()([y, z_ti]))
        else:
            pooling_wheres.append(None)
        
        # flatten and reshape
        z_ti_flat = keras.layers.Flatten()(z_ti)
        z_ti_flat = OnlyBatchNormalization(mode=1, name='encoder_noised_bn_{}'.format(i))(z_ti_flat)
        z_ti_flat = keras.layers.GaussianNoise(sigma, name='encoder_noised_noise_{}'.format(i))(z_ti_flat)
        forward_scale_and_shift_layers.append(ScaleAndShift(mode=2, name='encoder_scale_and_shift_{}'.format(i)))
        y = forward_scale_and_shift_layers[-1](z_ti_flat)
        y = keras.layers.Reshape(forward_layers[-1].output_shape[1:])(y)
        y = keras.layers.advanced_activations.LeakyReLU(0.1)(y)
        z_ti = keras.layers.Reshape(forward_layers[-1].output_shape[1:])(z_ti_flat)
        zs_ti.append(z_ti)
    y = keras.layers.Flatten()(y)
    forward_layers.append(keras.layers.Dense(10, name='encoder_noised_dense_{}'.format(len(layers))))
    z_ti = forward_layers[-1](y)
    z_ti = OnlyBatchNormalization(mode=2, name='encoder_noised_bn_{}'.format(len(layers)))(z_ti)
    z_ti = keras.layers.GaussianNoise(sigma, name='encoder_noised_noise_{}'.format(len(layers)))(z_ti)
    forward_scale_and_shift_layers.append(ScaleAndShift(mode=2, name='encoder_scale_and_shift_{}'.format(len(layers))))
    y = forward_scale_and_shift_layers[-1](z_ti)
    y = keras.layers.Activation('softmax', name='output_noised')(y)
    zs_ti.append(z_ti)
    output_noised = y

    # clean encoder
    y = x
    zs.append(y)
    zs_pre.append(y)
    for i, (forward_layer, forward_scale_and_shift_layer) in list(enumerate(zip(forward_layers, forward_scale_and_shift_layers)))[:-1]:
        z_pre = forward_layer(y)
        # flatten and reshape
        z_pre_flat = keras.layers.Flatten()(z_pre)
        z = OnlyBatchNormalization(mode=1, name='encoder_clean_bn_{}'.format(i))(z_pre_flat)
        y = forward_scale_and_shift_layer(z)
        y = keras.layers.Reshape(forward_layer.output_shape[1:])(y)
        y = keras.layers.advanced_activations.LeakyReLU(0.1)(y)
        z = keras.layers.Reshape(forward_layer.output_shape[1:])(z)
        zs_pre.append(z_pre)
        zs.append(z)
    flatten = keras.layers.Flatten()
    y_flatten = flatten(y)
    z_pre = forward_layers[-1](y_flatten)
    z = OnlyBatchNormalization(mode=2, name='encoder_clean_bn_{}'.format(len(forward_layers)))(z_pre)
    y = forward_scale_and_shift_layers[-1](z_ti)
    y = keras.layers.Activation('softmax', name='output')(y)
    zs_pre.append(z_pre)
    zs.append(z)

    # decoder
    output = y
    
    zs_bn_error = []
    zs_bn_error_shape = []
    z_hat = output_noised

    u = output_noised
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
    zs_bn_error_shape.append((None, 10,))

    for i, layer in list(enumerate(layers))[::-1]:
        if i != len(layers) - 1:
            if isinstance(layer, keras.layers.MaxPooling2D):
                #backward_layer = keras.layers.UpSampling2D(size=(2, 2), name='decoder_unpool_{}'.format(i))
                backward_layer = layers[i-1].__class__(name='decoder_{}_{}'.format(layers[i-1].name, i), **config_without_name(layers[i-1].get_config()))
                u = backward_layer(z_hat)
                print backward_layer.output_shape
            elif isinstance(layers[i+1], keras.layers.MaxPooling2D):
                backward_layer = keras.layers.UpSampling2D(size=(2, 2), name='decoder_unpool_{}'.format(i))
                u = backward_layer(z_hat)
                u = keras.layers.merge([u, pooling_wheres[i]], mode='mul', name='decoder_unpool_where_{}'.format(i))
            else:
                backward_layer = layer.__class__(name='decoder_{}_{}'.format(layer.name, i), **config_without_name(layer.get_config()))
                u = backward_layer(z_hat)
                u = keras.layers.Flatten()(u)
                u = OnlyBatchNormalization(mode=1, name='decoder_bn_{}'.format(i))(u)
                u = keras.layers.Reshape(backward_layer.output_shape[1:])(u)
        else:
            dense_size = flatten.output_shape[1]
            u = keras.layers.Dense(dense_size, name='decoder_dense_{}'.format(i))(z_hat)
            backward_layer = keras.layers.Reshape(forward_layers[-2].output_shape[1:])
            u = backward_layer(u)
        print zs_ti[i]._keras_shape, u._keras_shape
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
        zs_bn_error_shape.append(z_bn_error._keras_shape)

    return keras.models.Model(x, [output, output_noised] + zs_bn_error), zs_bn_error_shape[::-1]


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
            yield (
                    X_train_labeled_batch, 
                    merge({
                        "output": y_train_labeled_batch,
                        "output_noised": y_train_labeled_batch,
                        }, y_train_denoise_error)
                    )
            yield (
                X_train_unlabeled[batch_size * i:batch_size * (i + 1)],
                merge({
                    "output": y_train_unlabeled[batch_size * i:batch_size * (i + 1)],
                    "output_noised": y_train_unlabeled[batch_size * i:batch_size * (i + 1)],
                }, y_train_denoise_error)
            )


def main():
    nb_classes = 10
    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')

    """
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.0
    X_test /= 255.0
    """

    model, output_shapes = create_noised_model(0.50)
    print model.summary()
    print model.outputs
    print output_shapes

    # lambdas for denoising
    lams = [
        0,
        0.0,
        ] + [0.0 if isinstance(layer, keras.layers.MaxPooling2D) else 0.0 for layer in layers[2:]]+ [4.0]
    labeled_count = 4000

    (X_train_labeled, y_train_labeled), (X_train_unlabeled, y_train_unlabeled) = split_labeled_unlabeled(X_train, y_train[:, 0], labeled_count)


    y_train_all_denoise_error = dict([("denoise_error_{}".format(i), np.zeros((y_train.shape[0],) + shape[1:])) for i, shape in enumerate(output_shapes)])
    y_train_denoise_error = dict([("denoise_error_{}".format(i), np.zeros((100,) +  shape[1:])) for i, shape in enumerate(output_shapes)])
    y_test_denoise_error = dict([("denoise_error_{}".format(i), np.zeros((y_test.shape[0],) + shape[1:])) for i, shape in enumerate(output_shapes)])
    denoise_error_objective = dict([("denoise_error_{}".format(i), "mse") for i in range(len(output_shapes))])
    denoise_error_weights = dict([("denoise_error_{}".format(i), w) for i, w in zip(range(len(output_shapes)), lams)])


    y_test = np_utils.to_categorical(y_test, nb_classes)

    print X_train.shape, y_train.shape

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

    def scheduler(index):
        return 0.002 if index < 10 else max(0.0, 0.002 * (60 - index) / 20.0)

    model.fit_generator(
        generator(X_train_labeled, X_train_unlabeled, y_train_labeled, y_train_unlabeled, y_train_denoise_error),
        (X_train_unlabeled.shape[0]) * 2,
        nb_epoch=60,
        validation_data=(
            X_test,
            merge({
                "output": y_test,
                "output_noised": y_test,
            }, y_test_denoise_error)
        ),
        callbacks=[
            keras.callbacks.LearningRateScheduler(scheduler),
            PrintOutputValAccOnly(filepath="./results/ladderCNN_gamma/v1_flatten_log.txt"),
	    #keras.callbacks.EarlyStopping(monitor='val_loss', patience=1),
	    #keras.callbacks.ModelCheckpoint(filepath="./results/ladderCNN_gamma/v1_{epoch:02d}-{val_loss:.2f}model.hdf5",save_best_only=True)
            ],
        verbose=1,
        )




if __name__ == '__main__':
    main()

