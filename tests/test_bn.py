from keras.datasets import mnist
import keras
import keras.backend as K
from layers.batch_nomalize import OnlyBatchNormalization, ScaleAndShift
from layers.vanilla_conbinator import VanillaConbinator
import numpy as np

x = keras.layers.Input((28, 28))
y = keras.layers.Flatten()(x)
y = keras.layers.Dense(1024, activation='relu')(y)
y_bn = keras.layers.BatchNormalization(mode=2)(y)
y_bn_ss = OnlyBatchNormalization(mode=2)(y)
y_bn_ss = ScaleAndShift(mode=2)(y_bn_ss)
y_diff = keras.layers.merge([y_bn_ss, y_bn], mode=lambda xs: xs[0] - xs[1], output_shape=(1024,))
y_bn = keras.layers.Dense(10, activation='softmax')(y_bn)
y_bn_ss = keras.layers.Dense(10, activation='softmax')(y_bn_ss)

model = keras.models.Model(x, [y_bn_ss, y_diff])


def to_onehot(output, bins=10):
    return np.array(map(lambda x: [1.0 if i == x else 0.0 for i in range(bins)], output))


def to_zeros(output, bins=10):
    return np.zeros((len(output), bins))


def train_xor(model):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    model.summary()
    model.compile('adam', ['categorical_crossentropy', 'mse'], metrics=['accuracy'], loss_weights=[1.0, 0])
    model.fit(X_train,
              [to_onehot(y_train, 10), to_zeros(y_train, 1024)],
              validation_data=(X_test, [to_onehot(y_test, 10), to_zeros(y_test, 1024)])
              , nb_epoch=100)


if __name__ == '__main__':
    train_xor(model)
