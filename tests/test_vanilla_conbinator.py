from keras.datasets import mnist
import keras
from layers.vanilla_conbinator import VanillaConbinator
import numpy as np

x = keras.layers.Input((28, 28))
y = keras.layers.Flatten()(x)
y1 = keras.layers.Dense(1024, activation='relu')(y)
y2 = keras.layers.Dense(1024, activation='relu')(y)
y = VanillaConbinator()([y1, y2])
y = keras.layers.Activation('relu')(y)
y = keras.layers.Dense(512, activation='relu')(y)
y = keras.layers.Dense(10, activation='softmax')(y)

model = keras.models.Model(x, y)


def to_onehot(output, bins=10):
    return np.array(map(lambda x: [1.0 if i == x else 0.0 for i in range(bins)], output))
    #return np.zeros((len(output), bins))


def train_xor(model):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    model.summary()
    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, to_onehot(y_train, 10), validation_data=(X_test, to_onehot(y_test, 10)), nb_epoch=100)


if __name__ == '__main__':
    train_xor(model)
