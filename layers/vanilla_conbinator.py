from keras.layers import Layer, InputSpec, initializations
import keras.backend as K

class LadderConbinator(Layer):
    def __init__(self, init='glorot_uniform', weights=None, **kwargs):
        self.init = initializations.get(init)
        self.initial_weights = weights
        super(LadderConbinator, self).__init__(**kwargs)


    def build(self, input_shape):
        input_dim = input_shape[1][1:]

        self.output_dim = input_dim

        self.input_spec = [
            InputSpec(dtype=K.floatx(),
                      shape=(None,) + input_shape[0][1:]),
            InputSpec(dtype=K.floatx(),
                      shape=(None,) + input_shape[1][1:]),
        ]


        self.ua1 = self.init((input_dim), name='{}_ua1'.format(self.name))
        self.ua2 = self.init((input_dim), name='{}_ua2'.format(self.name))
        self.ua3 = self.init((input_dim), name='{}_ua3'.format(self.name))
        self.ua4 = self.init((input_dim), name='{}_ua4'.format(self.name))
        self.ua5 = self.init((input_dim), name='{}_ua5'.format(self.name))

        self.va1 = self.init((input_dim), name='{}_va1'.format(self.name))
        self.va2 = self.init((input_dim), name='{}_va2'.format(self.name))
        self.va3 = self.init((input_dim), name='{}_va3'.format(self.name))
        self.va4 = self.init((input_dim), name='{}_va4'.format(self.name))
        self.va5 = self.init((input_dim), name='{}_va5'.format(self.name))

        self.trainable_weights = [
            self.ua1,
            self.ua2,
            self.ua3,
            self.ua4,
            self.ua5,
            self.va1,
            self.va2,
            self.va3,
            self.va4,
            self.va5,
        ]

    def call(self, xs, mask=None):
        if not (type(xs) is list or len(xs) == 2 and xs[0].shape == xs[1].shape):
            raise Exception("conbinator must be called on a list of tensors")
        z, u = xs

        mu = self.ua1 * K.sigmoid(self.ua2 * u + self.ua3) + self.ua4 * u + self.ua5
        v = self.va1 * K.sigmoid(self.va2 * u + self.va3) + self.va4 * u + self.va5

        return (z - mu) * v + mu

    def get_output_shape_for(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'init': self.init.__name__,
            'input_dim': self.input_dim,
        }
        base_config = super(LadderConbinator, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class VanillaConbinator(Layer):
    def __init__(self, init='glorot_uniform', weights=None, **kwargs):
        self.init = initializations.get(init)
        self.initial_weights = weights
        super(VanillaConbinator, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1][1:]

        self.output_dim = input_dim

        self.input_spec = [
            InputSpec(dtype=K.floatx(),
                      shape=(None,) + input_shape[0][1:]),
            InputSpec(dtype=K.floatx(),
                      shape=(None,) + input_shape[1][1:]),
        ]


        self.b0 = self.init((input_dim), name='{}_b0'.format(self.name))
        self.w0z = self.init((input_dim), name='{}_w0z'.format(self.name))
        self.w0u = self.init((input_dim), name='{}_w0u'.format(self.name))
        self.w0zu = self.init((input_dim), name='{}_w0zu'.format(self.name))
        self.ws = self.init((input_dim), name='{}_ws'.format(self.name))

        self.b1 = self.init((input_dim), name='{}_b1'.format(self.name))
        self.w1z = self.init((input_dim), name='{}_w1z'.format(self.name))
        self.w1u = self.init((input_dim), name='{}_w1u'.format(self.name))
        self.w1zu = self.init((input_dim), name='{}_w1zu'.format(self.name))

        self.trainable_weights = [
            self.b0,
            self.w0z,
            self.w0u,
            self.w0zu,
            self.ws,
            self.b1,
            self.w1z,
            self.w1u,
            self.w1zu,
        ]

    def call(self, xs, mask=None):
        if not (type(xs) is list or len(xs) == 2 and xs[0].shape == xs[1].shape):
            raise Exception("conbinator must be called on a list of tensors")
        z, u = xs

        return self.b0 + self.w0z*z + self.w0u*u + self.w0zu*z*u + self.ws * K.sigmoid(self.b1 + self.w1z*z + self.w1u*u + self.w1zu*z*u)

    def get_output_shape_for(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'init': self.init.__name__,
            'input_dim': self.input_dim,
        }
        base_config = super(VanillaConbinator, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class AugumentPerceptron(Layer):
    def __init__(self, init='glorot_uniform', weights=None, **kwargs):
        self.init = initializations.get(init)
        self.initial_weights = weights
        super(VanillaConbinator, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1][1:]

        self.output_dim = input_dim

        self.input_spec = [
            InputSpec(dtype=K.floatx(),
                      shape=(None,) + input_shape[0][1:]),
            InputSpec(dtype=K.floatx(),
                      shape=(None,) + input_shape[1][1:]),
        ]


        self.b0 = self.init((input_dim), name='{}_b0'.format(self.name))
        self.w0z = self.init((input_dim), name='{}_w0z'.format(self.name))
        self.w0u = self.init((input_dim), name='{}_w0u'.format(self.name))
        self.w0zu = self.init((input_dim), name='{}_w0zu'.format(self.name))
        self.ws = self.init((input_dim), name='{}_ws'.format(self.name))

        self.b1 = self.init((input_dim), name='{}_b1'.format(self.name))
        self.w1z = self.init((input_dim), name='{}_w1z'.format(self.name))
        self.w1u = self.init((input_dim), name='{}_w1u'.format(self.name))
        self.w1zu = self.init((input_dim), name='{}_w1zu'.format(self.name))

        self.trainable_weights = [
            self.b0,
            self.w0z,
            self.w0u,
            self.w0zu,
            self.ws,
            self.b1,
            self.w1z,
            self.w1u,
            self.w1zu,
        ]

    def call(self, xs, mask=None):
        if not (type(xs) is list or len(xs) == 2 and xs[0].shape == xs[1].shape):
            raise Exception("conbinator must be called on a list of tensors")
        z, u = xs

        return self.b0 + self.w0z*z + self.w0u*u + self.w0zu*z*u + self.ws * K.sigmoid(self.b1 + self.w1z*z + self.w1u*u + self.w1zu*z*u)

    def get_output_shape_for(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'init': self.init.__name__,
            'input_dim': self.input_dim,
        }
        base_config = super(VanillaConbinator, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

