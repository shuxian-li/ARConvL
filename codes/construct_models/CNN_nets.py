from tensorflow.keras.layers import Add, Dense, Conv2D, BatchNormalization, Dropout
from tensorflow.keras.layers import Activation, MaxPooling2D, AveragePooling2D, Input, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model


def mnist_CNN(x_in, feature_dim):
    x = x_in
    channel = [32, 64]
    for ch in channel:
        x = Conv2D(ch, 3, activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = Conv2D(ch, 3, activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    x = Flatten()(x)
    x = Dense(feature_dim, kernel_initializer='he_uniform')(x)
    z = BatchNormalization()(x)

    return z


class Resnet:
    def __init__(self, size=44, stacks=3, starting_filter=16, input_shape=(32, 32, 3)):
        self.size = size
        self.stacks = stacks
        self.starting_filter = starting_filter
        self.residual_blocks = (size - 2) // 6
        self.input_shape = input_shape

    def get_model(self, input_shape=(32, 32, 3), n_classes=10):
        input_shape = self.input_shape
        n_filters = self.starting_filter

        inputs = Input(shape=input_shape)
        network = self.layer(inputs, n_filters)
        network = self.stack(network, n_filters, True)

        for _ in range(self.stacks - 1):
            n_filters *= 2
            network = self.stack(network, n_filters)

        network = Activation('elu')(network)
        network = AveragePooling2D(pool_size=network.shape[1])(network)
        network = Flatten()(network)
        # outputs = Dense(n_classes, activation='softmax',
        #                 kernel_initializer='he_normal')(network)
        outputs = network

        model = Model(inputs=inputs, outputs=outputs)

        return model

    def stack(self, inputs, n_filters, first_stack=False):
        stack = inputs

        if first_stack:
            stack = self.identity_block(stack, n_filters)
        else:
            stack = self.convolution_block(stack, n_filters)

        for _ in range(self.residual_blocks - 1):
            stack = self.identity_block(stack, n_filters)

        return stack

    def identity_block(self, inputs, n_filters):
        shortcut = inputs

        block = self.layer(inputs, n_filters, normalize_batch=False)
        block = self.layer(block, n_filters, activation=None)

        block = Add()([shortcut, block])

        return block

    def convolution_block(self, inputs, n_filters, strides=2):
        shortcut = inputs

        block = self.layer(inputs, n_filters, strides=strides,
                           normalize_batch=False)
        block = self.layer(block, n_filters, activation=None)

        shortcut = self.layer(shortcut, n_filters,
                              kernel_size=1, strides=strides,
                              activation=None)

        block = Add()([shortcut, block])

        return block

    def layer(self, inputs, n_filters, kernel_size=3,
              strides=1, activation='elu', normalize_batch=True):

        convolution = Conv2D(n_filters, kernel_size=kernel_size,
                             strides=strides, padding='same',
                             kernel_initializer="he_normal",
                             kernel_regularizer=l2(1e-4))

        x = convolution(inputs)

        if normalize_batch:
            x = BatchNormalization()(x)

        if activation is not None:
            x = Activation(activation)(x)

        return x
