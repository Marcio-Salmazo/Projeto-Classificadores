import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


def dense_layer(x, growth_rate):
    # Bottleneck: 1x1 conv com 4k canais
    x1 = BatchNormalization()(x)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(4 * growth_rate, kernel_size=1, padding='same', use_bias=False)(x1)

    # 3x3 conv
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(growth_rate, kernel_size=3, padding='same', use_bias=False)(x1)

    # Concatenação (ESSENCIAL da DenseNet)
    x = Concatenate()([x, x1])
    return x


def dense_block(x, num_layers, growth_rate):
    for _ in range(num_layers):
        x = dense_layer(x, growth_rate)
    return x


def transition_layer(x, compression=0.5):
    filters = int(x.shape[-1] * compression)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size=1, padding='same', use_bias=False)(x)
    x = AveragePooling2D(pool_size=2, strides=2)(x)

    return x


def DenseNet121(input_shape=(224, 224, 3), num_classes=1000):

    inputs = Input(shape=input_shape)

    # Initial Conv (ImageNet config)
    x = Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # Dense Blocks (121)
    block_layers = [6, 12, 24, 16]
    growth_rate = 32

    for i, num_layers in enumerate(block_layers):
        x = dense_block(x, num_layers, growth_rate)

        if i != len(block_layers) - 1:
            x = transition_layer(x, compression=0.5)

    # Final layers
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)