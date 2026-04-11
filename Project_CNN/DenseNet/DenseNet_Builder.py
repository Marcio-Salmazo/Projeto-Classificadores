import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


def dense_layer(x, growth_rate, dropout_rate=None):
    # Bottleneck: 1x1 conv com 4k canais
    x1 = BatchNormalization()(x)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(4 * growth_rate, 1, padding='same', use_bias=False)(x1)

    # 3x3 conv
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(growth_rate, 3, padding='same', use_bias=False)(x1)

    # Regularização
    if dropout_rate:
        x1 = Dropout(dropout_rate)(x1)

    # Concatenação (ESSENCIAL da DenseNet)
    return Concatenate()([x, x1])


def dense_block(x, num_layers, growth_rate):
    for _ in range(num_layers):
        x = dense_layer(x, growth_rate)
    return x


def transition_layer(x, compression=0.5):
    filters = int(x.shape[-1] * compression)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, 1, padding='same', use_bias=False)(x)
    x = AveragePooling2D(2, strides=2)(x)

    return x


def Shallow_densenet(input_shape=(128, 128, 3), num_classes=3, growth_rate=24):
    inputs = Input(shape=input_shape)

    # Entrada leve
    x = Conv2D(48, 3, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Configuração mais leve e Otimizada
    growth_rate = growth_rate
    block_layers = [6, 8, 12, 8]

    # Dense Block 1
    x = dense_block(x, block_layers[0], growth_rate)
    x = transition_layer(x)

    # Dense Block 2
    x = dense_block(x, block_layers[1], growth_rate)
    x = transition_layer(x)

    # Dense Block 3
    x = dense_block(x, block_layers[2], growth_rate)
    x = transition_layer(x)

    # Dense Block 4
    x = dense_block(x, block_layers[3], growth_rate)

    # Final
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    x = Dropout(0.3)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)


'''
# ARQUITETURA RASA COM BASE NA DENSENET-121 
def Shallow_densenet(input_shape=(128,128,3), num_classes=3, growth_rate=16):

    inputs = Input(shape=input_shape)

    # Entrada leve
    x = Conv2D(32, 3, strides=1, padding='same', use_bias=False)(inputs)

    # Configuração mais leve e Otimizada
    growth_rate = growth_rate
    block_layers = [4, 6, 8, 6]

    # Dense Block 1
    x = dense_block(x, block_layers[0], growth_rate)
    x = transition_layer(x)

    # Dense Block 2
    x = dense_block(x, block_layers[1], growth_rate)
    x = transition_layer(x)

    # Dense Block 3
    x = dense_block(x, block_layers[2], growth_rate)
    x = transition_layer(x)

    # Dense Block 4
    x = dense_block(x, block_layers[3], growth_rate)

    # Final
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    x = Dropout(0.3)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)

'''


# ARQUITETURA ORIGINAL DN-121 >> FIEL À IMPLEMENTAÇÃO DO ARTIGO
def DenseNet121(input_shape=(224, 224, 3), num_classes=1000, growth_rate=32):
    inputs = Input(shape=input_shape)

    # Initial Conv (ImageNet config)
    x = Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # Dense Blocks (121)
    block_layers = [6, 12, 24, 16]
    growth_rate = growth_rate

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
