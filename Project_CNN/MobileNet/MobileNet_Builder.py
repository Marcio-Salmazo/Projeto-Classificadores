import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


# ======================================================================================================================
# MOBILENET V1

def depthwise_separable_conv(x, filters, stride):
    # Depthwise
    x = DepthwiseConv2D(3, strides=stride, padding='same', use_bias=False,
                        depthwise_regularizer=tf.keras.regularizers.l2(1e-4))(x)

    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Pointwise
    x = Conv2D(filters, 1, padding='same', use_bias=False,
               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def MobileNetV1(input_shape=(224, 224, 3), num_classes=1000, alpha=1):

    def f(filters):
        return int(filters * alpha)

    inputs = Input(shape=input_shape)
    x = Conv2D(f(32), 3, strides=2, padding='same', use_bias=False,
               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = depthwise_separable_conv(x, f(64), 1)

    x = depthwise_separable_conv(x, f(128), 2)
    x = depthwise_separable_conv(x, f(128), 1)

    x = depthwise_separable_conv(x, f(256), 2)
    x = depthwise_separable_conv(x, f(256), 1)

    x = depthwise_separable_conv(x, f(512), 2)

    for _ in range(5):
        x = depthwise_separable_conv(x, f(512), 1)

    x = depthwise_separable_conv(x, f(1024), 2)
    x = depthwise_separable_conv(x, f(1024), 1)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax',
                    kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)

    return Model(inputs, outputs)


# ======================================================================================================================
# MOBILENET V2

def relu6(x):
    return tf.nn.relu6(x)


def bottleneck_block(x, out_channels, expansion_factor, stride):
    # Linear bottleneck (com uso de ReLU6) associado ao Inverted Resideual Block
    input_channels = x.shape[-1]

    # 1. Expansão (1x1 conv + ReLU6)
    expanded = Conv2D(input_channels * expansion_factor, 1, padding='same', use_bias=False)(x)
    expanded = BatchNormalization()(expanded)
    expanded = Activation(relu6)(expanded)

    # 2. Depthwise (3x3)
    depthwise = DepthwiseConv2D(3, strides=stride, padding='same', use_bias=False)(expanded)
    depthwise = BatchNormalization()(depthwise)
    depthwise = Activation(relu6)(depthwise)

    # 3. Projeção (1x1 linear — SEM ReLU)
    projected = Conv2D(out_channels, 1, padding='same', use_bias=False)(depthwise)
    projected = BatchNormalization()(projected)

    # 4. Residual (se possível)
    if stride == 1 and input_channels == out_channels:
        return Add()([x, projected])
    else:
        return projected


# Parâmetros padrões configurados para a ImageNet, seguindo o artigo
def MobileNetV2(input_shape=(224, 224, 3), num_classes=1000):
    inputs = Input(shape=input_shape)

    # Primeira camada (conv padrão)
    x = Conv2D(32, 3, strides=2, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation(relu6)(x)

    ''' 
        Tabela 2 do paper:
        
        t = fator de expansão
        c = output channels
        n = repetições da camada
        s = stride
        
        OBS: Cada registro descreve uma sequencia de 1 ou mais camadas
             identicas, repetidas 'n' vezes
    '''
    config = [
        # t, c, n, s
        (1, 16, 1, 1),
        (6, 24, 2, 2),
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]

    for t, c, n, s in config:
        for i in range(n):
            stride = s if i == 0 else 1
            x = bottleneck_block(x, c, t, stride)

    # Última camada
    x = Conv2D(1280, 1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation(relu6)(x)

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)
