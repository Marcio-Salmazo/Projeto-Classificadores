import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model

# Default weight decay conforme o paper
DEFAULT_WEIGHT_DECAY = 1e-4


# ======================================================================================================================
#                                               BLOCO RESIDUAL BÁSICO
# ======================================================================================================================
class BasicBlock(layers.Layer):
    """
    Bloco residual básico utilizado na ResNet-18 e ResNet-34.

    Estrutura:
        3x3 Conv → BN → ReLU
        3x3 Conv → BN
        + shortcut (identidade ou projeção)
        → ReLU

    Diferente do Bottleneck:
        - NÃO há convoluções 1x1 internas
        - NÃO há expansão de canais
    """

    def __init__(
            self,
            filters,
            stride=1,
            use_projection=False,
            weight_decay=DEFAULT_WEIGHT_DECAY,
            **kwargs
    ):
        super().__init__(**kwargs)

        # Primeira convolução 3x3
        self.conv1 = layers.Conv2D(
            filters,
            kernel_size=3,
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(weight_decay),
        )
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)

        # Segunda convolução 3x3
        self.conv2 = layers.Conv2D(
            filters,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(weight_decay),
        )
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)

        self.relu = layers.Activation("relu")

        # Shortcut por projeção (1x1) — Opção B do paper
        self.use_projection = use_projection
        if use_projection:
            self.shortcut_conv = layers.Conv2D(
                filters,
                kernel_size=1,
                strides=stride,
                padding="valid",
                use_bias=False,
                kernel_regularizer=regularizers.l2(weight_decay),
            )
            self.shortcut_bn = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        if self.use_projection:
            shortcut = self.shortcut_conv(inputs)
            shortcut = self.shortcut_bn(shortcut, training=training)
        else:
            shortcut = inputs

        x = layers.add([x, shortcut])
        x = self.relu(x)
        return x


# ======================================================================================================================
#                                           CONSTRUÇÃO DE STAGES
# ======================================================================================================================
def make_stage(filters, blocks, stride_first, weight_decay, name):
    """
    Constrói um stage da ResNet (conv2_x, conv3_x, conv4_x, conv5_x).

    - Primeiro bloco: downsampling + projeção
    - Blocos seguintes: identidade
    """
    stage = tf.keras.Sequential(name=name)

    stage.add(
        BasicBlock(
            filters=filters,
            stride=stride_first,
            use_projection=True,
            weight_decay=weight_decay,
        )
    )

    for _ in range(1, blocks):
        stage.add(
            BasicBlock(
                filters=filters,
                stride=1,
                use_projection=False,
                weight_decay=weight_decay,
            )
        )

    return stage


# ======================================================================================================================
#                                           BUILDER DA RESNET-10
# ======================================================================================================================
class ResNet10_Builder(Model):
    """
    Implementação da ResNet-10 baseada no mesmo padrão estrutural da ResNet-18,
    porém utilizando apenas 1 bloco residual por stage: (1,1,1,1).
    """

    def __init__(
            self,
            num_classes=1000,
            include_top=True,
            weight_decay=DEFAULT_WEIGHT_DECAY,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.include_top = include_top
        self.num_classes = num_classes

        # Camada inicial (igual à ResNet-18)
        self.conv1 = layers.Conv2D(
            64,
            kernel_size=7,
            strides=2,
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(weight_decay),
        )
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu = layers.Activation("relu")
        self.maxpool = layers.MaxPool2D(pool_size=3, strides=2, padding="same")

        # Stages — profundidade (1,1,1,1)
        self.conv2_x = make_stage(
            filters=64,
            blocks=1,
            stride_first=1,
            weight_decay=weight_decay,
            name="conv2_x",
        )
        self.conv3_x = make_stage(
            filters=128,
            blocks=1,
            stride_first=2,
            weight_decay=weight_decay,
            name="conv3_x",
        )
        self.conv4_x = make_stage(
            filters=256,
            blocks=1,
            stride_first=2,
            weight_decay=weight_decay,
            name="conv4_x",
        )
        self.conv5_x = make_stage(
            filters=512,
            blocks=1,
            stride_first=2,
            weight_decay=weight_decay,
            name="conv5_x",
        )

        # Cabeça de classificação
        self.avgpool = layers.GlobalAveragePooling2D()
        if include_top:
            self.fc = layers.Dense(
                num_classes,
                activation="softmax",
                kernel_regularizer=regularizers.l2(weight_decay),
            )
        else:
            self.fc = None

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2_x(x, training=training)
        x = self.conv3_x(x, training=training)
        x = self.conv4_x(x, training=training)
        x = self.conv5_x(x, training=training)

        x = self.avgpool(x)
        if self.include_top:
            x = self.fc(x)

        return x


# ======================================================================================================================
#                                           BUILDER DA RESNET-18
# ======================================================================================================================
class ResNet18_Builder(Model):
    """
    Implementação fiel da ResNet-18 conforme o paper original.
    """

    def __init__(
            self,
            num_classes=1000,
            include_top=True,
            weight_decay=DEFAULT_WEIGHT_DECAY,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.include_top = include_top
        self.num_classes = num_classes

        # Camada inicial (idêntica à ResNet-50)
        self.conv1 = layers.Conv2D(
            64,
            kernel_size=7,
            strides=2,
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(weight_decay),
        )
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu = layers.Activation("relu")
        self.maxpool = layers.MaxPool2D(pool_size=3, strides=2, padding="same")

        # Stages — profundidade (2,2,2,2)
        self.conv2_x = make_stage(
            filters=64,
            blocks=2,
            stride_first=1,
            weight_decay=weight_decay,
            name="conv2_x",
        )
        self.conv3_x = make_stage(
            filters=128,
            blocks=2,
            stride_first=2,
            weight_decay=weight_decay,
            name="conv3_x",
        )
        self.conv4_x = make_stage(
            filters=256,
            blocks=2,
            stride_first=2,
            weight_decay=weight_decay,
            name="conv4_x",
        )
        self.conv5_x = make_stage(
            filters=512,
            blocks=2,
            stride_first=2,
            weight_decay=weight_decay,
            name="conv5_x",
        )

        # Cabeça de classificação
        self.avgpool = layers.GlobalAveragePooling2D()
        if include_top:
            self.fc = layers.Dense(
                num_classes,
                activation="softmax",
                kernel_regularizer=regularizers.l2(weight_decay),
            )
        else:
            self.fc = None

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2_x(x, training=training)
        x = self.conv3_x(x, training=training)
        x = self.conv4_x(x, training=training)
        x = self.conv5_x(x, training=training)

        x = self.avgpool(x)
        if self.include_top:
            x = self.fc(x)

        return x


# ======================================================================================================================
#                                           BUILDER DA RESNET-34
# ======================================================================================================================

class ResNet34_Builder(Model):
    """
    Implementação fiel da ResNet-34 conforme o paper original.
    Usa BasicBlock (3x3 + 3x3).
    """

    def __init__(
            self,
            num_classes=1000,
            include_top=True,
            weight_decay=DEFAULT_WEIGHT_DECAY,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.include_top = include_top
        self.num_classes = num_classes

        # Camada inicial (idêntica à ResNet-18 / ResNet-50)
        self.conv1 = layers.Conv2D(
            64,
            kernel_size=7,
            strides=2,
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(weight_decay),
        )
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu = layers.Activation("relu")
        self.maxpool = layers.MaxPool2D(pool_size=3, strides=2, padding="same")

        # Stages — profundidade (3, 4, 6, 3)
        self.conv2_x = make_stage(
            filters=64,
            blocks=3,
            stride_first=1,
            weight_decay=weight_decay,
            name="conv2_x",
        )
        self.conv3_x = make_stage(
            filters=128,
            blocks=4,
            stride_first=2,
            weight_decay=weight_decay,
            name="conv3_x",
        )
        self.conv4_x = make_stage(
            filters=256,
            blocks=6,
            stride_first=2,
            weight_decay=weight_decay,
            name="conv4_x",
        )
        self.conv5_x = make_stage(
            filters=512,
            blocks=3,
            stride_first=2,
            weight_decay=weight_decay,
            name="conv5_x",
        )

        # Cabeça de classificação
        self.avgpool = layers.GlobalAveragePooling2D()
        if include_top:
            self.fc = layers.Dense(
                num_classes,
                activation="softmax",
                kernel_regularizer=regularizers.l2(weight_decay),
            )
        else:
            self.fc = None

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2_x(x, training=training)
        x = self.conv3_x(x, training=training)
        x = self.conv4_x(x, training=training)
        x = self.conv5_x(x, training=training)

        x = self.avgpool(x)
        if self.include_top:
            x = self.fc(x)

        return x


# ======================================================================================================================
#                                           FUNÇÃO DE CONSTRUÇÃO
# ======================================================================================================================
def build_resnet10(
        input_shape=(224, 224, 3),
        num_classes=1000,
        include_top=True,
        weight_decay=DEFAULT_WEIGHT_DECAY,
):
    """
    Função utilitária para construir a ResNet-10 no formato Keras Functional.
    """
    inputs = tf.keras.Input(shape=input_shape)
    backbone = ResNet10_Builder(
        num_classes=num_classes,
        include_top=include_top,
        weight_decay=weight_decay,
    )

    outputs = backbone(inputs, training=False)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="ResNet10_paper",
    )

    return model


def build_resnet18(
        input_shape=(224, 224, 3),
        num_classes=1000,
        include_top=True,
        weight_decay=DEFAULT_WEIGHT_DECAY,
):
    """
    Função utilitária para construir a ResNet-18 no formato Keras Functional.
    """
    inputs = tf.keras.Input(shape=input_shape)
    backbone = ResNet18_Builder(
        num_classes=num_classes,
        include_top=include_top,
        weight_decay=weight_decay,
    )
    outputs = backbone(inputs, training=False)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="ResNet18_paper",
    )
    return model


def build_resnet34(
        input_shape=(224, 224, 3),
        num_classes=1000,
        include_top=True,
        weight_decay=DEFAULT_WEIGHT_DECAY,
):
    """
    Função utilitária para construir a ResNet-34 no formato Keras Functional.
    """
    inputs = tf.keras.Input(shape=input_shape)

    backbone = ResNet34_Builder(
        num_classes=num_classes,
        include_top=include_top,
        weight_decay=weight_decay,
    )

    outputs = backbone(inputs, training=False)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="ResNet34_paper",
    )
    return model
