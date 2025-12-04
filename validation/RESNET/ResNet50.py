import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model

# Default weight decay: 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4

# ======================================================================================================================
#                                           CONSTRUÇÃO DA ARQUITETURA DA REDE
# ======================================================================================================================
class Bottleneck(layers.Layer):
    """
        If `stride != 1` or `in_channels != out_channels`, applies a conv1x1 shortcut.
        Applies Conv -> BN -> ReLU order (post-activation formulation used in original ResNet paper).
        Kernel regularizer uses L2 weight decay to mimic the paper's weight decay.
    """

    def __init__(self, filters, stride=1, use_shortcut_conv=False, weight_decay=DEFAULT_WEIGHT_DECAY, **kwargs):
        super().__init__(**kwargs)

        # Recebe o tamanho dos filtros, durante a criação das camadas por meio de uma tupla de valores
        # Exemplo: Conv2_x = 1x1, 64; 3x3, 64; 1x1, 256;
        # Na chamada da função temos: filters = (64, 64, 256).
        # Ver o construtor da class ResNet50 para a implementação
        filters1, filters2, filters3 = filters

        '''
            I) Bottleneck 1×1 → 3×3 → 1×1: O artigo define explicitamente o bloco bottleneck para 
            ResNet-50/101/152 exatamente com a sequência: redução via 1×1, conv 3×3 e expansão via 1×1. 
            Essa é a escolha que reduz custo computacional mantendo profundidade.
            
            II) Conv → BatchNormalization → ReLU (post-activation): essa ordem corresponde à formulação 
            usada no artigo original (Batch Normalization e ReLU são utilizados após cada 'conv '.
            
            III) use_bias=False: quando se usa BatchNorm, o bias na conv é redundante (BN cancela o deslocamento), 
            então é padrão de implementações reais (incluindo repositório Caffe) omitir bias. 
            O repositório oficial também usa convs sem bias. 

            IV) kernel_regularizer=regularizers.l2(weight_decay): implementa weight decay (L2) nos filtros
            das convoluções — prática equivalente ao weight decay descrito pelo artigo. 
            (Observação: Keras aplica L2 como parte da perda; comportamento prático corresponde a weight decay).
            
        '''
        self.conv1 = layers.Conv2D(filters1, kernel_size=1, strides=1, padding='valid', use_bias=False,
                                   kernel_regularizer=regularizers.l2(weight_decay))
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2D(filters2, kernel_size=3, strides=stride, padding='same', use_bias=False,
                                   kernel_regularizer=regularizers.l2(weight_decay))
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2D(filters3, kernel_size=1, strides=1, padding='valid', use_bias=False,
                                   kernel_regularizer=regularizers.l2(weight_decay))
        self.bn3 = layers.BatchNormalization()

        self.relu = layers.Activation('relu')

        '''
            A implementação canônica adota a projeção (1×1 conv) - Opção B descrita no artigo - para ajustar 
            as dimensões quando necessário. 'use_shortcut_conv' implementa esse tratamento.
        '''
        self.use_shortcut_conv = use_shortcut_conv
        if use_shortcut_conv:
            self.shortcut_conv = layers.Conv2D(filters3, kernel_size=1, strides=stride, padding='valid', use_bias=False,
                                               kernel_regularizer=regularizers.l2(weight_decay))
            self.shortcut_bn = layers.BatchNormalization()

    # Construção efetiva das camadas de bottleneck
    def call(self, inputs, training=False):

        """
            Executa exatamente as operações do bloco residual (bottleneck): três convoluções com BatchNorm + ReLU
            intercalados, seguido pela soma com o shortcut. Esse fluxo reproduz a equação descrita no artigo
            por meio da Figura 5.

            Se use_shortcut_conv for True, é aplicado a opção B descrita no artigo
            Se use_shortcut_conv for False, a rede utiliza atalhos identidade
        """
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)

        if self.use_shortcut_conv:
            shortcut = self.shortcut_conv(inputs)
            shortcut = self.shortcut_bn(shortcut, training=training)
        else:
            shortcut = inputs

        # O artigo descreve a soma residual seguida por ativação (Figura 5)
        x = layers.add([x, shortcut])
        x = self.relu(x)
        return x


def make_stage(filters, blocks, stride_first=2, weight_decay=DEFAULT_WEIGHT_DECAY, name=None):
    """
        Constrói uma stage da arquitetura (conv2_x, conv3_x, ...) com blocos bottleneck.
        de acordo com o que foi descrito pela tabela 1 do artigo.
            * filters é uma tuple/list com (f1, f2, f3) por bloco.
            * stride_first é o stride aplicado no primeiro bloco para downsampling.
    """
    stage_layers = tf.keras.Sequential(name=name)

    '''
        O primeiro bloco é construído à parte para implementar o 'stride_first' a fim de fazer o downsampling 
        entre stages — isto reproduz o mecanismo do artigo onde o primeiro bloco do stage realiza o 
        downsampling (stride 2) e usa projection shortcut para ajustar dimensões.
    '''
    stage_layers.add(Bottleneck(filters, stride=stride_first, use_shortcut_conv=True, weight_decay=weight_decay))

    # Cria os demais stages, conforme descrito pelo artigo
    for i in range(1, blocks):
        stage_layers.add(Bottleneck(filters, stride=1, use_shortcut_conv=False, weight_decay=weight_decay))
    return stage_layers


class ResNet50(Model):
    """
        Implementação efetiva da arquitetura ResNet-50, estando em conformidade
        com o que foi descrito pelo artigo Deep Residual Learning for Image Recognition
    """

    def __init__(self, num_classes=1000, include_top=True, weight_decay=DEFAULT_WEIGHT_DECAY, **kwargs):
        super().__init__(**kwargs)

        self.include_top = include_top  # Define a implementação da camada FC
        self.num_classes = num_classes  # Define o número de classes para a camada totalmente conectada
        self.weight_decay = weight_decay

        # --------------------------------------------------------------------------------------------------------------
        # CONSTRUÇÃO DE CAMADAS CONFORME A TABELA 1 DO ARTIGO:

        # O artigo descreve explicitamente a primeira camada como 7×7 conv com stride 2
        # seguida por 3×3 max pooling (redução espacial inicial).
        self.conv1 = layers.Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False,
                                   kernel_regularizer=regularizers.l2(self.weight_decay))
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        self.maxpool = layers.MaxPool2D(pool_size=3, strides=2, padding='same')

        # Stacks e filtros: os quatro stages e os filtros (64→256), (128→512), (256→1024), (512→2048)
        # com blocos (3,4,6,3) são exatamente a ResNet-50 conforme tabela do paper.
        self.conv2_x = make_stage(filters=(64, 64, 256), blocks=3, stride_first=1, weight_decay=self.weight_decay,
                                  name='conv2_x')
        self.conv3_x = make_stage(filters=(128, 128, 512), blocks=4, stride_first=2, weight_decay=self.weight_decay,
                                  name='conv3_x')
        self.conv4_x = make_stage(filters=(256, 256, 1024), blocks=6, stride_first=2, weight_decay=self.weight_decay,
                                  name='conv4_x')
        self.conv5_x = make_stage(filters=(512, 512, 2048), blocks=3, stride_first=2, weight_decay=self.weight_decay,
                                  name='conv5_x')

        # --------------------------------------------------------------------------------------------------------------

        # O artigo aplica global average pooling e FC - Fully connected layer - para classificação em 1000 classes).
        self.avgpool = layers.GlobalAveragePooling2D()
        if self.include_top:
            self.fc = layers.Dense(self.num_classes, activation='softmax',
                                   kernel_regularizer=regularizers.l2(self.weight_decay))
        else:
            self.fc = None

    def call(self, inputs, training=False, mask=None):

        """
            Esse encadeamento executa exatamente a sequência de operações do artigo,
            produzindo o mapa 7×7 (após conv5_x) que é reduzido por global average pooling
            para um vetor de 2048 antes do FC.
        """
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
#                                               COMPILAÇÃO E TREINAMENTO
# ======================================================================================================================

def build_resnet50(input_shape=(224, 224, 3), num_classes=1000, include_top=True, weight_decay=DEFAULT_WEIGHT_DECAY):
    """
    Helper to instantiate and build a ResNet50 model.
    """
    inputs = tf.keras.Input(shape=input_shape)
    model = ResNet50(num_classes=num_classes, include_top=include_top, weight_decay=weight_decay)
    outputs = model(inputs, training=False)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='ResNet50_paper')
    return model
