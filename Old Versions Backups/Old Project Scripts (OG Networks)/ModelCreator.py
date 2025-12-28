from PyQt5.QtCore import pyqtSignal
from tensorflow.keras.applications import ResNet50
from PatchOperations import PatchEncoder, PatchExtractor
from tensorflow.keras import layers, models, regularizers

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

'''
    NOVA IMPLEMENTAÇÃO DA RESNET-50:
        * Usa GlobalAveragePooling2D (reduz 100.000 parâmetros → ~2.000)
        * Adiciona BatchNorm
        * Adiciona L2 regularization
        * Adiciona Dropout moderado
        * Congela parcialmente o modelo base (melhor prática)
        * Não compila automaticamente 
'''


class Resnet50:
    log_signal = pyqtSignal(str)  # sinal para enviar mensagens de log ao PyQt

    def __init__(self, input_shape=(224, 224, 3), num_classes=3, last_layer_activation="softmax"):
        """
            Classe responsável por definir a arquitetura de rede neural
            parametro input_shape: Define as dimensões das entradas (imagens).
            parametro num_classes: Define a quantidade de classes na saída.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.last_layer_activation = last_layer_activation

    def resnet_classifier(self):
        base_model = ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=self.input_shape
        )

        # Congela camadas profundas — excelente para generalização
        for layer in base_model.layers[:140]:
            layer.trainable = False

        inputs = layers.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)

        # Substitui o Flatten (que causa overfitting)
        x = layers.GlobalAveragePooling2D()(x)

        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)

        x = layers.Dense(
            256,
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4)
        )(x)

        x = layers.Dropout(0.3)(x)

        outputs = layers.Dense(
            self.num_classes,
            activation=self.last_layer_activation,
            kernel_regularizer=regularizers.l2(1e-4)
        )(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model


# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

'''
    NOVA IMPLEMENTAÇÃO ViT
    
        * Adicionada regularização L2 (weight decay)
        * Dropout ajustado para 0.1–0.2 (o valor anterior 0.5 estava ALTÍSSIMO) 
        * Adicionada stochastic depth (probabilidade de pular camadas no treino → enorme antioverfitting)
        * MLP final com regularização 
        * Normalização final melhorada
'''


class VisionTransformer:

    def __init__(self, input_shape=None, patch_size=None, num_patches=None,
                 projection_dim=None, transformer_layers=None, num_heads=None,
                 mlp_units=None, num_classes=None):
        self.input_shape = input_shape
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.transformer_layers = transformer_layers
        self.num_heads = num_heads
        self.mlp_units = mlp_units
        self.num_classes = num_classes

    def vit_classifier(self):

        wd = 1e-4  # weight decay

        inputs = layers.Input(shape=self.input_shape)

        patches = PatchExtractor(self.patch_size)(inputs)
        x = PatchEncoder(self.num_patches, self.projection_dim)(patches)

        # stochastic depth (anti-overfitting)
        drop_path_rate = 0.1

        for i in range(self.transformer_layers):
            # Norma + Atenção
            y = layers.LayerNormalization(epsilon=1e-6)(x)
            y = layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.projection_dim,
                dropout=0.1
            )(y, y)
            y = layers.Dropout(0.1)(y)

            # Skip connection + stochastic depth
            x = layers.Add()([x, self.stochastic_depth(y, drop_path_rate)])

            # Norma + MLP
            y = layers.LayerNormalization(epsilon=1e-6)(x)
            y = layers.Dense(
                self.mlp_units,
                activation=tf.nn.gelu,
                kernel_regularizer=regularizers.l2(wd)
            )(y)
            y = layers.Dropout(0.1)(y)
            y = layers.Dense(
                self.projection_dim,
                kernel_regularizer=regularizers.l2(wd)
            )(y)
            x = layers.Add()([x, self.stochastic_depth(y, drop_path_rate)])

        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.GlobalAveragePooling1D()(x)

        x = layers.Dropout(0.2)(x)
        x = layers.Dense(
            self.mlp_units,
            activation=tf.nn.gelu,
            kernel_regularizer=regularizers.l2(wd)
        )(x)
        x = layers.Dropout(0.2)(x)

        outputs = layers.Dense(
            self.num_classes,
            activation="softmax",
            kernel_regularizer=regularizers.l2(wd)
        )(x)

        model = tf.keras.Model(inputs, outputs)
        return model

    def stochastic_depth(self, x, drop_prob):
        if drop_prob == 0.0:
            return x
        keep_prob = 1 - drop_prob
        shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)
        random_tensor = keep_prob + tf.random.uniform(shape)
        binary_mask = tf.floor(random_tensor)
        return (x / keep_prob) * binary_mask
