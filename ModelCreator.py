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

    def __init__(self, input_shape=(224, 224, 3), num_classes=3):
        """
            Classe responsável por definir a arquitetura de rede neural
            parametro input_shape: Define as dimensões das entradas (imagens).
            parametro num_classes: Define a quantidade de classes na saída.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes

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
            activation="softmax",
            kernel_regularizer=regularizers.l2(1e-4)
        )(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model


'''
IMPLEMENTAÇÃO ANTIGA 
ResNet está sendo usada SEM regularização praticamente nenhuma
Compilação dos modelos COM categorical_crossentropy sem smoothing
Os treinamentos não estão usando EarlyStopping, nem ModelCheckpoint, nem ReduceLR
-----------------------------------------------------------------------------------

class Resnet50:
    log_signal = pyqtSignal(str)  # sinal para enviar mensagens de log ao PyQt

    def __init__(self, input_shape=(128, 128, 3), num_classes=3):
        """
        Classe responsável por definir a arquitetura de rede neural
        parametro input_shape: Define as dimensões das entradas (imagens).
        parametro num_classes: Define a quantidade de classes na saída.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes

    def resnet_classifier(self):
        # Carregando a ResNet50 sem a camada de saída original
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)

        # Definição do modelo como sequêncial, onde as camadas são adicionadas uma após a outra.
        model = models.Sequential([

            base_model,

            # Transforma a saída das camadas convolucionais (um volume 3D) em um vetor 1D
            # para que possa ser passado para a camada densa.
            layers.Flatten(),

            # Definição da primeira camada Densa, totalmente conectada
            # layers.Dense(128, activation='relu'),
            #     * 128 → Quantidade de neurônios na camada
            #     * activation='relu' → Mantém a não linearidade para melhor aprendizado
            #
            # obs: Dropout desativa 50% dos neurônios de forma aleatória a cada iteração, o que evita overfitting,
            # ajudando a rede a generalizar melhor para novos dados

            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),

            # Camada de saída, com número de neurônios igual ao número de classes.
            # A função de ativação softmax transforma os valores de saída em probabilidades para cada classe.
            layers.Dense(self.num_classes, activation='softmax')  # 3 classes (dor, não dor, dor moderada)
        ])

        """
            Função responsável por compilar o modelo e dar inicio ao treinamento
            O metodo .compile() define as configurações do modelo antes do treinamento
                * optimizer='adam' → O Adam (Adaptive Moment Estimation) é um otimizador que
                  ajusta os pesos da rede durante o treinamento de maneira eficiente.
                * loss='categorical_crossentropy' → Essa é a função de erro usada para
                  problemas de classificação multiclasse.
                * metrics=['accuracy'] → Define que a acurácia será monitorada durante o treinamento.
        """
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        return model
    '''

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


'''
class VisionTransformer:

    def __init__(self, input_shape=None, patch_size=None, num_patches=None, projection_dim=None,
                 transformer_layers=None, num_heads=None, mlp_units=None, num_classes=None):
        self.input_shape = input_shape  # formato da imagem de entrada (ex: (224, 224, 3))
        self.patch_size = patch_size  # tamanho dos blocos em que a imagem será dividida (ex: 16)
        self.num_patches = num_patches  # total de patches da imagem (ex: 196)
        self.projection_dim = projection_dim  # dimensão dos vetores em que os patches serão projetados (ex: 512)
        self.transformer_layers = transformer_layers  # número de blocos do encoder Transformer (ex: 8)
        self.num_heads = num_heads  # número de cabeças de atenção (multi-head attention)
        self.mlp_units = mlp_units  # número de neurônios da MLP interna (ex: 128)
        self.num_classes = num_classes  # total de classes do problema de classificação

    def vit_classifier(self):
        # Aqui é definida a entrada do modelo, criando um placeholder para imagens com a
        # forma especificada por self.input_shape
        inputs = layers.Input(shape=self.input_shape)
        # Usa a camada personalizada PatchExtractor para dividir a imagem em patches não sobrepostos.
        patches = PatchExtractor(self.patch_size)(inputs)
        # Aplica a camada PatchEncoder, responsável por projetar cada patch em um vetor projection_dim
        # além de adiciona codificação posicional
        encoded_patches = PatchEncoder(self.num_patches, self.projection_dim)(patches)

        # Blocos Transformer: executa vários blocos de Transformer Encoder (quantidade definida por transformer_layers)
        for _ in range(self.transformer_layers):
            # Aqui ocorre o processo de normalização por camada, antes do processo de atenção
            # Importante salientar que epsilon = 1e-6 representa um pequeno valor para evitar divisão por zero.
            # Tal processo é importante por melhorar a estabilidade do treinamento
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

            # Aplica atenção multi-cabeça (multi-head self-attention), onde cada "cabeça" foca em partes
            # diferentes da sequência de patches. x1 é usado como query, key e value (self-attention) e
            # dropout=0.1: evita overfitting.
            attention_output = layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.projection_dim, dropout=0.1
            )(x1, x1)

            # Aplica uma conexão residual. Soma a saída da atenção com a entrada original, preservando
            # a informação inicial e melhora o fluxo de gradiente.
            x2 = layers.Add()([attention_output, encoded_patches])

            # Normaliza novamente a sequência, antes de passar por uma MLP.
            # A normalização antes de cada sub-bloco é prática padrão em Transformers.
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

            # Aplica um MLP com duas camadas:
            # Primeira camada: expande a capacidade (ex: 512 → 2048)
            # Segunda camada: retorna ao projection_dim (ex: 2048 → 512)
            # A ativação GELU é suave e eficaz para Transformers.
            x3 = layers.Dense(self.mlp_units, activation=tf.nn.gelu)(x3)
            x3 = layers.Dense(self.projection_dim)(x3)

            # Outra conexão residual, agora para o sub-bloco MLP.
            # Resultado é a entrada para o próximo bloco Transformer, se houver mais
            encoded_patches = layers.Add()([x3, x2])

        # Trecho referente à cabeça de classificação
        # Normaliza o vetor final de cada patch após os blocos do encoder.
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        # "Achata" todos os vetores em um único vetor plano por imagem.
        # Isso cria um vetor representando toda a imagem (todos os patches).
        representation = layers.Flatten()(representation)

        # Aplica dropout com 50% de taxa, ajudando a reduzir o overfitting.
        representation = layers.Dropout(0.5)(representation)

        # Aplica uma MLP final para extrair boas características antes da classificação.
        # Outro dropout é aplicado.
        features = layers.Dense(self.mlp_units, activation=tf.nn.gelu)(representation)
        features = layers.Dropout(0.5)(features)

        # Na última camada, temos:
        # Gera um vetor com num_classes valores (logits).
        # Cada valor representa o grau de associação com uma classe.
        logits = layers.Dense(self.num_classes)(features)

        # Cria o modelo final com entrada inputs e saída logits.
        model = tf.keras.Model(inputs=inputs, outputs=logits)

        # Realiza a chamada da função para compilar e treinar o modelo
        # self.vit_compile_train(model, train_generator, validation_generator, eps)
        return model
'''
