import tensorflow as tf
# TFDS é a biblioteca que disponibiliza datasets prontos, inclusive ImageNet.
# TFDS tem suporte para ImageNet2012 e facilita carregar o dataset com streaming (sem persistir 150GB localmente).
import tensorflow_datasets as tfds


class loader:

    # A ResNet opera em crops 224×224
    # usar 224 mantém compatibilidade com arquitetura e FC final (1000 classes)
    def __init__(self, image_size=224):
        self.image_size = image_size

    @staticmethod
    def resize_preserve(self, image, min_side=256, max_side=480):
        """
        Redimensiona a imagem preservando o aspecto, definindo o menor lado como
        um valor aleatório S ~ Uniform(min_side, max_side), como descrito no artigo:
        'The image is resized with its shorter side randomly sampled in [256, 480]
         for scale augmentation.'

        Parâmetros:
            image: Tensor uint8 ou float32 [H, W, 3]
            min_side: menor valor possível do lado curto (256)
            max_side: maior valor possível (480)
        Retorno:
            imagem redimensionada [H', W', 3], aspecto preservado
        """

        # Seleciona aleatoriamente o tamanho desejado do menor lado
        target_size = tf.random.uniform([], min_side, max_side + 1, dtype=tf.int32)

        # Obtém as dimensões originais
        shape = tf.cast(tf.shape(image)[:2], tf.float32)  # [H, W]
        height, width = shape[0], shape[1]

        # Descobre qual é o menor lado original
        shorter_side = tf.minimum(height, width)

        # Fator de escala para atingir target_size no menor lado
        scale = tf.cast(target_size, tf.float32) / shorter_side

        # Calcula novas dimensões preservando aspecto
        new_height = tf.cast(tf.round(height * scale), tf.int32)
        new_width = tf.cast(tf.round(width * scale), tf.int32)

        # Redimensiona com interpolação bilinear
        image = tf.image.resize(image, [new_height, new_width], method=tf.image.ResizeMethod.BILINEAR)

        return image

    def preprocess(self, image, label):
        """
            Aplica o pipeline de augmentation para treino (resize → random crop → flip → cast / escala).
            Tem o objetivo de aproximar-se da pipeline usada pelo artigo (data augmentation com crops e flips).
        """
        # Prática de escalar o menor lado aleatoriamente em [256,480] e então fazer crop 224.

        short_side = tf.random.uniform([], minval=256, maxval=480)  # scale jitter: draw short side in [256,480]
        image = self.resize_preserve(image, short_side)  # implementar preservando aspect ratio
        image = tf.image.random_crop(image, [224, 224, 3])  # random 224 crop

        # Faz um recorte aleatório 224×224 da imagem redimensionada.
        image = tf.image.random_crop(image, [self.image_size, self.image_size, 3])
        # Aplica uma inversão horizontal com probabilidade 50%.
        image = tf.image.random_flip_left_right(image)

        # Normalização mencionada pelo artigo “per-pixel mean subtracted”
        image = tf.cast(image, tf.float32) / 255.0
        imagenet_mean = tf.constant([0.485, 0.456, 0.406])
        imagenet_std = tf.constant([0.229, 0.224, 0.225])
        image = (image - imagenet_mean) / imagenet_std
        return image, label

    def get_imagenet(self, batch_size=256):
        # instruir TFDS a tentar ler o dataset do bucket público do TFDS / GCS, evitando
        # download local (streaming / leitura remota).
        read_config = tfds.ReadConfig()  # <- usa o bucket público GCS da ImageNet
        # retorna o split de treino da ImageNet2012.
        train_ds = tfds.load(
            "imagenet2012",
            split="train",
            shuffle_files=True,  # Pede que os shards sejam lidos embaralhados
            as_supervised=True,  # Retorna pares (image,label) em vez de dicts (facilita o .map(self.preprocess)).
            read_config=read_config,
        )

        # OBS:TFDS recomenda shuffle_files = True para grandes datasets fragmentados
        # (melhor aleatoriedade entre os fragmentos).

        # carrega o split de validação. Tipicamente não se utiliza o embaralhamento (shuffle) para avaliação,
        # uma vez que ela deve ser determinística (ordem não importa para métricas finais) - shuffle_files=False
        val_ds = tfds.load(
            "imagenet2012",
            split="validation",
            shuffle_files=False,
            as_supervised=True,
            read_config=read_config,
        )

        # Construção efetiva dos datasets, aplicando de fato as configurações estabelecidas préviamente
        # Observação: as práticas de aplicadas no pipeline de construção do dataset são recomendações do
        # próprio tensorflow (Isso é válido?)
        train_ds = (
            train_ds
            # Embaralha os elementos com buffer de 1024 (mantém 1024 itens em buffer e escolhe igualdade aleatória)
            # O embaralhamento durante treino evita ordem enviesada; tamanhos de buffer (1024) são escolhas heurísticas
            .shuffle(1024)
            # aplica a função preprocess em paralelo para acelerar CPU-bound preprocessing.
            # AUTOTUNE escolhe threads automaticamente.
            .map(self.preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            # Agrupa amostras em batches
            .batch(batch_size)
            # Permite que tf.data prepare o próximo batch enquanto a GPU treina no batch atual.
            # melhora throughput e mantém GPU ocupada
            .prefetch(tf.data.AUTOTUNE)
        )

        val_ds = (
            val_ds
            # Para validação redimensiona a imagem para 224×224
            .map(lambda x, y: (tf.image.resize(x, [self.image_size, self.image_size]), y),
                 num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        return train_ds, val_ds
