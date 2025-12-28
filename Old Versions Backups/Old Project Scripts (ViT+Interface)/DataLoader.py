import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Impede o TF de alocar toda a VRAM da GPU
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

import os
# Forçar TensorFlow a rodar no CPU, visto que a GPU não é necessária para carregar imagens.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class DataLoader:

    def __init__(self, path, img_size=224, batch_size=16, val_split=0.2):
        """
            Construtor da classe, aqui serão definidos
            o tamanho padrão das imagens e o batch size
            para a definição dos grupos de validação e
            treinamento. Adicionalmente são definidos
            os layers para augmentation.
        """

        self.path = path
        self.img_size = img_size
        self.batch_size = batch_size
        self.val_split = val_split

        # Augmentations
        self.augment = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.10),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
            tf.keras.layers.RandomBrightness(0.25),
            tf.keras.layers.RandomContrast(0.25),
            tf.keras.layers.GaussianNoise(0.05),
            tf.keras.layers.RandomCrop(img_size, img_size),
        ])

    """
        A função a baixo é responsável por gerenciar os dados
        presentes no diretório selecionado. Aqui as imagens são
        submetidas a um pré-processamento e separadas em grupos
        destinados ao treino e à validação
    """

    def process_data(self):

        """
            É uma ferramenta muito útil quando trabalhamos com muitos
            arquivos de imagem e é desejado:

                a - Evitar carregar tudo na memória.
                b - Automatizar o carregamento, normalização e divisão treino / validação.
                c - Aplicar transformações como rotação, zoom, flips, etc.
        """

        train_ds_raw = tf.keras.utils.image_dataset_from_directory(
            self.path,
            labels="inferred",
            label_mode="int",
            validation_split=self.val_split,
            subset="training",
            seed=123,
            image_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            shuffle=True,
        )

        val_ds_raw = tf.keras.utils.image_dataset_from_directory(
            self.path,
            labels="inferred",
            label_mode="int",
            validation_split=self.val_split,
            subset="validation",
            seed=123,
            image_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            shuffle=False,
        )

        class_names = train_ds_raw.class_names
        num_classes = len(class_names)

        AUTOTUNE = 1

        train_ds = (
            train_ds_raw
            .map(lambda x, y: (self.augment(x, training=True) / 255.0, y),
                 num_parallel_calls=AUTOTUNE)
            .prefetch(AUTOTUNE)
        )

        val_ds = (
            val_ds_raw
            .map(lambda x, y: (x / 255.0, y),
                 num_parallel_calls=AUTOTUNE)
            .prefetch(AUTOTUNE)
        )

        steps_train = tf.data.experimental.cardinality(train_ds_raw).numpy()
        steps_val = tf.data.experimental.cardinality(val_ds_raw).numpy()

        log_training_samples = (
            f"Dataset de treino com {steps_train * self.batch_size} imagens "
            f"em {num_classes} classes."
        )

        log_validation_samples = (
            f"Dataset de validação com {steps_val * self.batch_size} imagens."
        )

        class_names = train_ds_raw.class_names
        log_indexes = f"Classes: {class_names}"

        return (
            train_ds,
            val_ds,
            log_training_samples,
            log_validation_samples,
            log_indexes,
            num_classes,
            steps_train,
            steps_val
        )