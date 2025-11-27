import tensorflow as tf
import pandas as pd
import os


class CheXpertDataLoader:
    """
    Data loader oficial para o CheXpert-small dataset.
    Carrega imagens com labels multi-label (14 patologias),
    resolve incertezas (U-Zeros), normaliza, aplica augmentations
    e retorna tf.data.Dataset para treino e validação.
    """

    # 14 patologias do CheXpert na ordem oficial das colunas
    CHEXPERT_LABELS = [
        "No Finding",
        "Enlarged Cardiomediastinum",
        "Cardiomegaly",
        "Lung Opacity",
        "Lung Lesion",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural Effusion",
        "Pleural Other",
        "Fracture",
        "Support Devices"
    ]

    def __init__(
            self,
            data_dir,  # data_dir: caminho para a pasta CheXpert-v1.0-small
            train_csv="train.csv",  # arquivo CSV de treino
            valid_csv="valid.csv",  # arquivo CSV de validação
            image_size=(320, 320),  # tamanho final das imagens
            batch_size=32,  # tamanho do batch
            uncertainty_policy="zeros",  # "zeros", "ones" ou "ignore"
            augment=True  # se True, aplica augmentations no treino
    ):

        self.data_dir = data_dir
        self.train_csv_path = os.path.join(data_dir, train_csv)
        self.valid_csv_path = os.path.join(data_dir, valid_csv)
        self.image_size = image_size
        self.batch_size = batch_size
        self.uncertainty_policy = uncertainty_policy
        self.use_augmentation = augment

        # Normalização padrão ImageNet
        self.mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        self.std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

        # Augmentations recomendados pela literatura
        self.augmentation_layer = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1)
        ])

    # Carregar CSV e processar rótulos (U-Zeros)
    def load_csv(self, csv_path):
        df = pd.read_csv(csv_path)  # Lê o arquivo .csv
        df = df.fillna(0)  # valores NA viram 0

        # aplicar política para -1 (incertezas)
        if self.uncertainty_policy == "zeros":
            df = df.replace(-1, 0)
        elif self.uncertainty_policy == "ones":
            df = df.replace(-1, 1)
        elif self.uncertainty_policy == "ignore":
            # você selecionaria apenas exemplos com rótulos certos
            df = df[df[self.CHEXPERT_LABELS].isin([0, 1]).all(axis=1)]

        return df

    # Method 2: carregar imagem do disco
    def load_image(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, self.image_size)

        # normalização ImageNet
        img = tf.cast(img, tf.float32) / 255.0
        img = (img - self.mean) / self.std

        return img

    def normalize_path(self, p):
        # Remove prefixos duplicados no CSV, caso existam
        p = p.replace("CheXpert-v1.0-small/", "").lstrip("/\\")
        return os.path.join(self.data_dir, p)

    # Method 3: preparação do dataset tf.data
    def prepare_dataset(self, df, training=True):

        # image_paths = df["Path"].apply(lambda p: os.path.join(self.data_dir, p)).tolist()
        image_paths = df["Path"].apply(self.normalize_path).tolist()
        print("Exemplo de caminho montado:", image_paths[0])
        labels = df[self.CHEXPERT_LABELS].values.astype("float32")

        ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

        def process(path, label):
            img = self.load_image(path)

            if training and self.use_augmentation:
                img = self.augmentation_layer(img)

            return img, label

        AUTOTUNE = tf.data.AUTOTUNE

        ds = ds.shuffle(2000) if training else ds
        ds = ds.map(process, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(AUTOTUNE)

        return ds

    # Method 4: interface principal
    def get_datasets(self):
        print("Carregando CSVs CheXpert...")

        train_df = self.load_csv(self.train_csv_path)
        valid_df = self.load_csv(self.valid_csv_path)

        print(f"Treino: {len(train_df)} imagens")
        print(f"Validação: {len(valid_df)} imagens")

        train_ds = self.prepare_dataset(train_df, training=True)
        valid_ds = self.prepare_dataset(valid_df, training=False)

        return train_ds, valid_ds, len(train_df), len(valid_df)
