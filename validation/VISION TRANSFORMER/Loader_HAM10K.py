import tensorflow as tf
import pandas as pd
import os


class HAM10000Loader:
    """
    Loader oficial para o dataset HAM10000, com as seguintes reponsabilidades:

        ✔ Carrega train.csv / val.csv / test.csv
        ✔ Lê e normaliza as imagens
        ✔ Converte os rótulos (dx) para índices numéricos
        ✔ Aplica augmentations no treino
        ✔ Retorna tf.data.Dataset para treino, validação e teste

    """

    # As 7 classes oficiais do HAM10000 (estáveis e padronizadas)
    DX_LABELS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

    def __init__(self, root_dir, image_size=(224, 224), batch_size=32, augment=True):

        # 'root_dir' é a pasta principal que contém os arquivos train.csv, val.csv, test.csv e o diretório \images)
        self.root_dir = root_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.use_augment = augment

        # Normalização padrão do ViT (ImageNet mean/std)
        self.mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        self.std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

        # Augmentation para treino
        self.augment_layer = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1)
        ])

    # ==================================================================================================================

    # Leitura dos arquivos CSV's
    def load_csv(self, filename):

        csv_path = os.path.join(self.root_dir, filename)
        df = pd.read_csv(str(csv_path))

        # Converte dx (string) para índice
        df["label"] = df["dx"].apply(lambda x: self.DX_LABELS.index(x))

        return df

    # ==================================================================================================================

    # Pré-processamento da imagem
    def process_image(self, path):
        img = tf.io.read_file(path)  # Leitura do arquivo
        img = tf.image.decode_jpeg(img, channels=3)  # Decodificação do formato
        img = tf.image.resize(img, self.image_size)  # Redimensionamento

        # Normalização padrão
        img = tf.cast(img, tf.float32) / 255.0
        img = (img - self.mean) / self.std

        return img

    # ==================================================================================================================

    # Função usada dentro do map() do Dataset
    def process_sample(self, path, label, training):

        """
            Parametros:
                * path é um tensor tf.string que contém o caminho da imagem (ex.: "C:/.../ISIC_0001.jpg")
                * label é um tensor escalar inteiro (ex.: 3) — o índice da classe antes do one-hot.
                * training é um booleano Python passado a partir do make_dataset (indica se estamos no modo treino).

            Essa função é chamada dentro do .map() do tf.data.Dataset para pré-processar cada exemplo.
        """

        img = self.process_image(path)  # Retorno da imagem pré-processada

        # Caso o processo esteja em treinamento e caso a augmentation seja selecionada
        # aplica a pilha de augmentations (RandomFlip, RandomRotation, RandomZoom) definida no __init__.
        # OBSERVAÇÃO: essas layers são estocásticas (só alteram a imagem em tempo de execução).
        if training and self.use_augment:
            img = self.augment_layer(img)

        # Converte o índice escalar label num vetor one-hot: se label=3 e len(self.DX_LABELS)==7 → [0,0,0,1,0,0,0].
        label = tf.one_hot(label, depth=len(self.DX_LABELS))
        # Retorna a tupla (img, label) que tf.data agrupará em batches automaticamente (depois do .batch()).
        return img, label

    # ==================================================================================================================

    # Criação de tf.data.Dataset
    def make_dataset(self, df, training=True):

        # Cria um tensor tf.constant do tipo tf.string com todos os caminhos.
        paths = tf.constant(df["image_path"].tolist())
        # Mesmo raciocínio para labels: tensor escalar 1D com inteiros (ex.: dtype int64).
        labels = tf.constant(df["label"].tolist())
        # Cria um tf.data.Dataset que produz pares (path, label) por exemplo.
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))

        # Embaralha a ordem dos elementos com buffer size 5000.
        # Em geral: shuffle antes do map é recomendado para eficência.
        if training:
            # Buffer size: idealmente >= tamanho do dataset para perfeita aleatoriedade;
            # com 5000 você tem boa randomização sem esgotar memória.
            ds = ds.shuffle(5000)

        # Aplica a função process_sample a cada elemento do dataset.
        ds = ds.map(
            lambda p, l: self.process_sample(p, l, training),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Agrupa exemplos em batches. Resultado de img: (batch_size, H, W, 3); label: (batch_size, 7).
        ds = ds.batch(self.batch_size)

        if training:
            ds = ds.repeat()

        # Pré-carrega batches no background para manter GPU ocupada (pipeline assíncrono).
        # Muito importante para throughput.
        ds = ds.prefetch(tf.data.AUTOTUNE)

        return ds

    # ==================================================================================================================

    # Interface principal
    def get_datasets(self):
        """
            Retorna:
            train_ds, val_ds, test_ds,
            n_train, n_val, n_test
        """

        # Carregamento dos arquivos .csv
        train_df = self.load_csv("train.csv")
        val_df = self.load_csv("val.csv")
        test_df = self.load_csv("test.csv")

        # Criação dos datasets, usando os dados carregados pelo .csv
        train_ds = self.make_dataset(train_df, training=True)
        val_ds = self.make_dataset(val_df, training=False)
        test_ds = self.make_dataset(test_df, training=False)

        print("\n=== HAM10000 STATISTICS ===")
        print("Treino:", len(train_df))
        print("Val:", len(val_df))
        print("Teste:", len(test_df))
        print("============================\n")

        return train_ds, val_ds, test_ds, len(train_df), len(val_df), len(test_df)
