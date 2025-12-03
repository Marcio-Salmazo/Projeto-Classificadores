import tensorflow as tf
import pandas as pd
import os


class CheXpertDataLoader:
    """
        Data loader oficial para o CheXpert-small dataset.

        RESPONSABILIDADES:

        ✔ Ler os CSVs (train.csv e valid.csv)
        ✔ Aplicar política de incerteza (U-Zeros, U-Ones, U-Ignore)
        ✔ Corrigir paths das imagens
        ✔ Carregar imagens do disco (tf.io.read_file)
        ✔ Redimensionar + normalizar com estatísticas ImageNet
        ✔ Criar tf.data.Dataset para treino e validação
        ✔ Garantir ordens determinísticas (obrigatório no CheXpert)
    """

    # As 14 patologias do CheXpert na ordem EXACTA usada no artigo
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

    # ==================================================================================================================

    # Configuração inicial do loader (construtor)
    def __init__(
            self,
            data_dir,
            train_csv="train.csv",
            valid_csv="valid.csv",
            image_size=(320, 320),
            batch_size=32,
            uncertainty_policy="zeros",
            augment=True
    ):
        """
            Parâmetros:
                data_dir             Caminho da pasta contendo train.csv, valid.csv e pastas de imagens
                image_size           Tamanho final das imagens (CheXpert usa 320x320)
                batch_size           Batch size usado pelo tf.data.Dataset
                uncertainty_policy   Como tratar valores -1:
                                     - "zeros":  -1 => 0      (U -> Negative)
                                     - "ones":   -1 => 1      (U -> Positive)
                                     - "ignore": remover exemplos com U
                augment              Se True, ativa augmentação no treino
        """
        self.data_dir = data_dir
        self.train_csv_path = os.path.join(data_dir, train_csv)
        self.valid_csv_path = os.path.join(data_dir, valid_csv)
        self.image_size = image_size
        self.batch_size = batch_size
        self.uncertainty_policy = uncertainty_policy
        self.use_augmentation = augment

        # Normalização oficial ImageNet (usada no artigo CheXpert para ResNet50)
        self.mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        self.std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

        # Augmentações leves (o artigo usa horizontais leves, sem flips verticais ou rotações exageradas)
        self.augmentation_layer = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1)
        ])

    # ==================================================================================================================

    # Lê o arquivo .CSV do CheXpert e aplica a política de tratamento das incertezas (-1).
    def load_csv(self, csv_path):

        df = pd.read_csv(csv_path)  # Lê o arquivo .csv
        df = df.fillna(0)  # valores definidos como NA (Nulo) viram 0

        # Aplica a política para rótulos '-1' (incertezas)
        if self.uncertainty_policy == "zeros":
            df = df.replace(-1, 0)
        elif self.uncertainty_policy == "ones":
            df = df.replace(-1, 1)
        elif self.uncertainty_policy == "ignore":
            # Mantém apenas linhas onde todas as labels ∈ {0,1}
            df = df[df[self.CHEXPERT_LABELS].isin([0, 1]).all(axis=1)]

        return df

    # ==================================================================================================================

    ''' 
    # Carregamento das imagens que compõem o dataset
    def load_image(self, image_path):

        """
            Os comandos abaixo são responsáveis por ler as imagem do disco,
            decodificar o formato JPEG, redimensionar e normaliza com médias e desvios da ImageNet.
        """
        img = tf.io.read_file(image_path)  # Leitura do arquivo
        img = tf.image.decode_jpeg(img, channels=3)  # Decodificação do formato
        img = tf.image.resize(img, self.image_size)  # Redimensionamento

        # Normalização aos moldes da ImageNet
        img = tf.cast(img, tf.float32) / 255.0
        img = (img - self.mean) / self.std

        return img
    '''

    # ==================================================================================================================

    # Normalização de path (resolve prefixos duplicados do CSV)
    def normalize_path(self, p):
        """
            Os CSVs do CheXpert-small possuem paths como:
            'CheXpert-v1.0-small/train/patientXXXXX/studyY/viewZ.jpg'
            Esta função converte isso para um path absoluto válido no Windows.

            Por exemplo:
                Original -> CheXpert-v1.0-small/train/patient.../study.../view.jpg
                Normalizado -> C:/meu/dataset/train/patient.../study.../view.jpg

        """
        p = p.replace("CheXpert-v1.0-small/", "")  # Remove prefixo extra
        p = p.replace("\\", "/")  # Normaliza barras
        p = p.lstrip("/")  # Normaliza barras
        return os.path.join(self.data_dir, p).replace("\\", "/")  # Cria o path absoluto

    # ==================================================================================================================

    # Montagem do tf.data.Dataset (processamento + augment)
    def prepare_dataset(self, paths, labels, training=True):

        """
            Converte listas de paths e labels em Dataset eficiente para GPU.
            - Mantém ORDEM fixa no dataset de validação (essencial)
            - Usa augmentation no treino
            - Normaliza no padrão ImageNet
        """
        paths = tf.constant(paths)  # Fixa a ordem dos caminhos
        labels = tf.constant(labels)  # Fixa a ordem dos rótulos
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))

        def process(path, label):

            """
                Função aplicada a cada exemplo do dataset para:
                    - ler arquivo
                    - decodificar JPEG
                    - redimensionar
                    - normalizar
                    - aplicar augment (somente no treino)
            """

            # Os comandos abaixo são responsáveis por ler as imagem do disco,
            # decodificar o formato JPEG, redimensionar e normaliza com médias e desvios da ImageNet.

            img = tf.io.read_file(path) # Leitura do arquivo
            img = tf.image.decode_jpeg(
                img,
                channels=3,
                try_recover_truncated=True
            )  # Decodificação do formato

            # Normalização aos moldes da ImageNet
            img = tf.image.resize(img, self.image_size)  # Redimensionamento
            img = tf.cast(img, tf.float32) / 255.0
            img = (img - self.mean) / self.std

            # Augmentação apenas no treino
            if training and self.use_augmentation:
                img = self.augmentation_layer(img)

            return img, label

        AUTOTUNE = tf.data.AUTOTUNE

        # Aplica o shuffle (embaralhamento) + parallel map estritamente durante o treinamento
        if training:
            ds = ds.shuffle(buffer_size=10000)
            ds = ds.map(process, num_parallel_calls=AUTOTUNE)
        else:
            # Para a validação a ordem deve ser preservada, portanto
            # se usar paralelismo nesta etapa a ordem pode se perder.
            ds = ds.map(process)

        # Lotes (BATCH) + prefetch (permite pipeline GPU)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(AUTOTUNE)

        return ds

    # ==================================================================================================================

    # Método principal — devolve datasets e os respectivos metadados
    def get_datasets(self):
        print("Carregando CSVs...")
        """
                Retorna:
                - train_ds   (tf.data.Dataset)
                - valid_ds   (tf.data.Dataset)
                - n_train    (numero de amostras de treino)
                - n_valid    (numero de amostras de validação)
                - valid_paths (lista de paths usada para calcular AUC por estudo)
                    
            A ordem de valid_paths deve ser a mesma de valid_ds!
        """

        # Lê dataframes já com política U aplicada
        train_df = self.load_csv(self.train_csv_path)
        valid_df = self.load_csv(self.valid_csv_path)

        # Converte paths relativos em paths absolutos
        train_paths = train_df["Path"].apply(self.normalize_path).tolist()
        valid_paths = valid_df["Path"].apply(self.normalize_path).tolist()

        # Matriz de labels (float32 para BCE)
        train_labels = train_df[self.CHEXPERT_LABELS].values.astype("float32")
        valid_labels = valid_df[self.CHEXPERT_LABELS].values.astype("float32")

        # Montagem dos datasets
        train_ds = self.prepare_dataset(train_paths, train_labels, training=True)
        valid_ds = self.prepare_dataset(valid_paths, valid_labels, training=False)

        return train_ds, valid_ds, len(train_df), len(valid_df), valid_paths

    # ==================================================================================================================
