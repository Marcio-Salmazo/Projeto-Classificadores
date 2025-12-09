import tensorflow as tf
import numpy as np
import random
import os

from tensorflow.keras import mixed_precision
from ResNet50_pure import build_resnet50
from ResNet50_trainer import Trainer
from Network_Validation.Process_ImageNet import create_imagenet_tfrecords_streaming, load_tfrecords
from Process_Datase import apply_preprocessing


# 1. Seeds para reprodutibilidade
def set_global_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    print(f"🔒 Seeds fixados (seed={seed}) para reprodutibilidade.")


# 2. Mixed Precision (opcional, recomendado em GPUs RTX/Ampere)
def enable_mixed_precision():
    mixed_precision.set_global_policy("mixed_float16")
    print("Mixed precision ativada (float16) para acelerar o treinamento.")


def tfrecords_exist_safe(tfrecord_dir, num_train_shards=1024, num_val_shards=128):
    train_dir = os.path.join(tfrecord_dir, "train")
    val_dir = os.path.join(tfrecord_dir, "validation")

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        return False

    train_files = os.listdir(train_dir)
    val_files = os.listdir(val_dir)

    # Verifica contagem exata de shards
    train_ok = len(train_files) == num_train_shards
    val_ok = len(val_files) == num_val_shards

    if not (train_ok and val_ok):
        print("TFRecord directory exists, but shard count is incorrect.")
        print(f" Train shards: {len(train_files)} (expected: {num_train_shards})")
        print(f" Val shards:   {len(val_files)} (expected: {num_val_shards})")
        return False

    # Verifica se os nomes seguem o padrão correto
    if not all("train-" in f for f in train_files):
        return False

    if not all("validation-" in f for f in val_files):
        return False

    return True


class Main:
    """
        Classe principal que integra:
        - carregamento do dataset
        - construção da ResNet-50
        - treinamento do modelo
    """

    def __init__(self, image_size=224, batch_size=256, num_classes=1000, epochs=120, initial_lr=0.1,
                 momentum=0.9, weight_decay=1e-4):
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.epochs = epochs
        self.initial_lr = initial_lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.train_ds = None
        self.val_ds = None
        self.model = None

        # Diagnóstico GPU
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            print(f"GPU detectada: {gpus}")
        else:
            print("Nenhuma GPU detectada. Treinamento será lento.")

    def build_model(self):
        print("Construindo modelo ResNet-50 fiel ao paper...\n")

        model = build_resnet50(
            input_shape=(self.image_size, self.image_size, 3),
            num_classes=self.num_classes,
            include_top=True,
            weight_decay=self.weight_decay
        )

        model.summary()
        self.model = model
        print("\nLOG --- Modelo criado.\n")

    def load_data(self):

        print("\nPreparando ImageNet (TFRecords Streaming)\n")

        # Caminhos dos arquivos .tar originais
        train_tar = (r"C:/Users/marci_wawp/Desktop/Arquivos/Mestrado/Projeto-Classificadores/"
                     r"Datasets/DATASET IMAGENET/ILSVRC2012_img_train.tar")
        val_tar = (r"C:/Users/marci_wawp/Desktop/Arquivos/Mestrado/Projeto-Classificadores/"
                   r"Datasets/DATASET IMAGENET/ILSVRC2012_img_val.tar")

        # Arquivo oficial de anotações
        val_annotations = (r"C:/Users/marci_wawp/Desktop/Arquivos/Mestrado/Projeto-Classificadores/"
                           r"Network Validation/VISION TRANSFORMER/Validation_Notes.txt")

        # Diretório dos TFRecords
        tfrecord_dir = (r"C:/Users/marci_wawp/Desktop/Arquivos/Mestrado/"
                        r"Projeto-Classificadores/Datasets/DATASET IMAGENET")

        # Criação dos TFRecords se necessário
        if not tfrecords_exist_safe(tfrecord_dir):
            print("Nenhum TFRecord detectado...\n")
            create_imagenet_tfrecords_streaming(
                train_tar=train_tar,
                val_tar=val_tar,
                out_dir=tfrecord_dir,
                num_train_shards=1024,
                num_val_shards=128,
                val_annotations_file=val_annotations
            )
            print("\nTFRecords criados com sucesso!\n")
        else:
            print("TFRecords detectados - pulando etapa de criação.\n")

        print("Carregando TFRecords brutos...\n")

        # Carrega dados SEM PRE-PROCESSAMENTO
        train_ds = load_tfrecords(
            tfrecord_dir=tfrecord_dir,
            batch_size=self.batch_size,
            train=True,
            image_size=self.image_size
        )

        val_ds = load_tfrecords(
            tfrecord_dir=tfrecord_dir,
            batch_size=self.batch_size,
            train=False,
            image_size=self.image_size
        )

        # Aplica o pré-processamento fiel ao paper
        print("Aplicando pré-processamento (ResNet paper)...\n")
        train_ds, val_ds = apply_preprocessing(train_ds, val_ds)

        # Atribuição final
        self.train_ds = train_ds
        self.val_ds = val_ds
        print("\nImageNet carregado e pré-processado com sucesso.\n")

    def train(self):
        trainer = Trainer(
            model=self.model,
            train_ds=self.train_ds,
            val_ds=self.val_ds,
            epochs=self.epochs,
            initial_lr=self.initial_lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            patience=3,
            log_dir="logs",
            checkpoint_path="checkpoints/resnet50_best.h5"
        )

        print("Iniciando treinamento...\n")
        trainer.train()
        print("\nLOG --- Pipeline treinamento finalizado.\n")

    def run(self):

        set_global_seed(42)
        enable_mixed_precision()
        self.load_data()
        self.build_model()
        self.train()


# ======================================================================================================================
#                                             Execução direta do main.py
# ======================================================================================================================
if __name__ == "__main__":
    Main().run()
