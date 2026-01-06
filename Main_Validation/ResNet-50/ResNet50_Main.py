import tensorflow as tf
import numpy as np
import random
import os
from tensorflow.keras import mixed_precision
from ResNet50_pure import build_resnet50
from ResNet50_Trainer import ResNet50Trainer
from Process_Datase import apply_preprocessing
from Main_Validation.Process_ImageNet import load_tfrecords
import subprocess

# Diagnóstico GPU
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"GPU detectada: {gpus}")
else:
    print("Nenhuma GPU detectada. Treinamento será lento.")

# ======================================================================================================================
# Definição de seeds para reprodutibilidade

def set_global_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    print(f"🔒 Seeds fixados (seed={seed}) para reprodutibilidade.")

# ======================================================================================================================
# Ativação de mixed Precision (opcional, recomendado em GPUs RTX/Ampere)

def enable_mixed_precision():
    mixed_precision.set_global_policy("mixed_float16")
    print("Mixed precision ativada (float16) para acelerar o treinamento.")

# ======================================================================================================================
# CAMINHOS DE AMBIENTES, TFRECORDS E SCRIPTS ASSOCIADOS

TF_ENV_PYTHON = (r"C:/Users/marci_plgx30x/Desktop/Arquivos/Projetos do Mestrado (Git)/Projeto-Classificadores"
                 r"/.tf_venv/Scripts/python.exe")

TFRECORD_SCRIPT = (r"C:/Users/marci_plgx30x/Desktop/Arquivos/Projetos do Mestrado (Git)/Projeto-Classificadores"
                   r"/Validation/Create_TFRecords.py")

# Diretório onde serão criados: /train/*.tfrecord e /validation/*.tfrecord
TFRECORD_DIR = (r"C:/Users/marci_plgx30x/Desktop/Arquivos/Projetos do Mestrado (Git)/Projeto-Classificadores"
                r"/Datasets/ImageNet_TFRecords")

# Diretório de checkpoints do ViT
OUTPUT_DIR = (r"C:/Users/marci_plgx30x/Desktop/Arquivos/Projetos do Mestrado (Git)"
              r"/Projeto-Classificadores/Validation/Checkpoints"
)

# Caminhos para o checkpoint
CHECKPOINT_PATH = "checkpoints/resnet50_best.h5"

# ======================================================================================================================
# PARÂMETROS EXIGIDOS PELA RESNET

IMAGE_SIZE = 224
NUM_CLASSES = 1000
TRAIN_SIZE = 1281167    # Valor oficial da ImageNet (Modificar no futuro)
VAL_SIZE = 50000        # Valor oficial da ImageNet (Modificar no futuro)
BATCH_SIZE = 256

EPOCHS = 120
INITIAL_LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
TRAIN_SIZE = None
VAL_SIZE = None
LOG_DIR = "logs"
CHECKPOINT = CHECKPOINT_PATH

# ======================================================================================================================
# VERIFICAÇÃO DOS TFRECORDS

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

# ======================================================================================================================
# CONSTRUÇÃO DO MODELO

def build_model():

    print("==================================================================")
    print(">> CONSTRUINDO MODELO RESNET-50\n")

    model = build_resnet50(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        num_classes=NUM_CLASSES,
        include_top=True,
        weight_decay=WEIGHT_DECAY
    )

    model.summary()
    return model

# ======================================================================================================================
# CARREGAMENTO DOS TFRECORDS

def load_data():

    print("==================================================================")
    print(">> CARREGAMENTO E PREPARAÇÃO DOS TFRECORDS DA IMAGENET\n")

    # ----------------------------------------------------------------
    #       1. CRIAÇÃO DOS TFRECORDS (somente se ainda não existirem)
    # ----------------------------------------------------------------
    if not tfrecords_exist_safe(TFRECORD_DIR):
        print("==================================================================")
        print(">> TFRECORDS NÃO ENCONTRADOS, INICIANDO A GERAÇÃO DOS ARQUIVOS\n")

        result = subprocess.run(
            [TF_ENV_PYTHON, TFRECORD_SCRIPT],
            capture_output=True,
            text=True
        )

        print(result.stdout)
        print(result.stderr)

        print("==================================================================")
        print("\n>> TFRECORDS CRIADOS COM SUCESSO!\n")
    else:
        print("==================================================================")
        print(">> TFRECORDS JÁ EXISTEM, PULANDO A ETAPA DE CRIAÇÃO\n")

    # Carrega dados SEM PRE-PROCESSAMENTO
    train_ds = load_tfrecords(
        tfrecord_dir=TFRECORD_DIR,
        batch_size=BATCH_SIZE,
        train=True,
        image_size=IMAGE_SIZE
    )

    val_ds = load_tfrecords(
        tfrecord_dir=TFRECORD_DIR,
        batch_size=BATCH_SIZE,
        train=False,
        image_size=IMAGE_SIZE
    )

    # Aplica o pré-processamento fiel ao paper
    print("==================================================================")
    print("APLICANDO O PRÉ-PROCESSAMENTO DA BASE, DESCRITO PELO ARTIGO (...)\n")
    train_ds_processed, val_ds_processed = apply_preprocessing(train_ds, val_ds)

    print("==================================================================")
    print("BASE DEVIDAMENTE CARREGADA E PRÉ-PROCESSADA")
    return train_ds_processed, val_ds_processed

# ======================================================================================================================
# EXECUÇÃO PRINCIPAL

def main():

    train_dataset, val_dataset = load_data()
    model = build_model()

    trainer = ResNet50Trainer(

        model=model,
        train_ds=train_dataset,
        val_ds=val_dataset,

        num_classes=NUM_CLASSES,
        batch_size=BATCH_SIZE,
        epochs=120,
        initial_lr=0.1,
        momentum=0.9,
        weight_decay=1e-4,
        train_size=TRAIN_SIZE,
        val_size=VAL_SIZE,
        log_dir="logs",
        checkpoint_path="checkpoints/resnet50_best.h5"
    )

    print("==================================================================")
    print("INICIANDO TREINAMENTO (...)")
    trainer.train()
    print("==================================================================")
    print("PIPELINE DE TREINAMENTO FINALIZADO COM SUCESSO")















