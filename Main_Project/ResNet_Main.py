""" Execução AUTOMÁTICA do pipeline ResNet-50 segundo o artigo: """

import tensorflow as tf
import numpy as np
import random
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow.keras import mixed_precision
from Main_Project.DataLoader import DataLoader
from ResNet50.ResNet50_pure import build_resnet50
from ResNet50.ResNet50_Trainer import ResNet50Trainer

# ======================================================================================================================
# REMOVE WARNINGS E INFO DO LOG, MANTENDO APENAS ERROS CRÍTICOS

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ======================================================================================================================
# PARÂMETROS EXIGIDOS PELA RESNET

RUN_NAME = "Sanity_Check_5eps"

IMAGE_SIZE = 224
BATCH_SIZE = 32
VAL_SPLIT = 0.2

EPOCHS = 5
NUM_CLASSES = 3
INITIAL_LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4

# ======================================================================================================================
# CAMINHOS DE AMBIENTES, TFRECORDS E SCRIPTS ASSOCIADOS

# Diretório de logs de treinamento
LOG_DIR = f"ResNet50/Results/logs/{RUN_NAME}"

# Caminho e nome para o checkpoint do treinamento
CHECKPOINT_PATH = f"ResNet50/Results/Checkpoints_{RUN_NAME}/best_weights.h5"


# ======================================================================================================================
# DIAGNÓSTICO DO USO DA GPU (OPCIONAL)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"GPU detectada: {gpus}")
else:
    print("Nenhuma GPU detectada. Treinamento será lento.")

# ======================================================================================================================
# DEFINIÇÃO DE SEEDS PARA REPRODUTIBILIDADE

def set_global_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"Seeds fixados (seed={seed}) para reprodutibilidade.")

# ======================================================================================================================
# ATIVAÇÃO DO MIXED PRECISION

def enable_mixed_precision():
    mixed_precision.set_global_policy("mixed_float16")
    print("Mixed precision ativada (float16) para acelerar o treinamento.")

# ======================================================================================================================
# FUNÇÃO PARA OBTER O CAMINHO DE UM DIRETÓRIO VIA EXPLORER

def open_directory():
    """
        O tkinter é utilizado para exibir janela do explorer a fim de selecionar a pasta contendo o Dataset.
            * root = tk.Tk() - instância do tkinter
            * root.withdraw() -  Oculta a janela principal (para exibir apenas o pop-up)
            * filedialog.askdirectory(title="") - Abre a janela de seleção de pastas e retorna o caminho escolhido
    """
    root = tk.Tk()
    root.withdraw()

    path = filedialog.askdirectory(title="Selecione a pasta desejada")
    # Se o usuário cancelar ou fechar a janela, path será ""
    if not path:
        return None

    return path

# ======================================================================================================================
# CONSTRUÇÃO DO MODELO

def build_model():

    print("==================================================================")
    print(">> CONSTRUINDO MODELO RESNET-50")

    model = build_resnet50(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        num_classes=NUM_CLASSES,
        include_top=True,
        weight_decay=WEIGHT_DECAY
    )

    model.summary()
    return model

# ======================================================================================================================
# EXECUÇÃO PRINCIPAL

def main():

    # SELEÇÃO DO DIRETÓRIO CONTENDO O DATASET
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    messagebox.showinfo("Info", "Escolha o diretório contendo todo o Dataset para treino", parent=root)

    # VALIDAÇÃO DO CAMINHO DO DATASET
    while True:
        DATASET_PATH = open_directory()
        if not DATASET_PATH:
            print("Seleção de diretório cancelada pelo usuário")
            continue

        if not os.path.isdir(DATASET_PATH):
            print("Seleção de diretório inválido")
            continue
        break

    set_global_seed(42)
    enable_mixed_precision()

    root.destroy()
    print("-----------------------------------------------------------------------------------------------------------")
    print("                                  INICIANDO PIPELINE DE EXECUÇÃO                                           ")
    print("-----------------------------------------------------------------------------------------------------------")

    # ==================================================================================================================
    # CARREGAMENTO DOS DADOS

    # ==================================================================================================================
    # CARREGAMENTO DOS DADOS

    dataloader = DataLoader(
        path=DATASET_PATH,
        img_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT
    )

    (
        train_ds,
        val_ds,
        log_train,
        log_val,
        log_indexes,
        num_classes,
        steps_train,
        steps_val
    ) = dataloader.process_data()

    print("==================================================================")
    print(">> LOGS PROVENIENTES DO CARREGAMENTO DO DATASET")
    print("")
    print("Treinamento: ", log_train)
    print("Validação: ", log_val)
    print("Índices: ", log_indexes)
    print(f"Classes detectadas: {num_classes}")

    # CONSTRUÇÃO DO MODELO
    model = build_model()

    TRAIN_SIZE = steps_train * BATCH_SIZE
    VAL_SIZE = steps_val * BATCH_SIZE

    # CHAMADA DO TRAINER
    trainer = ResNet50Trainer(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,

        num_classes=NUM_CLASSES,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        initial_lr=INITIAL_LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        train_size=TRAIN_SIZE,
        val_size=VAL_SIZE,
        log_dir=LOG_DIR,
        checkpoint_path=CHECKPOINT_PATH
    )

    trainer.train()
    print("==================================================================")
    print("PIPELINE DE TREINAMENTO FINALIZADO COM SUCESSO")

if __name__ == "__main__":
    main()













