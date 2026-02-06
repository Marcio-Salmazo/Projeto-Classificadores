""" Execução AUTOMÁTICA do pipeline Vision Transformer (ViT) segundo o artigo: """
import shutil
import tkinter as tk
import os
from pathlib import Path

import Utils
# from ResNet.ResNet_DataLoader import load_data
from VisionTransformers.ViT_DataLoader import load_data

from Main_Project.VisionTransformers.ViT_Trainer import train_vit
from tkinter import messagebox

# Definição da variável de ambiente XLA_PYTHON_CLIENT_ALLOCATOR com o valor "platform" durante a execução do programa.
# Essa variável é usada por bibliotecas que usam XLA para controlar como a memória é alocada, especialmente em GPU.
# Evita erros de OOM. A memória tende a ser alocada sob demanda.
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# ======================================================================================================================
# PARÂMETROS PARA O CARREGAMENTO DE DADOS

IMAGE_INPUT_SIZE = 224
IMAGE_BATCH_SIZE = 32
DATASET_SPLIT = 0.2

# ======================================================================================================================
# PARÂMETROS EXIGIDOS PELA VIT
# OBSERVAÇÃO: CONFIGURAR O STEPS_PER_EPOCH MAIS A BAIXO NO CÓDIGO (CONFORME NECESSÁRIO)

PATCH_SIZE = 16  # OBRIGATÓRIO para seguir fielmente a ViT-B/16
HIDDEN_SIZE = 768  # ⚠️ crítico para a utilização dos pesos pré-treinados
TRANSFORMER_LAYERS = 12  # ⚠️ crítico para a utilização dos pesos pré-treinados
NUM_HEADS = 12  # ⚠️ crítico para a utilização dos pesos pré-treinados
MLP_UNITS = 3072  # ⚠️ crítico para a utilização dos pesos pré-treinados
BATCH_SIZE_VIT = IMAGE_BATCH_SIZE
EPOCHS = 5
WARMUP_STEPS = 0
BASE_LR = 1e-4
MODE = "finetune"

# Nome do diretório para armazenar o dataset organizado
DATA_DIR_NAME = f"Dataset_VAL{int(DATASET_SPLIT * 100)}%"

# ======================================================================================================================
# CAMINHOS DOS CHECKPOINTS

OUTPUT_DIR = Utils.resource_path("VisionTransformers\\Results")
print(str(OUTPUT_DIR))


# ======================================================================================================================
# EXECUÇÃO PRINCIPAL

def main():
    # ----------------------------------------------------------------
    # SELEÇÃO DO DIRETÓRIO CONTENDO O DATASET E VALIDAÇÃO DA ESTRUTURA
    # ----------------------------------------------------------------

    while True:
        base_datapath = Utils.open_directory('Selecione o diretório contendo a base de dados. Opte por escolher o'
                                             ' diretório já organizado com as divisões para treino e validação,'
                                             ' (se houver)')
        if not base_datapath:
            messagebox.showinfo("Info", "Seleção de diretório cancelada pelo usuário")
            continue
        break

    # AVALIA SE O CAMINHO SELECIONADO CONTÉM A DIVISÃO DE TRAIN E VAL
    if not os.path.isdir(f"{base_datapath}/train") or not os.path.isdir(f"{base_datapath}/val"):

        messagebox.showinfo("Info", "A base não contém originalmente a divisão entre treino e validação, "
                                    "essa estrutura será criada a seguir.")
        org_data = os.path.join(base_datapath, DATA_DIR_NAME)

        # Exclui o diretório caso ele já existe e recria-o
        if Path(org_data).exists():
            shutil.rmtree(Path(org_data))
        Path(org_data).mkdir(parents=True, exist_ok=True)

        TRAIN_PATH, VAL_PATH = Utils.split_dataset(base_datapath, org_data, val_split=DATASET_SPLIT,
                                                   seed=42, extensions=(".jpg", ".jpeg", ".png"))

    else:
        TRAIN_PATH = f"{base_datapath}/train"
        VAL_PATH = f"{base_datapath}/val"

    # ----------------------------------------------------------------
    # SELEÇÃO DO ARQUIVO CONTENDO OS PESOS PRÉ-TREINADOS
    # ----------------------------------------------------------------

    messagebox.showinfo("Info", "Escolha o arquivo de pesos pré-treinados para o Fine-Tuning")

    while True:
        WEIGHTS_PATH = Utils.open_file()
        if not WEIGHTS_PATH:
            print("Seleção do arquivo cancelado pelo usuário")
            continue
        break

    print(
        "\n-----------------------------------------------------------------------------------------------------------")
    print("                                  INICIANDO PIPELINE DE EXECUÇÃO                                           ")
    print("-----------------------------------------------------------------------------------------------------------")

    # ----------------------------------------------------------------
    #                   CARREGAMENTO DOS DADOS E LOG
    # ----------------------------------------------------------------

    train_ds, val_ds, class_names, num_classes = load_data(TRAIN_PATH, VAL_PATH, IMAGE_BATCH_SIZE)
    num_train_samples = Utils.count_images(TRAIN_PATH)
    num_val_samples = Utils.count_images(VAL_PATH)

    print(">> LOGS PROVENIENTES DO CARREGAMENTO DO DATASET\n")

    print("Treinamento: ", num_train_samples)
    print("Validação: ", num_val_samples)
    print("Índices: ", class_names)
    print(f"Classes detectadas: {num_classes}\n")

    # ==================================================================================================================
    # CHAMADA DO PRÉ-TREINO

    # Configuração automática de steps por época, com base nos valores obtidos pelo DataLoader
    # STEPS_PER_EPOCH = num_train_samples // BATCH_SIZE_VIT
    # STEPS_VAL = num_val_samples // BATCH_SIZE_VIT

    # Configuração manual  de steps por época para a condução de testes com uma parcela da base (Sanity-Check)
    STEPS_PER_EPOCH = 100
    STEPS_VAL = 50

    TOTAL_STEPS = STEPS_PER_EPOCH * EPOCHS

    print("Total Samples:", num_train_samples)
    print("Batch size :", BATCH_SIZE_VIT)
    print("Steps per epoch:", STEPS_PER_EPOCH)
    print("Total Steps:", TOTAL_STEPS)

    train_vit(
        train_ds=train_ds,
        val_ds=val_ds,
        output_dir=str(OUTPUT_DIR),
        patches=(PATCH_SIZE, PATCH_SIZE),
        hidden_size=HIDDEN_SIZE,
        depth=TRANSFORMER_LAYERS,
        num_heads=NUM_HEADS,
        mlp_dim=MLP_UNITS,
        num_classes=num_classes,
        total_steps=TOTAL_STEPS,
        warmup_steps=WARMUP_STEPS,
        base_lr=BASE_LR,
        mode=MODE,
        weights_path=WEIGHTS_PATH,
        steps_per_epoch=STEPS_PER_EPOCH,
        steps_val=STEPS_VAL,
        epochs=EPOCHS
    )


if __name__ == "__main__":
    main()
