# ======================================================================================================================
#                                              PACOTES E BIBLIOTECAS
# ======================================================================================================================

# Definição da variável de ambiente XLA_PYTHON_CLIENT_ALLOCATOR com o valor "platform" durante a execução do programa.
# Essa variável é usada por bibliotecas que usam XLA para controlar como a memória é alocada, especialmente em GPU.
# Evita erros de OOM. A memória tende a ser alocada sob demanda.
import os
import sys

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import math
import jax
import numpy as np
from ViT_Trainer import train_vit

# Validar que a GPU está sendo utilizada
print(jax.devices())

# ======================================================================================================================
#                             PARÂMETROS PARA A VIT E PARA O CARREGAMENTO DE DADOS
# ======================================================================================================================

# CAMINHOS
BASE_PATH = os.path.dirname(getattr(sys, '_MEIPASS', os.path.abspath(".")))
WEIGHTS_PATH = os.path.join(BASE_PATH, "Dataset and Weights", "imagenet21k_ViT-B_16.npz")
OUTPUT_DIR = os.path.join(BASE_PATH, "Main", "RESULTS")

print(">> CAMINHOS SELECIONADOS: ")
print(">> CAMINHO DOS PESOS PRE-TREINADOS: ", WEIGHTS_PATH)
print(">> CAMINHO DO DIRETORIO DE SAIDA: ", OUTPUT_DIR)
print("-----------------------------------------------------------")

# PARAMETROS CRÍTICOS PARA GARANTIR COMPATIBILIDADE COM OS PESOS
PATCH_SIZE = 16
HIDDEN_SIZE = 768
TRANSFORMER_LAYERS = 12
NUM_HEADS = 12
MLP_UNITS = 3072

# PARAMETROS CONFIGURÁVEIS
IMAGE_INPUT_SIZE = 224
BATCH_SIZE = 16
NUM_CLASSES = 3
EPOCHS = 100
WARMUP_STEPS = 1000
BASE_LR = 3e-5
MODE = "finetune"


# ======================================================================================================================
#                                       FUNÇÃO PRINCIPAL (ORQUESTRADOR)
# ======================================================================================================================


def main():
    print(">> INICIANDO PIPELINE DE EXECUCAO: ")
    print("-----------------------------------------------------------")

    # Carregamento dos dados préviamente processados
    XTRAIN = np.load("x_train.npy", mmap_mode="r")
    YTRAIN = np.load("y_train.npy", mmap_mode="r")
    XVAL = np.load("x_val.npy", mmap_mode="r")
    YVAL = np.load("y_val.npy", mmap_mode="r")

    # Configuração automática de steps por época, com base nos valores obtidos pelo DataLoader
    STEPS_PER_EPOCH = math.ceil(len(XTRAIN) / BATCH_SIZE)
    STEPS_VAL = len(XVAL) // BATCH_SIZE

    # Cálculo do total de steps do treino
    TOTAL_STEPS = STEPS_PER_EPOCH * EPOCHS

    # Log para o terminal
    print(">> PARAMETROS E CONFIGURACOES DA REDE: ")
    print(">> Batch size :", BATCH_SIZE)
    print(">> Steps per epoch:", STEPS_PER_EPOCH)
    print(">> Total Steps:", TOTAL_STEPS)
    print("-----------------------------------------------------------")

    train_vit(
        x_train=XTRAIN,
        y_train=YTRAIN,
        x_val=XVAL,
        y_val=YVAL,
        output_dir=str(OUTPUT_DIR),
        patches=(PATCH_SIZE, PATCH_SIZE),
        hidden_size=HIDDEN_SIZE,
        depth=TRANSFORMER_LAYERS,
        num_heads=NUM_HEADS,
        mlp_dim=MLP_UNITS,
        num_classes=NUM_CLASSES,
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

# COISAS A EXPLICAR:
# Para que serve o mmap_mode (load numpy)
# O que é o Warmup_steps
# Como aplicar seeds, sendo que não estou usando o tensorflow?
# Como salvar checkpoints e logs de parâmetros e configurações?
