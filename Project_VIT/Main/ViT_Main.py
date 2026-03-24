# ======================================================================================================================
#                                              PACOTES E BIBLIOTECAS
# ======================================================================================================================
import os
import jax
import Utils
import numpy as np
from ViT_Trainer import train_vit
from tkinter import messagebox


# Validar que a GPU está sendo utilizada
print(jax.devices())

# Definição da variável de ambiente XLA_PYTHON_CLIENT_ALLOCATOR com o valor "platform" durante a execução do programa.
# Essa variável é usada por bibliotecas que usam XLA para controlar como a memória é alocada, especialmente em GPU.
# Evita erros de OOM. A memória tende a ser alocada sob demanda.
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# ======================================================================================================================
#                             PARÂMETROS PARA A VIT E PARA O CARREGAMENTO DE DADOS
# ======================================================================================================================

IMAGE_INPUT_SIZE = 224
BATCH_SIZE = 32
DATASET_SPLIT = 0.2
NUM_CLASSES = 3
PATCH_SIZE = 16  # OBRIGATÓRIO para seguir fielmente a ViT-B/16
HIDDEN_SIZE = 768  # ⚠️ crítico para a utilização dos pesos pré-treinados
TRANSFORMER_LAYERS = 12  # ⚠️ crítico para a utilização dos pesos pré-treinados
NUM_HEADS = 12  # ⚠️ crítico para a utilização dos pesos pré-treinados
MLP_UNITS = 3072  # ⚠️ crítico para a utilização dos pesos pré-treinados
EPOCHS = 100
WARMUP_STEPS = 0
BASE_LR = 1e-4
MODE = "finetune"

# Nome do diretório para armazenar o dataset organizado
DATA_DIR_NAME = f"Dataset_VAL{int(DATASET_SPLIT * 100)}%"
# Caminho onde os resultados devem ser armazenados
OUTPUT_DIR = Utils.resource_path("Results")



# ======================================================================================================================
#                                       FUNÇÃO PRINCIPAL (ORQUESTRADOR)
# ======================================================================================================================

def main():

    messagebox.showinfo("Info", "Escolha o arquivo de pesos pré-treinados para o Fine-Tuning")

    while True:
        WEIGHTS_PATH = Utils.open_file()
        if not WEIGHTS_PATH:
            print("Seleção do arquivo cancelado pelo usuário")
            continue
        break

    print("\n---------------------------------------------------------------------------------------------------------")
    print("                                  INICIANDO PIPELINE DE EXECUÇÃO                                           ")
    print("\n---------------------------------------------------------------------------------------------------------")

    # Carregamento dos dados préviamente processados
    XTRAIN = np.load("x_train.npy")
    YTRAIN = np.load("y_train.npy")
    XVAL = np.load("x_val.npy")
    YVAL = np.load("y_val.npy")

    # Configuração automática de steps por época, com base nos valores obtidos pelo DataLoader
    STEPS_PER_EPOCH = len(XTRAIN) // BATCH_SIZE
    STEPS_VAL = len(XVAL) // BATCH_SIZE

    # Configuração manual  de steps por época para a condução de testes com uma parcela da base (Sanity-Check)
    # STEPS_PER_EPOCH = 100
    # STEPS_VAL = 50

    # Cálculo do total de steps do treino
    TOTAL_STEPS = STEPS_PER_EPOCH * EPOCHS

    # Log para o terminal
    print("Batch size :", BATCH_SIZE)
    print("Steps per epoch:", STEPS_PER_EPOCH)
    print("Total Steps:", TOTAL_STEPS)

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
