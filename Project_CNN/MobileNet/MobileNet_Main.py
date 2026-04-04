import shutil
import tensorflow as tf
import os
from tkinter import messagebox
from pathlib import Path

from MobileNet_Utils import open_directory, count_images, set_global_seed, split_dataset
from MobileNet_Builder import MobileNetV2
from MobileNet_Dataloader import load_data
from MobileNet_Trainer import compile_model, train_model
from tensorflow.keras import mixed_precision

# REMOVE WARNINGS E INFO DO LOG, MANTENDO APENAS ERROS CRÍTICOS
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ======================================================================================================================
# PARÂMETROS EXIGIDOS PELA MOBILENET

RUN_NAME = "Experiment"

INPUT_SIZE = 224
BATCH_SIZE = 32
VAL_SPLIT = 0.2
EPOCHS = 3000
NUM_CLASSES = 3
INITIAL_LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4

# ======================================================================================================================
# DIAGNÓSTICO DO USO DA GPU, DEFINIÇÃO DE SEEDS E ATIVAÇÃO DE MIXED PRECISION

mixed_precision.set_global_policy("mixed_float16")
print("Mixed precision ativada (float16) para acelerar o treinamento.")

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"GPU detectada: {gpus}")
else:
    print("Nenhuma GPU detectada. Treinamento será lento.")

set_global_seed(42)

# ======================================================================================================================
# CAMINHOS DE AMBIENTES E SCRIPTS ASSOCIADOS

# Nome do diretório para armazenar o dataset organizado
DATA_DIR_NAME = f"Dataset_VAL{int(VAL_SPLIT * 100)}%"
# Diretório de logs de treinamento
LOG_DIR = f"Results/logs/{RUN_NAME}"
# Caminho e nome para o checkpoint do treinamento
CHECKPOINT_PATH = f"Results/Checkpoints_{RUN_NAME}/best_weights.h5"


# ======================================================================================================================
# EXECUÇÃO PRINCIPAL
def main():

    # SOLICITA AO USUÁRIO O DIRETÓRIO CONTENDO A BASE DE DADOS (PREFERENCIALMENTE JÁ ORGANIZADA EM SUBSETS)
    while True:
        base_datapath = open_directory('Selecione o diretório contendo a base de dados. Opte por escolher o'
                                       ' diretório já organizado com as divisões para treino e validação,'
                                       ' (se houver)')
        if not base_datapath:
            messagebox.showinfo("Info", "Seleção de diretório cancelada pelo usuário")
            continue
        break

    # AVALIA SE O CAMINHO SELECIONADO CONTÉM A DIVISÃO TRAIN E VAL, CRIANDO-OS CASO NECESSÁRIO
    if not os.path.isdir(f"{base_datapath}/train") or not os.path.isdir(f"{base_datapath}/val"):

        messagebox.showinfo("Info", "A base não contém originalmente a divisão entre treino e validação, "
                                    "essa estrutura será criada a seguir.")
        org_data = os.path.join(base_datapath, DATA_DIR_NAME)

        # Exclui o diretório caso ele já existe e recria-o
        if Path(org_data).exists():
            shutil.rmtree(Path(org_data))
        Path(org_data).mkdir(parents=True, exist_ok=True)

        TRAIN_PATH, VAL_PATH = split_dataset(base_datapath, org_data, val_split=VAL_SPLIT,
                                             seed=42, extensions=(".jpg", ".jpeg", ".png"))

    else:
        TRAIN_PATH = f"{base_datapath}/train"
        VAL_PATH = f"{base_datapath}/val"

    print("\n")
    print("-----------------------------------------------------------------------------------------------------------")
    print("                                  INICIANDO PIPELINE DE EXECUÇÃO                                           ")
    print("-----------------------------------------------------------------------------------------------------------")

    # CARREGAMENTO DE DADOS E EXIBIÇÃO DE LOGS
    train_ds, val_ds, class_names, num_classes = load_data(TRAIN_PATH, VAL_PATH, BATCH_SIZE)
    num_train_samples = count_images(TRAIN_PATH)
    num_val_samples = count_images(VAL_PATH)

    print(">> LOGS PROVENIENTES DO CARREGAMENTO DO DATASET\n")

    print("Treinamento: ", num_train_samples)
    print("Validação: ", num_val_samples)
    print("Índices: ", class_names)
    print(f"Classes detectadas: {num_classes}\n")

    # CONSTRUÇÃO E COMPILAÇÃO DO MODELO
    model = MobileNetV2(input_shape=(INPUT_SIZE, INPUT_SIZE, 3), num_classes=NUM_CLASSES)
    model = compile_model(model)
    model.summary()

    # TREINAMENTO DO MODELO COMPILADO
    train_model(model, train_ds, val_ds, epochs=EPOCHS)

    print(">> PIPELINE DE TREINAMENTO FINALIZADO COM SUCESSO")


if __name__ == "__main__":
    main()
