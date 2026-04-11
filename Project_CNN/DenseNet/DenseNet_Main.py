import shutil
import tensorflow as tf
import os
from tkinter import messagebox
from pathlib import Path

from DenseNet_Utils import open_directory, count_images, set_global_seed, split_dataset
from DenseNet_Builder import Shallow_densenet, DenseNet121
from DenseNet_Dataloader import load_data
from DenseNet_Trainer import compile_model, train_model
from tensorflow.keras import mixed_precision

# REMOVE WARNINGS E INFO DO LOG, MANTENDO APENAS ERROS CRÍTICOS
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ======================================================================================================================
# PARÂMETROS EXIGIDOS PELA MOBILENET

RUN_NAME = "DENSENET testte2 BS16 3K"

# DenseNet se baseia na concatenação, fazendo com que a memória cresce expressivamente
# dessa forma, é necessário que o INPUT SIZE seja reduzido
INPUT_SIZE = 128
BATCH_SIZE = 16
VAL_SPLIT = 0.2
EPOCHS = 3000
GROWTH_RATE = 24
NUM_CLASSES = 3
INITIAL_LR = 0.01
MOMENTUM = 0.9

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
    train_ds, val_ds, class_names, num_classes = load_data(TRAIN_PATH, VAL_PATH, BATCH_SIZE, INPUT_SIZE)
    num_train_samples = count_images(TRAIN_PATH)
    num_val_samples = count_images(VAL_PATH)

    print("\n>> LOGS PROVENIENTES DO CARREGAMENTO DO DATASET")

    print("Treinamento: ", num_train_samples)
    print("Validação: ", num_val_samples)
    print("Índices: ", class_names)
    print(f"Classes detectadas: {num_classes}\n")

    # CONSTRUÇÃO E COMPILAÇÃO DO MODELO
    model = Shallow_densenet(input_shape=(INPUT_SIZE, INPUT_SIZE, 3), num_classes=NUM_CLASSES, growth_rate=GROWTH_RATE)
    model = compile_model(model, INITIAL_LR, MOMENTUM)
    model.summary()

    # TREINAMENTO DO MODELO COMPILADO
    train_model(model, train_ds, val_ds, epochs=EPOCHS, checkpoint_path=CHECKPOINT_PATH, log_dir=LOG_DIR)

    print(">> PIPELINE DE TREINAMENTO FINALIZADO COM SUCESSO")

    # Limpeza da memória após treino
    import gc
    del model
    gc.collect()
    tf.keras.backend.clear_session()


if __name__ == "__main__":
    main()
