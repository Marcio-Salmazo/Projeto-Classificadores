import shutil
import tensorflow as tf
import os

from tkinter import messagebox
from pathlib import Path
from ResNet50_Pure import build_resnet50
from ResNet_Trainer import ResNet_Trainer
from ResNet_Trainer2 import train

from ResNet_DataLoader import load_data
from ResNet_Shallow import build_resnet10
from ResNet_Shallow import build_resnet18
from ResNet_Shallow import build_resnet34
from ResNet_Utils import set_global_seed, enable_mixed_precision, open_directory, split_dataset, count_images

# REMOVE WARNINGS E INFO DO LOG, MANTENDO APENAS ERROS CRÍTICOSX
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ======================================================================================================================
# PARÂMETROS EXIGIDOS PELA RESNET

RUN_NAME = "teste old trainer"

IMAGE_SIZE = 224
BATCH_SIZE = 32
VAL_SPLIT = 0.2

EPOCHS = 10
NUM_CLASSES = 3
INITIAL_LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4

# ======================================================================================================================
# CAMINHOS DE AMBIENTES, TFRECORDS E SCRIPTS ASSOCIADOS

# Nome do diretório para armazenar o dataset organizado
DATA_DIR_NAME = f"Dataset_VAL{int(VAL_SPLIT * 100)}%"
# Diretório de logs de treinamento
LOG_DIR = f"Results/logs/{RUN_NAME}"
# Caminho e nome para o checkpoint do treinamento
CHECKPOINT_PATH = f"Results/Checkpoints_{RUN_NAME}/best_weights.h5"

# ======================================================================================================================
# DIAGNÓSTICO DO USO DA GPU, DEFINIÇÃO DE SEEDS E ATIVAÇÃO DE MIXED PRECISION

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"GPU detectada: {gpus}")
else:
    print("Nenhuma GPU detectada. Treinamento será lento.")

set_global_seed(42)
enable_mixed_precision()


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

    # AVALIA SE O CAMINHO SELECIONADO CONTÉM A DIVISÃO DE TRAIN E VAL
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

    '''
    # ----------------------------------------------------------------
    #                   CONSTRUÇÃO DO MODELO RESNET-50
    # ----------------------------------------------------------------
    print("==================================================================")
    print(">> CONSTRUINDO MODELO RESNET-50")

    model = build_resnet50(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        num_classes=NUM_CLASSES,
        include_top=True,
        weight_decay=WEIGHT_DECAY
    )
    model.summary()
    '''

    '''
    # ----------------------------------------------------------------
    #                   CONSTRUÇÃO DO MODELO RESNET-10
    # ----------------------------------------------------------------
    print("==================================================================")
    print(">> CONSTRUINDO MODELO RESNET-10")

    model = build_resnet10(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        num_classes=NUM_CLASSES,
        include_top=True,
        weight_decay=WEIGHT_DECAY
    )
    model.summary()
    '''

    '''
    # ----------------------------------------------------------------
    #                   CONSTRUÇÃO DO MODELO RESNET-18
    # ----------------------------------------------------------------

    print("==================================================================")
    print(">> CONSTRUINDO MODELO RESNET-18")

    model = build_resnet18(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        num_classes=NUM_CLASSES,
        include_top=True,
        weight_decay=WEIGHT_DECAY
    )
    model.summary()
    '''

    # ----------------------------------------------------------------
    #                   CONSTRUÇÃO DO MODELO RESNET-34
    # ----------------------------------------------------------------
    print("==================================================================")
    print(">> CONSTRUINDO MODELO RESNET-34")

    model = build_resnet34(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        num_classes=NUM_CLASSES,
        include_top=True,
        weight_decay=WEIGHT_DECAY
    )
    model.summary()

    # OBSERVAÇÃO: AMBOS OS TRAINER OPERAM DO MESMO MODO
    # O MAIS NOVO (TRAINER2) APENAS TEM UM CÓDIGO MAIS LIMPO
    '''
    # TREINAMENTO DO MODELO COMPILADO COM TRAINER ANTIGO
    trainer = ResNet_Trainer(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,

        num_classes=NUM_CLASSES,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        initial_lr=INITIAL_LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        train_size=num_train_samples,
        val_size=num_val_samples,
        log_dir=LOG_DIR,
        checkpoint_path=CHECKPOINT_PATH
    )
    
    trainer.train()
    '''

    # TREINAMENTO DO MODELO COMPILADO COM TRAINER NOVO
    train(model=model, train_ds=train_ds, val_ds=val_ds, batch_size=BATCH_SIZE, epochs=EPOCHS,
          train_size=num_train_samples, val_size=num_val_samples, initial_lr=INITIAL_LR, momentum=MOMENTUM,
          log_dir=LOG_DIR, checkpoint_path=CHECKPOINT_PATH)

    print("==================================================================")
    print("PIPELINE DE TREINAMENTO FINALIZADO COM SUCESSO")


if __name__ == "__main__":
    main()
