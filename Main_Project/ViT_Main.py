"""
    Execução AUTOMÁTICA do pipeline Vision Transformer (ViT) segundo o artigo:

     - Verifica se ImageNet está extraído. Se não estiver → extrai.
     - Verifica se TFRecords existem. Se não existirem → cria.
     - Remove os arquivos .tar e a pasta extraída após gerar os TFRecords (para economizar espaço).
     - Inicia o treinamento (pré-treino).
     - Opcionalmente avalia após o treino.
"""
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import sys


# Definição da variável de ambiente XLA_PYTHON_CLIENT_ALLOCATOR com o valor "platform" durante a execução do programa.
# Essa variável é usada por bibliotecas que usam XLA para controlar como a memória é alocada, especialmente em GPU.
# Evita erros de OOM. A memória tende a ser alocada sob demanda.
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

from Main_Project.DataLoader import DataLoader
from Main_Project.VisionTransformers.ViT_Trainer import train_vit

# ======================================================================================================================
# PARÂMETROS PARA O CARREGAMENTO DE DADOS

IMAGE_INPUT_SIZE = 224
IMAGE_BATCH_SIZE = 32
DATASET_SPLIT = 0.2

# ======================================================================================================================
# PARÂMETROS EXIGIDOS PELA VIT
# OBSERVAÇÃO: CONFIGURAR O STEPS_PER_EPOCH MAIS A BAIXO NO CÓDIGO (CONFORME NECESSÁRIO)

PATCH_SIZE = 16                 # OBRIGATÓRIO para seguir fielmente a ViT-B/16
HIDDEN_SIZE = 768               # ⚠️ crítico para a utilização dos pesos pré-treinados
TRANSFORMER_LAYERS = 12         # ⚠️ crítico para a utilização dos pesos pré-treinados
NUM_HEADS = 12                  # ⚠️ crítico para a utilização dos pesos pré-treinados
MLP_UNITS = 3072                # ⚠️ crítico para a utilização dos pesos pré-treinados
BATCH_SIZE_VIT = IMAGE_BATCH_SIZE
EPOCHS = 5
WARMUP_STEPS = 0
BASE_LR = 1e-4
MODE = "finetune"

# ======================================================================================================================
# FUNÇÃO PARA EXTRAIR O CAMINHO ABSOLUTO

def resource_path(relative_path):
    """ Retorna o caminho absoluto para o arquivo, compatível com PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
    return os.path.join(base_path, relative_path)

# ======================================================================================================================
# FUNÇÃO PARA EXTRAIR O CAMINHO DO ARQUIVO DE PESOS

def open_file():

    root = tk.Tk()
    root.withdraw()

    # Open the file explorer and get the full file path
    file_path = filedialog.askopenfilename(title="Selecione o arquivo de pesos")
    if not file_path:
        return None
    return file_path

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
# CAMINHOS DOS CHECKPOINTS

OUTPUT_DIR = resource_path("VisionTransformers\\Checkpoints")
print(str(OUTPUT_DIR))

# ======================================================================================================================
# EXECUÇÃO PRINCIPAL

def main():

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

    messagebox.showinfo("Info", "Escolha o arquivo de pesos pré-treinados para o Fine-Tuning", parent=root)
    # VALIDAÇÃO DO CAMINHO DOS PESOS
    while True:
        WEIGHTS_PATH = open_file()
        if not WEIGHTS_PATH:
           print("Seleção do arquivo cancelado pelo usuário")
           continue
        break

    root.destroy()
    print("-----------------------------------------------------------------------------------------------------------")
    print("INICIANDO PIPELINE DE EXECUÇÃO\n")
    print("-----------------------------------------------------------------------------------------------------------")

    # ==================================================================================================================
    # CARREGAMENTO DOS DADOS

    dataloader = DataLoader(
        path=DATASET_PATH,
        img_size=IMAGE_INPUT_SIZE,
        batch_size=IMAGE_BATCH_SIZE,
        val_split=DATASET_SPLIT,
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

    print("-----------------------------------------------------------------------------------------------------------")
    print(" ")
    print("LOGS PROVENIENTES DO CARREGAMENTO DO DATASET")
    print("")
    print("Treinamento: ", log_train)
    print("Validação: ", log_val)
    print("Índices: ", log_indexes)
    print(f"Classes detectadas: {num_classes}\n")

    print("-----------------------------------------------------------------------------------------------------------\n")

    # ==================================================================================================================
    # CHAMADA DO PRÉ-TREINO

    # Configuração automática de steps por época, com base nos valores obtidos pelo DataLoader
    # STEPS_PER_EPOCH = steps_train

    # Configuração manual  de steps por época para a condução de testes com uma parcela da base (Sanity-Check)
    STEPS_PER_EPOCH = 100

    TOTAL_STEPS = steps_train * EPOCHS


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
        epochs=EPOCHS
    )

    # ==================================================================================================================
    # VALIDAÇÃO DO TREINO

if __name__ == "__main__":
    main()
