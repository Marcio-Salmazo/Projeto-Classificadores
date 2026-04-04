# ******************************************************************************************************************** #
#                                                   IMPORTAÇÕES                                                        #
# ******************************************************************************************************************** #
import sys

import tensorflow as tf
import numpy as np
import random
import os
import tkinter as tk
import shutil
from pathlib import Path
from tkinter import filedialog, messagebox
from tensorflow.keras import mixed_precision


# ******************************************************************************************************************** #
#                                              FUNÇÕES AUXILIARES                                                      #
# ******************************************************************************************************************** #

# DEFINIÇÃO DE SEEDS PARA REPRODUTIBILIDADE
def set_global_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"Seeds fixados (seed={seed}) para reprodutibilidade.")


# ======================================================================================================================
# FUNÇÃO PARA OBTER O CAMINHO DE UM DIRETÓRIO VIA EXPLORER
def open_directory(msg):
    """
        O tkinter é utilizado para exibir janela do explorer a fim de selecionar a pasta contendo o Dataset.
            * root = tk.Tk() - instância do tkinter
            * root.withdraw() -  Oculta a janela principal (para exibir apenas o pop-up)
            * filedialog.askdirectory(title="") - Abre a janela de seleção de pastas e retorna o caminho escolhido
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    messagebox.showinfo("Info", msg, parent=root)

    path = filedialog.askdirectory(title="Selecione a pasta desejada")
    # Se o usuário cancelar ou fechar a janela, path será ""
    if not path:
        return None

    return path


# ======================================================================================================================
# FUNÇÃO PARA OBTER A CONTAGEM DE IMAGENS NA BASE
def count_images(dir_path):
    return sum(
        len(files)
        for _, _, files in os.walk(dir_path)
        if any(f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')) for f in files)
    )


# ======================================================================================================================
# ORGANIZAÇÃO DA BASE DE DADOS
def split_dataset(source_dir, output_dir, val_split=0.2, seed=42, extensions=(".jpg", ".jpeg", ".png")):
    """
        Divide automaticamente um dataset em train/val, copiando arquivos.

        Args:
            source_dir (str): diretório original com subpastas por classe
            output_dir (str): diretório de saída (train/ e val/ serão criados)
            val_split (float): fração para validação (ex: 0.2 = 20%)
            seed (int): seed para reprodutibilidade
            extensions (tuple): extensões de imagem aceitas
    """

    random.seed(seed)
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    train_dir = output_dir / "train"
    val_dir = output_dir / "val"

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    print(f">> Lendo dataset de: {source_dir}")
    print(f">> Criando diretórios para treino e validação em: {output_dir}")
    print(f">> Val split: {val_split * 100:.1f}% | Seed: {seed}\n")

    # Processa classe (representada por um diretório na pasta original do dataset)
    for class_dir in source_dir.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        print(f">> Processando classe: {class_name}")

        # Armazena uma lista com o total de imagens
        images = [
            f for f in class_dir.iterdir()
            if f.suffix.lower() in extensions
        ]

        if len(images) == 0:
            print(f">> Nenhuma imagem encontrada em {class_name}, pulando.")
            continue

        # Embaralha a lista automaticamente
        random.shuffle(images)

        # Divide a lista com base em val_split
        n_val = int(len(images) * val_split)
        val_images = images[:n_val]
        train_images = images[n_val:]

        # Cria diretórios da classe
        (train_dir / class_name).mkdir(exist_ok=True)
        (val_dir / class_name).mkdir(exist_ok=True)

        # Copia arquivos
        for img in train_images:
            shutil.copy2(img, train_dir / class_name / img.name)

        for img in val_images:
            shutil.copy2(img, val_dir / class_name / img.name)

        print(
            f">> Train: {len(train_images)} | "
            f"Val: {len(val_images)} | "
            f"Total: {len(images)}"
        )

    print("\n Divisão aplicada com sucesso!")
    return train_dir, val_dir
