# ======================================================================================================================
#                                        BIBLIOTECAS E CONFIGURAÇÕES INICIAIS
# ======================================================================================================================

import random
import shutil
import sys
import numpy as np
import os
from tkinter import messagebox
from pathlib import Path
import tensorflow as tf  # pyright: ignore[reportMissingModuleSource]

tf.config.set_visible_devices([], 'GPU')

AUTOTUNE = tf.data.AUTOTUNE
# IMAGENET_MEAN e IMAGENET_STD servem para normalização futura, de acordo com o artigo
IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
IMAGENET_STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)


# ======================================================================================================================
#                            FUNÇÃO PARA O PRE-PROCESSAMENTO DO CONJUNTO DE TREINO
# ======================================================================================================================

def preprocess_train(image, label):
    label = tf.cast(label, tf.int32)

    # Scale jitter: short side ∈ [224, 256]
    target_short = tf.random.uniform([], 224, 256, dtype=tf.int32)

    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    short = tf.minimum(h, w)

    scale = tf.cast(target_short, tf.float32) / tf.cast(short, tf.float32)
    new_h = tf.cast(tf.round(tf.cast(h, tf.float32) * scale), tf.int32)
    new_w = tf.cast(tf.round(tf.cast(w, tf.float32) * scale), tf.int32)

    image = tf.image.resize(image, [new_h, new_w])

    # Center crop 224x224
    image = tf.image.resize_with_crop_or_pad(image, 224, 224)

    # Horizontal flip
    image = tf.image.random_flip_left_right(image)

    # Random Brightness
    image = tf.image.random_brightness(image, 0.1)

    # Random Contrast
    image = tf.image.random_contrast(image, 0.9, 1.1)

    # Normalização ImageNet
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD

    return image, label


# ======================================================================================================================
#                               FUNÇÃO PARA O PRE-PROCESSAMENTO DO CONJUNTO DE VALIDAÇÃO
# ======================================================================================================================

def preprocess_val(image, label):
    label = tf.cast(label, tf.int32)

    # Resize mantendo aspecto: short side = 256
    h = tf.cast(tf.shape(image)[0], tf.float32)
    w = tf.cast(tf.shape(image)[1], tf.float32)
    short = tf.minimum(h, w)

    scale = 256.0 / short
    new_h = tf.cast(tf.round(h * scale), tf.int32)
    new_w = tf.cast(tf.round(w * scale), tf.int32)

    image = tf.image.resize(image, [new_h, new_w])

    # Center crop 224x224
    image = tf.image.resize_with_crop_or_pad(image, 224, 224)

    # Normalização ImageNet
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD

    return image, label


# ======================================================================================================================
#                                       FUNÇÃO PARA O CARREGAMENTO EFETIVO DA BASE
# ======================================================================================================================

def load_data(train_dir, val_dir):
    """
        Observação: batch_size=None inicialmente garante que o TensorFlow NÃO crie batches automaticamente.
        Dessa forma, o dataset fica element-wise, não batch-wise. O fluxo de atividades fica:

            * Primeiro: carrega imagem por imagem
            * Depois: aplica augmentação por imagem
            * Só então: cria o batch final
    """
    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        batch_size=None,
        shuffle=True
    )

    val_ds_raw = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="int",
        batch_size=None,
        shuffle=False
    )

    class_names = train_ds_raw.class_names
    num_classes = len(class_names)

    train_ds = (
        train_ds_raw
        .map(preprocess_train, num_parallel_calls=AUTOTUNE)
        .cache().prefetch(AUTOTUNE)
    )

    val_ds = (
        val_ds_raw
        .map(preprocess_val, num_parallel_calls=AUTOTUNE)
        .cache().prefetch(AUTOTUNE)
    )

    return (
        train_ds,
        val_ds,
        class_names,
        num_classes
    )


# ======================================================================================================================
#                                FUNÇÃO PARA A TRANSFORMAÇÃO DE TENSORES PARA NUMPY
# ======================================================================================================================

def tf_to_numpy(dataset):
    images, labels = [], []

    for img, lab in dataset:
        images.append(img.numpy())
        labels.append(lab.numpy())

    return np.array(images), np.array(labels)


# ======================================================================================================================
#                                          DIVISÃO DA BASE DE DADOS
# ======================================================================================================================

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


# ======================================================================================================================
#                      FUNÇÃO PRINCIPAL PARA PROCESSAR OS DADOS E GERAR OS ARQUIVOS NUMPY
# ======================================================================================================================

def main():

    # DEFINIÇÃO DE CAMINHOS
    # ---------------------------------------------------------------------------
    DATASET_SPLIT = 0.2
    BASE_PATH = os.path.dirname(getattr(sys, '_MEIPASS', os.path.abspath(".")))
    DATASET = r"C:\Users\marci_wawp\Desktop\REDUCED_DATASET"
    # ---------------------------------------------------------------------------

    # Avalia se o diretório selecionado para o dataset não possui a divisão entre treino e validação
    if not os.path.isdir(f"{DATASET}/train") or not os.path.isdir(f"{DATASET}/val"):

        print(">> Aplicando subsets de para treino e validação")
        PROCESSED_DATA = os.path.join(BASE_PATH, 'Processed Dataset')

        # Exclui o diretório caso ele já existe e recria-o
        if Path(PROCESSED_DATA).exists():
            shutil.rmtree(Path(PROCESSED_DATA))
        Path(PROCESSED_DATA).mkdir(parents=True, exist_ok=True)

        TRAIN_PATH, VAL_PATH = split_dataset(DATASET, PROCESSED_DATA, val_split=DATASET_SPLIT,
                                             seed=42, extensions=(".jpg", ".jpeg", ".png"))

    else:
        TRAIN_PATH = f"{DATASET}/train"
        VAL_PATH = f"{DATASET}/val"

    train_ds, val_ds, class_names, num_classes = load_data(TRAIN_PATH, VAL_PATH)

    print("\nConvertendo dados para treino...")
    x_train, y_train = tf_to_numpy(train_ds)
    print("Convertendo dados para validação...")
    x_val, y_val = tf_to_numpy(val_ds)

    np.save("x_train.npy", x_train)
    np.save("y_train.npy", y_train)
    np.save("x_val.npy", x_val)
    np.save("y_val.npy", y_val)

    print("Dados salvos!")
    print("Quantidade de classes encontradas: ", num_classes)
    print("Nome das classes encontradas: ", class_names)


if __name__ == "__main__":
    main()
