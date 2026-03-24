# ======================================================================================================================
#                                        BIBLIOTECAS E CONFIGURAÇÕES INICIAIS
# ======================================================================================================================

import shutil
import Utils
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

    # Scale jitter: short side ∈ [256, 480]
    target_short = tf.random.uniform([], 256, 481, dtype=tf.int32)

    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    short = tf.minimum(h, w)

    scale = tf.cast(target_short, tf.float32) / tf.cast(short, tf.float32)
    new_h = tf.cast(tf.round(tf.cast(h, tf.float32) * scale), tf.int32)
    new_w = tf.cast(tf.round(tf.cast(w, tf.float32) * scale), tf.int32)

    image = tf.image.resize(image, [new_h, new_w])

    # Random crop 224x224
    image = tf.image.random_crop(image, [224, 224, 3])

    # Horizontal flip
    image = tf.image.random_flip_left_right(image)

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
        image_size=(256, 256),
        batch_size=None,
        shuffle=True
    )

    val_ds_raw = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="int",
        image_size=(256, 256),
        batch_size=None,
        shuffle=False
    )

    class_names = train_ds_raw.class_names
    num_classes = len(class_names)

    train_ds = (
        train_ds_raw
        .map(preprocess_train, num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )

    val_ds = (
        val_ds_raw
        .map(preprocess_val, num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
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
#                      FUNÇÃO PRINCIPAL PARA PROCESSAR OS DADOS E GERAR OS ARQUIVOS NUMPY
# ======================================================================================================================

def main():
    # Parâmetros principais
    DATA_DIR_NAME = 'Processed Dataset'
    DATASET_SPLIT = 0.2

    while True:
        base_datapath = Utils.open_directory('Selecione o diretório contendo a base de dados. Opte por escolher o'
                                             ' diretório já organizado com as divisões para treino e validação,'
                                             ' (se houver)')
        if base_datapath:
            break
        messagebox.showinfo("Info", "Seleção de diretório cancelada pelo usuário")

    # Avalia se o diretório selecionado possui a divisão entre treino e validação
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