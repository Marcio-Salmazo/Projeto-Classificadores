"""
data_loader.py
--------------
Pipeline oficial de carregamento do ImageNet seguindo o paper ViT (Dosovitskiy et al.).
Inclui:
 - extração dos .tar
 - conversão para TFRecords
 - criação do dataset com mesmo preprocessamento usado no repositório oficial
 - batches prontos para JAX (shard + prefetch)
"""

import os
import tarfile
from typing import Dict, Tuple

import tensorflow as tf
import tensorflow_datasets as tfds
import jax
import jax.numpy as jnp


# ------------------------------------------------------------
# 1) Extração dos arquivos originais
# ------------------------------------------------------------
def extract_imagenet_tars(train_tar: str, val_tar: str, output_dir: str):
    """
    Extrai ILSVRC2012_img_train.tar e ILSVRC2012_img_val.tar
    no formato de diretórios esperados para criação de TFRecords.

    train_tar: caminho para ILSVRC2012_img_train.tar
    val_tar:   caminho para ILSVRC2012_img_val.tar
    output_dir: diretório destino (ex: imagenet_raw/)
    """
    os.makedirs(output_dir, exist_ok=True)

    # -------------------
    # 1. EXTRAIR TRAIN
    # -------------------
    print("Extraindo train...")
    train_dir = os.path.join(output_dir, "train")
    os.makedirs(train_dir, exist_ok=True)

    with tarfile.open(train_tar) as tar:
        tar.extractall(train_dir)

    # o train extrai múltiplos .tar internos por sinlabel → extrair também
    for fname in os.listdir(train_dir):
        if fname.endswith(".tar"):
            class_tar = os.path.join(train_dir, fname)
            class_name = fname.replace(".tar", "")
            class_dir = os.path.join(train_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            with tarfile.open(class_tar) as ct:
                ct.extractall(class_dir)
            os.remove(class_tar)

    # -------------------
    # 2. EXTRAIR VAL
    # -------------------
    print("Extraindo val...")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(val_dir, exist_ok=True)

    with tarfile.open(val_tar) as tar:
        tar.extractall(val_dir)

    print("Extração concluída.")


# ------------------------------------------------------------
# 2) Construtor oficial de TFRecords
# ------------------------------------------------------------

def write_tfrecord_example(image_bytes, label, writer):
    """Converte uma imagem para um tf.train.Example compatível com o pipeline JAX."""
    example = tf.train.Example(features=tf.train.Features(feature={
        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    }))
    writer.write(example.SerializeToString())


def create_imagenet_tfrecords(raw_dir: str, output_dir: str,
                              num_train_shards=1024, num_val_shards=128):
    """
    Cria TFRecords a partir do ImageNet extraído.
    Esse é o formato usado no paper e no repositório oficial.

    raw_dir: diretório que contém "train/" e "val/"
    output_dir: onde salvar os TFRecords
    """
    os.makedirs(output_dir, exist_ok=True)

    train_dir = os.path.join(raw_dir, "train")
    val_dir = os.path.join(raw_dir, "val")

    print("Gerando TFRecords de treino...")
    write_sharded_tfrecords(train_dir, os.path.join(output_dir, "train"),
                            num_shards=num_train_shards)

    print("Gerando TFRecords de validação...")
    write_sharded_tfrecords(val_dir, os.path.join(output_dir, "validation"),
                            num_shards=num_val_shards)

    print("TFRecords criados com sucesso.")


def write_sharded_tfrecords(img_dir: str, output_dir: str, num_shards: int):
    os.makedirs(output_dir, exist_ok=True)

    class_names = sorted(os.listdir(img_dir))
    label_map = {cls: i for i, cls in enumerate(class_names)}

    # lista todas imagens
    samples = []
    for cls in class_names:
        class_path = os.path.join(img_dir, cls)
        for fname in os.listdir(class_path):
            if fname.lower().endswith(".jpeg") or fname.lower().endswith(".jpg"):
                samples.append((os.path.join(class_path, fname), label_map[cls]))

    # shard
    shard_size = len(samples) // num_shards
    for shard_id in range(num_shards):
        start = shard_id * shard_size
        end = (shard_id + 1) * shard_size if shard_id < num_shards - 1 else len(samples)
        shard_path = os.path.join(output_dir, f"{shard_id:05d}-of-{num_shards:05d}.tfrecord")
        with tf.io.TFRecordWriter(shard_path) as writer:
            for img_path, label in samples[start:end]:
                with tf.io.gfile.GFile(img_path, "rb") as f:
                    img_bytes = f.read()
                write_tfrecord_example(img_bytes, label, writer)


# ------------------------------------------------------------
# 3) Pré-processamento fiel ao paper ViT
# ------------------------------------------------------------

IMAGENET_MEAN = jnp.array([0.5, 0.5, 0.5])
IMAGENET_STD  = jnp.array([0.5, 0.5, 0.5])


def preprocess_train(image, label, image_size=224):
    """Resize 256 → random crop 224 + flip + normalização."""
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    image = tf.image.random_crop(image, [image_size, image_size, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return image, label


def preprocess_val(image, label, image_size=224):
    """Resize menor |→ central crop 224."""
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    image = tf.image.central_crop(image, 224/256)
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return image, label


# ------------------------------------------------------------
# 4) Carregar TFRecords para JAX
# ------------------------------------------------------------

def load_tfrecords(tfrecord_dir: str, batch_size: int,
                   train: bool, image_size=224):
    """
    Lê TFRecords e retorna dataset pronto para JAX.

    - faz decode das features
    - aplica preprocessamento de acordo com (train/test)
    - faz batching e prefetch
    """
    files = tf.io.gfile.glob(os.path.join(tfrecord_dir, "*.tfrecord"))

    def _parse(example_proto):
        feature_description = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64)
        }
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        return parsed["image"], parsed["label"]

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)

    if train:
        ds = ds.shuffle(10000)
        ds = ds.map(lambda x, y: preprocess_train(x, y, image_size),
                    num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(lambda x, y: preprocess_val(x, y, image_size),
                    num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


# ------------------------------------------------------------
# 5) Transformar batch do TF → JAX
# ------------------------------------------------------------

def tf_to_jax(batch):
    """Converte batch (TensorFlow) → JAX arrays, mantendo shape intacto."""
    images, labels = batch
    images = jnp.asarray(images.numpy())
    labels = jnp.asarray(labels.numpy())
    return images, labels