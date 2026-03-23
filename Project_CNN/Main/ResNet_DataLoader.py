# ******************************************************************************************************************** #
#                                                   IMPORTAÇÕES                                                        #
# ******************************************************************************************************************** #

import tensorflow as tf
from tensorflow.keras import mixed_precision

# ******************************************************************************************************************** #
#                                              CARREGAMENTO DE DADOS                                                   #
# ******************************************************************************************************************** #

AUTOTUNE = tf.data.AUTOTUNE
# ImageNet normalization (canonical)
IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
IMAGENET_STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)


# ======================================================================================================================
# FUNÇÃO PARA O PRE-PROCESSAMENTO DO CONJUNTO DE TREINO
def preprocess_train(image, label):
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
# FUNÇÃO PARA O PRE-PROCESSAMENTO DO CONJUNTO DE VALIDAÇÃO

def preprocess_val(image, label):
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
# FUNÇÃO PARA O CARREGAMENTO EFETIVO DA BASE

def load_data(train_dir, val_dir, batch_size):
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
        .batch(batch_size)
        .repeat()
        .prefetch(AUTOTUNE)
    )

    val_ds = (
        val_ds_raw
        .map(preprocess_val, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .repeat()
        .prefetch(AUTOTUNE)
    )

    return train_ds, val_ds, class_names, num_classes
