import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE
IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
IMAGENET_STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)


# ======================================================================================================================
# FUNÇÃO PARA O PRE-PROCESSAMENTO DO CONJUNTO DE TREINO
def preprocess_train(image, label):
    # Resize direto (mantendo consistência com MobileNet)
    image = tf.image.resize(image, (160, 160))

    # Augmentation leve (SEM distorcer semântica)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)

    # Normalização ImageNet (mantém comparabilidade com ResNet)
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD

    return image, label


# ======================================================================================================================
# FUNÇÃO PARA O PRE-PROCESSAMENTO DO CONJUNTO DE VALIDAÇÃO
def preprocess_val(image, label):
    image = tf.image.resize(image, (160, 160))
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD

    return image, label


# ======================================================================================================================
# FUNÇÃO PARA O CARREGAMENTO EFETIVO DA BASE
def load_data(train_dir, val_dir, batch_size, img_size=160):
    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        image_size=(img_size, img_size),
        batch_size=None,
        shuffle=True
    )

    val_ds_raw = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="int",
        image_size=(img_size, img_size),
        batch_size=None,
        shuffle=False
    )

    class_names = train_ds_raw.class_names
    num_classes = len(class_names)

    train_ds = (
        train_ds_raw
        .map(preprocess_train, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    val_ds = (
        val_ds_raw
        .map(preprocess_val, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    return train_ds, val_ds, class_names, num_classes
