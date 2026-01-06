import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE

# ======================================================================================================================
# NORMALIZAÇÃO (ImageNet mean/std)

IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
IMAGENET_STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

# ======================================================================================================================
# TÉCNICA DE AUMGMENTATION ESTATÍSTICA PCA COLOR JITTER

def pca_color_jitter(image, alpha_std=0.1):
    """
        PCA color augmentation (AlexNet-style).
        image: Tensor float32 [H, W, 3], valores em [0, 1]
    """

    # Autovalores e autovetores pré-computados do ImageNet (AlexNet)
    eigenvalues = tf.constant([0.2175, 0.0188, 0.0045], dtype=tf.float32)
    eigenvectors = tf.constant([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ], dtype=tf.float32)

    # Amostra ruído Gaussiano
    alphas = tf.random.normal([3], mean=0.0, stddev=alpha_std)

    # Calcula perturbação
    delta = tf.reduce_sum(eigenvectors * (alphas * eigenvalues), axis=1)

    # Aplica jitter
    image = image + delta

    # Garante valores válidos
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image

# ======================================================================================================================
# PRÉ-PROCESSAMENTO PARA TREINO — FIEL AO PAPER RESNET

def preprocess_train(image, label):

    # 1) SCALE JITTER: lado curto ∈ [256, 480]
    target_short = tf.random.uniform([], minval=256, maxval=480, dtype=tf.int32)

    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    short = tf.minimum(h, w)

    scale = tf.cast(target_short, tf.float32) / tf.cast(short, tf.float32)

    new_h = tf.cast(tf.round(tf.cast(h, tf.float32) * scale), tf.int32)
    new_w = tf.cast(tf.round(tf.cast(w, tf.float32) * scale), tf.int32)

    image = tf.image.resize(image, [new_h, new_w])

    # 2) RANDOM CROP 224×224
    image = tf.image.random_crop(image, [224, 224, 3])

    # 3) RANDOM HORIZONTAL FLIP
    image = tf.image.random_flip_left_right(image)

    # 4) NORMALIZAÇÃO PARA [0,1]
    image = image / 255.0

    # 5) PCA COLOR JITTER (AlexNet-style)
    image = pca_color_jitter(image)

    # 6) NORMALIZAÇÃO ImageNet (mean/std)
    image = (image - IMAGENET_MEAN) / IMAGENET_STD

    return image, label


# ======================================================================================================================
# PRÉ-PROCESSAMENTO PARA VALIDAÇÃO — PADRÃO IMAGE NET
def preprocess_val(image, label):

    # 1) Resize mantendo aspecto: menor lado = 256
    h = tf.cast(tf.shape(image)[0], tf.float32)
    w = tf.cast(tf.shape(image)[1], tf.float32)
    short = tf.minimum(h, w)

    scale = 256.0 / short
    new_h = tf.cast(tf.round(h * scale), tf.int32)
    new_w = tf.cast(tf.round(w * scale), tf.int32)

    image = tf.image.resize(image, [new_h, new_w])

    # 2) Center crop 224×224
    image = tf.image.resize_with_crop_or_pad(image, 224, 224)

    # 3) Normalização
    image = image / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD

    return image, label


# ======================================================================================================================
# FUNÇÃO PLUG-AND-PLAY QUE AGREGA AS DEMAIS

def apply_preprocessing(train_ds, val_ds):
    train_ds = (
        train_ds
        .map(preprocess_train, num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )

    val_ds = (
        val_ds
        .map(preprocess_val, num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )

    return train_ds, val_ds
