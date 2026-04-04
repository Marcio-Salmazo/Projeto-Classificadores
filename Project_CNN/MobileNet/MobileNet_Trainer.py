import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, TensorBoard
import os
import datetime


def compile_model(model):
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=0.045,
        rho=0.9,
        momentum=0.9
    )
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_model(model, train_ds, val_ds, epochs=50):
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.98,
        patience=1
    )

    # CSV Logger
    csv_logger = CSVLogger(
        "training_metrics.csv",
        separator=',',
        append=False
    )

    # TensorBoard Logger
    log_dir = os.path.join(
        "logs",
        "fit",
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    tensorboard_callback = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,  # salva histogramas (pesos)
        write_graph=True,  # salva grafo do modelo
        write_images=False
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[lr_scheduler, csv_logger, tensorboard_callback]
    )

    return history
