import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, TensorBoard
import os
import datetime


class PlateauStopping(tf.keras.callbacks.Callback):
    def __init__(self, monitor='val_accuracy', window=50, threshold=0.005):
        super().__init__()
        self.monitor = monitor
        self.window = window
        self.threshold = threshold
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        value = logs.get(self.monitor)

        if value is None:
            return

        self.history.append(value)

        # Só começa a checar depois da janela cheia
        if len(self.history) >= self.window:
            recent = self.history[-self.window:]

            variation = max(recent) - min(recent)

            print(f"\n[Plateau Check] Variação últimas {self.window} épocas: {variation:.6f}")

            if variation < self.threshold:
                print(f"\nTreinamento interrompido por plateau (variação < {self.threshold})")
                self.model.stop_training = True


def compile_model(model, LR=1e-3, MOMENTUM=0.9):

    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=LR,
        rho=0.9,
        momentum=MOMENTUM
    )

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_model(model, train_ds, val_ds, epochs=3000, log_dir=None, checkpoint_path=None):

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-5
    )

    plateau_callback = PlateauStopping(
        monitor='val_accuracy',
        window=30,
        threshold=0.003
    )

    # CSV Logger
    csv_logger = CSVLogger(
        "training_metrics.csv",
        separator=',',
        append=False
    )

    '''
    # TensorBoard Logger
    log_dir = os.path.join(
        "logs",
        "fit",
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    '''

    tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="accuracy",
        mode="max",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    ),

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
        callbacks=[lr_scheduler,
                   csv_logger,
                   tensorboard_callback,
                   plateau_callback]
    )

    return history
