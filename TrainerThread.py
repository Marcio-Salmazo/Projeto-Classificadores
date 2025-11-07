from PyQt5.QtCore import QThread, pyqtSignal
from tensorflow.keras.callbacks import TensorBoard, Callback
from ApplicationModel import Model
import tensorflow as tf
import gc

'''
class ViT_TrainerThread(QThread):
    log_signal = pyqtSignal(str)
    training_finished = pyqtSignal(bool)

    def __init__(self, neural_network, train_data, val_data, epochs, logName):
        super().__init__()

        self.neural_network = neural_network
        self.train_data = train_data
        self.val_data = val_data
        self.epochs = epochs
        self.logName = logName
        self.history = None

        self.neural_network.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )

    class LogCallback(Callback):
        def __init__(self, writer, total_epochs, outer_instance):
            super().__init__()
            self.writer = writer
            self.total_epochs = total_epochs
            self.outer = outer_instance

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            msg = f"Época {epoch + 1}/{self.total_epochs}"

            for k, v in logs.items():
                if isinstance(v, (float, int)):
                    msg += f" | {k}: {v:.4f}"

            # envia log para a UI
            self.outer.log_signal.emit(msg)

            # salva para TensorBoard
            with self.writer.as_default():
                for k, v in logs.items():
                    if isinstance(v, (float, int)):
                        tf.summary.scalar(k, v, step=epoch)

    def run(self):
        try:
            model = Model()
            log_path = model.log_directory_manager(self.logName)

            self.log_signal.emit("Iniciando treinamento...")
            writer = tf.summary.create_file_writer(log_path)

            log_callback = self.LogCallback(writer, self.epochs, self)
            tensorboard_callback = TensorBoard(log_dir=log_path, histogram_freq=0)  # <= IMPORTANTE

            self.history = self.neural_network.fit(
                self.train_data,
                epochs=self.epochs,
                validation_data=self.val_data,
                callbacks=[tensorboard_callback, log_callback]
            )

            self.log_signal.emit("Treinamento finalizado com sucesso!")

        except Exception as e:
            self.log_signal.emit(f"Erro durante o treinamento: {str(e)}")

        finally:
            try:
                writer.close()
            except:
                pass

            tf.keras.backend.clear_session()
            gc.collect()
            self.training_finished.emit(True)

 '''


class ViT_TrainerThread(QThread):
    log_signal = pyqtSignal(str)
    training_finished = pyqtSignal(bool)

    def __init__(self, neural_network, train_data, val_data, epochs, logName):
        super().__init__()

        self.neural_network = neural_network
        self.train_data = train_data
        self.val_data = val_data
        self.epochs = epochs
        self.logName = logName
        self.history = None

        self.neural_network.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )

    class LogCallback(Callback):
        def __init__(self, writer, total_epochs, outer_instance):
            super().__init__()
            self.writer = writer
            self.total_epochs = total_epochs
            self.outer = outer_instance  # referência para a thread

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}

            msg = f"Época {epoch + 1}/{self.total_epochs}"
            for k, v in logs.items():
                if isinstance(v, (float, int)):
                    msg += f" | {k}: {v:.4f}"

            # usar o sinal da thread principal
            self.outer.log_signal.emit(msg)

            # registrar no TensorBoard também (opcional)
            with self.writer.as_default():
                for k, v in logs.items():
                    if isinstance(v, (float, int)):
                        tf.summary.scalar(k, v, step=epoch)

    def run(self):
        try:
            model = Model()
            log_path = model.log_directory_manager(self.logName)

            self.log_signal.emit("Iniciando treinamento...")
            self.log_signal.emit(f"Logs armazenados em: {log_path}")

            writer = tf.summary.create_file_writer(log_path)

            # cria o callback interno
            log_callback = self.LogCallback(
                writer=writer,
                total_epochs=self.epochs,
                outer_instance=self
            )

            tensorboard_callback = TensorBoard(
                log_dir=log_path,
                histogram_freq=1
            )

            self.history = self.neural_network.fit(
                self.train_data,
                epochs=self.epochs,
                validation_data=self.val_data,
                callbacks=[tensorboard_callback, log_callback]
            )

            self.log_signal.emit("Treinamento finalizado com sucesso!")
            self.training_finished.emit(True)

        except Exception as e:
            self.log_signal.emit(f"Erro durante o treinamento: {str(e)}")
            self.training_finished.emit(False)


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

class ResNet_TrainerThread(QThread):
    log_signal = pyqtSignal(str)
    training_finished = pyqtSignal(bool)

    def __init__(self, neural_network, train_data, val_data, epochs, logName):
        super().__init__()
        self.neural_network = neural_network
        self.train_data = train_data
        self.val_data = val_data
        self.epochs = epochs
        self.logName = logName
        self.history = None

    class LogCallback(Callback):
        def __init__(self, writer, total_epochs, outer_instance):
            super().__init__()
            self.writer = writer
            self.total_epochs = total_epochs
            self.outer = outer_instance  # referência para a thread

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}

            msg = f"Época {epoch + 1}/{self.total_epochs}"
            for k, v in logs.items():
                if isinstance(v, (float, int)):
                    msg += f" | {k}: {v:.4f}"

            # usar o sinal da thread principal
            self.outer.log_signal.emit(msg)

            # registrar no TensorBoard também (opcional)
            with self.writer.as_default():
                for k, v in logs.items():
                    if isinstance(v, (float, int)):
                        tf.summary.scalar(k, v, step=epoch)

    def run(self):
        try:
            model = Model()
            log_path = model.log_directory_manager(self.logName)

            self.log_signal.emit("Iniciando treinamento...")
            self.log_signal.emit(f"Logs armazenados em: {log_path}")

            writer = tf.summary.create_file_writer(log_path)

            # cria o callback interno
            log_callback = self.LogCallback(
                writer=writer,
                total_epochs=self.epochs,
                outer_instance=self
            )

            tensorboard_callback = TensorBoard(
                log_dir=log_path,
                histogram_freq=1
            )

            self.history = self.neural_network.fit(
                self.train_data,
                epochs=self.epochs,
                validation_data=self.val_data,
                callbacks=[tensorboard_callback, log_callback]
            )

            self.log_signal.emit("Treinamento finalizado com sucesso!")
            self.training_finished.emit(True)

        except Exception as e:
            self.log_signal.emit(f"Erro durante o treinamento: {str(e)}")
            self.training_finished.emit(False)