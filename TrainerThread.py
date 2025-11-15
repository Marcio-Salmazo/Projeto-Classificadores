import gc
import os
import traceback

from PyQt5.QtCore import QThread, pyqtSignal
from tensorflow.keras.callbacks import TensorBoard, Callback
from ApplicationModel import Model
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


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
            # console message
            msg = f"Época {epoch + 1}/{self.total_epochs}"
            for k, v in logs.items():
                if isinstance(v, (float, int)):
                    msg += f" | {k}: {v:.4f}"
            self.outer.log_signal.emit(msg)

            # write scalars only (no histograms)
            with self.writer.as_default():
                for k, v in logs.items():
                    if isinstance(v, (float, int)):
                        tf.summary.scalar(k, v, step=epoch)
                # flush so TensorBoard files are updated frequently
                try:
                    self.writer.flush()
                except Exception:
                    pass

    def run(self):
        writer = None
        try:
            model_helper = Model()  # seu gerenciador de diretórios / logs
            log_path = model_helper.log_directory_manager(self.logName)

            self.log_signal.emit("Iniciando treinamento...")
            self.log_signal.emit(f"Logs armazenados em: {log_path}")

            # Create tf summary writer
            writer = tf.summary.create_file_writer(log_path)

            # callbacks
            log_callback = self.LogCallback(writer=writer, total_epochs=self.epochs, outer_instance=self)

            # IMPORTANT: disable expensive features that can cause OOM/crash
            tensorboard_callback = TensorBoard(
                log_dir=log_path,
                histogram_freq=0,  # desabilita histogramas (muito pesado para ViT)
                write_graph=False,  # não desenhar o grafo
                update_freq='epoch',
                profile_batch=0  # desabilita profiling
            )

            callbacks = [tensorboard_callback, log_callback]

            # Fit (executa dentro da thread que tem o modelo em uso)
            self.history = self.neural_network.fit(
                self.train_data,
                epochs=self.epochs,
                validation_data=self.val_data,
                callbacks=callbacks,
            )

            # SALVAR PESOS AQUI DENTRO DA MESMA THREAD
            self.log_signal.emit("Treinamento concluído. Salvando pesos...")
            try:
                self.neural_network.save_weights(f"{self.logName}_weights.h5")
                self.log_signal.emit(f"Pesos de treinamento salvos como {self.logName}_weights.h5")

            except Exception as save_err:
                self.log_signal.emit(f"Pesos de treinamento salvos como {self.logName}_weights.h5")
                self.training_finished.emit(False)

            # finally, clean up
            try:
                writer.close()
            except Exception:
                pass

            # clear TF session & free memory
            try:
                tf.keras.backend.clear_session()
                gc.collect()
            except Exception:
                pass

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

    #   CALLBACK INTERNO
    class LogCallback(Callback):
        def __init__(self, writer, total_epochs, outer_instance):
            super().__init__()
            self.writer = writer
            self.total_epochs = total_epochs
            self.outer = outer_instance  # referência para a thread

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}

            # Criar mensagem para exibir no PyQt
            msg = f"Época {epoch + 1}/{self.total_epochs}"
            for k, v in logs.items():
                if isinstance(v, (float, int)):
                    msg += f" | {k}: {v:.4f}"

            # Enviar mensagem para a UI
            self.outer.log_signal.emit(msg)

            # Registrar no TensorBoard
            try:
                with self.writer.as_default():
                    for k, v in logs.items():
                        if isinstance(v, (float, int)):
                            tf.summary.scalar(k, v, step=epoch)
            except Exception as tb_err:
                self.outer.log_signal.emit(f"[TensorBoard] Falha ao registrar logs: {tb_err}")

    #   EXECUÇÃO DA THREAD
    def run(self):
        try:

            # VALIDAÇÕES INICIAIS
            if self.neural_network is None:
                raise ValueError("O modelo (neural_network) não foi fornecido.")
            if self.train_data is None:
                raise ValueError("train_data está vazio ou não foi fornecido.")
            if self.val_data is None:
                raise ValueError("val_data está vazio ou não foi fornecido.")
            if not isinstance(self.epochs, int) or self.epochs <= 0:
                raise ValueError("O número de épocas deve ser um inteiro maior que zero.")

            # GERAR DIRETÓRIO DE LOG
            model = Model()
            log_path = model.log_directory_manager(self.logName)

            self.log_signal.emit("Iniciando treinamento da ResNet...")
            self.log_signal.emit(f"Logs serão salvos em: {log_path}")
            writer = tf.summary.create_file_writer(log_path)

            # CALLBACKS
            log_callback = self.LogCallback(
                writer=writer,
                total_epochs=self.epochs,
                outer_instance=self
            )

            tensorboard_callback = TensorBoard(
                log_dir=log_path,
                histogram_freq=0
            )

            # TREINAMENTO
            self.history = self.neural_network.fit(
                self.train_data,
                epochs=self.epochs,
                validation_data=self.val_data,
                callbacks=[tensorboard_callback, log_callback]
            )

            # SALVAR PESOS AQUI DENTRO DA MESMA THREAD
            self.log_signal.emit("Treinamento concluído. Salvando pesos...")
            try:
                self.neural_network.save_weights(f"{self.logName}_weights.h5")
                self.log_signal.emit(f"Pesos de treinamento salvos como {self.logName}_weights.h5")

            except Exception as save_err:
                self.log_signal.emit(f"Pesos de treinamento salvos como {self.logName}_weights.h5")
                self.training_finished.emit(False)

            # finally, clean up
            try:
                writer.close()
            except Exception:
                pass

            # clear TF session & free memory
            try:
                tf.keras.backend.clear_session()
                gc.collect()
            except Exception:
                pass

            self.log_signal.emit("Treinamento finalizado com sucesso!")
            self.training_finished.emit(True)

        except Exception as e:
            self.log_signal.emit(f"Erro durante o treinamento: {str(e)}")
            self.training_finished.emit(False)
