
# import traceback
from ApplicationModel import Model
from PyQt5.QtCore import QThread, pyqtSignal
from tensorflow.keras.callbacks import TensorBoard, Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import gc
import os


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class ViT_TrainerThread(QThread):
    log_signal = pyqtSignal(str)
    training_finished = pyqtSignal(bool)

    def __init__(self, neural_network, train_data, val_data, epochs, logName, steps_train, steps_val):

        # Construtor da classe
        super().__init__()
        self.neural_network = neural_network
        self.train_data = train_data
        self.val_data = val_data
        self.epochs = epochs
        self.logName = logName
        self.history = None
        self.steps_train = steps_train
        self.steps_val = steps_val

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
            # envia para a UI
            try:
                self.outer.log_signal.emit(msg)
            except Exception:
                pass

            # escreve scalars no tensorboard
            try:
                with self.writer.as_default():
                    for k, v in logs.items():
                        if isinstance(v, (float, int)):
                            tf.summary.scalar(k, v, step=epoch)
                    try:
                        self.writer.flush()
                    except Exception:
                        pass
            except Exception:
                # não deixar o callback quebrar o treino
                try:
                    self.outer.log_signal.emit("[TensorBoard] falha ao registrar métricas")
                except Exception:
                    pass

    def run(self):
        writer = None
        success = False
        try:
            model_helper = Model()  # gerenciador de diretórios / logs
            log_path = model_helper.log_directory_manager(self.logName)
            os.makedirs(log_path, exist_ok=True)

            self.log_signal.emit("Iniciando treinamento ViT...")
            self.log_signal.emit(f"Logs armazenados em: {log_path}")

            # create writer
            writer = tf.summary.create_file_writer(log_path)

            # recompila o modelo com configurações robustas
            # Observação: assume que a saída do modelo é softmax; ajustar from_logits se necessário.
            opt = tf.keras.optimizers.Adam(learning_rate=5e-5)
            loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1, from_logits=False)

            self.neural_network.compile(
                optimizer=opt,
                loss=loss,
                metrics=["accuracy"]
            )

            # callbacks essenciais
            checkpoint_path = os.path.join(log_path, "best_weights_vit.h5")
            checkpoint = ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
                verbose=0
            )

            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=8,
                restore_best_weights=True,
                verbose=1
            )

            reduce_lr = ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=4,
                min_lr=1e-6,
                verbose=1
            )

            log_callback = self.LogCallback(writer=writer, total_epochs=self.epochs, outer_instance=self)

            tensorboard_callback = TensorBoard(
                log_dir=log_path,
                histogram_freq=0,    # desabilita histogramas (muito pesado)
                write_graph=False,
                update_freq='epoch',
                profile_batch=0
            )

            callbacks = [tensorboard_callback, checkpoint, early_stop, reduce_lr, log_callback]

            # Fit
            self.history = self.neural_network.fit(
                self.train_data,
                epochs=self.epochs,
                validation_data=self.val_data,
                steps_per_epoch=self.steps_train,
                validation_steps=self.steps_val,
                callbacks=callbacks,
                verbose=1
            )

            # Se chegou aqui, os callbacks (EarlyStopping) provavelmente já restauraram os melhores pesos
            # Tentamos salvar os pesos finais (garantir que são os melhores)
            try:
                # preferimos salvar os best_weights do checkpoint se existir
                if os.path.exists(checkpoint_path):
                    # load checkpoint into model (só para garantir)
                    self.neural_network.load_weights(checkpoint_path)
                # agora salvar com nome mais amigável (opcional)
                final_path = f"{self.logName}_vit_best_weights.h5"
                self.neural_network.save_weights(final_path)
                self.log_signal.emit(f"Pesos de treinamento salvos como {final_path}")
                success = True
            except Exception as save_err:
                self.log_signal.emit(f"Erro ao salvar pesos ViT: {save_err}")
                success = False

        except Exception as e:
            # captura qualquer erro durante o fit
            try:
                self.log_signal.emit(f"Erro durante o treinamento ViT: {str(e)}")
            except Exception:
                pass
            success = False

        finally:
            # fechar writer e liberar memória
            try:
                if writer is not None:
                    writer.close()
            except Exception:
                pass

            try:
                tf.keras.backend.clear_session()
            except Exception:
                pass
            try:
                gc.collect()
            except Exception:
                pass

            # emitir sinal final
            try:
                self.training_finished.emit(bool(success))
            except Exception:
                pass

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


class ResNet_TrainerThread(QThread):
    log_signal = pyqtSignal(str)
    training_finished = pyqtSignal(bool)

    def __init__(self, neural_network, train_data, val_data, epochs, logName, steps_train, steps_val):
        super().__init__()
        self.neural_network = neural_network
        self.train_data = train_data
        self.val_data = val_data
        self.epochs = epochs
        self.logName = logName
        self.history = None
        self.steps_train = steps_train
        self.steps_val = steps_val

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
            try:
                self.outer.log_signal.emit(msg)
            except Exception:
                pass

            try:
                with self.writer.as_default():
                    for k, v in logs.items():
                        if isinstance(v, (float, int)):
                            tf.summary.scalar(k, v, step=epoch)
                    try:
                        self.writer.flush()
                    except Exception:
                        pass
            except Exception:
                try:
                    self.outer.log_signal.emit("[TensorBoard] falha ao registrar métricas")
                except Exception:
                    pass

    def run(self):
        writer = None
        success = False
        try:
            # validações básicas
            if self.neural_network is None:
                raise ValueError("O modelo (neural_network) não foi fornecido.")
            if self.train_data is None or self.val_data is None:
                raise ValueError("Dados de treino/validação não informados.")
            if not isinstance(self.epochs, int) or self.epochs <= 0:
                raise ValueError("Número de épocas inválido.")

            model_helper = Model()
            log_path = model_helper.log_directory_manager(self.logName)
            os.makedirs(log_path, exist_ok=True)

            self.log_signal.emit("Iniciando treinamento ResNet...")
            self.log_signal.emit(f"Logs serão salvos em: {log_path}")

            writer = tf.summary.create_file_writer(log_path)

            # recompilar o modelo aqui com hiperparâmetros robustos
            opt = tf.keras.optimizers.Adam(learning_rate=3e-5)

            # ----------------------------------------------------------------------------------------------------------
            '''
            # COMENTAR O SGUINTE 'LOSS' E 'COMPILE' AO UTILIZAR O TESTE COM CHEXPERT
            loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1, from_logits=False)

            self.neural_network.compile(
                optimizer=opt,
                loss=loss,
                metrics=["accuracy"]
            )
            '''

            # COMENTAR O SGUINTE 'LOSS' E 'COMPILE' AO UTILIZAR O DATASET DOS RATOS
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

            self.neural_network.compile(
                optimizer=opt,
                loss=loss,
                metrics=[tf.keras.metrics.AUC(multi_label=True)]
            )
            # ----------------------------------------------------------------------------------------------------------

            # callbacks
            checkpoint_path = os.path.join(log_path, "best_weights_resnet.h5")
            checkpoint = ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
                verbose=0
            )

            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=8,
                restore_best_weights=True,
                verbose=1
            )

            reduce_lr = ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=4,
                min_lr=1e-6,
                verbose=1
            )

            log_callback = self.LogCallback(writer=writer, total_epochs=self.epochs, outer_instance=self)

            tensorboard_callback = TensorBoard(
                log_dir=log_path,
                histogram_freq=0,
                write_graph=False,
                update_freq='epoch',
                profile_batch=0
            )

            callbacks = [tensorboard_callback, checkpoint, early_stop, reduce_lr, log_callback]

            '''
            # COMENTAR O SGUINTE 'FIT' AO UTILIZAR O TESTE COM CHEXPERT
            # fit
            self.history = self.neural_network.fit(
                self.train_data,
                epochs=self.epochs,
                validation_data=self.val_data,
                steps_per_epoch=self.steps_train,
                validation_steps=self.steps_val,
                callbacks=callbacks,
                verbose=1
            )
            '''

            # COMENTAR O SGUINTE 'FIT' AO UTILIZAR O DATASET DOS RATOS
            self.history = self.neural_network.fit(
                self.train_data,
                epochs=self.epochs,
                validation_data=self.val_data,
                callbacks=callbacks,
                verbose=1
            )

            # salvar pesos (preferir o checkpoint best)
            try:
                if os.path.exists(checkpoint_path):
                    self.neural_network.load_weights(checkpoint_path)
                final_path = f"{self.logName}_resnet_best_weights.h5"
                self.neural_network.save_weights(final_path)
                self.log_signal.emit(f"Pesos de treinamento salvos como {final_path}")
                success = True
            except Exception as save_err:
                self.log_signal.emit(f"Erro ao salvar pesos ResNet: {save_err}")
                success = False

        except Exception as e:
            try:
                self.log_signal.emit(f"Erro durante o treinamento ResNet: {str(e)}")
            except Exception:
                pass
            success = False

        finally:
            try:
                if writer is not None:
                    writer.close()
            except Exception:
                pass

            try:
                tf.keras.backend.clear_session()
            except Exception:
                pass

            try:
                gc.collect()
            except Exception:
                pass

            try:
                self.training_finished.emit(bool(success))
            except Exception:
                pass
