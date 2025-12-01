import os
import gc
import numpy as np
import tensorflow as tf

# Callbacks do Keras
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
    Callback
)
# Camadas e regularizadores
from tensorflow.keras import layers, regularizers
# Backbone pré-treinado
from tensorflow.keras.applications import ResNet50
# Classe auxiliar usada apenas para gerenciar diretórios de logs
from ApplicationModel import Model
# Funções de métrica pós-treino
from chexpert_metrics import aggregate_by_study, auc_per_class, auc_mean_5


# ======================================================================================================================
# ======================================================================================================================

# Cria o modelo de rede com a arquitetura ResNet50 - SEM DEPENDÊNCIA DO PYQT
class Resnet50:
    """
        Construtor da arquitetura ResNet50 para uso no experimento CheXpert.

        Responsabilidades:
        -------------------
        ✔ Carregar ResNet50 pré-treinada pela ImageNet
        ✔ Congelar parte inicial das camadas (padrão do fine-tuning)
        ✔ Adicionar cabeçalho customizado para classificação multi-label (14)
        ✔ Retornar um modelo Keras pronto para compilação e treino

        Observações:
        - Última camada deve ser SIGMOID (uma probabilidade por patologia)
        - Função de perda (Loss) usado será BCE, não Softmax/Categorical
    """

    def __init__(self, input_shape=(224, 224, 3), num_classes=14, last_layer_activation="sigmoid"):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.last_layer_activation = last_layer_activation

    def resnet_classifier(self):
        # Carrega backbone pré-treinado, sem a fully-connected final (include_top=False)
        base_model = ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=self.input_shape
        )

        # Congela primeiras camadas do backbone:
        # Isso impede que os pesos do ImageNet sejam destruídos logo no começo.
        # Fine-tuning parcial típico usa ~50% das camadas congeladas.
        for layer in base_model.layers[:140]:
            layer.trainable = False

        # Entrada do modelo
        inputs = layers.Input(shape=self.input_shape)

        # O backbone é aplicado SEM "training=False" para que BN se comporte corretamente
        x = base_model(inputs)
        # GlobalAveragePooling reduz feature map de (H,W,2048) → (2048)
        x = layers.GlobalAveragePooling2D()(x)
        # BatchNormalization ajuda estabilidade do treinamento
        x = layers.BatchNormalization()(x)
        # Dropout contra overfitting no cabeçalho
        x = layers.Dropout(0.4)(x)
        # Camada densa intermediária
        x = layers.Dense(
            256,
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4)
        )(x)

        x = layers.Dropout(0.3)(x)

        # Camada final multi-label (14 probabilidades independentes)
        outputs = layers.Dense(
            self.num_classes,
            activation=self.last_layer_activation,
            kernel_regularizer=regularizers.l2(1e-4)
        )(x)

        # Constrói o modelo final
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model


# ======================================================================================================================
# ======================================================================================================================


class EpochAUCCallback(Callback):
    """
        Callback personalizada que calcula AUC ao final de cada época,
        seguindo o protocolo oficial do CheXpert.

            * Roda predições por view
            * Agrega por estudo usando max pooling
            * Calcula AUC por classe com sklearn
            * Calcula mean AUC das 5 patologias oficiais

        OBSERVAÇÃO: Não foi utilizado  tf.keras.metrics.AUC porque o artigo calcula AUC
        'por estudo', não por view. Isso exige agrupar múltiplas imagens do mesmo exame
        e só então calcular AUC (sklearn), algo impossível de fazer dentro do TF.
    """

    def __init__(self, val_ds, val_paths, log_dir=None):
        super().__init__()
        self.val_ds = val_ds  # dataset SEM shuffle
        self.val_paths = np.array(val_paths)  # paths no MESMO ORDENAMENTO que val_ds
        self.log_dir = log_dir

        # Writer do TensorBoard
        if log_dir:
            self.writer = tf.summary.create_file_writer(log_dir)
        else:
            self.writer = None

    # Coração do callback, definindo o comportamento ao final de cada época
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # Predições na ordem do conjunto de validação
        preds = self.model.predict(self.val_ds, verbose=0)

        # Extração de labels na mesma ordem do val_ds
        labels = []
        for _, batch_labels in self.val_ds:
            labels.append(batch_labels.numpy())
        labels = np.vstack(labels)

        # Agregar por estudo (max pooling por estudo)
        preds_study, labels_study, keys = aggregate_by_study(
            self.val_paths.tolist(),
            preds,
            labels,
            mode="max"
        )

        # Calcular AUC por classe + AUC média das 5 doenças
        aucs = auc_per_class(preds_study, labels_study)
        mean5 = auc_mean_5(aucs)

        print(f"\n[EPOCH {epoch + 1}] Mean AUC (5 diseases): {mean5:.4f}")
        print(f"[EPOCH {epoch + 1}] AUCs (14 classes):", ["{:.4f}".format(x) if x == x else "nan" for x in aucs])

        # Registro no TensorBoard
        if self.writer:
            with self.writer.as_default():
                tf.summary.scalar("val_auc_mean5", mean5, step=epoch)
                for i, auc_val in enumerate(aucs):
                    if not np.isnan(auc_val):
                        tf.summary.scalar(f"val_auc_class_{i}", auc_val, step=epoch)
            try:
                self.writer.flush()
            except Exception:
                pass

# ======================================================================================================================


class ResNet_Trainer:

    def __init__(self, neural_network, train_data, val_data, val_paths, epochs, logName):
        """
            Classe responsável pelo treino da rede:
                ✔ compilar modelo (com BCE e sem métricas internas)
                ✔ configurar callbacks
                ✔ rodar model.fit()
                ✔ salvar pesos finais e melhores pesos
                ✔ registrar no TensorBoard
                ✔ garantir limpeza de sessão após o treino
        """
        self.neural_network = neural_network
        self.train_data = train_data
        self.val_data = val_data
        self.val_paths = val_paths  # obrigatório para max pooling por estudo
        self.epochs = epochs
        self.logName = logName
        self.history = None

    def run(self):
        writer = None

        try:
            if self.neural_network is None:
                raise ValueError("Modelo não fornecido.")
            if self.train_data is None or self.val_data is None:
                raise ValueError("Dados de treino/validação não fornecidos.")
            if self.epochs <= 0:
                raise ValueError("Número de épocas inválido.")

            print("\n[TRAIN] Iniciando treinamento CheXpert-ResNet50...\n")

            # Diretório de logs
            model_helper = Model()
            log_path = model_helper.log_directory_manager(self.logName)
            os.makedirs(log_path, exist_ok=True)

            writer = tf.summary.create_file_writer(log_path)

            # Compilação — usando SOMENTE BinaryCrossentropy
            # (AUC será computada externamente, como no artigo)
            optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

            self.neural_network.compile(
                optimizer=optimizer,
                loss=loss
            )

            # CALLBACKS — padrão CheXpert + AUC custom
            checkpoint_path = os.path.join(log_path, "best_weights_resnet.h5")

            callbacks = [
                ModelCheckpoint(
                    filepath=checkpoint_path,
                    monitor="val_loss",
                    save_best_only=True,
                    save_weights_only=True,
                    verbose=1
                ),
                EarlyStopping(
                    monitor="val_loss",
                    patience=8,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=4,
                    min_lr=1e-6,
                    verbose=1
                ),
                TensorBoard(log_dir=log_path),
                EpochAUCCallback(
                    val_ds=self.val_data,
                    val_paths=self.val_paths,
                    log_dir=log_path
                )
            ]

            # Treinamento de fato
            self.history = self.neural_network.fit(
                self.train_data,
                epochs=self.epochs,
                validation_data=self.val_data,
                callbacks=callbacks,
                verbose=1
            )

            # Salvamento final
            final_path = os.path.join(log_path, "final_weights.h5")
            self.neural_network.save_weights(final_path)
            print(f"\n[TRAIN] Pesos finais salvos em: {final_path}")

        except Exception as e:
            print(f"\n[ERRO] {e}\n")

        finally:
            if writer:
                writer.close()
            tf.keras.backend.clear_session()
            gc.collect()

        return self.history
