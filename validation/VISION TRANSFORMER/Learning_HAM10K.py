"""
    Learning_HAM10K.py

    Script / módulo responsável por:
        - Construir a arquitetura ViT (usando sua classe VisionTransformer)
        - Carregar HAM10000 via HAM10000Loader
        - Treinar o modelo com callbacks (checkpoint, early-stop, reduce-lr, tensorboard)
        - Avaliar o modelo ao final com ham_metrics.evaluate_ham10000

    Observações:
        - Este arquivo NÃO usa PyQt/QThread — é um runner console-friendly,
          voltado para testes de validação científica da sua implementação ViT.
        - Assume que PatchExtractor e PatchEncoder (usados pela sua VisionTransformer)
"""

import os
import gc
import tensorflow as tf

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard
)
from ModelCreator import VisionTransformer


class ViTBuilder:

    # Construtor
    def __init__(self, input_shape=(224, 224, 3), patch_size=16, projection_dim=64, transformer_layers=8,
                 num_heads=4, mlp_units=128, num_classes=7):
        self.input_shape = input_shape
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.transformer_layers = transformer_layers
        self.num_heads = num_heads
        self.mlp_units = mlp_units
        self.num_classes = num_classes

    # Função utilitária para criar o ViT usando sua classe e parâmetros típicos
    def build_vit(self):
        """
            Constrói um modelo ViT usando a classe VisionTransformer construída préviamente.

            OBSERVAÇÃO:
                * num_patches é calculado como (H*W) / (patch_size^2) e precisa ser inteiro.
        """

        # calculo do total de patches de acordo com o que foi descrito no artigo
        H, W, C = self.input_shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "input_shape deve ser divisível por patch_size"
        num_patches = (H // self.patch_size) * (W // self.patch_size)

        # Instância da classe VisionTransformer proveniente do ModelCreator.py (Implementação original)
        vit_instance = VisionTransformer(
            input_shape=self.input_shape,
            patch_size=self.patch_size,
            num_patches=num_patches,
            projection_dim=self.projection_dim,
            transformer_layers=self.transformer_layers,
            num_heads=self.num_heads,
            mlp_units=self.mlp_units,
            num_classes=self.num_classes
        )

        # O modelo é criado a partir da função vit_classifier()
        # Nela são definidos as operações com os Patches
        model = vit_instance.vit_classifier()
        return model


# Classe Trainer limpa — sem associação com PyQt/QThread
class ViTTrainer:

    # Construtor
    def __init__(self, model, train_ds, val_ds, n_train, n_val, log_name="vit_ham10000_run",
                 epochs=30, batch_size=32, lr=5e-5, label_smoothing=0.1, checkpoint_dir="logs"):

        # Instância do modelo Keras
        self.vit_model = model

        # tf.data.Dataset para treino e validação
        self.train_ds = train_ds
        self.val_ds = val_ds

        # contagens (usadas para steps_per_epoch / info)
        self.n_train = n_train
        self.n_val = n_val

        # parâmetros de treino / paths
        self.batch_size = batch_size
        self.epochs = epochs
        self.log_name = log_name
        self.lr = lr
        self.label_smoothing = label_smoothing
        self.checkpoint_dir = checkpoint_dir

        # calculo dos steps automaticamente (inteiro)
        self.steps_per_epoch = max(1, self.n_train // self.batch_size)
        self.validation_steps = max(1, self.n_val // self.batch_size)

    def compile_model(self):
        """
            Compila o modelo com otimização e loss apropriadas.
            CategoricalCrossentropy: assumimos one-hot labels no loader.
        """

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        loss = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=self.label_smoothing,
            from_logits=False
        )

        # Métrica simples: accuracy (top-1), suficiente para HAM10000
        self.vit_model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=["accuracy"]
        )

    def make_callbacks(self):
        """
            Cria callbacks comuns: checkpoint (melhor val_loss), early stopping,
            ReduceLROnPlateau e TensorBoard.
        """
        # Diretório para armazenamento dos logs (Utilizáveis no TensorBoard)
        log_path = os.path.join(self.checkpoint_dir, self.log_name)
        os.makedirs(log_path, exist_ok=True)

        checkpoint_path = os.path.join(log_path, "best_weights_vit.h5")
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
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

        tensorboard = TensorBoard(
            log_dir=log_path,
            update_freq="epoch",
            profile_batch=0
        )

        return [checkpoint, early_stop, reduce_lr, tensorboard], checkpoint_path, log_path

    def train(self):
        """
            Executa o fluxo de treino:
                * compila o modelo
                * cria callbacks
                * chama model.fit(...)
                * salva pesos finais
                * retorna history
        """

        # Compilar
        print("Compilando modelo...")
        self.compile_model()

        # Callbacks
        callbacks, checkpoint_path, log_path = self.make_callbacks()
        print(f"Logs e checkpoints em: {log_path}")

        # Treinamento: Usamos `steps_per_epoch` calculado a partir do n_train
        # (evita problemas de término precoce do dataset)
        print(f"Iniciando treinamento por {self.epochs} épocas.")
        print(f"steps_per_epoch = {self.steps_per_epoch}, validation_steps = {self.validation_steps}")

        history = self.vit_model.fit(
            self.train_ds,
            epochs=self.epochs,
            validation_data=self.val_ds,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            callbacks=callbacks,
            verbose=1
        )

        # Certificar que os melhores pesos do checkpoint estão carregados
        try:
            if os.path.exists(checkpoint_path):
                self.vit_model.load_weights(checkpoint_path)
        except Exception as e:
            print(f"Aviso: falha ao carregar checkpoint: {e}")

        # Armazenamento dos pesos finais
        final_weights_path = os.path.join(log_path, f"{self.log_name}_final_weights.h5")
        try:
            self.vit_model.save_weights(final_weights_path)
            print(f"Pesos finais salvos em: {final_weights_path}")
        except Exception as e:
            print(f"Erro ao salvar pesos finais: {e}")

        # limpar sessão para liberar memória GPU (bom em scripts)
        tf.keras.backend.clear_session()
        gc.collect()
        return history
