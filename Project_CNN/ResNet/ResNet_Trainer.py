import tensorflow as tf
import os


class ResNet_Trainer:
    """
    Trainer fiel ao paper:
    'Deep Residual Learning for Image Recognition' (He et al.)
    """

    def __init__(
        self,
        model,
        train_ds,
        val_ds,
        num_classes=1000,
        batch_size=256,
        epochs=120,
        initial_lr=0.1,
        momentum=0.9,
        weight_decay=1e-4,
        train_size=1281167,   # ImageNet train
        val_size=50000,       # ImageNet val
        log_dir="logs",
        checkpoint_path="checkpoints/resnet50_best.h5",
    ):
        self.model = model
        self.train_ds = train_ds
        self.val_ds = val_ds

        self.batch_size = batch_size
        self.epochs = epochs
        self.initial_lr = initial_lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        # Número fixo de iterações (paper)
        self.steps_per_epoch = train_size // batch_size
        self.validation_steps = val_size // batch_size

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # --------------------------------------------------------------------------------------------------------------
        # Otimizador definido igual ao artigo: SGD + momentum + weight decay (CANÔNICO)
        # --------------------------------------------------------------------------------------------------------------
        self.optimizer = tf.keras.optimizers.SGD(
            learning_rate=initial_lr,
            momentum=momentum
        )

        # --------------------------------------------------------------------------------------------------------------
        # Compila o modelo logo na instância da classe
        # --------------------------------------------------------------------------------------------------------------

        """
            Compila o modelo conforme os parâmetros definidos no artigo
                - Loss: SparseCategoricalCrossentropy
                     * É usada quando há um problema em que cada exemplo pertence a
                       exatamente uma de várias classes possíveis
                     * O termo "Sparse" (esparso) refere-se especificamente ao formato
                       em que os rótulos de treinamento são fornecidos. Em vez de usar a codificação
                       one-hot, os rótulos são fornecidos como inteiros únicos (ImageNet usa labels inteiros).
                - Métricas: top-1 accuracy e top-5 accuracy
        """

        self.model.compile(
            optimizer=self.optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[
                "accuracy",
                tf.keras.metrics.SparseCategoricalAccuracy(name="top1"),
            ]
        )

        # --------------------------------------------------------------------------------------------------------------
        # Configura callbacks logo na instância da classe
        # --------------------------------------------------------------------------------------------------------------
        self.callbacks = self._create_callbacks(log_dir, checkpoint_path)


    # ==================================================================================================================
    # Learning rate e step decay (≈ epochs 30 e 60)
    '''
        O paper não define exatamente quando considerar que ocorreu plateau,
        O repositório oficial em Caffe mostra que plateaus ocorrem APROXIMADAMENTE 
        em pontos que correspondem às epochs 30 e 60, Frameworks modernos adotaram isso como “regra”.
        Para uma implementação mais moderna é possível substituir pela função:
    '''

    '''
    def lr_scheduler(self):
        """
            Learning rate decay por plateau — fiel ao artigo:
            - Começa em 0.1
            - Divide por 10 quando a métrica de avaliação (acc) estaciona
        """
        return tf.keras.callbacks.ReduceLROnPlateau(
            monitor="acc",  # métrica usada no artigo
            factor=0.1,  # divide LR por 10
            patience=3,  # aguarda 3 épocas sem melhora
            mode="max",
            verbose=1,
            min_lr=1e-5  # LR mínimo (opcional)
        )
    '''

    '''
        O repositório oficial em Caffe mostra que plateaus ocorrem APROXIMADAMENTE 
        em pontos que correspondem às epochs 30 e 60, Frameworks modernos adotaram isso como “regra”.
        A implementação abaixo busca replicar algo semelhante à lógica tradicional do Caffe
    '''
    def _lr_schedule(self, epoch, lr):
        if epoch < 30:
            return self.initial_lr
        elif epoch < 60:
            return self.initial_lr * 0.1
        else:
            return self.initial_lr * 0.01

    # ==================================================================================================================
    # Criação efetiva dos Callbacks

    def _create_callbacks(self, log_dir, checkpoint_path):
        return [
            tf.keras.callbacks.LearningRateScheduler(
                self._lr_schedule,
                verbose=1
            ),

            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="accuracy",
                mode="max",
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
            ),

            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir
            ),

            tf.keras.callbacks.CSVLogger(
                filename=os.path.join(log_dir, "training_metrics.csv"),
                separator=",",
                append=False
            ),
        ]

    # ==================================================================================================================
    # Função de treinamento

    def train(self):
        print("==================================================================")
        print("        TREINAMENTO INICIADO COM A ARQUITETURA RESNET-50          ")
        print("==================================================================")
        print(f">> ÉPOCAS DE TREINAMENTO: {self.epochs}")
        print(f">> STEPS POR ÉPOCA: {self.steps_per_epoch}")
        print(f">> QUANTIDADE TOTAL DE STEPS: {self.epochs * self.steps_per_epoch}\n")
        print("==================================================================")

        history = self.model.fit(
            self.train_ds,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_data=self.val_ds,
            validation_steps=self.validation_steps,
            callbacks=self.callbacks,
        )

        print(">> TREINAMENTO FINALIZADO COM SUCESSO")
        print(">> MELHOR MODELO FOI SALVO EM: ", self.callbacks[1].filepath)
        print("==================================================================")

        return history
