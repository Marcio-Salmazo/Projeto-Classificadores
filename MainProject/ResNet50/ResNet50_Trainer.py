import tensorflow as tf
import os


class Trainer:
    """
        Classe responsável por treinar o modelo ResNet-50 criado.
        Ela não cria o modelo nem o loader; apenas recebe ambos.
    """

    def __init__(self, model, train_ds, val_ds, epochs=90, initial_lr=0.1, momentum=0.9, weight_decay=1e-4, patience=3,
                 log_dir="logs", checkpoint_path="checkpoints/resnet50_best.h5"
                 ):
        self.model = model
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.epochs = epochs
        self.initial_lr = initial_lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.patience = patience
        self.log_dir = log_dir
        self.checkpoint_path = checkpoint_path

        # Garantir que diretórios para armazenamento de pesos e logs existam
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Otimizador definido igual ao artigo (SGD - Stochastic Gradient Descent + momentum)
        self.optimizer = tf.keras.optimizers.SGD(
            learning_rate=initial_lr,
            momentum=momentum
        )

        # Compila o modelo logo na instância da classe
        self.compile_model()
        # Configura callbacks logo na instância da classe
        self.callbacks = self.create_callbacks()

    # ------------------------------------------------------------------------------------------------------------------
    def compile_model(self):
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
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5")
            ]
        )

    def lr_scheduler(self):
        """
        Learning rate decay por plateau — fiel ao artigo:
        - Começa em 0.1
        - Divide por 10 quando a métrica de avaliação (val_top5) estaciona
        """
        return tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_top5",  # métrica usada no paper (erro/top-5)
            factor=0.1,  # divide LR por 10
            patience=self.patience,  # aguarda 3 épocas sem melhora
            mode="max",
            verbose=1,
            min_lr=1e-5  # LR mínimo (opcional)
        )

    '''
    # O paper não define exatamente quando considerar que ocorreu plateau,
    # O repositório oficial em Caffe mostra que plateaus ocorrem APROXIMADAMENTE 
    # em pontos que correspondem às epochs 30 e 60, Frameworks modernos adotaram isso como “regra”.
    
    def lr_scheduler(self):
        """
        Step decay learning rate schedule igual ao paper:

        O paper usa LR inicial 0.1 e reduz por 10× nas épocas 30 e 60,
        treinando por 90 épocas.
        """

        def schedule(epoch, lr):
            if epoch < 30:
                return self.initial_lr
            elif epoch < 60:
                return self.initial_lr * 0.1
            else:
                return self.initial_lr * 0.01

        return tf.keras.callbacks.LearningRateScheduler(schedule, verbose=1)
    '''

    def create_callbacks(self):
        """
            Cria todos os callbacks usados no treinamento.
        """
        return [
            self.lr_scheduler(),
            # Salva o melhor modelo baseado em val_top5
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.checkpoint_path,
                monitor="val_top5",
                mode="max",
                # Salva apenas quando a métrica melhora
                save_best_only=True,
                # Salva o modelo completo (arquitetura + pesos).
                save_weights_only=False,
                verbose=1
            ),
            # Registra logs para visualização (perda, métricas, histogramas).
            tf.keras.callbacks.TensorBoard(log_dir=self.log_dir),

            # 🔵 CSV Logger — salva métricas a cada epoch
            tf.keras.callbacks.CSVLogger(
                filename="training_metrics.csv",
                separator=",",
                append=False
            )
        ]

    def train(self):
        """
            Executa o processo completo de treinamento.
        """
        print("\nIniciando treinamento da ResNet-50...\n")

        history = self.model.fit(
            self.train_ds,
            epochs=self.epochs,
            validation_data=self.val_ds,
            callbacks=self.callbacks
        )

        print("\nTreinamento finalizado!")
        print(f"Melhor modelo salvo em: {self.checkpoint_path}\n")

        return history
