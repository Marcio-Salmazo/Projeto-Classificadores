import tensorflow as tf

from ImageNet_Loader import loader
from ResNet50_pure import build_resnet50
from ResNet50_trainer import Trainer


class Main:
    """
        Classe principal que integra:
        - carregamento do dataset
        - construção da ResNet-50
        - treinamento do modelo
    """

    def __init__(self, image_size=224, batch_size=256, num_classes=1000, epochs=120, initial_lr=0.1,
                  momentum=0.9, weight_decay=1e-4):
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.epochs = epochs
        self.initial_lr = initial_lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.train_ds = None
        self.val_ds = None
        self.model = None

    def load_data(self):
        print("Carregando ImageNet-2012 via TFDS...")
        loaded_data = loader(image_size=self.image_size)
        train_ds, val_ds = loaded_data.get_imagenet(batch_size=self.batch_size)

        # Diagnóstico rápido:
        for x, y in train_ds.take(1):
            print(f"\nVerificação do batch size: {x.shape} (esperado: {self.batch_size})\n")

        self.train_ds = train_ds
        self.val_ds = val_ds
        print("LOG --- Dataset carregado.\n")

    def build_model(self):
        print("Construindo modelo ResNet-50 fiel ao paper...\n")

        model = build_resnet50(
            input_shape=(self.image_size, self.image_size, 3),
            num_classes=self.num_classes,
            include_top=True,
            weight_decay=self.weight_decay
        )

        model.summary()
        self.model = model
        print("\nLOG --- Modelo criado.\n")

    def train(self):

        trainer = Trainer(
            model=self.model,
            train_ds=self.train_ds,
            val_ds=self.val_ds,
            epochs=self.epochs,
            initial_lr=self.initial_lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            patience=3,
            log_dir="logs",
            checkpoint_path="checkpoints/resnet50_best.h5"
        )

        print("Iniciando treinamento...\n")
        trainer.train()
        print("\nLOG --- Pipeline treinamento finalizado.\n")

    def run(self):
        self.load_data()
        self.build_model()
        self.train()


# ======================================================================================================================
#                                             Execução direta do main.py
# ======================================================================================================================
if __name__ == "__main__":
    Main().run()
