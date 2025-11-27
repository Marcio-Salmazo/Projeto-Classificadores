import os
import tensorflow as tf

from chexpert_loader import CheXpertDataLoader
from ModelCreator import Resnet50
from TrainerThread import ResNet_TrainerThread

# OBSERVAÇÃO: PARA EFETUAR O TESTE DE INTEGRIDADE DA REDE É NECESSÁRIO MODIFICAR MANUALMENTE
# A FUNÇÃO DE PERDA E MÉTRICA NO ARQUIVO TrainerThread.py (NA FUNÇÃO REFERENTE À RESNET).
# ALÉM DISSO, É NECESSÁRIO ATENTAR-SE PARA O CAMINHO DIFINIDO PARA O DATASET CHEXPERT E DEFINIR O NOME DE LOG

# Configurações manuais de treinamento
DATA_DIR = r"C:\Users\marci_wawp\Desktop\Arquivos\Mestrado\Projeto-Classificadores\impl. tests\cheXpert-small"
BATCH_SIZE = 32
EPOCHS = 10
INPUT_SHAPE = (320, 320, 3)
LOG_NAME = "chexpert_resnet50_experimento_01"


# 1) Carregamento do dataset
print("\n[1] Carregando dataset CheXpert-small...\n")

loader = CheXpertDataLoader(
    data_dir=DATA_DIR,
    image_size=(320, 320),
    batch_size=BATCH_SIZE,
    uncertainty_policy="zeros",
    augment=True
)

train_ds, val_ds, n_train, n_val = loader.get_datasets()
steps_train = n_train // BATCH_SIZE
steps_val = n_val // BATCH_SIZE

print(f"Total treino: {n_train} | steps: {steps_train}")
print(f"Total validação: {n_val} | steps: {steps_val}")

# 2) Construção da rede
print("\n[2] Construindo modelo ResNet50...\n")

network_builder = Resnet50(
    input_shape=INPUT_SHAPE,
    num_classes=14,
    last_layer_activation="sigmoid"
)
model = network_builder.resnet_classifier()
model.summary()

# 3) Treinamento
print("\n[3] Inicializando treinamento...\n")

trainer = ResNet_TrainerThread(
    neural_network=model,
    train_data=train_ds,
    val_data=val_ds,
    epochs=EPOCHS,
    logName=LOG_NAME,
    steps_train=steps_train,
    steps_val=steps_val,
)

# Para testes fora do PyQt:
trainer.run()

print("\nTreinamento finalizado!\n")