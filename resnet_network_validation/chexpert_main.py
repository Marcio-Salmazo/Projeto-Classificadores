import os
import random
import numpy as np

'''
    A seed busca manter a REPRODUTIBILIDADE, ela precisa ser definida antes de tudo para garantir que:
        * TF não inicialize com estados internos aleatórios. 
        * A ordem de carregamento do dataset não seja diferente a cada execução.
        * os pesos aleatórios iniciais não sejam diferentes a cada run.
'''
SEED = 42

os.environ['PYTHONHASHSEED'] = str(SEED)  # controla hash randomizado do Python
random.seed(SEED)  # seed do módulo random
np.random.seed(SEED)  # seed do numpy

# Opcional, mas recomendado:
# Força operações determinísticas (pode deixar mais lento):
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import tensorflow as tf  # Importação do TF vem após a definição da seed
tf.random.set_seed(SEED)  # seed do tensorflow

# Imports do projeto
from chexpert_loader import CheXpertDataLoader
from chexpert_metrics import evaluate_chexpert
from chexpert_learning import Resnet50, ResNet_Trainer

# ======================================================================================================================

# Configurações manuais para o treinamento com o dataset Chexpert (Nomes auto-explicativos)
DATA_DIR = r"C:\Users\marci_wawp\Desktop\Arquivos\Mestrado\Projeto-Classificadores\network_validation\cheXpert-small"
BATCH_SIZE = 32
EPOCHS = 3
INPUT_SHAPE = (320, 320, 3)
LOG_NAME = "run_resnet50_chexpert"

# ======================================================================================================================
# Carregamento do dataset pela função definida em chexpert_loader

print("\nCarregando dataset CheXpert-small...\n")

loader = CheXpertDataLoader(
    data_dir=DATA_DIR,
    image_size=(320, 320),
    batch_size=BATCH_SIZE,
    uncertainty_policy="zeros",
    augment=True
)

train_ds, val_ds, n_train, n_val, val_paths = loader.get_datasets()
print(f"Treino: {n_train} imagens")
print(f"Validação: {n_val} imagens")

# ======================================================================================================================
# Construção da rede pela função definida em chexpert_learning

print("\nConstruindo modelo ResNet50...\n")
network_builder = Resnet50(
    input_shape=INPUT_SHAPE,
    num_classes=14,  # CheXpert tem 14 labels (multi-label)
    last_layer_activation="sigmoid" # multi-label -> sigmoid
)
model = network_builder.resnet_classifier()  # Constrói e retorna um tf.keras.Model pronto
model.summary()  # Mostra arquitetura / parâmetros

# ======================================================================================================================
# Treinamento aplicado pela classe ResNet_Trainer, responsável pela compilação, callbacks e treinamento.

print("\nInicializando treinamento...\n")
trainer = ResNet_Trainer(
    neural_network=model,
    train_data=train_ds,
    val_data=val_ds,
    val_paths=val_paths,  # necessário para callback que calcula AUC por estudo
    epochs=EPOCHS,
    logName="chexpert_test"
)

# Executa o treinamento (bloqueante). O trainer cuida dos callbacks e do salvamento.
trainer.run()
print("\nTreinamento finalizado!\n")

# ======================================================================================================================

# Avaliação Final (Pós-Treino)
# A função evaluate_chexpert roda predict, agrupa por study e retorna AUCs
results = evaluate_chexpert(model, val_ds, val_paths)

print("\nAUC média das 5 patologias:", results["auc_mean_5"])
print("AUC por classe (14):", results["auc_per_class"])

# ======================================================================================================================
