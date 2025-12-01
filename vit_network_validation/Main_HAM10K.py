# -------------------------------------------------------------------------------------------------------------------- #
#      ANTES DE EXECUTAR, VERIFICAR SE OS DADOS ESTÃO DEVIDADEMENTE PREPARADOS DE ACORDO COM O Prepare_HAM10K.py
# -------------------------------------------------------------------------------------------------------------------- #

"""
    Main_HAM10K.py

        Arquivo principal responsável por integrar os demais módulos de teste para a ViT:
        - Carregamento do dataset, seguindo o modelo do artigo
        - Construção da ViT, em conformidade com a implementação prévia
        - Treinamento da rede e subsequente avaliação dos resultados
"""

import os
import random
import numpy as np
import tensorflow as tf

'''
    A seed busca manter a REPRODUTIBILIDADE, ela precisa ser definida antes de tudo para garantir que:
        * TF não inicialize com estados internos aleatórios. 
        * A ordem de carregamento do dataset não seja diferente a cada execução.
        * os pesos aleatórios iniciais não sejam diferentes a cada run.
'''
SEED = 42

os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

if "TF_DETERMINISTIC_OPS" in os.environ:
    del os.environ["TF_DETERMINISTIC_OPS"]
# os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Imports do projeto
from Loader_HAM10K import HAM10000Loader
from Metrics_HAM10K import evaluate_ham10000
from Learning_HAM10K import ViTTrainer, ViTBuilder

# ======================================================================================================================

# Configurações manuais para o treinamento com o dataset HAM10000 (Nomes auto-explicativos)
ROOT = r"C:\Users\marci_wawp\Desktop\Arquivos\Mestrado\Projeto-Classificadores\vit_network_validation\HAM10000"
BATCH_SIZE = 32
EPOCHS = 3
INPUT_SHAPE = (224, 224, 3)
PATCH_SIZE = 16
LOG_NAME = "vit_ham10000_test"
CHECKPOINT_DIR = "logs"

# ======================================================================================================================
# Carregamento do dataset pela função definida em Loader_HAM10K

print("\nCarregando HAM10000 datasets...\n")
loader = HAM10000Loader(
    root_dir=ROOT,
    image_size=INPUT_SHAPE[:2],
    batch_size=BATCH_SIZE,
    augment=True
)

train_ds, val_ds, test_ds, n_train, n_val, n_test = loader.get_datasets()

# ======================================================================================================================
# Construção da rede pela função definida em Learning_HAM10K

print("Construindo rede com arquitetura ViT...\n")
model_instance = ViTBuilder(
    input_shape=INPUT_SHAPE,
    patch_size=PATCH_SIZE,
    projection_dim=64,
    transformer_layers=8,
    num_heads=4,
    mlp_units=128,
    num_classes=len(loader.DX_LABELS)
)
vit_model = model_instance.build_vit()
# vit_model.summary()

# ======================================================================================================================
# Treinamento aplicado pela classe ViTTrainer, responsável pela compilação, callbacks e treinamento.

trainer = ViTTrainer(
    model=vit_model,
    train_ds=train_ds,
    val_ds=val_ds,
    n_train=n_train,
    n_val=n_val,
    log_name=LOG_NAME,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=5e-5,
    label_smoothing=0.1,
    checkpoint_dir=CHECKPOINT_DIR
)

history = trainer.train()

# ======================================================================================================================
# Avaliação dos resultados no conjunto de validação / teste

print("\nAvaliando no conjunto de VALIDAÇÃO...\n")
val_results = evaluate_ham10000(vit_model, val_ds)
print("Val accuracy:", val_results["accuracy"])
print("Val F1 macro:", val_results["f1_macro"])
print("Matriz de confusão (VAL):\n", val_results["confusion_matrix"], "\n")

print("\nAvaliando no conjunto de TESTE...\n")
test_results = evaluate_ham10000(vit_model, test_ds)
print("Test accuracy:", test_results["accuracy"])
print("Test F1 macro:", test_results["f1_macro"])
print("Matriz de confusão (TEST):\n", test_results["confusion_matrix"])

print("\nDone.\n")
