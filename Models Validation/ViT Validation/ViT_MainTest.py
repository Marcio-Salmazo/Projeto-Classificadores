"""
    Execução AUTOMÁTICA do pipeline Vision Transformer (ViT) segundo o artigo:

     - Verifica se ImageNet está extraído. Se não estiver → extrai.
     - Verifica se TFRecords existem. Se não existirem → cria.
     - Remove os arquivos .tar e a pasta extraída após gerar os TFRecords (para economizar espaço).
     - Inicia o treinamento (pré-treino).
     - Opcionalmente avalia após o treino.
"""

# OBSERVAÇÃO: AO CONDUZIR UM EXPERIMENTO COM A VIT, É NECESSÁRIO DESCOMENTAR O TRECHO INDICADO NA LINHA
# 334 DO ARQUIVO PROCESS_IMAGENET.py

import os
# Definição da variável de ambiente XLA_PYTHON_CLIENT_ALLOCATOR com o valor "platform" durante a execução do programa.
# Essa variável é usada por bibliotecas que usam XLA para controlar como a memória é alocada, especialmente em GPU.
# Evita erros de OOM. A memória tende a ser alocada sob demanda.
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import subprocess
from VisionTransformer_trainer import train_vit, evaluate_vit, evaluate_vit_from_iterator

# ======================================================================================================================
# CAMINHOS DE AMBIENTES, TFRECORDS E SCRIPTS ASSOCIADOS

TF_ENV_PYTHON = (r"C:/Users/marci_plgx30x/Desktop/Arquivos/Projetos do Mestrado (Git)/Projeto-Classificadores"
                 r"/.tf_venv/Scripts/python.exe")

TFRECORD_SCRIPT = (r"C:/Users/marci_plgx30x/Desktop/Arquivos/Projetos do Mestrado (Git)/Projeto-Classificadores"
                   r"/Validation/Create_TFRecords.py")

# Diretório onde serão criados: /train/*.tfrecord e /validation/*.tfrecord
TFRECORD_DIR = (r"C:/Users/marci_plgx30x/Desktop/Arquivos/Projetos do Mestrado (Git)/Projeto-Classificadores"
                r"/Datasets/ImageNet_\TFRecords")

# Diretório de checkpoints do ViT
OUTPUT_DIR = (r"C:/Users/marci_plgx30x/Desktop/Arquivos/Projetos do Mestrado (Git)"
              r"/Projeto-Classificadores/Main_Validation/ResNet-50/Checkpoints"
)

# ======================================================================================================================
# FLAGS DE EXECUÇÃO

RUN_PRETRAIN = False
RUN_FINETUNE = True
RUN_EVALUATE = True


# ======================================================================================================================
# FUNÇÕES AUXILIARES

def tfrecords_exist_safe(tfrecord_dir, num_train_shards=1024, num_val_shards=128):
    train_dir = os.path.join(tfrecord_dir, "train")
    val_dir = os.path.join(tfrecord_dir, "validation")

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        return False

    train_files = os.listdir(train_dir)
    val_files = os.listdir(val_dir)

    # Verifica contagem exata de shards
    train_ok = len(train_files) == num_train_shards
    val_ok = len(val_files) == num_val_shards

    if not (train_ok and val_ok):
        print("TFRecord directory exists, but shard count is incorrect.")
        print(f" Train shards: {len(train_files)} (expected: {num_train_shards})")
        print(f" Val shards:   {len(val_files)} (expected: {num_val_shards})")
        return False

    # Verifica se os nomes seguem o padrão correto
    if not all("train-" in f for f in train_files):
        return False

    if not all("validation-" in f for f in val_files):
        return False

    return True


# ======================================================================================================================
# EXECUÇÃO PRINCIPAL

def main():
    print("\n PIPELINE DE EXECUÇÃO INICIADO...\n")

    # ------------------------------------------------------------
    #       1. TFRECORDS (somente se ainda não existirem)
    # ------------------------------------------------------------
    if not tfrecords_exist_safe(TFRECORD_DIR):
        print(">> TFRecords não encontrados. Criando agora (streaming)...\n")

        result = subprocess.run(
            [TF_ENV_PYTHON, TFRECORD_SCRIPT],
            capture_output=True,
            text=True
        )

        print(result.stdout)
        print(result.stderr)

        print("\n>> TFRecords criados com sucesso!\n")
    else:
        print(">> TFRecords já existem. Pulando criação.\n")

    # ------------------------------------------------------------
    #                        2. PRÉ-TREINO
    # ------------------------------------------------------------
    if RUN_PRETRAIN:
        print("\nINICIANDO PRÉ-TREINO ViT...\n")

        train_vit(
            tfrecord_train_dir=TFRECORD_DIR,
            tfrecord_val_dir=TFRECORD_DIR,
            output_dir=OUTPUT_DIR,
            mode="pretrain",
            total_steps=100000,
            warmup_steps=10000,
            batch_size=16,
            base_lr=2e-4,
        )

    # ------------------------------------------------------------
    #                        3. FINE-TUNING
    # ------------------------------------------------------------
    if RUN_FINETUNE:
        print("\nINICIANDO FINE-TUNING...\n")

        train_vit(
            tfrecord_train_dir=TFRECORD_DIR,
            tfrecord_val_dir=TFRECORD_DIR,
            output_dir=OUTPUT_DIR,
            mode="finetune",
            total_steps=20000,
            warmup_steps=0,
            batch_size=32,
            base_lr=2e-4
        )

    # ------------------------------------------------------------
    #                       4. AVALIAÇÃO
    # ------------------------------------------------------------
    if RUN_EVALUATE:
        print("\nAVALIANDO O MODELO...\n")

        results = evaluate_vit(
            checkpoint_dir=OUTPUT_DIR,
            tfrecord_val_dir=TFRECORD_DIR,
            batch_size=64,
            num_batches=200
        )

        print("\nRESULTADOS FINAIS:")
        print(results)


if __name__ == "__main__":
    main()
