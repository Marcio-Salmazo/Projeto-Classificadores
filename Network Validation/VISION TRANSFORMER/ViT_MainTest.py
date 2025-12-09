"""
    Execução AUTOMÁTICA do pipeline Vision Transformer (ViT) segundo o artigo:

     - Verifica se ImageNet está extraído. Se não estiver → extrai.
     - Verifica se TFRecords existem. Se não existirem → cria.
     - Remove os arquivos .tar e a pasta extraída após gerar os TFRecords (para economizar espaço).
     - Inicia o treinamento (pré-treino).
     - Opcionalmente avalia após o treino.
"""

import os
import shutil
from Process_and_Load_ImageNet import create_imagenet_tfrecords_streaming
from VisionTransformer_trainer import train_vit, evaluate_vit

# ======================================================================================================================
# CONFIGURAÇÃO DE CAMINHOS

TRAIN_TAR = (r"C:/Users/marci_wawp/Desktop/Arquivos/Mestrado/Projeto-Classificadores/"
             r"Datasets/DATASET IMAGENET/ILSVRC2012_img_train.tar")
VAL_TAR = (r"C:/Users/marci_wawp/Desktop/Arquivos/Mestrado/Projeto-Classificadores/"
           r"Datasets/DATASET IMAGENET/ILSVRC2012_img_val.tar")

# Diretório onde serão criados: /train/*.tfrecord e /validation/*.tfrecord
TFRECORD_DIR = r"C:/Users/marci_wawp/Desktop/Arquivos/Mestrado/Projeto-Classificadores/Datasets/DATASET IMAGENET"

# Diretório de checkpoints do ViT
OUTPUT_DIR = (
    r"C:/Users/marci_wawp/Desktop/Arquivos/Mestrado/Projeto-Classificadores/"
    r"Network Validation/VISION TRANSFORMER/Models and checkpoints"
)

VAL_ANNOTATIONS = (
    r"C:/Users/marci_wawp/Desktop/Arquivos/Mestrado/Projeto-Classificadores/"
    r"Network Validation/VISION TRANSFORMER/Validation_Notes.txt"
)

# ======================================================================================================================
# FLAGS DE EXECUÇÃO

RUN_PRETRAIN = True
RUN_FINETUNE = False
RUN_EVALUATE = False
DELETE_TARS_AFTER_TFRECORDS = True


# ======================================================================================================================
# FUNÇÕES AUXILIARES

def tfrecords_exist():
    train_dir = os.path.join(TFRECORD_DIR, "train")
    val_dir = os.path.join(TFRECORD_DIR, "validation")

    return (
            os.path.isdir(train_dir) and len(os.listdir(train_dir)) > 0 and
            os.path.isdir(val_dir) and len(os.listdir(val_dir)) > 0
    )


def delete_tar_files():
    """Remove os arquivos .tar originais para economizar espaço."""
    if os.path.exists(TRAIN_TAR):
        try:
            os.remove(TRAIN_TAR)
            print(f"Removido: {TRAIN_TAR}")
        except PermissionError:
            print(f"Não foi possível apagar {TRAIN_TAR}")

    if os.path.exists(VAL_TAR):
        try:
            os.remove(VAL_TAR)
            print(f"Removido: {VAL_TAR}")
        except PermissionError:
            print(f"Não foi possível apagar {VAL_TAR}")


# ======================================================================================================================
# EXECUÇÃO PRINCIPAL

def main():
    print("\n PIPELINE DE EXECUÇÃO INICIADO...\n")

    # ------------------------------------------------------------
    #       1. TFRECORDS (somente se ainda não existirem)
    # ------------------------------------------------------------
    if not tfrecords_exist():
        print(">> TFRecords não encontrados. Criando agora (streaming)...\n")

        os.makedirs(TFRECORD_DIR, exist_ok=True)

        create_imagenet_tfrecords_streaming(
            train_tar=TRAIN_TAR,
            val_tar=VAL_TAR,
            out_dir=TFRECORD_DIR,
            num_train_shards=1024,
            num_val_shards=128,
            val_annotations_file=VAL_ANNOTATIONS
        )

        print("\n>> TFRecords criados com sucesso!\n")

        if DELETE_TARS_AFTER_TFRECORDS:
            print(">> Removendo .tar para economizar espaço...")
            delete_tar_files()
            print(">> Remoção concluída.\n")

    else:
        print(">> TFRecords já existem. Pulando criação.\n")

    # ------------------------------------------------------------
    #                        2. PRÉ-TREINO
    # ------------------------------------------------------------
    if RUN_PRETRAIN:
        print("\nINICIANDO PRÉ-TREINO ViT...\n")

        train_vit(
            tfrecord_train_dir=os.path.join(TFRECORD_DIR, "train"),
            tfrecord_val_dir=os.path.join(TFRECORD_DIR, "validation"),
            output_dir=OUTPUT_DIR,
            mode="pretrain",
            total_steps=100000,
            warmup_steps=10000,
            batch_size=256,
            base_lr=2e-4,
        )

    # ------------------------------------------------------------
    #                        3. FINE-TUNING
    # ------------------------------------------------------------
    if RUN_FINETUNE:
        print("\nINICIANDO FINE-TUNING...\n")

        train_vit(
            tfrecord_train_dir=os.path.join(TFRECORD_DIR, "train"),
            tfrecord_val_dir=os.path.join(TFRECORD_DIR, "validation"),
            output_dir=OUTPUT_DIR,
            mode="finetune",
            total_steps=20000,
            warmup_steps=0,
            batch_size=512,
            base_lr=0.01,
        )

    # ------------------------------------------------------------
    #                       4. AVALIAÇÃO
    # ------------------------------------------------------------
    if RUN_EVALUATE:
        print("\nAVALIANDO O MODELO...\n")

        results = evaluate_vit(
            state=None,  # trainer JIT já lida com carregamento interno
            val_iter=None,
            num_batches=200
        )

        print("\nRESULTADOS FINAIS:")
        print(results)


if __name__ == "__main__":
    main()
