"""
    Este script roda no AMBIENTE TF (.tf_venv), evitando incompatibilidade com o
    ambiente virtual da ViT que utiliza uma versão mais recente do NumPy

        * Ele apenas chama create_imagenet_tfrecords_streaming.
"""

import os
from Process_ImageNet import create_imagenet_tfrecords_streaming

TRAIN_TAR = (r"C:/Users/marci_plgx30x/Desktop/Arquivos/Projetos do Mestrado (Git)"
             r"/Projeto-Classificadores/Datasets/ImageNet_1K/ILSVRC2012_img_train.tar")

VAL_TAR = (r"C:/Users/marci_plgx30x/Desktop/Arquivos/Projetos do Mestrado (Git)"
             r"/Projeto-Classificadores/Datasets/ImageNet_1K/ILSVRC2012_img_val.tar")

# Diretório onde serão criados: /train/*.tfrecord e /validation/*.tfrecord
TFRECORD_DIR = (r"C:/Users/marci_plgx30x/Desktop/Arquivos/Projetos do Mestrado (Git)/Projeto-Classificadores"
                r"/Datasets/ImageNet_1K")

# Diretório de checkpoints do ViT
OUTPUT_DIR = (r"C:/Users/marci_plgx30x/Desktop/Arquivos/Projetos do Mestrado (Git)"
              r"/Projeto-Classificadores/Network_Validation/CHECKPOINTS")

# Diretório de checkpoints do ViT
VAL_ANNOTATIONS = (r"C:/Users/marci_plgx30x/Desktop/Arquivos/Projetos do Mestrado (Git)"
              r"/Projeto-Classificadores/Network_Validation/CHECKPOINTS/Validation_Notes.txt")

DELETE_TARS_AFTER_TFRECORDS = True


def delete_tar_files():
    """
        Remove os arquivos .tar originais para economizar espaço.
    """
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


print("== TFRecords concluídos com sucesso! ==")

if __name__ == "__main__":
    print("== Criando TFRecords no ambiente TensorFlow ==")

    create_imagenet_tfrecords_streaming(
        train_tar=TRAIN_TAR,
        val_tar=VAL_TAR,
        out_dir=TFRECORD_DIR,
        num_train_shards=1024,
        num_val_shards=128,
        val_annotations_file=VAL_ANNOTATIONS
    )

    if DELETE_TARS_AFTER_TFRECORDS:
        print(">> Removendo .tar para economizar espaço...")
        delete_tar_files()
        print(">> Remoção concluída.\n")

    delete_tar_files()
