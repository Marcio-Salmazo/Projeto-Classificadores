"""
    Execução AUTOMÁTICA do pipeline Vision Transformer (ViT) segundo o artigo:

     - Verifica se ImageNet está extraído. Se não estiver → extrai.
     - Verifica se TFRecords existem. Se não existirem → cria.
     - Remove os arquivos .tar e a pasta extraída após gerar os TFRecords (para economizar espaço).
     - Inicia o treinamento (pré-treino).
     - Opcionalmente avalia após o treino.
"""

import os

from VisionTransformer_ImageNet import (
    create_imagenet_tfrecords_streaming
)
from VisionTransformer_trainer import train_vit
from VisionTransformer_evaluate import evaluate_vit

# ======================================================================================================================
# CONFIGURAÇÕES CAMINHOS

# Caminho do dataset original (compactado como .tar)
TRAIN_TAR = (r"C:/Users/marci_wawp/Desktop/Arquivos/Mestrado/Projeto-Classificadores/Datasets"
             r"/DATASET IMAGENET/ILSVRC2012_img_train.tar")
VAL_TAR = (r"C:/Users/marci_wawp/Desktop/Arquivos/Mestrado/Projeto-Classificadores/Datasets"
           r"/DATASET IMAGENET/ILSVRC2012_img_val.tar")
# Onde os TFRecords serão gravados
TFRECORD_DIR = r"C:/Users/marci_wawp/Desktop/Arquivos/Mestrado/Projeto-Classificadores/Datasets/DATASET IMAGENET"
# Onde modelos e checkpoints serão salvos
OUTPUT_DIR = (r"C:/Users/marci_wawp/Desktop/Arquivos/Mestrado/Projeto-Classificadores/"
              r"Network Validation/VISION TRANSFORMER/Models and checkpoints")
CHECKPOINT_DIR = OUTPUT_DIR
# Arquivo de anotações da validação (opcional porém recomendado para labels corretas)
VAL_ANNOTATIONS = (r"C:/Users/marci_wawp/Desktop/Arquivos/Mestrado/Projeto-Classificadores/"
                   r"Network Validation/VISION TRANSFORMER/Validation_Notes.txt")

# ======================================================================================================================
# CONFIGURAÇÃO DE FLAGS

# Flags para controlar comportamento automático.
# Devem ser alteradas antes da execução, definindo o que deve (ou não ser feito)
RUN_PRETRAIN = True
RUN_FINETUNE = False
RUN_EVALUATE = False

# se True, remove os .tar após criação bem-sucedida dos TFRecords (economia de espaço).
DELETE_TARS_AFTER_TFRECORDS = True


# ======================================================================================================================
# FUNÇÕES AUXILIARES

def tfrecords_exist():
    """
        Função responsável por verificar se já existem TFRecords no diretório

        * os.path.exists(...) e os.path.isdir(...) garantem que a pasta existe e é diretório, respectivamente.
        * len(os.listdir(...)) > 0 garante que há arquivos (shards) dentro.
        * Retorna True apenas se ambas as pastas existirem e contiverem arquivos
    """
    train_dir = os.path.join(TFRECORD_DIR, "train")
    val_dir = os.path.join(TFRECORD_DIR, "validation")

    return (
            os.path.exists(train_dir)
            and os.path.isdir(train_dir)
            and len(os.listdir(train_dir)) > 0
            and os.path.exists(val_dir)
            and os.path.isdir(val_dir)
            and len(os.listdir(val_dir)) > 0
    )


def checkpoint_exists():
    """
        Verifica se a pasta de saída existe:
            * Em caso positivo, verifica se algum arquivo/dir dentro tem a palavra "checkpoint" no nome
            (inspeciona rapidamente se há checkpoints salvos).
            * Em caso negativo, retorna False.

    """
    if not os.path.exists(OUTPUT_DIR):
        return False
    return any("checkpoint" in name.lower() for name in os.listdir(OUTPUT_DIR))


# ======================================================================================================================
# EXECUÇÃO PRINCIPAL (INICIADA PELO 'RUN' DA IDE)

def main():
    print("\n INICIANDO PIPELINE DO VISION TRANSFORMER... \n")

    # ------------------------------------------------------------------------------------------------------------------
    # VERIFICAÇÃO INICIAL E CRIAÇÃO DOS TFRECORDS

    if not tfrecords_exist():

        print(">> TFRecords NÃO encontrados. Criando agora via streaming...")

        # Cria o caminho onde os TFRecords serão armazenados
        os.makedirs(TFRECORD_DIR, exist_ok=True)

        # Chama a função responsável por ler TRAIN_TAR e VAL_TAR e criar os shards
        '''
           Parâmetros notáveis:
            
            * num_train_shards = 1024 e num_val_shards = 128 representam o padrão usado pelo repositório oficial. 
              OBS: Shards menores ajudam carregamento paralelo.
                
            * val_annotations_file = VAL_ANNOTATIONS — fornece o arquivo de mapeamento da validação.
        '''

        create_imagenet_tfrecords_streaming(
            train_tar=TRAIN_TAR,
            val_tar=VAL_TAR,
            out_dir=TFRECORD_DIR,
            num_train_shards=1024,
            num_val_shards=128,
            val_annotations_file=VAL_ANNOTATIONS
        )

        print("\n>> TFRecords criados com sucesso!\n")

        # Apagar os .tar originais após criar os TFRecords (caso a flag esteja habilitada)
        if DELETE_TARS_AFTER_TFRECORDS:
            print(">> Removendo arquivos .tar para economizar espaço...")
            try:
                if os.path.exists(TRAIN_TAR):
                    os.remove(TRAIN_TAR)
                if os.path.exists(VAL_TAR):
                    os.remove(VAL_TAR)
                print(">> Arquivos .tar removidos!\n")
            except PermissionError:
                print("Aviso: não foi possível apagar os .tar (permissão negada). Faça manualmente.\n")

    else:
        print(">> TFRecords já existem. Pulando criação.\n")

    # ------------------------------------------------------------------------------------------------------------------
    # ETAPA DE PRÉ-TREINO

    '''
        o Vision Transformer é treinado em duas fases distintas, segundo o artigo 
        "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020).
        
        * Na primeira fase, o modelo é treinado em um dataset gigantesco, como JFT-300M ou ImageNet-21k (14M de imagens)
          O objetivo central é aprender representações gerais e robustas, não específicas de um conjunto pequeno.
        * Na segunda fase, o modelo é ajustado (refinado) em um dataset menor, como a ImageNet-1k (ILSVRC2012),
          com o objetivo de especializar o modelo para a tarefa de classificação.
        
        Por limitações de espaço em hardware, ambos o pré-treino e o refinamento são aplicados na ImageNet-1k  
        o que é esperado um resultado inferior ao artigo. O Fine-tuning também em ImageNet-1k melhora um pouco.
    '''
    if RUN_PRETRAIN:
        print("\n INICIANDO PRÉ-TREINO ViT... \n")
        train_vit(
            tfrecord_train_dir=os.path.join(TFRECORD_DIR, "train"),
            tfrecord_val_dir=os.path.join(TFRECORD_DIR, "validation"),
            output_dir=OUTPUT_DIR,
            mode="pretrain",
            total_steps=100000,
            warmup_steps=10000,
            batch_size=256,
            base_lr=2e-4
        )

    # ------------------------------------------------------------------------------------------------------------------
    # ETAPA DE FINE-TUNING (opcional)

    '''
        Observação: Nos transformers, os modelos são treinados em steps, não em epochs. Neste caso, temos:
                    1 step = 1 atualização de gradiente = 1 batch passado pelo modelo.
    '''

    if RUN_FINETUNE:
        print("\n INICIANDO FINE-TUNING ViT... \n")
        train_vit(
            tfrecord_train_dir=os.path.join(TFRECORD_DIR, "train"),
            tfrecord_val_dir=os.path.join(TFRECORD_DIR, "validation"),
            output_dir=OUTPUT_DIR,
            mode="finetune",
            total_steps=20000,
            warmup_steps=0,
            batch_size=512,
            base_lr=0.01
        )

    # ------------------------------------------------------------------------------------------------------------------
    # ETAPA DE AVALIAÇÃO

    if RUN_EVALUATE:
        print("\n AVALIANDO O MODELO... \n")

        evaluate_vit(
            checkpoint_dir=CHECKPOINT_DIR,
            tfrecord_val_dir=os.path.join(TFRECORD_DIR, "validation"),
            batch_size=512,
            num_batches=200
        )


if __name__ == "__main__":
    main()
