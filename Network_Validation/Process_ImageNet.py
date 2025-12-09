"""
    Arquivo responsável por:
        * Ler o ImageNet diretamente dos arquivos .tar
        * Criar TFRecords, o formato usado pelos autores do ViT
        * Fazer o processo em streaming, evitando a extração completa no SSD

    O processo aqui descrito está alinhado com o artigo (localizado no Appendix B – Training Details),
    o qual cita que a arquitetura ViT foi treinada usando pipelines otimizados com TFRecords.
"""

import os
import tarfile
import tensorflow as tf

import numpy as np
import jax.numpy as jnp
from typing import Optional, Dict, Tuple  # tipos para documentação de código

AUTOTUNE = tf.data.AUTOTUNE


# ======================================================================================================================
# ESCREVER IMAGEM NO TFRECORD

def write_example(writer: tf.io.TFRecordWriter, image_bytes: bytes, label: int):
    """
        Função responsável por escrever uma imagem no TFRecord
        Cada imagem será representada como:
            * bytes puros (image_bytes)
            * label numérico (int64)
        O formato TFRecord é o mesmo usado no repositório oficial do ViT.
    """
    feature = {
        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    }
    # Converte em TFRecord serializado e grava.
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())


# ======================================================================================================================
# ITERAR SOBRE O ARQUIVO TRAIN.TAR

def open_train_tar_members(train_tar_path: str):
    """
        Função responsável por iterar sobre o arquivo train.tar
            * Cada arquivo .tar interno é o conjunto de imagens de uma classe.
            * O ImageNet foi estruturado assim desde sua criação, então isso respeita exatamente o dataset utilizado
        Essa etapa é critica por não extrair nada no disco, lendo cada classe por streaming
    """
    with tarfile.open(train_tar_path, "r:") as tar:  # Abre o arquivo externo.
        for member in tar:  # percorre cada arquivo interno.

            # Garante que cada item é um arquivo e tem a extensão .tar, exemplo: 'n01440764.tar'
            if member.isfile() and member.name.endswith(".tar"):
                f = tar.extractfile(member)  # Retorna um file-like object sem extrair no disco.
                yield member.name, f  # Retorna para o chamador (nome, fileobj).


# ======================================================================================================================
# CRIA OS TFRECORDS PARA IMAGENET

def create_imagenet_tfrecords_streaming(
        train_tar: str,
        val_tar: str,
        out_dir: str,
        num_train_shards: int = 1024,
        num_val_shards: int = 128,
        val_annotations_file: Optional[str] = None):
    """
        Cria TFRecords para ImageNet transmitindo dados de train_tar e val_tar.
            - train_tar: caminho para ILSVRC2012_img_train.tar (contém os arquivos .tar de cada classe)
            - val_tar: caminho para ILSVRC2012_img_val.tar
            - out_dir: diretório base de saída; criará os diretórios out_dir/train e out_dir/validation
            - num_train_shards, num_val_shards: número de shards de saída
            - val_annotations_file: caminho opcional para um arquivo de anotações de validação
              (mapeia a imagem de validação para o synset)

        Formatos comuns aceitos:
            * linhas de "val_00000001.JPEG n01440764"
            * linhas contendo apenas synsets (um por linha) que correspondam à ordem dos nomes dos arquivos de validação
    """

    # Garantem que os diretórios existem.
    # Nada é sobrescrito se já existir (importante para reexecuções).
    os.makedirs(out_dir, exist_ok=True)
    train_out = os.path.join(out_dir, "train")
    val_out = os.path.join(out_dir, "validation")
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(val_out, exist_ok=True)

    # Criação dos shards (padrão usado no artigo)
    # “TFRecords are split into many small shards to maximize parallel reading throughput.”
    train_writers = []
    for i in range(num_train_shards):
        shard_path = os.path.join(train_out, f"train-{i:05d}-of-{num_train_shards:05d}.tfrecord")
        train_writers.append(tf.io.TFRecordWriter(shard_path))

    '''
        Construção do dicionário synset → label
        Cada classe (synset) recebe um ID numérico.
        No ImageNet:
            * synset = identificador tipo "n01440764"
            * label = int64 correspondente
    '''
    class_to_label: Dict[str, int] = {}
    next_label = 0
    global_img_idx = 0  # used to round-robin assign shards

    print("Streaming train.tar and writing TFRecords (round-robin across shards)...")

    # Loop principal sobre o trainer.tar (Aqui ocorre de fato a parte do 'streaming')
    # Recebe o fileobj da classe e lê seus arquivos internamente sem extrair no disco
    for member_name, inner_fileobj in open_train_tar_members(train_tar):

        # Obtém rótulo synset a partir de member_name e.g. "n01440764.tar" -> n01440764
        synset = os.path.splitext(os.path.basename(member_name))[0]
        '''
            * Labels são ordenados pela ordem encontrada no tar
            * compatível com o pipeline ViT (que não exige ordem específica)
        '''
        if synset not in class_to_label:
            class_to_label[synset] = next_label
            next_label += 1
        label = class_to_label[synset]

        try:
            #  Lê a classe inteira direto da memória, não do disco (.tar interno).
            with tarfile.open(fileobj=inner_fileobj, mode="r:") as inner_tar:

                # Lê cada um dos arquivos internos (imagens) localizados no .tar interno
                for img_member in inner_tar:
                    # Verfifca se é um arquivo válido
                    if not img_member.isfile():
                        continue
                    # Processa a imagem interna, recebendo seus bytes crus
                    try:
                        img_f = inner_tar.extractfile(img_member)
                        if img_f is None:
                            continue
                        img_bytes = img_f.read()

                        '''
                            Distribuição round-robin nos shards:
                                O round-robin espalha uniformemente as imagens nos shards, 
                                evitando que cada shard fique com classes agrupadas.
                            
                            O artigo deixa explícto:
                                “Shuffling and mixing samples across shards improves training 
                                stability and TPU parallelism.”
                        '''
                        shard_idx = global_img_idx % num_train_shards
                        write_example(train_writers[shard_idx], img_bytes, label)
                        global_img_idx += 1
                    except Exception as e_img:
                        print(f"Warning: skipping image {img_member.name} in {synset}: {e_img}")

        except Exception as e_inner:
            print(f"Warning: could not open inner tar {member_name}: {e_inner}")
        finally:
            # Garante que fileobj seja fechado ao final do processo
            try:
                inner_fileobj.close()
            except Exception:
                pass

    # Fecha train writers
    for w in train_writers:
        w.close()

    print(f"Train TFRecords written to: {train_out}")
    print(f"Number of classes (train): {len(class_to_label)}")

    '''
    Processamento do validation set:
        A segunda metade do arquivo faz a mesma coisa para o arquivo ILSVRC2012_img_val.tar
        Contudo, O ImageNet validation não vem organizado por pastas. Por isso, são necessárias 
        heurísticas e/ou arquivo de anotações (val_annotations.txt) para mapear val_00000001.JPEG → synset → label
    '''

    # Criando os writers do validation:
    # Shards menores → leitura paralela mais eficiente → alinhado ao artigo.
    val_writers = []
    for i in range(num_val_shards):
        shard_path = os.path.join(val_out, f"validation-{i:05d}-of-{num_val_shards:05d}.tfrecord")
        val_writers.append(tf.io.TFRecordWriter(shard_path))

    # Constrói o mapeamento de anotações de validação (Inicia vazio):
    # nome do arquivo -> rótulo (índice inteiro por ordem de classe do treino)
    val_label_map: Dict[str, int] = {}

    # Análise do arquivo de anotações (se existir)
    if val_annotations_file and os.path.exists(val_annotations_file):

        print(f"Parsing val annotations from {val_annotations_file}")
        # Abre o arquivo para iniciar a análise:
        with open(val_annotations_file, "r") as f:
            # Percorre cada uma das linhas do arquivo
            # Obtendo os tokens puros  (removendo todos os caracteres de espaço em branco iniciais e finais)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Obtém os tokens puros (removendo todos os caracteres de espaço em branco iniciais e finais)
                # tokens é uma lista das labels
                tokens = line.split()

                # Sendo: "val_00000001.JPEG n01440764"
                # Temos que  fname -> val_00000001.JPEG e syn -> n01440764
                # Salva no dicionário
                if len(tokens) >= 2:
                    fname = tokens[0]
                    syn = tokens[1]
                    if syn in class_to_label:
                        val_label_map[fname] = class_to_label[syn]
                    else:
                        # if syn not known (rare), ignore or warn
                        print(f"Warning: val synset {syn} not found among train classes.")
                else:
                    # If only synset per line, we cannot map without ordering knowledge
                    # fallthrough handled below
                    pass

    # Se não houver mapeamento, tenta reconstruir utilizando ordem lexicográfica dos arquivos
    if not val_label_map:
        print("No explicit val annotations parsed. Attempting to create mapping heuristically...")
        # Estratégia: iterar val_tar, coletar a lista de nomes de arquivos

        val_filenames = []  # Lista de nomes de arquivos
        with tarfile.open(val_tar, "r:") as vtar:
            for member in vtar:
                if member.isfile():
                    val_filenames.append(member.name)

        # Se o número de nomes de arquivo for zero -> erro
        if len(val_filenames) == 0:
            print("Error: no files found in val tar.")
        else:
            # Se o arquivo val_annotations_file contiver apenas linhas com synsets, tenta usá-las:
            if val_annotations_file and os.path.exists(val_annotations_file):

                syn_lines = [ln.strip() for ln in open(val_annotations_file) if ln.strip()]
                if len(syn_lines) == len(val_filenames):

                    # Mapeia em ordem
                    for fname, syn in zip(sorted(val_filenames), syn_lines):
                        syn = syn.split()[0]
                        val_label_map[os.path.basename(fname)] = class_to_label.get(syn, 0)
                else:
                    # Como último recurso: todos os rótulos são definidos como 0
                    # (isso interromperá a avaliação), por isso há o aviso ao usuário.
                    print("Warning: couldn't map val labels reliably. All val labels set to 0.")
                    for fname in val_filenames:
                        val_label_map[os.path.basename(fname)] = 0
            else:
                # Sem não houver arquivo de anotações: tenta mapear pela ordem do nome base para as
                # chaves class_to_label (melhor esforço)
                # Isso não é perfeito; a melhor prática é fornecer um arquivo val_annotations_file.
                print("No val annotations file provided. Creating fallback labels (may be incorrect).")
                for fname in val_filenames:
                    val_label_map[os.path.basename(fname)] = 0

    # Stream val_tar e escreve os TFRecords
    print("Streaming val.tar and writing validation TFRecords...")
    val_img_idx = 0
    with tarfile.open(val_tar, "r:") as vtar:
        for member in vtar:
            if not member.isfile():
                continue
            try:
                f = vtar.extractfile(member)
                if f is None:
                    continue
                img_bytes = f.read()
                basename = os.path.basename(member.name)
                label = val_label_map.get(basename, 0)
                shard_idx = val_img_idx % num_val_shards
                write_example(val_writers[shard_idx], img_bytes, label)
                val_img_idx += 1
            except Exception as e:
                print(f"Warning: skipping val image {member.name}: {e}")

    for w in val_writers:
        w.close()

    print(f"Validation TFRecords written to: {val_out}")
    print("Done.")


# ======================================================================================================================
# CRIA OS TFRECORDS PARA IMAGENET

def parse_example(serialized_example, image_size=224):
    """
    Lê um tf.train.Example e converte:
        * bytes → imagem float32 normalizada em [0,1]
        * label → int32
    Este pipeline é leve e segue o paper do ViT (resize para 224x224).
    """
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }

    ex = tf.io.parse_single_example(serialized_example, features)
    img = tf.io.decode_jpeg(ex["image"], channels=3)

    # Convert to float32 em [0,1], como no paper
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [image_size, image_size], method=tf.image.ResizeMethod.BILINEAR)

    label = tf.cast(ex["label"], tf.int32)
    return img, label


# ======================================================================================================================
# CARREGA OS TFRECORDS EM BATCHES PARA O VIT

def load_tfrecords(tfrecord_dir, batch_size, train=True, image_size=224, shuffle_buffer=10000):
    split = "train" if train else "validation"
    pattern = tf.io.gfile.glob(f"{tfrecord_dir}/{split}/*.tfrecord")

    if not pattern:
        raise ValueError(f"Nenhum TFRecord encontrado em: {tfrecord_dir}/{split}")

    ds = tf.data.TFRecordDataset(pattern, num_parallel_reads=AUTOTUNE)
    ds = ds.map(lambda x: parse_example(x, image_size=image_size), num_parallel_calls=AUTOTUNE)

    if train:
        ds = ds.shuffle(shuffle_buffer)
        ds = ds.map(lambda im, lab: (tf.image.random_flip_left_right(im), lab), num_parallel_calls=AUTOTUNE)
        ds = ds.repeat()  # treinamento precisa de dataset infinito

    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(AUTOTUNE)

    return ds


# ======================================================================================================================
# CONVERTE AS IMAGENS E LABELS DO TENSORFLOW PARA JAX ARRAYS

def tf_to_jax(batch_tf):
    images_tf, labels_tf = batch_tf
    images_np = np.asarray(images_tf)
    labels_np = np.asarray(labels_tf)

    # Formato das imagens -> (BATCH,HEIGHT,WIDTH,CHANNELS)
    return jnp.array(images_np), jnp.array(labels_np)
