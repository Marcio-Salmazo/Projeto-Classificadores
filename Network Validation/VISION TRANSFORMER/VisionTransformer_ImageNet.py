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
# tipos para documentação de código
from typing import Optional, Dict, Tuple


def _write_example(writer: tf.io.TFRecordWriter, image_bytes: bytes, label: int):
    """Write one tf.train.Example with image bytes and int64 label."""
    feature = {
        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())


def _open_train_tar_members(train_tar_path: str):
    """
    Yield members for class tars inside the top-level train tar.
    Each returned item is (member_name, fileobj) where fileobj is a file-like
    object for the inner class tar bytes.
    """
    with tarfile.open(train_tar_path, "r:") as tar:
        for member in tar:
            # In ImageNet's train tar, top-level members are files like 'n01440764.tar'
            if member.isfile() and member.name.endswith(".tar"):
                f = tar.extractfile(member)  # file-like for the inner tar
                yield member.name, f


def create_imagenet_tfrecords_streaming(
        train_tar: str,
        val_tar: str,
        out_dir: str,
        num_train_shards: int = 1024,
        num_val_shards: int = 128,
        val_annotations_file: Optional[str] = None):
    """
    Create TFRecords for ImageNet by streaming data from train_tar and val_tar.

    - train_tar: path to ILSVRC2012_img_train.tar (contains per-class .tar members)
    - val_tar: path to ILSVRC2012_img_val.tar
    - out_dir: base output dir; will create out_dir/train and out_dir/validation
    - num_train_shards, num_val_shards: number of output shards
    - val_annotations_file: optional path to a validation annotations file (maps val image -> synset)
        common formats accepted:
          * lines of "val_00000001.JPEG n01440764"
          * lines with only synsets (one per line) matching sorted validation filenames order
    """
    os.makedirs(out_dir, exist_ok=True)
    train_out = os.path.join(out_dir, "train")
    val_out = os.path.join(out_dir, "validation")
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(val_out, exist_ok=True)

    # -------------------------
    # TRAIN: open shard writers (round-robin)
    # -------------------------
    train_writers = []
    for i in range(num_train_shards):
        shard_path = os.path.join(train_out, f"train-{i:05d}-of-{num_train_shards:05d}.tfrecord")
        train_writers.append(tf.io.TFRecordWriter(shard_path))

    # We'll build class -> label map in the order we encounter class tars.
    class_to_label: Dict[str, int] = {}
    next_label = 0
    global_img_idx = 0  # used to round-robin assign shards

    print("Streaming train.tar and writing TFRecords (round-robin across shards)...")
    # iterate top-level train tar members (each should be a .tar for a synset/class)
    for member_name, inner_fileobj in _open_train_tar_members(train_tar):
        # member_name e.g. "n01440764.tar"
        synset = os.path.splitext(os.path.basename(member_name))[0]  # n01440764
        if synset not in class_to_label:
            class_to_label[synset] = next_label
            next_label += 1
        label = class_to_label[synset]

        # Open the inner tar (class images) from the fileobj (in-memory file-like)
        try:
            with tarfile.open(fileobj=inner_fileobj, mode="r:") as inner_tar:
                for img_member in inner_tar:
                    if not img_member.isfile():
                        continue
                    try:
                        img_f = inner_tar.extractfile(img_member)
                        if img_f is None:
                            continue
                        img_bytes = img_f.read()
                        shard_idx = global_img_idx % num_train_shards
                        _write_example(train_writers[shard_idx], img_bytes, label)
                        global_img_idx += 1
                    except Exception as e_img:
                        # skip unreadable images but log
                        print(f"Warning: skipping image {img_member.name} in {synset}: {e_img}")
        except Exception as e_inner:
            print(f"Warning: could not open inner tar {member_name}: {e_inner}")
        finally:
            # ensure the fileobj is closed
            try:
                inner_fileobj.close()
            except Exception:
                pass

    # close train writers
    for w in train_writers:
        w.close()

    print(f"Train TFRecords written to: {train_out}")
    print(f"Number of classes (train): {len(class_to_label)}")

    # -------------------------
    # VAL: create writers and mapping
    # -------------------------
    val_writers = []
    for i in range(num_val_shards):
        shard_path = os.path.join(val_out, f"validation-{i:05d}-of-{num_val_shards:05d}.tfrecord")
        val_writers.append(tf.io.TFRecordWriter(shard_path))

    # Build val annotation mapping: filename -> label (int index per train's class ordering)
    val_label_map: Dict[str, int] = {}

    # If a val_annotations_file is provided, parse it.
    if val_annotations_file and os.path.exists(val_annotations_file):
        print(f"Parsing val annotations from {val_annotations_file}")
        with open(val_annotations_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tokens = line.split()
                # Many devkits: "val_00000001.JPEG n01440764"
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

    # If val_label_map still empty, try to infer by enumerating val tar members and assigning
    # labels based on filenames order if val_annotations_file contained only synsets.
    # Fallback: if val_annotations_file absent, attempt simple heuristic: if val images are stored within
    # per-class folders inside val tar (rare), extract mapping; otherwise, will attempt to map by
    # filename using the train class order (best-effort).
    if not val_label_map:
        print("No explicit val annotations parsed. Attempting to create mapping heuristically...")
        # Strategy: iterate val tar, gather filenames list
        val_filenames = []
        with tarfile.open(val_tar, "r:") as vtar:
            for member in vtar:
                if member.isfile():
                    val_filenames.append(member.name)
        # if number of filenames is zero -> error
        if len(val_filenames) == 0:
            print("Error: no files found in val tar.")
        else:
            # If val_annotations_file had lines with synsets only, try to use them:
            if val_annotations_file and os.path.exists(val_annotations_file):
                # read synsets lines
                syn_lines = [ln.strip() for ln in open(val_annotations_file) if ln.strip()]
                if len(syn_lines) == len(val_filenames):
                    # map in order
                    for fname, syn in zip(sorted(val_filenames), syn_lines):
                        syn = syn.split()[0]
                        val_label_map[os.path.basename(fname)] = class_to_label.get(syn, 0)
                else:
                    # last resort: set all labels to 0 (will break evaluation); warn user
                    print("Warning: couldn't map val labels reliably. All val labels set to 0.")
                    for fname in val_filenames:
                        val_label_map[os.path.basename(fname)] = 0
            else:
                # No annotation file: attempt to map by basename order to class_to_label keys (best effort)
                # This is imperfect; best practice is to provide val_annotations_file.
                print("No val annotations file provided. Creating fallback labels (may be incorrect).")
                for fname in val_filenames:
                    val_label_map[os.path.basename(fname)] = 0

    # Now stream val tar and write TFRecords
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
                _write_example(val_writers[shard_idx], img_bytes, label)
                val_img_idx += 1
            except Exception as e:
                print(f"Warning: skipping val image {member.name}: {e}")

    # close val writers
    for w in val_writers:
        w.close()

    print(f"Validation TFRecords written to: {val_out}")
    print("Done.")
