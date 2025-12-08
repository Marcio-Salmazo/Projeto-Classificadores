"""
main.py
-------
Orquestrador geral para:
 - extração do ImageNet
 - criação de TFRecords
 - pré-treino do ViT
 - fine-tuning do ViT
 - avaliação final

Execute:
 python main.py --mode pretrain
 python main.py --mode finetune
 python main.py --mode evaluate
"""

import argparse
import os

from data_loader import extract_imagenet_tars, create_imagenet_tfrecords
from train_vit import train_vit
from evaluate_vit import evaluate_vit


# ------------------------------------------------------------
# MAIN DISPATCHER
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True,
                        choices=["extract", "tfrecords", "pretrain", "finetune", "evaluate"])

    parser.add_argument("--raw_dir", type=str, default="D:/datasets/imagenet_raw")
    parser.add_argument("--tfrecord_dir", type=str, default="D:/datasets/imagenet_tfrecords")

    parser.add_argument("--train_tar", type=str, default="D:/datasets/ILSVRC2012_img_train.tar")
    parser.add_argument("--val_tar",   type=str, default="D:/datasets/ILSVRC2012_img_val.tar")

    parser.add_argument("--output_dir", type=str, default="D:/outputs/vit")
    parser.add_argument("--checkpoint_dir", type=str, default="D:/outputs/vit")

    args = parser.parse_args()

    # -------------------------
    # 1) EXTRAIR IMAGENET
    # -------------------------
    if args.mode == "extract":
        extract_imagenet_tars(args.train_tar, args.val_tar, args.raw_dir)
        return

    # -------------------------
    # 2) CRIAR TFRECORDS
    # -------------------------
    if args.mode == "tfrecords":
        create_imagenet_tfrecords(args.raw_dir, args.tfrecord_dir)
        return

    # -------------------------
    # 3) PRÉ-TREINO
    # -------------------------
    if args.mode == "pretrain":
        train_vit(
            tfrecord_train_dir=os.path.join(args.tfrecord_dir, "train"),
            tfrecord_val_dir=os.path.join(args.tfrecord_dir, "validation"),
            output_dir=args.output_dir,
            mode="pretrain",
            total_steps=100000,
            warmup_steps=10000,
            batch_size=256,
            base_lr=2e-4
        )
        return

    # -------------------------
    # 4) FINE-TUNING
    # -------------------------
    if args.mode == "finetune":
        train_vit(
            tfrecord_train_dir=os.path.join(args.tfrecord_dir, "train"),
            tfrecord_val_dir=os.path.join(args.tfrecord_dir, "validation"),
            output_dir=args.output_dir,
            mode="finetune",
            total_steps=20000,
            warmup_steps=0,
            batch_size=512,
            base_lr=0.01
        )
        return

    # -------------------------
    # 5) AVALIAÇÃO
    # -------------------------
    if args.mode == "evaluate":
        evaluate_vit(
            checkpoint_dir=args.checkpoint_dir,
            tfrecord_val_dir=os.path.join(args.tfrecord_dir, "validation"),
            batch_size=512,
            num_batches=200
        )
        return


if __name__ == "__main__":
    main()