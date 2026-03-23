# -------------------------------------------------------------------------------------------------------------------- #
#                   Arquivo responsável por criar um ponte entre a interface e a execução da ViT                       #
#                           Se faz necessário em razão da mudança de ambientes virtuais                                #
# -------------------------------------------------------------------------------------------------------------------- #

import json
import sys
import os

from ViT_Trainer import train_vit
from MainProject.DataLoader import DataLoader
from MainProject.VisionTransformers import ViT_Utils


def main(config_path: str):

    # ==================================================================================================================
    #                                   LEITURA DO JSON DE CONFIGURAÇÃO E ENVIO DE LOGS
    # ==================================================================================================================
    with open(config_path, "r") as f:
        cfg = json.load(f)

    print("---------------------------------------------------------------")
    print("AMBIENTE DE EXECUCAO DO VISION TRANSFORMER")
    print("EXECUTÁVEL: ", sys.executable)
    print("VERSÃO DO PYTHON: ", sys.version)
    print("---------------------------------------------------------------")
    print("INICIANDO TREINAMENTO DA REDE")
    print(f"Modo: {cfg['mode']}")
    print(f"Batch size: {cfg['batch_size']}")
    print(f"Total steps: {cfg['total_steps']}")

    # ==================================================================================================================
    #                                   CRIAÇÃO DO DIRETÓRIO DE OUTPUT DE RESULTADOS
    # ==================================================================================================================
    output_dir = ViT_Utils.resource_path("MainProject\\VisionTransformers\\Outputs\\ViT_Runs")
    os.makedirs(output_dir, exist_ok=True)


    # ==================================================================================================================
    #                                    CARREGAMENTO DOS DADOS, VIA DATALOADER
    # ==================================================================================================================
    dataloader = DataLoader(
        path=cfg["dataset_path"],
        img_size=cfg["input_size"],
        batch_size=cfg["image_batch_size"],
        val_split=cfg["split"],
    )

    (
        train_ds,
        val_ds,
        log_train,
        log_val,
        log_indexes,
        num_classes,
        steps_train,
        steps_val
    ) = dataloader.process_data()

    print("LOGS PROVENIENTES DO CARREGAMENTO DO DATASET")
    print(" ")
    print("Treinamento: ", log_train)
    print("Validação: ", log_val)
    print("Índices: ", log_indexes)
    print(f"Classes detectadas: {num_classes}\n")

    # ==================================================================================================================
    #                                           CHAMADA DO TRAINER DA REDE
    # ==================================================================================================================
    train_vit(
        train_ds=train_ds,
        val_ds=val_ds,
        output_dir=output_dir,
        patches=(cfg["patch_size"], cfg["patch_size"]),
        hidden_size=cfg["hidden_size"],
        depth=cfg["transformer_layers"],
        num_heads=cfg["num_heads"],
        mlp_dim=cfg["mlp_units"],
        num_classes=num_classes,
        total_steps=cfg["total_steps"],
        warmup_steps=cfg["warmup_steps"],
        # batch_size=cfg["batch_size"],
        base_lr=cfg["base_lr"],
        mode=cfg["mode"],
        weights_path=cfg["weights"]
    )

# ======================================================================================================================
# ENTRY POINT DO SCRIPT

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Uso correto:")
        print("python ViT_EntryPoint.py <config.json>")
        sys.exit(1)

    main(sys.argv[1])