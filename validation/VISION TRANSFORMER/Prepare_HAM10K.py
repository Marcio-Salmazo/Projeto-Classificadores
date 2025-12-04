import os
import pandas as pd
from sklearn.model_selection import train_test_split

"""
    Esse código deve ser executado uma única vez para organizar os dados.
    A ideia é gerar os arquivos arquivos train.csv, val.csv, test.csv, para
    serem carregados pelo loader, a partir do arquivo de metadados 'HAM10000_metadata.csv'.
    
    O script abaixo tem a responsabilidade de :
        ✔ Ler HAM10000_metadata.csv
        ✔ Gerar o campo image_path
        ✔ Faz split estratificado 70/15/15
        ✔ Salva train.csv, val.csv, test.csv
"""

# Definição manual de caminhos para o dataset (Modificar conforme necessário)
DATASET_DIR = r"C:\Users\marci_wawp\Desktop\Arquivos\Mestrado\Projeto-Classificadores\vit_network_validation\HAM10000"
IMG_DIR = os.path.join(DATASET_DIR, "images")
META_PATH = os.path.join(DATASET_DIR, "HAM10000_metadata.csv")

# Carregamento dos metadados
df = pd.read_csv(META_PATH)

# Criação de coluna com caminho absoluto da imagem
df["image_path"] = df["image_id"].apply(lambda x: os.path.join(IMG_DIR, f"{x}.jpg"))

# Remoção imagens ausentes (robustez)
df = df[df["image_path"].apply(os.path.exists)]

# split estratificado Entre TREINO / VALIDAÇÃO / TESTE (70/15/15)
train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df["dx"])
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df["dx"])

# Armazenamento dos splits em formato CSV
train_df.to_csv(os.path.join(DATASET_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(DATASET_DIR, "val.csv"), index=False)
test_df.to_csv(os.path.join(DATASET_DIR, "test.csv"), index=False)

print("Preparação concluída!")
print(f"Treino: {len(train_df)}")
print(f"Val: {len(val_df)}")
print(f"Teste: {len(test_df)}")