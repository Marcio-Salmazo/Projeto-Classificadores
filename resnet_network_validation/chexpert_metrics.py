import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ======================================================================================================================

# Definição da lista oficial das 14 patologias do CheXpert na ordem exata usada pelo dataset
CHEXPERT_LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices"
]

# As cinco patologias principais usadas para avaliação oficial
CHEXPERT_5_COMPETITION = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion"
]


# ======================================================================================================================

# Função de agregação por estudo (o núcleo do artigo base)
def aggregate_by_study(paths, preds, labels, mode="max"):
    """
        paths: lista de paths absolutos das imagens de validação
        preds: matriz (N,14) — predições da rede
        labels: matriz (N,14) — labels na mesma ordem
        mode: método de agregação entre views de um mesmo study:
              - "max" (padrão do artigo)
              - "mean" (alternativo usado em algumas reimplementações)

        O artigo CheXpert explica:
        - um study (exame) pode ter múltiplas imagens (views) do mesmo paciente
        - a predição do study = max pooling (pior caso clínico)
    """

    '''
        Extrai o nome do estudo a partir do caminho
        Exemplo:
           patient00001/study1/view1_frontal.jpg
        Extraímos o valor do estudo:  study1
    '''
    study_ids = [
        p.split("/")[-3] if "/" in p else p.split("\\")[-3]
        for p in paths
    ]

    # Monta um dataframe auxiliar para agrupar por study
    df = pd.DataFrame({
        "study": study_ids,
        "pred": list(preds),
        "label": list(labels)
    })

    # Função interna para agregar (max ou mean)
    def agg_func(x):
        arr = np.vstack(x)  # empilha views do mesmo estudo
        if mode == "mean":
            return arr.mean(axis=0)
        return arr.max(axis=0)  # padrão do artigo: max pooling entre views

    # Agrupa predictions e labels por estudo
    agg = df.groupby("study").agg({"pred": agg_func, "label": agg_func})

    preds_study = np.vstack(agg["pred"].values)  # empilha as predições do mesmo estudo
    labels_study = np.vstack(agg["label"].values)  # empilha as labels do mesmo estudo
    study_keys = list(agg.index)

    return preds_study, labels_study, study_keys


# ======================================================================================================================

# Geração de AUC por classe (14 patologias)
def auc_per_class(preds_study, labels_study):
    """
        Computa AUC ROC para cada uma das 14 patologias.
        Se alguma patologia não aparece em labels (só zeros),
        o sklearn dispara erro e substitui por NaN.
    """

    aucs = []
    for i in range(14):
        try:
            auc = roc_auc_score(labels_study[:, i], preds_study[:, i])
        except ValueError:
            # Classe não presente no ground-truth → impossível calcular AUC
            auc = np.nan
        aucs.append(auc)
    return aucs


# ======================================================================================================================


# AUC média das 5 patologias oficiais
def auc_mean_5(auc_list):
    # Usa somente as 5 patologias usadas para a métrica oficial:
    # Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion.
    indexes = [CHEXPERT_LABELS.index(x) for x in CHEXPERT_5_COMPETITION]
    subset = [auc_list[i] for i in indexes]
    return np.nanmean(subset)


# ======================================================================================================================


# 5. Função principal de avaliação pós-treino
def evaluate_chexpert(model, valid_ds, valid_paths):
    """
        Executa o pipeline completo de avaliação CheXpert:

        1) Roda predições batch por batch em valid_ds
        2) Junta labels na mesma ordem dos batches
        3) Agrega por estudo (max pooling por padrão)
        4) Calcula:
            - AUC por classe (14)
            - AUC média das 5 patologias
        5) Retorna todas as informações para debug

        Parâmetros:
            model       : modelo Keras treinado (multi-label sigmoid)
            valid_ds    : tf.data.Dataset sem shuffle
            valid_paths : paths ordenados exatamente como valid_ds produz
    """

    # Obtém predições do modelo (N,14)
    preds = model.predict(valid_ds, verbose=1)

    # Obtém labels na mesma ordem dos batches
    labels = []
    for batch_imgs, batch_labels in valid_ds:
        labels.append(batch_labels.numpy())
    labels = np.vstack(labels)

    # Consolida views por estudo (max pooling)
    preds_study, labels_study, study_keys = aggregate_by_study(
        valid_paths, preds, labels
    )

    # Calculo de AUC por classe
    aucs = auc_per_class(preds_study, labels_study)

    # Calculo AUC média oficial (5 patologias)
    auc_mean = auc_mean_5(aucs)

    # Retornar tudo de forma organizada
    return {
        "auc_per_class": aucs,
        "auc_mean_5": auc_mean,
        "study_ids": study_keys,
        "preds_study": preds_study,
        "labels_study": labels_study
    }


# ======================================================================================================================

'''
    EXPLICAÇÃO DO FLUXO DE AVALIÇÃO DO CHEXPERT, COM A DIDÁTICA ORIENTADA AO ARTIGO
    
    1) A rede prevê VIEWs, não estudos

        Uma radiografia no CheXpert pode ter várias views para o mesmo exame:
            * view1_frontal
            * view2_lateral
        O modelo produz predições por view.

    2) O paper faz AGGREGATION por estudo (max pooling)
    
        No artigo é dito "For each study with multiple views, we take the maximum probability across views."
        Isso é implementado aqui por return arr.max(axis=0)

    3) AUC é calculada APÓS a agregação
    
        Não faz sentido calcular AUC por view — o artigo não faz isso.
        Por isso, a sequência é: predict → agrupar por estudo → calcular AUC

    4) As 5 patologias oficiais formam a métrica principal
        
        CHEXPERT_5_COMPETITION = [...]
        Essas cinco dão a “CheXpert Competition Metric”.
'''
