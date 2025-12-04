import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score
)

# Classes oficiais do HAM10000 na ordem do dataset
HAM10K_LABELS = [
    "akiec",  # carcinoma acantolítico
    "bcc",  # carcinoma basocelular
    "bkl",  # queratose benigna
    "df",  # dermatofibroma
    "mel",  # melanoma
    "nv",  # nevo melanocítico
    "vasc"  # lesão vascular
]


def evaluate_ham10000(model, val_ds):
    """
    Avalia um modelo treinado no HAM10000 usando métricas clássicas
    de classificação multiclass.
        * model: modelo Keras (Arquitetura ViT)
        * val_ds: tf.data.Dataset contendo batches (img, label_one_hot)

    Retorno:
        dicionário com métricas e valores organizados.
    """

    # ==================================================================================
    #                       Obter predições e labels na mesma ordem
    # ==================================================================================

    all_preds = []
    all_labels = []

    # Obtém predições do modelo na mesma ordem em que o dataset fornece exemplos.
    for batch_imgs, batch_labels in val_ds:
        preds = model.predict(batch_imgs, verbose=0)
        all_preds.append(preds)
        all_labels.append(batch_labels.numpy())

    # np.vstack transforma a lista de batches em uma matriz única.
    all_preds = np.vstack(all_preds)  # shape: (N, 7)
    all_labels = np.vstack(all_labels)  # shape: (N, 7)

    # labels reais devem virar inteiros (argmax)
    # Como o treinamento foi feito com one-hot, é necessário retornar ao rótulo inteiro usando argmax.
    true_labels = np.argmax(all_labels, axis=1)  # shape: (N,)
    pred_labels = np.argmax(all_preds, axis=1)  # shape: (N,)

    # ==================================================================================
    #                               Métricas principais
    # ==================================================================================

    # accuracy_score → métrica convencional
    acc = accuracy_score(true_labels, pred_labels)
    # balanced_accuracy_score → muito importante porque HAM10000 é desbalanceado
    balanced_acc = balanced_accuracy_score(true_labels, pred_labels)

    '''
        F1 macro/micro/weighted → geralmente relatados em artigos
        F1 macro → trata todas as classes igualmente
        F1 weighted → peso proporcional ao tamanho de cada classe
    '''
    f1_macro = f1_score(true_labels, pred_labels, average="macro")
    f1_weighted = f1_score(true_labels, pred_labels, average="weighted")
    f1_micro = f1_score(true_labels, pred_labels, average="micro")

    # Precisão e recall
    precision_macro = precision_score(true_labels, pred_labels, average="macro")
    recall_macro = recall_score(true_labels, pred_labels, average="macro")

    # Confusion matrix → essencial para análise
    cm = confusion_matrix(true_labels, pred_labels)

    # ==================================================================================
    #                       Organizar tudo num dicionário limpo
    # ==================================================================================
    results = {
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "f1_micro": f1_micro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "confusion_matrix": cm,
        "true_labels": true_labels,
        "pred_labels": pred_labels,
        "probabilities": all_preds,
        "class_names": HAM10K_LABELS
    }

    return results
