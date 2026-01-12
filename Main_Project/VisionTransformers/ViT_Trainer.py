# -------------------------------------------------------------------------------------------------------------------- #
#                                                                                                                      #
#            Arquivo responsável pelo treino + fine-tuning do Vision Transformer puro,                                 #
#            seguindo fielmente o paper "An Image is Worth 16x16 Words".                                               #
#            Aqui são definidos:                                                                                       #
#                * perda (cross-entropy) e métricas;                                                                   #
#                * train_step e eval_step escritos em JAX e decorados com jax.pmap para execução distribuída;          #
#                * função train_vit(...) que monta dataset, modelo, inicializa parâmetros, cria otimizador,            #
#                  replica estado para dispositivos e faz o loop de treinamento,                                       #
#                  salvando checkpoints e chamando avaliação periódica;                                                #
#                * função evaluate_vit(...) que avalia o modelo em batches do conjunto de validação.                   #
#                                                                                                                      #
# -------------------------------------------------------------------------------------------------------------------- #
import os
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
import jax
import jax.numpy as jnp
import optax
import json
import csv
import numpy as np

from datetime import datetime
from flax.training import train_state, checkpoints
from Main_Project.VisionTransformers.ViT_Model import VisionTransformer
from Main_Project.VisionTransformers.ViT_Utils import load_vit_npz

# ======================================================================================================================
# AUXILIAR DE LOGS (PARA TREINO E VALIDAÇÃO)
'''
def save_metrics_json(output_dir, step, metrics):
    """Salva métricas em JSON."""
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    filepath = os.path.join(logs_dir, f"step_{step:06d}.json")
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=4)
'''

def to_python_type(x):
    if hasattr(x, "item"):
        return x.item()
    return x

def save_metrics_json(output_dir, step_epoch, metrics, mode):

    """Salva métricas em JSON."""
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    if mode == 'steps':
        filepath = os.path.join(logs_dir, f"step_{step_epoch:06d}.json")
    if mode == 'epochs':
        filepath = os.path.join(logs_dir, f"epoch_{step_epoch:06d}.json")

    # Converte tudo para tipos nativos
    safe_metrics = {
        k: to_python_type(v) for k, v in metrics.items()
    }

    with open(filepath, "w") as f:
        json.dump(safe_metrics, f, indent=4)

def append_metrics_csv(output_dir, metrics):
    """Acrescenta linha ao CSV principal de logs."""
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    csv_path = os.path.join(logs_dir, "train_log.csv")
    file_exists = os.path.exists(csv_path)

    # Criar o CSV com cabeçalho se ainda não existir
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(metrics)


# ======================================================================================================================
# CONVERTE AS IMAGENS E LABELS DO BATCH PARA JAX ARRAYS

def tf_to_jax(batch_tf):
    images_tf, labels_tf = batch_tf
    images_np = np.asarray(images_tf)
    labels_np = np.asarray(labels_tf)

    # Formato das imagens -> (BATCH,HEIGHT,WIDTH,CHANNELS)
    return jnp.array(images_np), jnp.array(labels_np)


# ======================================================================================================================
# CRIAÇÃO DO OTIMIZADOR DE PRÉ-TREINO IDÊNTICO AO APRESENTADO NO ARTIGO

def create_optimizer_pretrain(base_lr, warmup_steps, total_steps):
    """
        * Cria um learning rate schedule com warmup seguido de cosine decay (warmup->peak->cosine decay até end_value).
        * Passa esse schedule para optax.adam com beta1=0.9, beta2=0.999, eps=1e-8.

        * 0 artigo usa Adam (β₁=0.9, β₂=0.999) com warmup de 10k steps e cosine decay (por padrão),
          e LR base ~2e-4 no pré-treino. Isso reproduz essas escolhas diretamente.
    """
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=base_lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=0.0,
    )
    optimizer = optax.adam(
        learning_rate=schedule,
        b1=0.9, b2=0.999, eps=1e-8
    )
    return optimizer


# ======================================================================================================================
# CRIAÇÃO DO OTIMIZADOR DE FINE-TUNNING IDÊNTICO AO APRESENTADO NO ARTIGO

def create_optimizer_finetune(base_lr):
    """
        Retorna um otimizador SGD com momentum 0.9, sem Nesterov.
        O artigo recomenda SGD+momentum 0.9 para fine-tuning.
    """
    optimizer = optax.sgd(
        learning_rate=base_lr,
        momentum=0.9,
        nesterov=False
    )
    return optimizer


# ======================================================================================================================
# DEFINIÇÃO DA FUNÇÃO DE PERDA E DAS MÉTRICAS

def cross_entropy_loss(logits, labels):
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))


def compute_train_metrics(logits, labels):
    # Métricas rápidas usadas durante o treinamento
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return {"loss": loss, "accuracy": accuracy}


def compute_eval_metrics(logits, labels):

    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    log_probs = jax.nn.log_softmax(logits)     # cross-entropy
    loss = -jnp.mean(log_probs[jnp.arange(labels.size), labels])
    return {
        "loss": loss,
        "accuracy": accuracy
    }


# ======================================================================================================================
# FUNÇÃO DE TREINO VIA JAX

@jax.jit
def train_step_jit(state, batch, rng):
    images, labels = batch
    dropout_rng = rng
    """
            1 - Decorador jax.jit: compila a função para execução eficiente em um único dispositivo (GPU).

            2 - images, labels = batch: os dados já chegaram sharded (cada dispositivo recebe seu sub-batch).

            3 - loss_fn(params):
                * Aplica o modelo (state.apply_fn) com esses params em modo train=True.
                * Fornece RNG fixo jax.random.PRNGKey(0) para rngs={"dropout": ...} — isso é um ponto a melhorar
                * Computa a função de loss.

            4 - jax.value_and_grad(loss_fn, has_aux=True) calcula loss + logits
                e o gradiente da loss em relação aos parâmetros.

            5 - grads = jax.lax.pmean(grads, axis_name="batch") — média dos gradientes entre os
                dispositivos (síncrona). Essencial para data-parallel training consistente.

            6 - new_state = state.apply_gradients(grads=grads) — aplica o passo do otimizador e retorna novo TrainState.

            7 - metrics = compute_metrics(...) → jax.lax.pmean(...) — agregação média das métricas entre dispositivos.
        """

    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params},
            images,
            train=True,
            rngs={"dropout": dropout_rng}
        )
        loss = cross_entropy_loss(logits, labels)
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    new_state = state.apply_gradients(grads=grads)
    metrics = compute_train_metrics(logits, labels)
    return new_state, metrics


# ==================================================================================================================== #
#                                                      OBSERVAÇÃO
#     O CÓDIGO ORIGINAL UTILIZA @PMAP PARA PARALELIZAR E DISTRIBUIR O TREINAMENTO ENTRE MÚLTIPLOS DISPOSITIVOS
#     COMO O TREINAMENTO FOI FEITO UTILIZANDO UMA ÚNICA GPU (DO MEU COMPUTADOR PESSOAL), FORAM APLICADAS AS
#     MODIFICAÇÕES NECESSÁRIAS PARA RETIRAR ESTE PARALELISMO (UTILIZANDO O DECORADOR JIT)
# ==================================================================================================================== #

# ======================================================================================================================
# LOOP PRINCIPAL DE TREINAMENTO

def train_vit(

        # Parâmetros base
        train_ds,
        val_ds,
        output_dir: str,
        patches=(16, 16),
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_dim=3072,
        num_classes=1000,   # (VALOR ALEATÓRIO)
        total_steps=100000, # (VALOR ALEATÓRIO)
        warmup_steps=10000, # (VALOR ALEATÓRIO)
        base_lr=2e-4,
        mode="finetune",
        weights_path=None,
        steps_per_epoch=100, # (VALOR ALEATÓRIO)
        steps_val = 100,     # (VALOR ALEATÓRIO)
        epochs=5 # (VALOR ALEATÓRIO)
    ):

    train_iter = iter(train_ds)
    val_iter = iter(val_ds)

    # Criação do modelo e definição das configurações
    transformer_cfg = dict(
        num_layers=depth,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
        add_position_embedding=True,
    )

    class PatchConfig:
        pass

    pc = PatchConfig()
    pc.size = patches

    model = VisionTransformer(
        num_classes=num_classes,
        patches=pc,
        transformer=transformer_cfg,
        hidden_size=hidden_size,
        representation_size=None,
        classifier="token"
    )

    # Inicialização de parâmetros
    rng = jax.random.PRNGKey(0)
    dummy = jnp.ones([1, 224, 224, 3], dtype=jnp.float32)
    # Inicializa parâmetros com pesos aleatórios (default do Flax)
    params = model.init({"params": rng, "dropout": rng}, dummy, train=True)["params"]

    if mode == "finetune":
        # Carregamento dos pesos pré-treinados
        pretrained_path = weights_path

        print(f">> Carregando pesos pré-treinados de {pretrained_path}")
        params = load_vit_npz(params, pretrained_path)
        print(">> Pesos pré-treinados carregados com sucesso!")

    # Escolha do otimizador
    if mode == "pretrain":
        optimizer = create_optimizer_pretrain(base_lr, warmup_steps, total_steps)
    else:
        optimizer = create_optimizer_finetune(base_lr)

    # Criação do TrainState: encapsula params, apply_fn e otimizador (tx)
    # num estado que facilita updates e checkpointing.
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )

    # Inicialização do melhor valor de perda
    best_val_loss = float("inf")

    print("=====================================================")
    print("           INICIANDO LOOP DE TREINAMENTO             ")
    print("=====================================================")

    for epoch in range(epochs):

        # Buffers para métricas de treino (nível epoch)
        epoch_train_losses = []
        epoch_train_accs = []

        # Reinicia o dataset a cada época
        train_iter = iter(train_ds)

        for step in range(steps_per_epoch):
            try:
                batch_tf = next(train_iter)
            except StopIteration:
                # segurança extra
                break

            batch = tf_to_jax(batch_tf)

            # gerar RNG novo para dropout
            rng, dropout_rng = jax.random.split(rng)
            # Apresenta apenas o batch atual e as métricas locais daquele batch
            state, metrics = train_step_jit(state, batch, dropout_rng)

            # Acumula métricas do treino
            epoch_train_losses.append(float(metrics["loss"]))
            epoch_train_accs.append(float(metrics["accuracy"]))

            global_step = epoch * steps_per_epoch + step

            if global_step % 5 == 0:
                print(
                    f"-----------------------------------------------------\n"
                    f"[{global_step:06d}]"
                    f"loss={float(metrics['loss']):.4f}, "
                    f"acc={float(metrics['accuracy']):.4f}"
                )

        # Logging ocasional por épocas
        # Aqui são feitos a média dos últimos steps_per_epoch batches
        # As métricas agregadas de treino são armazenadas em JSON e no CSV
        # Isso representa como o modelo se comportou durante essa época de treino
        # Essa NÃO é uma etapa de validação
        epoch_log = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "train_loss_epoch": float(np.mean(epoch_train_losses)),
            "train_accuracy_epoch": float(np.mean(epoch_train_accs)),
        }

        # Log estruturado por epoch
        save_metrics_json(output_dir, epoch, epoch_log, mode='epochs')
        append_metrics_csv(output_dir, epoch_log)

        print(
            f"-----------------------------------------------------"
            f"\n[EPOCH {epoch}]\n"
            f"train_loss={epoch_log['train_loss_epoch']:.4f}, "
            f"train_acc={epoch_log['train_accuracy_epoch']:.4f}"
        )

        # ==============================================================================
        # Avaliação global aproximada, não pontual a cada 20 épocas
        # la avalia varios batches do conjunto de validação (definido por num_batches)
        # É uma validação em uma amostra consistente para acompanhar o comportamento

        if epoch % 20 == 0 and epoch != 0:

            val_iter = iter(val_ds)
            eval_results = evaluate_epoch(
                state,
                val_iter,
                num_batches=steps_val,
                num_classes=num_classes,
            )

            if eval_results is None:
                print("Avaliação ignorada: dataset de validação esgotado.")
            else:

                val_log = {
                    "epoch": int(epoch),
                    "timestamp": datetime.now().isoformat(),
                    "val_loss": float(eval_results["loss"]),
                    "accuracy": float(eval_results["accuracy"]),
                    "val_balanced_acc": float(eval_results["balanced_accuracy"]),
                    "val_precision_macro": float(eval_results["precision_macro"]),
                    "val_recall_macro": float(eval_results["recall_macro"]),
                    "val_f1_macro": float(eval_results["f1_macro"]),
                }
                # Salvar JSON para este step
                save_metrics_json(output_dir, epoch, val_log,  mode='epochs')
                # Escrever linha no CSV
                append_metrics_csv(output_dir, val_log)

                print(
                    f"val_loss={val_log['val_loss']:.4f}, "
                    f"accuracy={val_log['val_accuracy']:.4f}, "
                )

                if eval_results["loss"] < best_val_loss:
                    best_val_loss = eval_results["loss"]

                    # Armazenamento dos Checkpoints
                    # São responsáveis por armazenar pesos do modelo, estado do otimizador e step
                    # Importantes para recuperação em caso de falha e seleção do melhor modelo
                    # OBS: O melhor modelo quase nunca é o último step.
                    # Permitem a continuidade do treino
                    checkpoints.save_checkpoint(
                        output_dir,
                        state,
                        epoch,
                        prefix="best_",
                        overwrite=True
                    )
                    print("Novo melhor modelo salvo!")

                if "confusion_matrix" in eval_results:
                    save_confusion_matrix(
                        output_dir,
                        epoch,
                        eval_results["confusion_matrix"]
                    )

        # Resetar buffers
        epoch_train_losses.clear()
        epoch_train_accs.clear()

    print("=====================================================")
    print("             AVALIAÇÃO FINAL DO MODELO               ")
    print("=====================================================")

    # Reinicializar o val_iter
    val_iter = iter(val_ds)

    final_results = evaluate_epoch(
        state,
        val_iter,
        num_batches=steps_val,  # ou None se quiser tudo
        num_classes=num_classes,
    )

    if final_results is None:
        print("Avaliação final ignorada: dataset de validação esgotado.")
    else:

        final_log = {
            "step": int(total_steps),
            "timestamp": datetime.now().isoformat(),

            "val_loss_final": float(final_results["loss"]),
            "val_accuracy_final": float(final_results["accuracy"]),

            "val_balanced_acc_final": float(final_results["balanced_accuracy"]),
            "val_precision_macro_final": float(final_results["precision_macro"]),
            "val_recall_macro_final": float(final_results["recall_macro"]),
            "val_f1_macro_final": float(final_results["f1_macro"]),
        }

        save_metrics_json(output_dir, total_steps, final_log,  mode='steps')
        append_metrics_csv(output_dir, final_log)

        if "confusion_matrix" in final_results:
            save_confusion_matrix(
                output_dir,
                total_steps,
                final_results["confusion_matrix"]
            )

        print(
            f">> AVALIÇÃO FINAL: \n"
            f"loss={final_log['val_loss_final']:.4f}, "
            f"accuracy={final_log['val_accuracy_final']:.4f}"
        )

    # salvar final
    checkpoints.save_checkpoint(output_dir, state, total_steps, prefix="final_", overwrite=True)
    print(">> TREINO CONCLUÍDO")

# ======================================================================================================================
#                           FUNÇÕES VOLTADAS PARA A AVALIAÇÃO DO TREINAMENTO
# ======================================================================================================================

# ======================================================================================================================
# Accuracy balanceada (importante se dataset não for equilibrado)
def balanced_accuracy(cm):
    recalls = np.diag(cm) / (cm.sum(axis=1) + 1e-8)
    return np.mean(recalls)

# Precision, Recall e F1-score (macro), Sem usar sklearn (boa prática no JAX):
def precision_recall_f1(cm):
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "precision_macro": np.mean(precision),
        "recall_macro": np.mean(recall),
        "f1_macro": np.mean(f1),
    }

# Cálculo da matriz de confusão
def compute_confusion_matrix(labels, preds, num_classes):
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for l, p in zip(labels, preds):
        cm[l, p] += 1

    return cm

def save_confusion_matrix(output_dir, step, cm):
    """
    Salva a matriz de confusão como arquivo .npy
    """
    cm_dir = os.path.join(output_dir, "logs", "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)

    path = os.path.join(cm_dir, f"cm_step_{step:06d}.npy")
    np.save(path, cm)

@jax.jit
def eval_step_jit(state, batch):
    images, labels = batch

    logits = state.apply_fn(
        {"params": state.params},
        images,
        train=False,
    )
    metrics = compute_eval_metrics(logits, labels)
    return metrics, logits, labels

def evaluate_epoch(
    state,
    val_iter,
    num_batches=50,
    num_classes=3,
):
    """
    Avalia o modelo no conjunto de validação.

    - num_batches = None  → usa o dataset de validação
    - num_batches = N     → usa apenas N batches
    """

    losses = []
    accs = []
    all_preds = []
    all_labels = []

    batch_count = 0

    while True:

        # Caso num_batches seja definido, ao alcançar o total de iterações, o loop se encerra
        if num_batches is not None and batch_count >= num_batches:
            break
        # Caso contrário, tenta-se obter o próximo batch do dataset.
        try:
            batch_tf = next(val_iter)
        # Quando o dataset acaba? O iterador lança automaticamente StopIteration
        except StopIteration:
            break

        batch = tf_to_jax(batch_tf)
        metrics, logits, labels = eval_step_jit(state, batch)

        losses.append(float(metrics["loss"]))
        accs.append(float(metrics["accuracy"]))

        preds = jnp.argmax(logits, axis=-1)
        all_preds.append(preds)
        all_labels.append(labels)

        batch_count += 1

    if not losses:
        return None

    all_preds = np.asarray(jnp.concatenate(all_preds))
    all_labels = np.asarray(jnp.concatenate(all_labels))

    results = {
        "loss": float(np.mean(losses)),
        "accuracy": float(np.mean(accs))
    }

    if num_classes is not None:
        cm = compute_confusion_matrix(all_labels, all_preds, num_classes)
        results["confusion_matrix"] = cm
        results["balanced_accuracy"] = balanced_accuracy(cm)
        results.update(precision_recall_f1(cm))

    return results