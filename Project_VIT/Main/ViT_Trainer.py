# ======================================================================================================================
#                                              PACOTES E BIBLIOTECAS
# ======================================================================================================================
import os

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import jax.numpy as jnp
import optax
import json
import csv
import numpy as np

from flax.training import train_state, checkpoints
from flax.traverse_util import flatten_dict, unflatten_dict
from ViT_Model import VisionTransformer
from ViT_Utils import load_vit_npz


# ======================================================================================================================
#                                           FUNÇÃO GERADORA DE BATCHES
# ======================================================================================================================

def create_batches(x, y, batch_size, shuffle=True):
    indices = np.arange(len(x))

    if shuffle:
        np.random.shuffle(indices)

    for i in range(0, len(x), batch_size):
        batch_idx = indices[i:i + batch_size]
        yield x[batch_idx], y[batch_idx]


# ======================================================================================================================
#                               FUNÇÃO AUXILIAR PARA CONVERSÃO DE TIPOS NATIVOS
# ======================================================================================================================

def to_python_type(x):
    if hasattr(x, "item"):
        return x.item()
    return x


# ======================================================================================================================
#                           FUNÇÃO AUXILIAR PARA ARMAZENAR MÉTRICAS DE TREINO EM JSON
# ======================================================================================================================

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


# ======================================================================================================================
#                           FUNÇÃO AUXILIAR PARA ARMAZENAR MÉTRICAS NO DOCUMENTO .CSV
# ======================================================================================================================

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
#                    CRIAÇÃO DO OTIMIZADOR DE PRÉ-TREINO ANÁLOGO AO APRESENTADO NO ARTIGO
# ======================================================================================================================

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
#                                          CRIAÇÃO DO OTIMIZADOR DE FINE-TUNNING
# ======================================================================================================================

def create_optimizer_finetune(base_lr, total_steps, warmup_steps, params):
    # O artigo original utiliza SGD, contudo, AdamW opera melhor dataset reduzido, além de
    # o otimizador mais utilizado em transformers modernos
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=base_lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=1e-6,
    )

    # Optimizer real
    trainable_tx = optax.adamw(
        learning_rate=schedule,
        b1=0.9,
        b2=0.999,
        eps=1e-8,
        weight_decay=0.01,
    )

    # Frozen = gradiente zero
    frozen_tx = optax.set_to_zero()

    mask = create_frozen_mask(params)

    optimizer = optax.multi_transform(
        {
            "trainable": trainable_tx,
            "frozen": frozen_tx,
        },
        mask
    )

    return optimizer


# ======================================================================================================================
#                                   DEFINIÇÃO DAS FUNÇÕES DE PERDA E DAS MÉTRICAS
# ======================================================================================================================

'''def cross_entropy_loss(logits, labels):
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))'''


def cross_entropy_loss(logits, labels, smoothing=0.1):
    num_classes = logits.shape[-1]
    one_hot = jax.nn.one_hot(labels, num_classes)

    one_hot = (one_hot * (1.0 - smoothing) + smoothing / num_classes)
    log_probs = jax.nn.log_softmax(logits)

    return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))


def compute_train_metrics(logits, labels):
    # Métricas rápidas usadas durante o treinamento
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(
        (jnp.argmax(logits, axis=-1) == labels).astype(jnp.float32)
    )
    return {"loss": loss, "accuracy": accuracy}


def compute_eval_metrics(logits, labels):
    accuracy = jnp.mean(
        (jnp.argmax(logits, axis=-1) == labels).astype(jnp.float32)
    )
    log_probs = jax.nn.log_softmax(logits)  # cross-entropy
    loss = -jnp.mean(log_probs[jnp.arange(labels.size), labels])
    return {
        "loss": loss,
        "accuracy": accuracy
    }


# ======================================================================================================================
#                                                FUNÇÃO DE TREINO VIA JAX
# ======================================================================================================================

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


# =====================================================================================================================
#                                                      OBSERVAÇÃO:
#     O CÓDIGO ORIGINAL UTILIZA @PMAP PARA PARALELIZAR E DISTRIBUIR O TREINAMENTO ENTRE MÚLTIPLOS DISPOSITIVOS
#     COMO O TREINAMENTO FOI FEITO UTILIZANDO UMA ÚNICA GPU (DO MEU COMPUTADOR PESSOAL), FORAM APLICADAS AS
#     MODIFICAÇÕES NECESSÁRIAS PARA RETIRAR ESTE PARALELISMO (UTILIZANDO O DECORADOR JIT)
# =====================================================================================================================

# ======================================================================================================================
#                                   FUNÇÕES VOLTADAS PARA A AVALIAÇÃO DO TREINAMENTO
# ======================================================================================================================
def evaluate(state, x, y, batch_size):
    losses = []
    accuracies = []

    for images, labels in create_batches(
            x, y,
            batch_size=batch_size,
            shuffle=False):
        images = jnp.array(images)
        labels = jnp.array(labels)

        metrics, _, _ = eval_step_jit(state, (images, labels))

        losses.append(float(metrics["loss"]))
        accuracies.append(float(metrics["accuracy"]))

    return {
        "loss": np.mean(losses),
        "accuracy": np.mean(accuracies),
    }


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


# Avaliação da época
def evaluate_epoch(
        state,
        x_val,
        y_val,
        batch_size=32,
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

    for i, (images, labels) in enumerate(create_batches(x_val, y_val, batch_size)):

        if num_batches is not None and i >= num_batches:
            break

        images = jnp.array(images)
        labels = jnp.array(labels)

        metrics, logits, labels = eval_step_jit(state, (images, labels))

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

    correct = (all_preds == all_labels).sum()
    total = all_labels.shape[0]

    results = {
        "loss": float(np.mean(losses)),
        "accuracy": float(correct / total)
    }

    return results


# ======================================================================================================================
#                                            MÁSCARA PARA CONGELAR CAMADAS
# ======================================================================================================================

def create_frozen_mask(params):
    """
    Define quais parâmetros serão treináveis
    e quais serão congelados.
    """

    flat_params = flatten_dict(params)

    mask = {}

    for path, _ in flat_params.items():

        path_str = "/".join(path)

        # -----------------------------
        # CAMADAS TREINÁVEIS
        # -----------------------------
        if (
                "encoderblock_10" in path_str
                or "encoderblock_11" in path_str
                or "encoder_norm" in path_str
                or "head" in path_str
        ):

            mask[path] = "trainable"

        # -----------------------------
        # RESTANTE CONGELADO
        # -----------------------------
        else:
            mask[path] = "frozen"

    return unflatten_dict(mask)


# ======================================================================================================================
#                                            LOOP PRINCIPAL DE TREINAMENTO
# ======================================================================================================================

def train_vit(
        x_train,
        y_train,
        x_val,
        y_val,
        output_dir: str,
        patches=(16, 16),
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_dim=3072,
        num_classes=3,
        batch_size=32,
        total_steps=0,
        warmup_steps=1000,
        base_lr=2e-4,
        mode="finetune",
        weights_path=None,
        steps_per_epoch=0,
        steps_val=0,
        epochs=0
):
    # Criação do modelo e definição das configurações
    transformer_cfg = dict(
        num_layers=depth,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        dropout_rate=0.2,
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
    dummy = jnp.ones((1, 224, 224, 3), dtype=jnp.float32)
    # Inicializa parâmetros com pesos aleatórios (default do Flax)
    params = model.init({"params": rng, "dropout": rng}, dummy, train=True)["params"]

    if mode == "finetune":
        # Carregamento dos pesos pré-treinados
        pretrained_path = weights_path
        params = load_vit_npz(params, pretrained_path)
        print(">> PESOS PRE-TREINADOS CARREGADOS COM SUCESSO!")

    flat_params = flatten_dict(params)
    for path in flat_params.keys():
        print("/".join(path))

    # Escolha do otimizador
    if mode == "pretrain":
        optimizer = create_optimizer_pretrain(base_lr, warmup_steps, total_steps)
    else:
        optimizer = create_optimizer_finetune(
            base_lr=base_lr,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            params=params
        )

        print("\n==============================")
        print("FINE-TUNING PARCIAL ATIVADO")
        print("==============================")
        print("Treinando:")
        print("- encoderblock_8 → encoderblock_11")
        print("- encoder_norm")
        print("- head")
        print("==============================\n")

    # Criação do TrainState: encapsula params, apply_fn e otimizador (tx)
    # num estado que facilita updates e checkpointing.
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )

    print("=====================================================")
    print("           INICIANDO LOOP DE TREINAMENTO             ")
    print("=====================================================")

    for epoch in range(epochs):

        # Buffers para métricas de treino (nível epoch)
        train_losses = []
        train_accs = []

        train_batches = create_batches(
            x_train,
            y_train,
            batch_size=batch_size
        )

        for step, (images, labels) in enumerate(train_batches):
            images = jnp.array(images)
            labels = jnp.array(labels)

            # gerar RNG novo para dropout
            rng, dropout_rng = jax.random.split(rng)
            # Apresenta apenas o batch atual e as métricas locais daquele batch
            state, metrics = train_step_jit(state, (images, labels), dropout_rng)

            # Acumula métricas do treino
            train_losses.append(float(metrics["loss"]))
            train_accs.append(float(metrics["accuracy"]))

        # Médias da época
        train_loss = np.mean(train_losses)
        train_acc = np.mean(train_accs)

        val_metrics = evaluate(
            state,
            x_val,
            y_val,
            batch_size=batch_size
        )

        log = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
        }

        append_metrics_csv(output_dir, log)

        print(
            f"[EPOCH {epoch}] "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        if epoch % 10 == 0:
            checkpoints.save_checkpoint(
                ckpt_dir=os.path.join(output_dir, "checkpoints"),
                target=state,
                step=epoch,
                prefix="epoch_",
                overwrite=False,
                keep=5
            )

        train_losses.clear()
        train_accs.clear()

    print("=====================================================")
    print("             AVALIAÇÃO FINAL DO MODELO               ")
    print("=====================================================")

    final_results = evaluate(
        state,
        x_val,
        y_val,
        batch_size=batch_size
    )

    final_log = {
        "val_loss_final": float(final_results["loss"]),
        "val_accuracy_final": float(final_results["accuracy"]),
    }

    print(final_log)
