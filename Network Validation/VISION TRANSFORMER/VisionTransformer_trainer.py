"""
train_vit.py
------------
Treino + fine-tuning do Vision Transformer puro,
seguindo fielmente o paper "An Image is Worth 16x16 Words".

Dependências:
 - models_vit_pure.py
 - data_loader.py
 - JAX + Flax + Optax
"""

import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import optax
import flax
from flax.training import train_state, checkpoints
from flax.core import freeze, unfreeze

from VisionTransformer_pure import VisionTransformer
from VisionTransformer_ImageNet import load_tfrecords, tf_to_jax


# ------------------------------------------------------------
# 1) Otimizadores idênticos ao paper
# ------------------------------------------------------------

def create_optimizer_pretrain(base_lr, warmup_steps, total_steps):
    """
    Adam + warmup + cosine decay (como no paper).
    """
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=base_lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=0.0
    )
    optimizer = optax.adam(
        learning_rate=schedule,
        b1=0.9, b2=0.999, eps=1e-8
    )
    return optimizer


def create_optimizer_finetune(base_lr):
    """
    SGD + momentum (0.9), sem weight decay
    conforme fine-tuning do paper.
    """
    optimizer = optax.sgd(
        learning_rate=base_lr,
        momentum=0.9,
        nesterov=False
    )
    return optimizer


# ------------------------------------------------------------
# 2) Loss (cross-entropy)
# ------------------------------------------------------------

def cross_entropy_loss(logits, labels):
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.mean(jnp.sum(one_hot * optax.log_softmax(logits), axis=-1))


def compute_metrics(logits, labels):
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return {"loss": loss, "accuracy": accuracy}


# ------------------------------------------------------------
# 3) Função de treino JAX (pmap-ready)
# ------------------------------------------------------------

@partial(jax.pmap, axis_name="batch")
def train_step(state, batch):
    images, labels = batch
    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params},
            images,
            train=True,
            rngs={"dropout": jax.random.PRNGKey(0)}
        )
        loss = cross_entropy_loss(logits, labels)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    grads = jax.lax.pmean(grads, axis_name="batch")

    new_state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits, labels)
    metrics = jax.lax.pmean(metrics, axis_name="batch")
    return new_state, metrics


@partial(jax.pmap, axis_name="batch")
def eval_step(state, batch):
    images, labels = batch
    logits = state.apply_fn(
        {"params": state.params},
        images,
        train=False,
        rngs={"dropout": jax.random.PRNGKey(0)}
    )
    metrics = compute_metrics(logits, labels)
    metrics = jax.lax.pmean(metrics, axis_name="batch")
    return metrics


# ------------------------------------------------------------
# 4) Loop principal de treino
# ------------------------------------------------------------

def train_vit(
    tfrecord_train_dir: str,
    tfrecord_val_dir: str,
    output_dir: str,
    patches=(16,16),
    hidden_size=768,
    depth=12,
    num_heads=12,
    mlp_dim=3072,
    num_classes=1000,
    batch_size=256,
    total_steps=100000,
    warmup_steps=10000,
    base_lr=2e-4,
    mode="pretrain"   # ou "finetune"
):
    """
    Executa treino ou fine-tuning.
    """

    # 1) Criar dataset
    train_ds = load_tfrecords(tfrecord_train_dir, batch_size, train=True)
    val_ds   = load_tfrecords(tfrecord_val_dir,   batch_size, train=False)

    train_iter = iter(train_ds)
    val_iter   = iter(val_ds)

    # 2) Criar modelo
    transformer_cfg = dict(
        num_layers=depth,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
        add_position_embedding=True,
    )

    class PatchConfig: pass
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

    # 3) Inicializar parâmetros
    rng = jax.random.PRNGKey(0)
    dummy = jnp.ones([1,224,224,3], dtype=jnp.float32)
    params = model.init({"params":rng,"dropout":rng}, dummy, train=True)["params"]

    # 4) Otimizador
    if mode == "pretrain":
        optimizer = create_optimizer_pretrain(base_lr, warmup_steps, total_steps)
    else:
        optimizer = create_optimizer_finetune(base_lr)

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )

    # Replicar para multi-GPU
    state = jax.device_put_replicated(state, jax.local_devices())

    # 5) Loop de treino
    for step in range(total_steps):

        batch_tf = next(train_iter)
        batch = tf_to_jax(batch_tf)

        # shard: (n_devices, batch_per_dev, ...)
        batch = jax.tree.map(
            lambda x: x.reshape((jax.local_device_count(), -1) + x.shape[1:]),
            batch
        )

        state, metrics = train_step(state, batch)

        if step % 100 == 0:
            print(f"[{step}/{total_steps}] loss={metrics['loss'][0]:.4f}, acc={metrics['accuracy'][0]:.4f}")

        if step % 1000 == 0 and step > 0:
            checkpoints.save_checkpoint(output_dir, state, step, overwrite=True)

        if step % 5000 == 0:
            val_metrics = evaluate_vit(state, val_iter)
            print(f"VAL: loss={val_metrics['loss']:.4f}, acc={val_metrics['accuracy']:.4f}")

    # salvar final
    checkpoints.save_checkpoint(output_dir, state, total_steps, overwrite=True)
    print("Treino concluído.")


# ------------------------------------------------------------
# Avaliação
# ------------------------------------------------------------

def evaluate_vit(state, val_iter, num_batches=50):
    metrics_list = []
    for _ in range(num_batches):
        try:
            batch_tf = next(val_iter)
        except StopIteration:
            break
        batch = tf_to_jax(batch_tf)
        batch = jax.tree.map(
            lambda x: x.reshape((jax.local_device_count(), -1) + x.shape[1:]),
            batch
        )
        metrics = eval_step(state, batch)
        metrics_list.append(metrics)

    loss = jnp.mean(jnp.stack([m["loss"][0] for m in metrics_list]))
    acc  = jnp.mean(jnp.stack([m["accuracy"][0] for m in metrics_list]))
    return {"loss": float(loss), "accuracy": float(acc)}


# ------------------------------------------------------------
# Execução direta
# ------------------------------------------------------------

if __name__ == "__main__":
    train_vit(
        tfrecord_train_dir="D:/datasets/imagenet_tfrecords/train",
        tfrecord_val_dir="D:/datasets/imagenet_tfrecords/validation",
        output_dir="D:/outputs/vit_pretrain",
        mode="pretrain",
        batch_size=256,
        total_steps=100000,
        warmup_steps=10000,
        base_lr=2e-4
    )