"""
evaluate_vit.py
---------------
Avaliação do Vision Transformer no conjunto de validação ImageNet,
seguindo fielmente o paper ViT e compatível com o modelo puro.

Requer:
 - models_vit_pure.py
 - data_loader.py
 - checkpoint gerado por train_vit.py
"""

import os
import jax
import jax.numpy as jnp
from flax.training import checkpoints
from functools import partial

from models_vit_pure import VisionTransformer
from data_loader import load_tfrecords, tf_to_jax


# ------------------------------------------------------------
# Métricas (top-1 e top-5 accuracy)
# ------------------------------------------------------------

def compute_metrics(logits, labels):
    top1 = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    top5 = jnp.mean(jnp.any(
        jnp.argsort(logits, axis=-1)[:, -5:] == labels[:, None], axis=-1))

    loss = -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(labels.size), labels])

    return {
        "loss": loss,
        "top1": top1,
        "top5": top5
    }


# ------------------------------------------------------------
# Passo de avaliação pmap-ready
# ------------------------------------------------------------

@partial(jax.pmap, axis_name="batch")
def eval_step(state, batch):
    images, labels = batch
    logits = state.apply_fn({"params": state.params}, images, train=False)
    metrics = compute_metrics(logits, labels)
    metrics = jax.lax.pmean(metrics, axis_name="batch")
    return metrics


# ------------------------------------------------------------
# Função principal de avaliação
# ------------------------------------------------------------

def evaluate_vit(
    checkpoint_dir: str,
    tfrecord_val_dir: str,
    batch_size: int = 256,
    patches=(16, 16),
    hidden_size=768,
    depth=12,
    num_heads=12,
    mlp_dim=3072,
    num_classes=1000,
    image_size=224,
    num_batches=200
):
    """
    Executa avaliação do modelo ViT em ImageNet validation.
    """

    # Criar dataset
    val_ds = load_tfrecords(tfrecord_val_dir, batch_size, train=False, image_size=image_size)
    val_iter = iter(val_ds)

    # Criar modelo
    class PatchCfg: pass
    pc = PatchCfg(); pc.size = patches

    transformer_cfg = dict(
        num_layers=depth,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        dropout_rate=0.0,
        attention_dropout_rate=0.0,
        add_position_embedding=True,
    )

    model = VisionTransformer(
        num_classes=num_classes,
        patches=pc,
        transformer=transformer_cfg,
        hidden_size=hidden_size,
        representation_size=None,
        classifier="token"
    )

    # Dummy init
    dummy = jnp.zeros([1, image_size, image_size, 3], dtype=jnp.float32)
    rng = jax.random.PRNGKey(0)
    variables = model.init({"params": rng}, dummy, train=False)

    # Carregar checkpoint
    print(f"Carregando checkpoint de: {checkpoint_dir}")
    state = checkpoints.restore_checkpoint(checkpoint_dir, target=None)
    if state is None:
        raise ValueError("Nenhum checkpoint encontrado.")

    # Replicar para múltiplas GPUs
    state = jax.device_put_replicated(state, jax.local_devices())

    # Loop de avaliação
    all_metrics = []
    print("Iniciando avaliação...")

    for i in range(num_batches):
        try:
            batch_tf = next(val_iter)
        except StopIteration:
            break

        batch = tf_to_jax(batch_tf)

        # Separar batch entre GPUs
        batch = jax.tree_map(
            lambda x: x.reshape((jax.local_device_count(), -1) + x.shape[1:]),
            batch
        )

        metrics = eval_step(state, batch)
        all_metrics.append(metrics)

        if i % 20 == 0:
            print(f"Lote {i}/{num_batches}")

    # Agregar métricas
    loss  = float(jnp.mean(jnp.stack([m["loss"][0]  for m in all_metrics])))
    top1  = float(jnp.mean(jnp.stack([m["top1"][0]  for m in all_metrics])))
    top5  = float(jnp.mean(jnp.stack([m["top5"][0]  for m in all_metrics])))

    print("=== RESULTADOS ===")
    print(f"Val Loss : {loss:.4f}")
    print(f"Top-1 Acc: {top1*100:.2f}%")
    print(f"Top-5 Acc: {top5*100:.2f}%")

    return loss, top1, top5