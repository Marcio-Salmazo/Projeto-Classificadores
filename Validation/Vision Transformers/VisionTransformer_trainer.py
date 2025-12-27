"""
    Arquivo responsável pelo treino + fine-tuning do Vision Transformer puro,
    seguindo fielmente o paper "An Image is Worth 16x16 Words".
    Aqui são definidos:
        * perda (cross-entropy) e métricas;
        * train_step e eval_step escritos em JAX e decorados com jax.pmap para execução distribuída;
        * função train_vit(...) que monta dataset, modelo, inicializa parâmetros, cria otimizador,
          replica estado para dispositivos e faz o loop de treinamento,
          salvando checkpoints e chamando avaliação periódica;
        * função evaluate_vit(...) que avalia o modelo em batches do conjunto de validação.
"""
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
from Validation.Process_ImageNet import load_tfrecords
from ViT_ModelCreator import VisionTransformer
from ViT_CheckpointLoader import load_vit_npz


# ======================================================================================================================
# AUXILIAR DE LOGS (PARA TREINO E VALIDAÇÃO)

def save_metrics_json(output_dir, step, metrics):
    """Salva métricas em JSON."""
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    filepath = os.path.join(logs_dir, f"step_{step:06d}.json")
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=4)


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
# CONVERTE AS IMAGENS E LABELS DO TENSORFLOW PARA JAX ARRAYS

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
    # Métricas completas usadas na validação (top-1 e top-5)

    top1 = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    top5 = jnp.mean(
        jnp.any(
            jnp.argsort(logits, axis=-1)[:, -5:] == labels[:, None],
            axis=-1
        )
    )
    # cross-entropy
    log_probs = jax.nn.log_softmax(logits)
    loss = -jnp.mean(log_probs[jnp.arange(labels.size), labels])
    return {
        "loss": loss,
        "top1": top1,
        "top5": top5,
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
        tfrecord_train_dir: str,
        tfrecord_val_dir: str,
        output_dir: str,
        patches=(16, 16),
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_dim=3072,
        num_classes=1000,
        batch_size=32, # ORIGINAL = 256
        total_steps=100000,
        warmup_steps=10000,
        base_lr=2e-4,
        mode="finetune"
    ):
    """
        Carregamento dos datasets:
            * load_tfrecords retorna um tf.data.Dataset com batches;
            * tf_to_jax converte cada batch para arrays JAX.

        Importante: O loader precisa retornar batches com drop_remainder=True
        para que pmap receba batches divisíveis por n_devices (Caso seja utilizado).
        O seu data_loader já fazia ds.batch(..., drop_remainder=True).
    """

    train_ds = load_tfrecords(tfrecord_train_dir, batch_size, train=True)
    val_ds = load_tfrecords(tfrecord_val_dir, batch_size, train=False)

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
        # ------------------------------------------------------------------------------------------------------
        #                       DEFINIR AQUI O CAMINHO PARA OS PESOS PRÉ-TREINADOS
        # ------------------------------------------------------------------------------------------------------
        pretrained_path = (r"C:/Users/marci_plgx30x/Desktop/Arquivos/Projetos do Mestrado (Git)/"
                           r"Projeto-Classificadores/Datasets/Pre_Trained/imagenet21k_ViT-B_16.npz")

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

    # Loop de treinamento
    for step in range(total_steps):

        batch_tf = next(train_iter)
        batch = tf_to_jax(batch_tf)

        # gerar RNG novo para dropout
        rng, dropout_rng = jax.random.split(rng)
        # JIT treino
        state, metrics = train_step_jit(state, batch, dropout_rng)

        # Logging ocasional
        if step % 100 == 0:
            train_log = {
                "step": step,
                "timestamp": datetime.now().isoformat(),
                "train_loss": float(metrics["loss"]),
                "train_accuracy": float(metrics["accuracy"]),
            }

            # Salvar JSON do step
            save_metrics_json(output_dir, step, train_log)

            # Adicionar ao CSV
            append_metrics_csv(output_dir, train_log)

            # Print no console
            print(f"[{step:06d}/{total_steps}] "
                  f"loss={train_log['train_loss']:.4f}, "
                  f"acc={train_log['train_accuracy']:.4f}")

        # Checkpoint
        if step % 1000 == 0 and step > 0:
            checkpoints.save_checkpoint(output_dir, state, step, overwrite=True)

        # Avaliação ocasional
        if step % 5000 == 0 and step > 0:
            val_iter = iter(val_ds)  # reset seguro
            val = evaluate_vit_from_iterator(state, val_iter, num_batches=25)

            val_log = {
                "step": step,
                "timestamp": datetime.now().isoformat(),
                "val_loss": float(val["loss"]),
                "val_top1": float(val["top1"]),
                "val_top5": float(val["top5"]),
            }

            # Salvar JSON para este step
            save_metrics_json(output_dir, step, val_log)

            # Escrever linha no CSV
            append_metrics_csv(output_dir, val_log)

            print("\nVALIDAÇÃO")
            print(f"VAL >> loss={val_log['val_loss']:.4f}, "
                  f"top1={val_log['val_top1']:.4f}, "
                  f"top5={val_log['val_top5']:.4f}\n")

    # salvar final
    checkpoints.save_checkpoint(output_dir, state, total_steps, overwrite=True)
    print("Treino concluído.")


# ======================================================================================================================
# AVALIAÇÃO DO TREINAMENTO

@jax.jit
def eval_step_jit(state, batch):
    images, labels = batch
    logits = state.apply_fn(
        {"params": state.params},
        images,
        train=False,
    )
    metrics = compute_eval_metrics(logits, labels)
    return metrics


def evaluate_vit(
        checkpoint_dir: str,
        tfrecord_val_dir: str,
        batch_size: int = 256,
        num_batches: int = 50,
        patches=(16, 16),
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_dim=3072,
        num_classes=1000,
        image_size=224
):
    """
        Avaliação completa: carrega dataset, modelo, checkpoint e roda a avaliação.
    """

    # ---- carregar dataset ----
    val_ds = load_tfrecords(
        tfrecord_val_dir,
        batch_size=batch_size,
        train=False,
        image_size=image_size
    )
    val_iter = iter(val_ds)

    # ---- construir modelo ----
    class PatchCfg:
        pass

    pc = PatchCfg()
    pc.size = patches

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

    # ---- init dummy ----
    dummy = jnp.zeros([1, image_size, image_size, 3], dtype=jnp.float32)
    rng = jax.random.PRNGKey(0)
    _ = model.init({"params": rng}, dummy, train=False)

    # ---- carregar checkpoint ----
    state = checkpoints.restore_checkpoint(checkpoint_dir, target=None)
    if state is None:
        raise ValueError("Nenhum checkpoint encontrado.")

    # ---- avaliação ----
    metrics_list = []

    for _ in range(num_batches):
        try:
            batch_tf = next(val_iter)
        except StopIteration:
            break

        batch = tf_to_jax(batch_tf)
        metrics = eval_step_jit(state, batch)
        metrics_list.append(metrics)

    if not metrics_list:
        return {"loss": float("nan"), "top1": float("nan"), "top5": float("nan")}

    loss = float(jnp.mean(jnp.array([m["loss"] for m in metrics_list])))
    top1 = float(jnp.mean(jnp.array([m["top1"] for m in metrics_list])))
    top5 = float(jnp.mean(jnp.array([m["top5"] for m in metrics_list])))

    print("\n=== RESULTADOS ===")
    print(f"Val Loss : {loss:.4f}")
    print(f"Top-1 Acc: {top1 * 100:.2f}%")
    print(f"Top-5 Acc: {top5 * 100:.2f}%")

    return {"loss": loss, "top1": top1, "top5": top5}

def evaluate_vit_from_iterator(state, val_iter, num_batches=50):
    """
         Essa função usa o dataset já carregado
         Não toca em load_tfrecords
         Nunca mais gera o erro do step 5000
    """

    metrics_list = []

    for _ in range(num_batches):
        try:
            batch_tf = next(val_iter)
        except StopIteration:
            break

        batch = tf_to_jax(batch_tf)
        metrics = eval_step_jit(state, batch)
        metrics_list.append(metrics)

    if not metrics_list:
        return {"loss": float("nan"), "top1": float("nan"), "top5": float("nan")}

    loss = float(jnp.mean(jnp.array([m["loss"] for m in metrics_list])))
    top1 = float(jnp.mean(jnp.array([m["top1"] for m in metrics_list])))
    top5 = float(jnp.mean(jnp.array([m["top5"] for m in metrics_list])))

    return {"loss": loss, "top1": top1, "top5": top5}