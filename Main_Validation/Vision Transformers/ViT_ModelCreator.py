# VisionTransformer_pure.py
# Estilo 1 (fiel ao repositório Google Research)
# Dividido em seções com explicações inline.

from typing import Any, Callable, Optional, Tuple, Type
import flax.linen as nn
import jax.numpy as jnp

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any


# ======================================================================================
# 1) UTILITÁRIOS / CAMADAS SIMPLES
# - IdentityLayer: mantém API consistente quando não existe a representação 'pre_logits'.
# - AddPositionEmbs: embeddings posicionais aprendíveis (mesmo design do repositório).
# ======================================================================================

class IdentityLayer(nn.Module):
    """Camada identidade — usada quando não há 'pre_logits'."""

    @nn.compact
    def __call__(self, x):
        return x


class AddPositionEmbs(nn.Module):
    """
    Adiciona embeddings posicionais aprendíveis.
    - posemb_init: inicializador para o parâmetro pos_embedding.
    - nota: vem do repositório oficial (positional embeddings aprendíveis).
    """
    posemb_init: Callable[[PRNGKey, Shape, Dtype], Array]
    dtype: Dtype = jnp.bfloat16
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs):
        # inputs: (batch, seq_len, emb_dim)
        assert inputs.ndim == 3, f"inputs.ndim must be 3, got {inputs.ndim}"
        pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])

        pe = self.param("pos_embedding", self.posemb_init, pos_emb_shape, self.param_dtype)
        pe = pe.astype(self.dtype)
        return inputs + pe


# ======================================================================================
# 2) BLOCO MLP (Feed-Forward) do Transformer
# - Implementa: Dense -> GELU -> Dropout -> Dense -> Dropout
# - Nomes dos submódulos seguem convenção (útil para checkpointing).
# ======================================================================================

class MlpBlock(nn.Module):
    """
    Bloco MLP do Transformer.
    Parâmetros:
      - mlp_dim: dimensão intermediária (D_ff)
      - dropout_rate: taxa de dropout
      - out_dim: opcional (normalmente igual ao hidden_size)
    """
    mlp_dim: int
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32
    out_dim: Optional[int] = None
    dropout_rate: float = 0.1
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.xavier_uniform()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.normal(stddev=1e-6)

    @nn.compact
    def __call__(self, inputs, *, deterministic: bool):
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim

        # Dense (proj para mlp_dim)
        x = nn.Dense(
            features=self.mlp_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="Dense_0")(inputs)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

        # Dense (proj de volta)
        x = nn.Dense(
            features=actual_out_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="Dense_1")(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

        return x


# ======================================================================================
# 3) BLOCO DE ENCODER (Attention + MLP + Residuals)
# - Estrutura padrão: LayerNorm -> MHA -> Dropout -> Residual -> LayerNorm -> MLP -> Residual
# - Usamos nomes de sub-módulos que espelham o repositório oficial para compatibilidade.
# ======================================================================================

class Encoder1DBlock(nn.Module):
    """
    Um único bloco Encoder do Transformer.
    Parâmetros:
      - mlp_dim, num_heads, dropout_rate, attention_dropout_rate
    O nome dos submódulos (ex: 'LayerNorm_0', 'MultiHeadDotProductAttention_1', 'MlpBlock_3') é
    escolhido para coincidir com o estilo de nomenclatura do repositório oficial, facilitando o mapeamento
    de checkpoints.
    """
    mlp_dim: int
    num_heads: int
    dtype: Dtype = jnp.float32
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, inputs, *, deterministic: bool):
        assert inputs.ndim == 3, f"Expected (batch, seq, hidden) got {inputs.shape}"

        # -------------------------
        # Attention block (pre-LN)
        # -------------------------
        x = nn.LayerNorm(dtype=self.dtype, name="LayerNorm_0")(inputs)  # nome deliberado
        x = nn.MultiHeadDotProductAttention(
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            broadcast_dropout=False,
            deterministic=deterministic,
            dropout_rate=self.attention_dropout_rate,
            num_heads=self.num_heads,
            name="MultiHeadDotProductAttention_1")(x, x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = x + inputs  # residual

        # -------------------------
        # MLP block (pre-LN)
        # -------------------------
        y = nn.LayerNorm(dtype=self.dtype, name="LayerNorm_2")(
            x)  # nome deliberado ("LayerNorm_2" conforme checkpoint warnings)
        y = MlpBlock(
            mlp_dim=self.mlp_dim,
            dtype=self.dtype,
            dropout_rate=self.dropout_rate,
            name="MlpBlock_3")(y, deterministic=deterministic)

        out = x + y  # final residual
        return out


# ======================================================================================
# 4) ENCODER (empilha L blocos)
# - Adiciona positional embeddings (se configurado)
# - Faz dropout inicial conforme especificado
# ======================================================================================

class Encoder(nn.Module):
    """
    Empilha num_layers blocos Encoder1DBlock.
    A opção add_position_embedding controla a adição de embeddings posicionais aprendíveis.
    """
    num_layers: int
    mlp_dim: int
    num_heads: int
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    add_position_embedding: bool = True
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, *, train: bool):
        # x : (batch, seq_len, emb)
        assert x.ndim == 3

        if self.add_position_embedding:
            x = AddPositionEmbs(
                posemb_init=nn.initializers.normal(stddev=0.02),
                dtype=self.dtype,
                name="posembed_input")(x)
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

        # empilha os blocos
        for lyr in range(self.num_layers):
            x = Encoder1DBlock(
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dtype=self.dtype,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                name=f"encoderblock_{lyr}")(x, deterministic=not train)

        # final layer norm (nome 'encoder_norm' para facilitar checkpoints)
        encoded = nn.LayerNorm(dtype=self.dtype, name="encoder_norm")(x)
        return encoded


# ======================================================================================
# 5) VISION TRANSFORMER (ViT) — implementação "pura"
# - Patch embedding usando Conv2D (kernel = patch_size, stride = patch_size)
# - Flatten grid -> sequence
# - (opcional) adicionar cls token
# - Executar encoder
# - Selecionar pooling/classifier (token ou gap)
# - (opcional) representation_size + pre_logits
# - Cabeça final (head) com inicialização zero (como no paper)
# ======================================================================================

class VisionTransformer(nn.Module):
    """
    VisionTransformer (puro) - estilo e nomes alinhados ao repositório Google Research.
    Parâmetros principais:
      - num_classes: número de classes finais
      - patches: objeto com atributo 'size' (ex: patches.size = (16,16))
      - transformer: dicionário com (num_layers, mlp_dim, num_heads, ...)
      - hidden_size: dimensão do embedding (D)
      - representation_size: dimensão adicional pré-head (opcional)
      - classifier: 'token' (CLS) ou 'gap'
    """
    num_classes: int
    patches: Any
    transformer: Any
    hidden_size: int
    representation_size: Optional[int] = None
    classifier: str = "token"
    head_bias_init: float = 0.0
    encoder: Type[nn.Module] = Encoder
    model_name: Optional[str] = None

    # Define o mixed precision
    dtype: Any = jnp.bfloat16
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, inputs, *, train: bool):
        # inputs: (batch, H, W, C)
        x = inputs

        # -------------------------
        # Patch embedding (Conv with kernel=patch_size and stride=patch_size)
        # -------------------------
        x = nn.Conv(
            features=self.hidden_size,
            kernel_size=self.patches.size,
            strides=self.patches.size,
            padding='VALID',
            name='embedding',
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(x)

        # x now (batch, h, w, c)
        n, h, w, c = x.shape
        if self.transformer is not None:
            # flatten spatial grid to sequence (batch, num_patches, hidden)
            x = jnp.reshape(x, [n, h * w, c])

            # Add class token if requested
            if self.classifier in ["token", "token_unpooled"]:
                cls = self.param("cls", nn.initializers.zeros, (1, 1, c))
                cls = jnp.tile(cls, [n, 1, 1])
                x = jnp.concatenate([cls, x], axis=1)

            # call encoder (Transformer)
            x = self.encoder(name="Transformer", **self.transformer)(x, train=train)

        # -------------------------
        # Pooling / classifier selection
        # -------------------------
        if self.classifier == "token":
            # use cls token
            x = x[:, 0]
        elif self.classifier == "gap":
            x = jnp.mean(x, axis=1)
        elif self.classifier in ["unpooled", "token_unpooled"]:
            # leave as-is (caller handles)
            pass
        else:
            raise ValueError(f"Invalid classifier='{self.classifier}'")

        # -------------------------
        # Optional representation (pre_logits)
        # -------------------------
        if self.representation_size is not None:
            x = nn.Dense(features=self.representation_size, name="pre_logits")(x)
            x = nn.tanh(x)
        else:
            x = IdentityLayer(name="pre_logits")(x)

        # -------------------------
        # Final head (class projection)
        # - Initialize kernel and bias to zero per the paper (helps fine-tuning).
        # -------------------------
        if self.num_classes:

            x = nn.Dense(
                features=self.num_classes,
                name='head',
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.constant(self.head_bias_init)
            )(x)

        return x
