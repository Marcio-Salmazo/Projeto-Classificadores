# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Optional, Tuple, Type

'''
    Usamos Flax porque:
        * Flax é o framework oficial recomendado pela Google Research 
          para implementação do ViT em JAX.
        * jax.numpy é o backend principal de tensores.
        
    No artigo, temos: “We provide an implementation based on JAX and Flax.”
'''
import flax.linen as nn
import jax.numpy as jnp

# Inicialização de variáveis globais
Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any

'''
    OBSERVAÇÃO: @nn.compact é um decorador na biblioteca Flax para JAX, 
    que permite definir módulos e parâmetros de rede neural diretamente 
    dentro do método __call__ da classe do módulo. 
    
    O decorador simplifica a definição do modelo, permitindo que a 
    arquitetura da rede seja expressa de forma mais concisa e legível, 
    semelhante à notação matemática ou a como os dados fluem através das camadas. 
'''


# ======================================================================================================================

class IdentityLayer(nn.Module):
    """
        Uma camada que simplesmente retorna o que recebe.
        Ela existe porque No ViT, há duas possibilidades antes da cabeça final:
            1 - Ter pre_logits (hidden representation)
            2 - Não ter pre_logits, usando a representação da última camada diretamente

        Portanto, essa camada serve para:
            * Manter consistência da API
            * Dar nome aos tensores no checkpoint
    """

    @nn.compact
    def __call__(self, x):
        return x


# ======================================================================================================================

class AddPositionEmbs(nn.Module):
    # Adição dos embeddings posicionais aprendidos
    # posemb_init: positional embedding initializer.
    posemb_init: Callable[[PRNGKey, Shape, Dtype], Array]
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs):
        """
            Applies the AddPositionEmbs module.
            Args:
              inputs: Inputs to the layer.

            Returns:
              Output tensor with shape `(bs, timesteps, in_dim)`.
        """
        # Verifica se inputs possui tem 3 dimensões, retornando mensagem em caso negativo
        assert inputs.ndim == 3, ('Number of dimensions should be 3, but it is: %d' % inputs.ndim)

        '''
            * Cria parâmetro pos_embedding, com as Dimensões: (1, num_patches+1, hidden_dim)
            * Soma ao embedding dos patches
            * Converte dtype para compatibilidade
        '''
        pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
        pe = self.param('pos_embedding', self.posemb_init, pos_emb_shape, self.param_dtype)
        pe = pe.astype(inputs.dtype)
        return inputs + pe


# ======================================================================================================================

class MlpBlock(nn.Module):
    """
        Implementa à parte MLP da camada Transformer. O artigo define essa estrutura da seguinte forma:
        “Each encoder block consists of a Multi-head Self-Attention (MSA) layer,
        followed by an MLP block with a GELU activation.”
    """

    mlp_dim: int
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32
    out_dim: Optional[int] = None
    dropout_rate: float = 0.1
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.xavier_uniform()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.normal(stddev=1e-6)

    @nn.compact
    def __call__(self, inputs, *, deterministic):
        # ================================== #
        # ENTENDER MELHOR ESSA IMPLEMENTAÇÃO #
        # ================================== #

        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim

        # Primeira camada Densa
        x = nn.Dense(
            features=self.mlp_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)(inputs)
        # Ativação com GELU seguido por Dropout
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

        # Segunda camada Densa
        output = nn.Dense(
            features=actual_out_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)(x)
        output = nn.Dropout(rate=self.dropout_rate)(output, deterministic=deterministic)

        # Retorna nova projeção ao hidden_size
        return output


# ======================================================================================================================

class Encoder1DBlock(nn.Module):
    """
        Uma camada do Transformer Encoder.
        Atributos:

            * mlp_dim: dimensão do MLP no bloco de atenção.
            * dtype: o tipo de dado da computação (padrão: float32).
            * dropout_rate: taxa de dropout.
            * attention_dropout_rate: dropout para as cabeças de atenção.
            * deterministic: bool, determinístico ou não (para aplicar dropout).
            * num_heads: número de cabeças em nn.MultiHeadDotProductAttention
    """

    mlp_dim: int
    num_heads: int
    dtype: Dtype = jnp.float32
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, inputs, *, deterministic):
        """
            Aplica o módulo Encoder1DBlock.
            Args:
              inputs: Dados de entrada para a camada.
              deterministic: bool, determinístico ou não (para aplicar dropout). Caso True, não aplica Dropout

            O artigo diz: “We use the standard Transformer Encoder architecture from Vaswani et al., 2017.”
            Essa estrutura padrão consiste em:

                * LayerNorm
                * Multi-Head Self-Attention
                * Residual
                * LayerNorm
                * MLP
                * Residual
        """

        # Bloco de atenção:
        assert inputs.ndim == 3, f'Expected (batch, seq, hidden) got {inputs.shape}'  # Verifica as dimensões de entrada

        # LayerNorm antes da atenção
        x = nn.LayerNorm(dtype=self.dtype)(inputs)
        # Multi-head attention
        x = nn.MultiHeadDotProductAttention(
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            broadcast_dropout=False,
            deterministic=deterministic,
            dropout_rate=self.attention_dropout_rate,
            num_heads=self.num_heads)(
            x, x)
        # Dropout + Residual
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = x + inputs

        # LayerNorm antes da MLP
        y = nn.LayerNorm(dtype=self.dtype)(x)
        # Chama a função que cria o bloco MLP
        y = MlpBlock(
            mlp_dim=self.mlp_dim, dtype=self.dtype, dropout_rate=self.dropout_rate)(
            y, deterministic=deterministic)

        # Residual final
        return x + y


# ======================================================================================================================

class Encoder(nn.Module):
    """
        Encoder do Modelo Transformer para tradução de sequência para sequência.
        Responsável por empilhar L camadas de Encoder1DBlock
        Atributos:
            * num_layers: número de camadas
            * mlp_dim: dimensão do MLP no topo do bloco de atenção
            * num_heads: Número de cabeças em nn.MultiHeadDotProductAttention
            * dropout_rate: taxa de dropout.
            * attention_dropout_rate: taxa de dropout na autoatenção.

        O artigo cita:
            “The Transformer encoder consists of L layers of MSA + MLP blocks.”
      """

    num_layers: int
    mlp_dim: int
    num_heads: int
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    add_position_embedding: bool = True

    @nn.compact
    def __call__(self, x, *, train):
        """
            Aplica o Transformer às entradas.
            Argumentos:
                x: Entradas da camada.
                train: Define como `True` durante o treinamento.

            Retorno:
                Saída de um codificador Transformer.
        """

        # Verifica as dimensões de entrada (batch, len, emb)
        assert x.ndim == 3

        # Inserção do positional embedding antes do encoder
        if self.add_position_embedding:
            x = AddPositionEmbs(
                posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
                name='posembed_input')(
                x)
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

        # Aplica o encoder efetivamente
        for lyr in range(self.num_layers):
            x = Encoder1DBlock(
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                name=f'encoderblock_{lyr}',
                num_heads=self.num_heads)(
                x, deterministic=not train)
        encoded = nn.LayerNorm(name='encoder_norm')(x)

        return encoded


# ======================================================================================================================

class VisionTransformer(nn.Module):
    # Arquitetura completa da VisionTransformer

    num_classes: int
    patches: Any
    transformer: Any
    hidden_size: int
    representation_size: Optional[int] = None
    classifier: str = 'token'
    head_bias_init: float = 0.
    encoder: Type[nn.Module] = Encoder
    model_name: Optional[str] = None

    @nn.compact
    def __call__(self, inputs, *, train):

        # inputs: formato do tensor de imagem (lote, H, W, C)
        x = inputs  # expect shape (n, H, W, C)

        # Pure ViT: patch embedding com uma camada Conv2D (kernel=patch_size, stride=patch_size)
        # No artigo é descrito: “A patch of size P×P is mapped to a D-dimensional embedding using a linear projection.”
        x = nn.Conv(
            features=self.hidden_size,
            kernel_size=self.patches.size,
            strides=self.patches.size,
            padding='VALID',
            name='embedding')(x)

        # Now x is a grid of embeddings: (n, h, w, c)
        n, h, w, c = x.shape

        # Transformando o grid de embeddings em sequência de patch embeddings: (n, h*w, c)
        # No artigo temos: “We reshape the image into a sequence of flattened patches.”
        if self.transformer is not None:
            x = jnp.reshape(x, [n, h * w, c])

            # Se requisitado, é adicionado o Class Token, o qual:
            #   - Recebe os gradientes
            #   - Representa toda a imagem
            #   - Serve como entrada para o MLP final
            if self.classifier in ['token', 'token_unpooled']:
                cls = self.param('cls', nn.initializers.zeros, (1, 1, c))
                cls = jnp.tile(cls, [n, 1, 1])
                x = jnp.concatenate([cls, x], axis=1)

            # Aplica o Transform encoder com L camadas, Attention + MLP e Positional embedding já adicionado
            x = self.encoder(name='Transformer', **self.transformer)(x, train=train)

        # Classificação via token CLS:
        if self.classifier == 'token':
            x = x[:, 0]
        elif self.classifier == 'gap':
            x = jnp.mean(x, axis=1)
        elif self.classifier in ['unpooled', 'token_unpooled']:
            pass
        else:
            raise ValueError(f'Invalid classifier={self.classifier}')

        # Representation layer (opcional)
        # Está de acordo com o artigo: "We add an optional MLP layer before classification."
        if self.representation_size is not None:
            x = nn.Dense(features=self.representation_size, name='pre_logits')(x)
            x = nn.tanh(x)
        else:
            x = IdentityLayer(name='pre_logits')(x)

        # Cabeça final (head)
        # No artigo: “The final classification layer is a linear projection. We initialize the head to zeros.”
        if self.num_classes:
            x = nn.Dense(
                features=self.num_classes,
                name='head',
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.constant(self.head_bias_init))(x)
        return x
