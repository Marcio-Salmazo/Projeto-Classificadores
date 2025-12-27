# -------------------------------------------------------------------------------------------------------------------- #
#                      Agregado de utilitários destinados ao carregamento de pesos pré-treinados                       #
#                      Os pesos estão disponibilizados no repositório oficial do Google Research                       #
# -------------------------------------------------------------------------------------------------------------------- #

import numpy as np
import jax.numpy as jnp

# ======================================================================================================================
#  FUNÇÕES UTILITÁRIAS DE FLATTEN E UNFLATTEN

def flatten_dict(d, prefix=""):
    """Transforma dict hierárquico em dict de chaves 'a/b/c'."""
    out = {}
    for k, v in d.items():
        full = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_dict(v, full))
        else:
            out[full] = v
    return out


def unflatten_dict(flat):
    """Transforma dict achatado de volta em estrutura hierárquica."""
    root = {}
    for k, v in flat.items():
        parts = k.split("/")
        d = root
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v
    return root


# ======================================================================================================================
#  REGRAS DE REMAPEAMENTO ESPECÍFICAS (Google → Flax)

def map_checkpoint_key(key):
    """
    Função de mapeamento EXATA baseada na diferença observada entre:
    - Nomes do checkpoint Google
    - Nomes reais do seu modelo (verificados via params init)

    Diferenças encontradas:
        Google usa: encoderblock0, encoderblock1, ..., encoderblock11
        Seu modelo usa: encoderblock_0, encoderblock_1, ..., encoderblock_11

    Esta função corrige somente isso, NADA mais.
    """
    # Patch de encoderblockN → encoderblock_N
    import re

    key = re.sub(r"encoderblock(\d+)", r"encoderblock_\1", key)

    # Nenhuma outra modificação é necessária
    return key


# ======================================================================================================================
#  LOADER PRINCIPAL DE PESOS PRE-TREINADOS

def load_vit_npz(params, npz_path):
    """
    Carrega pesos pré-treinados para ViT-B/16 do Google.

    REGRAS:
    - Remapeia somente encoderblock0 → encoderblock_0
    - Ignora parâmetros da head (shape diferente ou inexistentes)
    - Ignora shapes incompatíveis
    - Log detalhado resumido e preciso
    """

    print(f">> Lendo checkpoint: {npz_path}")
    data = np.load(npz_path)

    flat_params = flatten_dict(params)
    new_params = {}

    # Contadores de estatística
    count_loaded = 0
    count_missing = 0
    count_ignored_head = 0
    count_shape_mismatch = 0
    count_remapped = 0

    remap_log = []

    for ckpt_key in data.keys():
        original_key = ckpt_key
        mapped_key = map_checkpoint_key(ckpt_key)

        if mapped_key != original_key:
            count_remapped += 1
            remap_log.append((original_key, mapped_key))

        # Ignorar HEAD do Google
        if mapped_key.startswith("head/") or "/head/" in mapped_key:
            count_ignored_head += 1
            continue

        # Parâmetro não existe no modelo
        if mapped_key not in flat_params:
            count_missing += 1
            continue

        ckpt_value = data[ckpt_key]
        model_value = flat_params[mapped_key]

        # Shape incompatível
        if ckpt_value.shape != model_value.shape:
            count_shape_mismatch += 1
            continue

        new_params[mapped_key] = jnp.asarray(ckpt_value)
        count_loaded += 1

    # Reconstruir parâmetros finais
    flat_out = {
        k: new_params.get(k, v)
        for k, v in flat_params.items()
    }

    # ==================================================================
    #  Relatório final
    # ==================================================================
    print("\n================== CHECKPOINT REPORT ==================")
    print(f"Total parâmetros modelo:      {len(flat_params)}")
    print(f"Carregados com sucesso:       {count_loaded}")
    print(f"Chaves remapeadas:            {count_remapped}")
    print(f"Ignorados (head):             {count_ignored_head}")
    print(f"Ignorados (missing):          {count_missing}")
    print(f"Ignorados (shape mismatch):   {count_shape_mismatch}")
    print("========================================================\n")

    if remap_log:
        print(">> Remapeamentos aplicados:")
        for old, new in remap_log[:20]:  # Mostra só os primeiros 20
            print(f"   {old}  →  {new}")
        if len(remap_log) > 20:
            print("   ... (demais omitidos)")

    print(f"\n>> TOTAL carregado: {count_loaded} parâmetros pré-treinados.")
    return unflatten_dict(flat_out)
