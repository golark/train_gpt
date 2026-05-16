"""Checkpoint loading helpers: adapt common key-name variants to local model state_dict.

Provides:
- adapt_and_load(checkpoint_path, model, map_location='cpu') -> dict with load diagnostics

Strategy:
- Load full checkpoint (weights_only=False) with map_location
- Locate model blob under common keys ("model", "state_dict", ...)
- For each checkpoint key try variants: strip 'module.' / 'model.' prefixes, add 'transformer.' prefix, etc., and match to model.state_dict() keys
- If a matched target parameter has mismatched shape but transposed shape matches, transpose before copying.
- Load into model with strict=False and return diagnostics.
"""

from typing import Dict, Tuple
import torch
import os

COMMON_MODEL_KEYS = ('model', 'state_dict', 'model_state_dict', 'state')


def adapt_and_load(checkpoint_path: str, model: torch.nn.Module, map_location='cpu') -> Dict[str, object]:
    """Load a checkpoint and adapt its keys to `model`.

    Returns a dict with keys: 'matched', 'missing', 'unexpected', 'failed_keys', 'loaded_state'
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=map_location)

    # find model blob
    model_blob = None
    for k in COMMON_MODEL_KEYS:
        if k in ckpt:
            model_blob = ckpt[k]
            break
    if model_blob is None:
        # maybe the entire checkpoint is a state_dict
        if isinstance(ckpt, dict) and any(isinstance(v, torch.Tensor) for v in list(ckpt.values())[:5]):
            model_blob = ckpt
        else:
            raise RuntimeError('Could not locate model state in checkpoint; top-level keys: ' + str(list(ckpt.keys())))

    src_keys = list(model_blob.keys())
    tgt_sd = model.state_dict()
    tgt_keys_set = set(tgt_sd.keys())

    mapped = {}
    unmapped = {}

    def key_variants(k: str):
        # generate plausible variants in order of preference
        v = [k]
        if k.startswith('module.'):
            v.append(k[len('module.'):])
        if k.startswith('model.'):
            v.append(k[len('model.'):])
        if k.startswith('transformer.'):
            v.append(k[len('transformer.'):])
        # add transformer prefix if missing
        if not k.startswith('transformer.'):
            v.append('transformer.' + k)
        # also try prefixed with 'model.' + k
        v.append('model.' + k)
        v.append('module.' + k)
        # remove duplicates preserving order
        seen = set()
        out = []
        for it in v:
            if it not in seen:
                out.append(it); seen.add(it)
        return out

    transposed_suffixes = ('attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight', 'c_fc.weight', 'c_proj.weight', 'c_attn.weight')

    for k in src_keys:
        val = model_blob[k]
        found = False
        for cand in key_variants(k):
            if cand in tgt_keys_set:
                tgt_shape = tgt_sd[cand].shape
                src_shape = getattr(val, 'shape', None)
                # if shapes equal, copy
                if src_shape == tgt_shape:
                    mapped[cand] = val
                    found = True
                    break
                # if transposed shapes match, transpose
                if src_shape is not None and src_shape[::-1] == tgt_shape:
                    mapped[cand] = val.t()
                    found = True
                    break
                # otherwise, still try to map (will fail later)
                mapped[cand] = val
                found = True
                break
        if not found:
            unmapped[k] = val

    # build a new state dict for loading
    new_sd = tgt_sd.copy()
    # replace mapped entries
    for k, v in mapped.items():
        try:
            new_sd[k] = v
        except Exception:
            # skip on error
            pass

    # attempt to load
    res = model.load_state_dict(new_sd, strict=False)

    diagnostics = {
        'matched_target_keys': list(mapped.keys()),
        'unmapped_checkpoint_keys': list(unmapped.keys()),
        'missing_keys': res.missing_keys,
        'unexpected_keys': res.unexpected_keys,
        'error_msg': None,
    }
    return diagnostics

