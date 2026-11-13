# Modified for SplatAnything: inference-only release.
"""Load complete inference checkpoints; reject missing or mismatched model tensors."""
from pathlib import Path
import torch
from safetensors.torch import load_file

def load_checkpoint(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if path.suffix == '.safetensors':
        state = load_file(str(path))
    else:
        checkpoint = torch.load(str(path), map_location='cpu', mmap=True, weights_only=True)
        state = checkpoint.get('state_dict', checkpoint)
        state = {k.removeprefix('model.'): v for k, v in state.items()}
    return state

def load_model_weights(model, path):
    state = load_checkpoint(path)
    adapt_depth_mask_checkpoint(model, state)
    expected = model.state_dict()
    missing = [k for k in expected if k not in state]
    mismatched = [k for k in expected if k in state and expected[k].shape != state[k].shape]
    if missing or mismatched:
        raise ValueError(f"Incomplete/incompatible checkpoint {path}: "
                         f"missing={missing[:10]}, shape mismatches={mismatched[:10]}. "
                         "Use complete stage-1 / WorldMirror-2 geometry weights.")
    model.load_state_dict({k: state[k] for k in expected}, strict=True)
    print(f"[Weights] {type(model).__name__}: {len(expected)} tensors loaded from {path}")

def adapt_depth_mask_checkpoint(model: torch.nn.Module, state_dict: dict) -> None:
    """Expand a 2-channel pretrained depth output conv to the model's 3-channel mask head.

    Mirrors wrapper._adapt_depth_mask_checkpoint: when the model was built with
    enable_depth_mask=True but the base pretrained weights only have the 2-channel
    (depth, conf) output conv, copy those two channels into the 3-channel target and leave
    the extra mask channel at init. Without this the whole conv shape-mismatches and is
    skipped -> depth channels stay random and predicted depth is garbage. In-place on
    state_dict; no-op if the model has no depth mask.
    """
    if not getattr(model, "enable_depth_mask", False):
        return
    current = model.state_dict()
    for suffix in ("weight", "bias"):
        key = f"depth_head.scratch.output_conv2.2.{suffix}"
        if key not in state_dict or key not in current:
            continue
        ckpt_t, cur_t = state_dict[key], current[key]
        if ckpt_t.shape[0] == cur_t.shape[0] - 1 and ckpt_t.shape[1:] == cur_t.shape[1:]:
            new_t = cur_t.clone()
            new_t[: ckpt_t.shape[0]] = ckpt_t
            state_dict[key] = new_t
