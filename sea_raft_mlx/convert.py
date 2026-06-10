"""Convert a SEA-RAFT torch checkpoint (HF model.safetensors) to MLX NHWC safetensors.

Transforms:
1. Conv2d weights (O, I, kh, kw) -> (O, kh, kw, I)  [incl. depthwise].
2. Sequential activation-slot compaction: upsample_weight.2 -> .1, flow_head.2 -> .1.
3. BasicBlock shared-module dedup: downsample.1.* (== bn3) -> bn3.*.
4. Drop num_batches_tracked.
Linear / LayerNorm / BatchNorm / gamma pass through unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx


def convert_key(key: str) -> str | None:
    if key.endswith("num_batches_tracked"):
        return None
    key = key.replace("upsample_weight.2.", "upsample_weight.1.")
    key = key.replace("flow_head.2.", "flow_head.1.")
    key = key.replace("downsample.1.", "bn3.")
    return key


def is_conv_weight(key: str, shape) -> bool:
    # 4-D weights are conv kernels (Linear weights are 2-D here).
    return key.endswith(".weight") and len(shape) == 4


def convert(src: str, dst: str) -> dict:
    raw = dict(mx.load(str(src)))
    out = {}
    for k, v in raw.items():
        nk = convert_key(k)
        if nk is None:
            continue
        if is_conv_weight(nk, v.shape):
            v = v.transpose(0, 2, 3, 1)  # (O, I, kh, kw) -> (O, kh, kw, I)
        out[nk] = v
    # Materialize before saving (lazy tensors save as zeros).
    mx.eval(list(out.values()))
    mx.save_safetensors(str(dst), out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    a = ap.parse_args()
    converted = convert(a.src, a.dst)
    print(f"converted {len(converted)} tensors -> {a.dst}")
