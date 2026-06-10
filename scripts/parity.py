"""Parity: torch SEA-RAFT reference (pinned 9137517) vs the MLX port, real Tartan480x640-S
weights, identical input pair. Compares the final flow + per-iteration predictions."""

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REF = "/Users/dustinnielson/Development/sea-raft-reference/core"
CKPT = ("/Users/dustinnielson/.cache/huggingface/hub/models--MemorySlices--Tartan480x640-S/"
        "snapshots/054c3e29b6ead517bef63e8c80a911fcb1803df7/model.safetensors")

sys.path.insert(0, REF)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from safetensors.torch import load_file as load_torch_st

from raft import RAFT  # reference

import mlx.core as mx
import os
if os.environ.get('MLX_CPU'):
    mx.set_default_device(mx.cpu)
from sea_raft_mlx.model import SEARAFT, SEARAFTConfig
from sea_raft_mlx.convert import convert


def stats(name, a, b):
    diff = float(np.max(np.abs(a - b)))
    cos = float((a * b).sum() / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
    epe = float(np.mean(np.sqrt(((a - b) ** 2).sum(axis=-1)))) if a.shape[-1] == 2 else float("nan")
    print(f"  {name:16s} cosine={cos:.7f}  max_abs={diff:.5g}  epe={epe:.5g}")
    return cos, diff


def main():
    # ---- torch reference ----
    args = SimpleNamespace(
        dim=128, initial_dim=64, block_dims=[64, 128, 256], pretrain="resnet18",
        radius=4, num_blocks=2, iters=4, use_var=True, var_min=0, var_max=10,
    )
    tmodel = RAFT(args)
    sd = load_torch_st(CKPT)
    missing, unexpected = tmodel.load_state_dict(sd, strict=False)
    print(f"torch load: missing={len(missing)} unexpected={len(unexpected)}")
    assert not unexpected, unexpected[:5]
    tmodel.eval()

    # ---- input pair: structured frames with real motion, 0..255, [1, 3, H, W] ----
    rng = np.random.default_rng(5)
    H, W = 256, 448

    def frame(shift):
        img = np.zeros((H, W, 3), dtype=np.float32)
        yy, xx = np.mgrid[0:H, 0:W]
        img[..., 0] = xx / W * 255
        img[..., 1] = yy / H * 255
        img[60 + shift : 140 + shift, 160 + shift : 280 + shift, 2] = 255.0
        img += rng.normal(0, 2.0, img.shape).astype(np.float32)
        return np.clip(img, 0, 255)

    i1 = frame(0)[None]
    i2 = frame(6)[None]

    with torch.no_grad():
        tout = tmodel(torch.from_numpy(i1).permute(0, 3, 1, 2).contiguous(),
                      torch.from_numpy(i2).permute(0, 3, 1, 2).contiguous(), test_mode=True)
    t_final = tout["final"].permute(0, 2, 3, 1).numpy()           # [N, H, W, 2]
    t_flows = [f.permute(0, 2, 3, 1).numpy() for f in tout["flow"]]

    # ---- MLX port ----
    out_path = "/tmp/sea_raft_tartan_s_mlx.safetensors"
    convert(CKPT, out_path)
    model = SEARAFT(SEARAFTConfig())
    weights = list(dict(mx.load(out_path)).items())
    model.load_weights(weights, strict=True)
    model.eval()
    mx.eval(model.parameters())

    mout = model(mx.array(i1), mx.array(i2))
    mx.eval(mout["final"])
    m_final = np.array(mout["final"])
    m_flows = [np.array(f) for f in mout["flow"]]

    print("\n--- parity (torch vs mlx) ---")
    ok = True
    for i, (tf, mf) in enumerate(zip(t_flows, m_flows)):
        cos, diff = stats(f"flow[{i}]", tf, mf)
        ok &= cos > 0.999
    cos, diff = stats("final", t_final, m_final)
    print("\nflow magnitude (torch):", float(np.abs(t_final).max()))
    print("VERDICT:", "PARITY OK" if (cos > 0.9999 and diff < 0.05) else "DIVERGED")


if __name__ == "__main__":
    main()
