# sea-raft-mlx

Apple [MLX](https://github.com/ml-explore/mlx) port of **[SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT)**
(Wang, Lipson, Deng — ECCV 2024, BSD-3-Clause) — simple, efficient, accurate optical flow on
Apple Silicon. Translated file-isomorphically from the pinned reference
(`princeton-vl/SEA-RAFT @ 9137517`), NHWC throughout.

> **Parity (fp32, CPU, real Tartan480x640-S weights, identical input pair):**
> final flow **cosine 1.0000, max_abs 1.2e-3 px, EPE 1.1e-4 px** across all iteration outputs
> (`scripts/parity.py`). On Metal GPU, accumulation-order drift gives mean EPE ≈ 0.1 px on
> ~31 px flows — sub-pixel and production-fine.

## Architecture (S/M, resnet18 backbone)

ResNet-FPN context+feature encoders (1/8 scale) → initial flow head → all-pairs correlation
pyramid (4 levels, recomputed per level; zeros-padding bilinear lookup, r=4) → `iters`×
[motion encoder → **ConvNeXt** update blocks (no ConvGRU) → flow head] → convex-combination
8× upsample. Hand-rolled NHWC ops: `grid_sample` (zeros, align_corners=true), bilinear
`interpolate`, `unfold3x3`.

## Usage

```python
import mlx.core as mx
from sea_raft_mlx import SEARAFT, SEARAFTConfig
from sea_raft_mlx.convert import convert

convert("model.safetensors", "sea_raft_s_mlx.safetensors")   # torch HF ckpt -> MLX NHWC
model = SEARAFT(SEARAFTConfig())                              # S config (dim=128, iters=4)
model.load_weights(list(dict(mx.load("sea_raft_s_mlx.safetensors")).items()), strict=True)
model.eval()

out = model(img1, img2)        # [N, H, W, 3] in 0..255 -> {'final': [N, H, W, 2] flow (px)}
```

## Weights

Original checkpoints: [`MemorySlices/*`](https://huggingface.co/MemorySlices) (the first author's
uploads, `bsd-3-clause`-tagged; e.g. `Tartan480x640-S` — TartanAir-only, the provenance-cleanest
stage). Converted MLX weights will be published to `mlx-community` once
[princeton-vl/SEA-RAFT#31](https://github.com/princeton-vl/SEA-RAFT/issues/31) (explicit
checkpoint-license confirmation) is answered.

## License

MIT (this port). Upstream SEA-RAFT code: BSD-3-Clause (Princeton Vision & Learning Lab).
