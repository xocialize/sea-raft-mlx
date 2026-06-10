"""Hand-rolled NHWC spatial ops for SEA-RAFT — MLX equivalents of the torch calls.

- grid_sample_bilinear: torch F.grid_sample(mode='bilinear', padding_mode='zeros',
  align_corners=True) — NOTE zeros padding (RAFT's corr lookup), unlike RIFE's border.
- interpolate_bilinear: F.interpolate(mode='bilinear') for the corr-pyramid downsample
  (align_corners=False).
- unfold3x3: F.unfold(kernel=3, padding=1) for the convex upsample, returned as
  [N, H, W, C, 9] (patch index p = ky*3+kx, row-major — torch channel-major order).
"""

from __future__ import annotations

import mlx.core as mx


def grid_sample_bilinear(inp: mx.array, grid: mx.array, align_corners: bool = True) -> mx.array:
    """inp: [B, H, W, C]; grid: [B, gH, gW, 2] normalized [-1, 1] (x, y). Zeros padding."""
    B, H, W, C = inp.shape
    _, gH, gW, _ = grid.shape

    gx = grid[..., 0]
    gy = grid[..., 1]
    if align_corners:
        ix = (gx + 1) * 0.5 * (W - 1)
        iy = (gy + 1) * 0.5 * (H - 1)
    else:
        ix = ((gx + 1) * W - 1) * 0.5
        iy = ((gy + 1) * H - 1) * 0.5

    x0 = mx.floor(ix)
    y0 = mx.floor(iy)
    x1 = x0 + 1
    y1 = y0 + 1
    wx1 = ix - x0
    wx0 = 1.0 - wx1
    wy1 = iy - y0
    wy0 = 1.0 - wy1

    inp_flat = inp.reshape(B, H * W, C)

    def corner(xc, yc, w):
        # zeros padding: out-of-range neighbours contribute 0
        valid = (xc >= 0) & (xc <= W - 1) & (yc >= 0) & (yc <= H - 1)
        xs = mx.clip(xc, 0, W - 1).astype(mx.int32)
        ys = mx.clip(yc, 0, H - 1).astype(mx.int32)
        idx = (ys * W + xs).reshape(B, gH * gW, 1)
        idx = mx.broadcast_to(idx, (B, gH * gW, C))
        v = mx.take_along_axis(inp_flat, idx, axis=1).reshape(B, gH, gW, C)
        return v * (w * valid.astype(inp.dtype))[..., None]

    return (corner(x0, y0, wy0 * wx0) + corner(x1, y0, wy0 * wx1)
            + corner(x0, y1, wy1 * wx0) + corner(x1, y1, wy1 * wx1))


def _coords_1d(out: int, in_: int, align_corners: bool) -> mx.array:
    dst = mx.arange(out, dtype=mx.float32)
    if align_corners:
        return dst * ((in_ - 1) / (out - 1)) if out > 1 else dst * 0.0
    return (dst + 0.5) * (in_ / out) - 0.5


def _bilinear_1d(x: mx.array, axis: int, out: int, align_corners: bool) -> mx.array:
    in_ = x.shape[axis]
    if in_ == out:
        return x
    src = _coords_1d(out, in_, align_corners)
    i0 = mx.floor(src)
    w1 = src - i0
    w0 = 1.0 - w1
    i0c = mx.clip(i0, 0, in_ - 1).astype(mx.int32)
    i1c = mx.clip(i0 + 1, 0, in_ - 1).astype(mx.int32)
    g0 = mx.take(x, i0c, axis=axis)
    g1 = mx.take(x, i1c, axis=axis)
    shape = [1] * x.ndim
    shape[axis] = out
    return g0 * w0.reshape(shape) + g1 * w1.reshape(shape)


def interpolate_bilinear(x: mx.array, scale_factor: float, align_corners: bool = False) -> mx.array:
    """x: [N, H, W, C]."""
    N, H, W, C = x.shape
    oh = int(round(H * scale_factor))
    ow = int(round(W * scale_factor))
    x = _bilinear_1d(x, 1, oh, align_corners)
    x = _bilinear_1d(x, 2, ow, align_corners)
    return x


def unfold3x3(x: mx.array) -> mx.array:
    """x: [N, H, W, C] -> [N, H, W, C, 9] (zero-padded 3x3 patches; p = ky*3+kx)."""
    N, H, W, C = x.shape
    xp = mx.pad(x, [(0, 0), (1, 1), (1, 1), (0, 0)])
    patches = []
    for ky in range(3):
        for kx in range(3):
            patches.append(xp[:, ky:ky + H, kx:kx + W, :])
    return mx.stack(patches, axis=-1)  # [N, H, W, C, 9]
