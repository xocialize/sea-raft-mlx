"""SEA-RAFT — MLX port, isomorphic to princeton-vl/SEA-RAFT @ 9137517 (core/{raft,update,
extractor,layer,corr}.py). NHWC throughout; only PyTorch ops -> MLX ops and NCHW -> NHWC change.

Key map vs the torch checkpoint (see convert.py): conv weights transposed (O,I,kh,kw) ->
(O,kh,kw,I); nn.Sequential activation slots compacted (upsample_weight.2 -> .1, flow_head.2 ->
.1, downsample.1 stays — see convert); BatchNorm running stats kept; num_batches_tracked dropped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .ops import grid_sample_bilinear, interpolate_bilinear, unfold3x3


@dataclass
class SEARAFTConfig:
    dim: int = 128
    initial_dim: int = 64
    block_dims: tuple = (64, 128, 256)
    pretrain: str = "resnet18"
    radius: int = 4
    num_blocks: int = 2
    iters: int = 4

    @property
    def corr_levels(self) -> int:
        return 4

    @property
    def corr_channel(self) -> int:
        return self.corr_levels * (self.radius * 2 + 1) ** 2


# ---------- layer.py ----------

class ConvNextBlock(nn.Module):
    def __init__(self, dim: int, output_dim: int, layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * output_dim)
        self.pwconv2 = nn.Linear(4 * output_dim, dim)
        self.gamma = layer_scale_init_value * mx.ones((dim,))
        self.final = nn.Conv2d(dim, output_dim, kernel_size=1, padding=0)

    def __call__(self, x: mx.array) -> mx.array:
        inp = x
        x = self.dwconv(x)            # NHWC already — no permutes needed
        x = self.norm(x)
        x = self.pwconv1(x)
        x = nn.gelu(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        return self.final(inp + x)


class BasicBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm(planes)
        self.bn2 = nn.BatchNorm(planes)
        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.bn3 = nn.BatchNorm(planes)
            # torch: Sequential(conv1x1, bn3) with keys downsample.0 / downsample.1(=bn3)
            self.downsample = [nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride)]

    def __call__(self, x: mx.array) -> mx.array:
        y = nn.relu(self.bn1(self.conv1(x)))
        y = nn.relu(self.bn2(self.conv2(y)))
        if self.downsample is not None:
            x = self.bn3(self.downsample[0](x))
        return nn.relu(x + y)


# ---------- extractor.py ----------

class ResNetFPN(nn.Module):
    """ResNet18-style backbone, output resolution 1/8."""

    def __init__(self, config: SEARAFTConfig, input_dim: int, output_dim: int):
        super().__init__()
        block_dims = list(config.block_dims)
        initial_dim = config.initial_dim
        assert config.pretrain == "resnet18", "only the S/M (resnet18) variants are ported"
        n_block = [2, 2, 2]

        self.conv1 = nn.Conv2d(input_dim, initial_dim, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm(initial_dim)

        in_planes = initial_dim

        def make_layer(dim, stride, num):
            nonlocal in_planes
            layers = [BasicBlock(in_planes, dim, stride=stride)]
            for _ in range(num - 1):
                layers.append(BasicBlock(dim, dim, stride=1))
            in_planes = dim
            return layers

        self.layer1 = make_layer(block_dims[0], 1, n_block[0])   # 1/2
        self.layer2 = make_layer(block_dims[1], 2, n_block[1])   # 1/4
        self.layer3 = make_layer(block_dims[2], 2, n_block[2])   # 1/8
        self.final_conv = nn.Conv2d(block_dims[2], output_dim, kernel_size=1, stride=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.bn1(self.conv1(x)))
        for blk in self.layer1:
            x = blk(x)
        for blk in self.layer2:
            x = blk(x)
        for blk in self.layer3:
            x = blk(x)
        return self.final_conv(x)


# ---------- update.py ----------

class BasicMotionEncoder(nn.Module):
    def __init__(self, config: SEARAFTConfig, dim: int = 128):
        super().__init__()
        cor_planes = config.corr_channel
        self.convc1 = nn.Conv2d(cor_planes, dim * 2, kernel_size=1, padding=0)
        self.convc2 = nn.Conv2d(dim * 2, dim + dim // 2, kernel_size=3, padding=1)
        self.convf1 = nn.Conv2d(2, dim, kernel_size=7, padding=3)
        self.convf2 = nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1)
        self.conv = nn.Conv2d(dim * 2, dim - 2, kernel_size=3, padding=1)

    def __call__(self, flow: mx.array, corr: mx.array) -> mx.array:
        cor = nn.relu(self.convc1(corr))
        cor = nn.relu(self.convc2(cor))
        flo = nn.relu(self.convf1(flow))
        flo = nn.relu(self.convf2(flo))
        out = nn.relu(self.conv(mx.concatenate([cor, flo], axis=-1)))
        return mx.concatenate([out, flow], axis=-1)


class BasicUpdateBlock(nn.Module):
    def __init__(self, config: SEARAFTConfig, hdim: int, cdim: int):
        super().__init__()
        self.encoder = BasicMotionEncoder(config, dim=cdim)
        self.refine = [ConvNextBlock(2 * cdim + hdim, hdim) for _ in range(config.num_blocks)]

    def __call__(self, net: mx.array, inp: mx.array, corr: mx.array, flow: mx.array) -> mx.array:
        motion_features = self.encoder(flow, corr)
        inp = mx.concatenate([inp, motion_features], axis=-1)
        for blk in self.refine:
            net = blk(mx.concatenate([net, inp], axis=-1))
        return net


# ---------- corr.py ----------

def coords_grid(batch: int, ht: int, wd: int) -> mx.array:
    """[N, H, W, 2] (x, y) pixel coords."""
    ys, xs = mx.meshgrid(mx.arange(ht), mx.arange(wd), indexing="ij")
    coords = mx.stack([xs, ys], axis=-1).astype(mx.float32)  # (x, y) — torch stacks reversed
    return mx.broadcast_to(coords[None], (batch, ht, wd, 2))


class CorrBlock:
    def __init__(self, fmap1: mx.array, fmap2: mx.array, config: SEARAFTConfig):
        self.num_levels = config.corr_levels
        self.radius = config.radius
        self.corr_pyramid = []
        # All-pairs correlation, recomputed per level against a downsampled fmap2
        # (matches the reference: corr is rebuilt each level, NOT pooled).
        N, h1, w1, d = fmap1.shape
        a = fmap1.reshape(N, h1 * w1, d)
        scale = 1.0 / math.sqrt(d)
        f2 = fmap2
        for _ in range(self.num_levels):
            _, h2, w2, _ = f2.shape
            b = f2.reshape(N, h2 * w2, d)
            corr = (a @ b.transpose(0, 2, 1)) * scale          # [N, h1w1, h2w2]
            corr = corr.reshape(N * h1 * w1, h2, w2, 1)        # NHWC for sampling
            self.corr_pyramid.append(corr)
            f2 = interpolate_bilinear(f2, 0.5, align_corners=False)
        self.h1, self.w1, self.batch = h1, w1, N

    def __call__(self, coords: mx.array) -> mx.array:
        """coords: [N, H, W, 2] pixel coords at 1/8 scale -> [N, H, W, corr_channel]."""
        r = self.radius
        N, h1, w1, _ = coords.shape
        dx = mx.linspace(-r, r, 2 * r + 1)
        dy = mx.linspace(-r, r, 2 * r + 1)
        dyg, dxg = mx.meshgrid(dy, dx, indexing="ij")
        # torch stacks meshgrid(dy, dx) -> (..., 2) with last dim (dy_val, dx_val)?? No:
        # delta = stack(meshgrid(dy, dx), axis=-1) gives (dy, dx) BUT bilinear_sampler reads
        # coords[..., 0] as x. The reference adds delta to centroid (x, y) — so delta[..., 0]
        # pairs with x. meshgrid(dy, dx) -> first grid varies over dy. stack => [..., (dy, dx)].
        # centroid is (x, y): x + dy?? The reference relies on the SYMMETRIC window (-r..r in
        # both axes), making (dy, dx) vs (dx, dy) equivalent for the sampled grid POSITIONS but
        # transposing the window enumeration order — which permutes the corr_channel ordering
        # consumed by trained conv weights. Match torch exactly: delta = (dy, dx) added to
        # (x, y) coords.
        delta = mx.stack([dyg, dxg], axis=-1).reshape(1, 2 * r + 1, 2 * r + 1, 2)

        out_pyramid = []
        centroid_base = coords.reshape(N * h1 * w1, 1, 1, 2)
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            _, h2, w2, _ = corr.shape
            coords_lvl = centroid_base / (2 ** i) + delta       # pixel coords (x, y)
            # normalize for grid_sample (align_corners=True)
            gx = 2 * coords_lvl[..., 0] / (w2 - 1) - 1
            gy = 2 * coords_lvl[..., 1] / (h2 - 1) - 1
            grid = mx.stack([gx, gy], axis=-1)
            sampled = grid_sample_bilinear(corr, grid, align_corners=True)  # [B', 2r+1, 2r+1, 1]
            out_pyramid.append(sampled.reshape(N, h1, w1, -1))
        return mx.concatenate(out_pyramid, axis=-1)


# ---------- raft.py ----------

class SEARAFT(nn.Module):
    def __init__(self, config: SEARAFTConfig | None = None):
        super().__init__()
        self.config = config or SEARAFTConfig()
        c = self.config
        self.output_dim = c.dim * 2

        self.cnet = ResNetFPN(c, input_dim=6, output_dim=2 * c.dim)
        self.init_conv = nn.Conv2d(2 * c.dim, 2 * c.dim, kernel_size=3, padding=1)
        # torch Sequential(conv3x3, ReLU, conv1x1) — MLX list keys 0/1 (ReLU slot compacted)
        self.upsample_weight = [
            nn.Conv2d(c.dim, c.dim * 2, kernel_size=3, padding=1),
            nn.Conv2d(c.dim * 2, 64 * 9, kernel_size=1, padding=0),
        ]
        self.flow_head = [
            nn.Conv2d(c.dim, 2 * c.dim, kernel_size=3, padding=1),
            nn.Conv2d(2 * c.dim, 6, kernel_size=3, padding=1),
        ]
        self.fnet = ResNetFPN(c, input_dim=3, output_dim=self.output_dim)
        self.update_block = BasicUpdateBlock(c, hdim=c.dim, cdim=c.dim)

    def _flow_head(self, net: mx.array) -> mx.array:
        return self.flow_head[1](nn.relu(self.flow_head[0](net)))

    def _upsample_weight(self, net: mx.array) -> mx.array:
        return self.upsample_weight[1](nn.relu(self.upsample_weight[0](net)))

    def upsample_data(self, flow: mx.array, info: mx.array, mask: mx.array):
        """Convex-combination 8x upsample. flow [N,H,W,2], info [N,H,W,4], mask [N,H,W,576]."""
        N, H, W, _ = flow.shape

        # mask channels: torch view (N, 1, 9, 8, 8, H, W) from (N, 576, H, W)
        # => channel c = k*64 + i*8 + j. NHWC reshape [N,H,W,9,8,8] preserves that order.
        m = mask.reshape(N, H, W, 9, 8, 8)
        m = mx.softmax(m, axis=3)

        def convex(x: mx.array) -> mx.array:
            C = x.shape[-1]
            patches = unfold3x3(x)                              # [N,H,W,C,9]
            out = mx.einsum("nhwck,nhwkij->nhwijc", patches, m)  # [N,H,W,8,8,C]
            out = out.transpose(0, 1, 3, 2, 4, 5)                # [N,H,8,W,8,C]
            return out.reshape(N, H * 8, W * 8, C)

        return convex(8 * flow), convex(info)

    def __call__(self, image1: mx.array, image2: mx.array, iters: int | None = None) -> dict:
        """image1/2: [N, H, W, 3] in 0..255. Returns {'final': [N, H, W, 2] flow, ...}."""
        c = self.config
        if iters is None:
            iters = c.iters
        N, H0, W0, _ = image1.shape

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        # InputPadder (sintel mode, /8, replicate)
        pad_h = (((H0 // 8) + 1) * 8 - H0) % 8
        pad_w = (((W0 // 8) + 1) * 8 - W0) % 8
        pads = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        if pad_h or pad_w:
            widths = [(0, 0), (pads[2], pads[3]), (pads[0], pads[1]), (0, 0)]
            image1 = mx.pad(image1, widths, mode="edge")
            image2 = mx.pad(image2, widths, mode="edge")
        _, H, W, _ = image1.shape

        cnet = self.cnet(mx.concatenate([image1, image2], axis=-1))
        cnet = self.init_conv(cnet)
        net, context = cnet[..., : c.dim], cnet[..., c.dim :]

        flow_update = self._flow_head(net)
        weight_update = 0.25 * self._upsample_weight(net)
        flow_8x = flow_update[..., :2]
        info_8x = flow_update[..., 2:]
        flow_up, info_up = self.upsample_data(flow_8x, info_8x, weight_update)
        flow_predictions = [flow_up]
        info_predictions = [info_up]

        fmap1 = self.fnet(image1)
        fmap2 = self.fnet(image2)
        corr_fn = CorrBlock(fmap1, fmap2, c)

        h8, w8 = H // 8, W // 8
        base = coords_grid(N, h8, w8)
        for _ in range(iters):
            coords2 = base + flow_8x
            corr = corr_fn(coords2)
            net = self.update_block(net, context, corr, flow_8x)
            flow_update = self._flow_head(net)
            weight_update = 0.25 * self._upsample_weight(net)
            flow_8x = flow_8x + flow_update[..., :2]
            info_8x = flow_update[..., 2:]
            flow_up, info_up = self.upsample_data(flow_8x, info_8x, weight_update)
            flow_predictions.append(flow_up)
            info_predictions.append(info_up)

        if pad_h or pad_w:
            def unpad(x):
                return x[:, pads[2] : H - pads[3], pads[0] : W - pads[1], :]
            flow_predictions = [unpad(f) for f in flow_predictions]
            info_predictions = [unpad(f) for f in info_predictions]

        return {"final": flow_predictions[-1], "flow": flow_predictions, "info": info_predictions}
