import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.jit import Final


class RelPosMultiHeadAttn2D(nn.Module):
    d_head: Final[int]
    d_model: Final[int]
    n_head: Final[int]

    def __init__(self, d_model, d_head, n_head, h, w):
        super().__init__()
        self.d_head = d_head
        self.n_head = n_head
        self.d_model = d_model
        self.register_buffer("scale", torch.tensor(1.0 / (self.d_head ** 0.5)))

        self.qkv = nn.Conv2d(d_model, d_head * n_head * 3, kernel_size=1)

        self.rh_k = nn.Parameter(
            torch.randn(h * 2 - 1, n_head, d_head) * self.scale
        )
        self.rw_k = nn.Parameter(
            torch.randn(w * 2 - 1, n_head, d_head) * self.scale
        )

        self.rh_v = nn.Parameter(
            torch.randn(1, h * 2 - 1, n_head, d_head) * self.scale
        )
        self.rw_v = nn.Parameter(
            torch.randn(1, w * 2 - 1, n_head, d_head) * self.scale
        )

        self.fc = nn.Conv2d(d_head * n_head, d_model, kernel_size=1)

        self.norm = nn.GroupNorm(16, d_model)

    @torch.jit.export
    def rel_shift(self, x, qlen: int):
        zero_pad_shape = (x.size(0), 1) + x.size()[2:]
        zero_pad = torch.zeros(zero_pad_shape, device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded_shape = (x.size(1) + 1, x.size(0)) + x.size()[2:]
        x_padded = x_padded.view(x_padded_shape)

        return x_padded[1:].view_as(x)[:, 0:qlen]

    @torch.jit.export
    def rel_shift_2d(self, xh, xw, h: int, w: int):
        xh = self.rel_shift(xh, h)  # [h*w, h, b, n_head]
        xw = self.rel_shift(xw, w)  # [h*w, w, b, n_head]

        xh = xh.repeat(1, w, 1, 1)
        xw = xw.repeat(1, h, 1, 1)

        return xh + xw

    def forward(self, x: torch.Tensor):
        residual = x

        b, c, h, w = x.size()
        t = h * w

        q, k, v = torch.chunk(
            self.qkv(x).view(b, self.n_head * 3, self.d_head, t), 3, dim=1
        )

        ac = torch.einsum("bndi, bndj -> bnij", q, k)

        bd_h = torch.einsum("bndi, jnd -> ijbn", q, self.rh_k)
        bd_w = torch.einsum("bndi, jnd -> ijbn", q, self.rw_k)

        bd = self.rel_shift_2d(bd_h, bd_w, h, w)  # [h*w, h*w, b, n_head]
        bd = bd.permute(2, 3, 0, 1)

        attn = F.softmax((ac + bd) * self.scale, dim=1)

        pos_emb_v = self.rel_shift_2d(
            self.rh_v.repeat(t, 1, 1, 1), self.rw_v.repeat(t, 1, 1, 1), h, w
        )  # [h*w, h*w, n_head, d_head]

        x = torch.einsum("bnij, bndj -> bndi", attn, v) + torch.einsum(
            "bnij, ijnd -> bndi", attn, pos_emb_v
        )

        x = x.reshape(b, self.n_head * self.d_head, h, w)
        x = self.fc(x)

        return self.norm(x + residual)


class QueriedAttn2D(nn.Module):
    d_head: Final[int]
    d_model: Final[int]
    n_head: Final[int]

    def __init__(self, d_model, d_head, n_head):
        super().__init__()
        self.d_head = d_head
        self.n_head = n_head
        self.d_model = d_model
        self.register_buffer("scale", torch.tensor(1.0 / (self.d_head ** 0.5)))

        self.kv = nn.Conv2d(d_model, d_head * n_head * 2, kernel_size=1)
        self.q = nn.Linear(d_model, d_head * n_head)

        self.fc = nn.Sequential(
            nn.Linear(d_head * n_head, d_model), nn.ReLU(True)
        )

    def forward(self, x: torch.Tensor, query: torch.Tensor):
        b, c, h, w = x.size()
        t = h * w

        k, v = torch.chunk(
            self.kv(x).view(b, self.n_head * 2, self.d_head, t), 2, dim=1
        )
        q = self.q(query).view(b, self.n_head, self.d_head)

        logits = torch.einsum("bnd, bndj -> bnj", q, k)

        attn = F.softmax(logits * self.scale, dim=2)

        x = torch.einsum("bnj, bndj -> bnd", attn, v)

        x = x.reshape(b, self.n_head * self.d_head)
        x = self.fc(x)

        return x


class SpatialCompressionAttnModule(nn.Module):
    layers: Final[nn.ModuleList]

    def __init__(
        self, d_model, height, width, n_head=8, d_head=None, n_layers=3
    ):
        super().__init__()
        if d_head is None:
            d_head = d_model // n_head

        self.layers = nn.ModuleList(
            [
                RelPosMultiHeadAttn2D(d_model, d_head, n_head, height, width)
                for _ in range(n_layers)
            ]
        )
        self.input_norm = nn.GroupNorm(16, d_model)

        self.register_buffer(
            "selection_point", torch.tensor([0.0, 0.0]).view(1, 1, 1, 2)
        )

    def forward(self, x):
        b, c, h, w = x.size()
        x = self.input_norm(x)

        for layer in self.layers:
            x = layer(x)

        return F.grid_sample(
            x, self.selection_point.expand(b, 1, 1, 2), align_corners=False
        ).view(b, c)


class QueriedSpatialCompressionAttnModule(SpatialCompressionAttnModule):
    layers: Final[nn.ModuleList]

    def __init__(
        self, d_model, height, width, n_head=8, d_head=None, n_layers=3
    ):
        if d_head is None:
            d_head = d_model // n_head
        super().__init__(
            d_model=d_model,
            height=height,
            width=width,
            n_head=n_head,
            d_head=d_head,
            n_layers=n_layers,
        )

        del self.selection_point

        self.query_compress = QueriedAttn2D(d_model, d_head, n_head)

    def forward(self, x, query):
        b, c, h, w = x.size()
        x = self.input_norm(x)

        for layer in self.layers:
            x = layer(x)

        return self.query_compress(x, query)


if __name__ == "__main__":
    x = torch.randn(8, 512, 4, 4)
    query = torch.randn(8, 512)
    tgt = torch.relu_(torch.randn(8, 512))
    model = torch.jit.script(
        QueriedSpatialCompressionAttnModule(
            512, n_head=8, n_layers=3, height=4, width=4
        )
    )
    print(sum(param.numel() for param in model.parameters()))
    optim = optim.Adam(model.parameters(), lr=5e-4)

    for _ in range(200):
        optim.zero_grad()

        preds = model(x, query)
        loss = F.smooth_l1_loss(preds, tgt)

        loss.backward()

        optim.step()

        print(loss.item())
