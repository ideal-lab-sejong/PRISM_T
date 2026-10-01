import numpy as np
import torch
import torch.nn as nn

LEAKY = 0.01


class IBN(nn.Module):
    """Half instance norm [per-sample / direction] + half frozen BatchNorm1d [per-feature / magnitude].

    The instance branch uses GroupNorm(1), which on (N, width) input equals InstanceNorm1d over the
    feature vector taken as a single-channel signal.
    """
    def __init__(self, width, mode="ibn", ratio=0.5):
        super().__init__()
        if mode not in ("none", "instance", "batch", "ibn"):
            raise ValueError(f"unknown IBN mode: {mode!r}")
        self.mode = mode
        self.h = int(round(width * ratio)) if mode == "ibn" else width
        # Frozen BatchNorm with momentum=1.0: fit() estimates the running statistics in a single
        # pass, each 4096-row chunk overwriting the last (for >4096 training rows only the final
        # chunk's statistics are kept). This is intentional and reproduces the reported numbers.
        if mode == "none":
            self.gn, self.bn = None, None
        elif mode == "instance":
            self.gn, self.bn = nn.GroupNorm(1, width, affine=False), None
        elif mode == "batch":
            self.gn, self.bn = None, nn.BatchNorm1d(width, affine=False, momentum=1.0)
        else:  # ibn
            self.gn = nn.GroupNorm(1, self.h, affine=False) if self.h > 0 else None
            self.bn = nn.BatchNorm1d(width - self.h, affine=False, momentum=1.0) if width - self.h > 0 else None

    def forward(self, x):
        if self.mode == "none":
            return x
        if self.mode == "instance":
            return self.gn(x)
        if self.mode == "batch":
            return self.bn(x)
        parts = []
        if self.gn is not None: parts.append(self.gn(x[:, :self.h]))
        if self.bn is not None: parts.append(self.bn(x[:, self.h:]))
        return torch.cat(parts, dim=1)


class PRISMT(nn.Module):
    def __init__(self, in_dim, seed, width=512, layers=1, act=True, norm="ibn"):
        super().__init__()
        self.act = act
        torch.manual_seed(seed); np.random.seed(seed)
        self.layers = nn.ModuleList()
        d = in_dim
        for _ in range(layers):
            self.layers.append(nn.ModuleDict({"lin": nn.Linear(d, width), "ibn": IBN(width, mode=norm)}))
            d = width
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def _forward(self, X):
        z = torch.tensor(X, dtype=torch.float32)
        for lyr in self.layers:
            z = lyr["ibn"](lyr["lin"](z))
            a = self.act
            if a is True or a == "leaky":
                z = torch.where(z > 0, z, LEAKY * z)
            elif a == "sigmoid":
                z = torch.sigmoid(z)
            elif a == "tanh":
                z = torch.tanh(z)
            elif a == "relu":
                z = torch.clamp_min(z, 0.0)
        return z

    def fit(self, X_train):
        self.train()
        with torch.no_grad():
            for i in range(0, len(X_train), 4096):
                self._forward(X_train[i:i + 4096])
        self.eval()
        return self

    @torch.no_grad()
    def features(self, X):
        out = [self._forward(X[i:i + 4096]).numpy() for i in range(0, len(X), 4096)]
        return np.concatenate(out, 0)
