# 
from typing import Tuple, Union
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

LayerNorm = nn.LayerNorm
logger = logging.getLogger(__name__)


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=LayerNorm,
            out_act_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = (bias, bias) if isinstance(bias, bool) else bias
        drop_probs = (drop, drop) if isinstance(drop, float) else drop

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])
        self.out_act = out_act_layer() if out_act_layer is not None else nn.Identity()

        # debug flags
        self.debug_checks = False
        self.debug_name = 'Mlp'
        self.debug_sanitize = False
        self.debug_clamp = None

        # --- new: parameter check flags ---
        self._debug_params_checked_once = False

        # --- new: guard for fan_in == 0 to avoid NaN bias from default init ---
        if self.fc1.in_features == 0 and self.fc1.bias is not None:
            # Default init would make bound=inf -> NaN bias; zero it to avoid NaN propagation
            with torch.no_grad():
                self.fc1.bias.zero_()
            logger.warning("[MlpDebug] %s::init fc1.in_features=0 -> zeroed fc1.bias as a guard", self.debug_name)

    def _check_params_once(self):
        if self._debug_params_checked_once:
            return
        self._debug_params_checked_once = True
        msgs = []
        with torch.no_grad():
            def stat_ok(t, name):
                if t is None:
                    msgs.append(f"{name}: None")
                    return
                finite = torch.isfinite(t)
                if not finite.all():
                    num_bad = (~finite).sum().item()
                    msgs.append(f"{name}: non-finite {num_bad}/{t.numel()}")
                else:
                    msgs.append(f"{name}: ok")
            stat_ok(self.fc1.weight, "fc1.weight")
            stat_ok(self.fc1.bias,   "fc1.bias")
            stat_ok(self.fc2.weight, "fc2.weight")
            stat_ok(self.fc2.bias,   "fc2.bias")
            msgs.append(f"in_features={self.fc1.in_features}, hidden_features={self.fc2.in_features}, out_features={self.fc2.out_features}")
        logger.warning("[MlpDebug] %s::param_check -> %s", self.debug_name, " | ".join(msgs))

    def forward(self, x):
        # Guard against malformed concatenation or empty feature dimensions.
        if x.dim() == 0:
            raise RuntimeError(f"{self.debug_name} received scalar input.")
        
        # --- new: print param status once ---
        # self._check_params_once()

        if self.debug_checks:
            x = self._check_finite(x, "input")
            x = self.fc1(x)
            x = self._check_finite(x, "fc1")
            x = self.act(x)
            x = self._check_finite(x, "act")

            x = self.drop1(x)
            x = self._check_finite(x, "drop1")

            x = self.norm(x)
            x = self._check_finite(x, "norm")

            x = self.fc2(x)
            x = self._check_finite(x, "fc2")

            x = self.drop2(x)
            x = self._check_finite(x, "drop2")

            x = self.out_act(x)
            x = self._check_finite(x, "out")
        else:
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop1(x)
            x = self.norm(x)
            x = self.fc2(x)
            x = self.drop2(x)
            x = self.out_act(x)
        return x
    
    @torch.no_grad()
    def _check_finite(self, x: torch.Tensor, tag: str) -> torch.Tensor:
        if not torch.is_tensor(x):
            return x
        finite = torch.isfinite(x)
        if not finite.all():
            num_bad = (~finite).sum().item()
            total = x.numel()
            # Compute finite ranges without letting NaN/Inf pollute min/max.
            finite_vals = x[finite]
            fin_min = finite_vals.min().item() if finite_vals.numel() > 0 else float("nan")
            fin_max = finite_vals.max().item() if finite_vals.numel() > 0 else float("nan")
            logger.warning(
                "[MlpDebug] %s::%s non-finite %s/%s, finite[min=%s, max=%s]",
                self.debug_name,
                tag,
                num_bad,
                total,
                fin_min,
                fin_max,
            )
            if self.debug_sanitize:
                clamp = self.debug_clamp if self.debug_clamp is not None else 1e6
                x = torch.nan_to_num(x, nan=0.0, posinf=clamp, neginf=-clamp)
                if self.debug_clamp is not None:
                    x = torch.clamp(x, -self.debug_clamp, self.debug_clamp)
        return x
    
class MixerBlock(nn.Module):
    """Residual Block w/ token mixing and channel MLPs.

    Based on: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    def __init__(
            self,
            dim: int,
            seq_len: int,
            mlp_ratio: Union[float, Tuple[float, float]] = (0.25, 0.5),
            act_layer: type = nn.GELU,
            drop: float = 0.,
            drop_path: float = 0.,
            residual_scale: float = 1.0,
    ) -> None:
        """Initialize MixerBlock.

        Args:
            dim: Dimension of input features.
            seq_len: Sequence length.
            mlp_ratio: Expansion ratios for token mixing and channel MLPs.
            mlp_layer: MLP layer class.
            norm_layer: Normalization layer.
            act_layer: Activation layer.
            drop: Dropout rate.
            drop_path: Drop path rate.
            residual_scale: Scaling factor for residual connections.
        """
        super().__init__()
        tokens_dim, channels_dim = [int(x * dim) for x in mlp_ratio]
        self.norm1 = LayerNorm(dim)
        self.mlp_tokens = Mlp(seq_len, tokens_dim, act_layer=act_layer, drop=drop)
        self.drop_path = nn.Identity()
        self.norm2 = LayerNorm(dim)
        self.mlp_channels = Mlp(dim, channels_dim, act_layer=act_layer, drop=drop)
        self.residual_scale = residual_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        orig_dtype = x.dtype
        # path 1: token mixing
        y = self.norm1(x)
        with torch.autocast(device_type='cuda', enabled=False):
            y = self.mlp_tokens(y.transpose(1, 2)).transpose(1, 2)
        x = x + self.residual_scale * y
        # path 2: channel mixing
        y2 = self.norm2(x)
        with torch.autocast(device_type='cuda', enabled=False):
            y2 = self.mlp_channels(y2)
        x = x + self.residual_scale * y2
        x = x.to(orig_dtype)
        return x


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x, mask=None):
        # x: (B, T, hidden_dim)
        weights = self.attention(x)  # (B, T, 1)
        if mask is not None:
            weights = weights.masked_fill(~mask.unsqueeze(-1), -1e9)
        weights = F.softmax(weights, dim=1)
        pooled = torch.sum(x * weights, dim=1)  # (B, hidden_dim)
        return pooled
