from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F

from fla.layers.utils import get_layer_cache, get_unpad_data, index_first_axis, pad_input, update_layer_cache
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from ccq.ops.ccq_gdn import (
    chunk_ccq_gdn,
    fused_chunk_ccq_gdn,
    fused_recurrent_ccq_gdn,
)

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


@torch.compile
def elu_p1(x):
    return (F.elu(x, 1., False) + 1.).to(x)


@torch.compile
def sum_norm(x):
    return (x / x.sum(-1, keepdim=True)).to(x)


class CCQGDN(nn.Module):
    """
    The layer implementaion for [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464).  # noqa

    Similar to Mamba2, each layer contains around 6*hidden_size*hidden_size parameters.

    Parameter alloation when use_gate=True:
        - 0.75 * hidden_size * hidden_size for the q_proj and k_proj each
        - 1.5 * hidden_size * hidden_size for the v_proj, g_proj and o_proj each
        - Others are ignorably small.
        - In total = 0.75 * 2 + 1.5 * 3 = 6 * hidden_size * hidden_size
    NOTE: num_heads * head_dim = 0.75 * hidden_size, please make sure to set the correct num_heads and head_dim.

    Parameter allocation when use_gate=False:
        - 1 * hidden_size * hidden_size for the q_proj and k_proj each
        - 2 * hidden_size * hidden_size for the v_proj and o_proj each
        - Others are ignorably small.
        - In total = 1 * 2 + 2 * 2 = 6 * hidden_size * hidden_size

    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 2048.
        expand_v (float, Optional):
            The expansion ratio for the value dim. Default: 2.0.
        head_dim (int, Optional):
            The dimension of each head. Default: 256.
        num_heads (int, Optional):
            The number of heads. Default: 4.
        num_v_heads (int, Optional):
            The number of heads for the value projection, equal to `num_heads` if `None`.
            GVA is applied if `num_v_heads` > `num_heads`. Default: `None`.
        mode (str, Optional):
            Which Gated DeltaNet kernel to use.
            Currently available: `chunk` and `fused_recurrent`.
            Default: `chunk`.
        use_beta (bool, Optional):
            Whether to use beta. Default: `True`.
        use_gate (bool, Optional):
            Whether to use output gate. Default: `True`.
        use_short_conv (bool, Optional):
            Whether to use short convolutions. Default: `True`.
        allow_neg_eigval (bool, Optional):
            Allow negative eigenvalues. Default: `False`. If set to `True`, the beta will be multiplied by 2.
            See reference: [Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues](https://arxiv.org/abs/2411.12537)
        conv_size (int, Optional):
            The kernel size of the short convolution, only used when `use_short_conv` is `True`. Default: 4.
        conv_bias (bool, Optional):
            Whether to use bias in the short convolution, only used when `use_short_conv` is `True`. Default: `False`.
        layer_idx (int, Optional):
            The index of the layer. Default: None.
        norm_eps (float, Optional):
            The epsilon value for the normalization layer. Default: 1e-5.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 2,
        head_dim: int = 256,
        num_heads: int = 6,
        num_v_heads: int = None,
        num_kv_heads: int = None,
        mode: str = 'chunk',
        use_gate: bool = True,
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        erase_lambda_init: float = 0.01,
        subtract_mean: bool = True,
        enable_ccq: bool = True,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        **kwargs,
    ) -> CCQGDN:
        super().__init__()

        self.mode = mode
        self.subtract_mean = subtract_mean
        self.enable_ccq = enable_ccq
        self.allow_neg_eigval = allow_neg_eigval
        self.hidden_size = hidden_size
        self.expand_v = expand_v

        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads
        # GQA: num_kv_heads <= num_heads. K and V projections produce
        # num_kv_heads * head_dim; K is repeated to num_heads and V to
        # num_v_heads inside forward(). Default preserves MHA behaviour.
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        if self.num_kv_heads > self.num_heads:
            raise ValueError(
                f"num_kv_heads={self.num_kv_heads} must be <= "
                f"num_heads={self.num_heads}. For more V heads than Q heads "
                f"use num_v_heads (GVA), not num_kv_heads.",
            )
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads={self.num_heads} must be a multiple of "
                f"num_kv_heads={self.num_kv_heads}.",
            )

        self.head_k_dim = head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)
        # Projection-side dims (smaller when GQA is enabled).
        self.kv_key_dim = int(self.num_kv_heads * self.head_k_dim)
        self.kv_value_dim = int(self.num_kv_heads * self.head_v_dim)
        self.layer_idx = layer_idx

        # Consistency check: Ensure expand_v produces integer values
        if not math.isclose(self.num_v_heads * self.head_dim * expand_v, self.value_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by key_dim={self.key_dim}. "
                f"Resulting value_dim would be {self.num_v_heads * self.head_dim * expand_v}, which is invalid for nn.Linear.",
            )
        if self.num_v_heads > self.num_heads and self.num_v_heads % self.num_heads != 0:
            raise ValueError(
                f"num_v_heads={self.num_v_heads} must be divisible by num_heads={self.num_heads}.",
            )

        if not math.isclose(head_dim * expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by head_dim={head_dim}. "
                f"Resulting head_v_dim would be {head_dim * expand_v}, which is invalid for FusedRMSNormGated.",
            )
        assert mode in ['chunk', 'fused_chunk', 'fused_recurrent'], f"Not supported mode `{mode}`."

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.kv_key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.kv_value_dim, bias=False)
        self.erase_lambda_proj = nn.Linear(self.key_dim, self.num_heads, bias=True)
        # λ_h = sigmoid(erase_lambda_proj(q)). At init, we want E[λ_h] ≈ erase_lambda_init
        # but with per-head variation AND mild input dependence — this gives the
        # CCQ cleaning a non-degenerate gradient signal across heads from step 1.
        # Bias is drawn around logit(erase_lambda_init) so each head starts at a slightly
        # different λ; weight is small Gaussian so q can move λ in either direction.
        _bias_mean = math.log(erase_lambda_init / (1.0 - erase_lambda_init))
        nn.init.normal_(self.erase_lambda_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.erase_lambda_proj.bias,   mean=_bias_mean, std=0.1)
        self.a_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
        self.b_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)

        A = torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        # hard coded for now
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.num_v_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min),
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.kv_key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.kv_value_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
            )
        else:
            warnings.warn(
                "ShortConvolution is crucial to the performance. "
                "Do not turn it off, i.e., setting `use_short_conv=False` unless you know what you are doing.",
            )
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps, dtype=torch.float32)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        # change to inference mode.
        mode = 'fused_recurrent' if (q_len <= 64 and not self.training) else self.mode
        if self.training:
            assert mode in ('chunk', 'fused_chunk'), "Only chunk and fused_chunk modes are supported in training."

        last_state = get_layer_cache(self, past_key_values)

        cu_seqlens = kwargs.get('cu_seqlens')
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))
        lambda_h = F.sigmoid(self.erase_lambda_proj(q))
        q = rearrange(q, '... (h d) -> ... h d', d=self.head_k_dim)
        k = rearrange(k, '... (h d) -> ... h d', d=self.head_k_dim)
        lambda_h = rearrange(lambda_h, '... h -> ... h 1')
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)

        # GQA: repeat K and V from num_kv_heads up to num_heads so the
        # kernel sees full per-Q-head K/V. No-op when num_kv_heads == num_heads.
        if self.num_kv_heads < self.num_heads:
            g_kv = self.num_heads // self.num_kv_heads
            k = repeat(k, '... h d -> ... (h g) d', g=g_kv)
            v = repeat(v, '... h d -> ... (h g) d', g=g_kv)

        if self.num_v_heads > self.num_heads:
            q, k = map(
                lambda x: repeat(x, '... h d -> ... (h g) d', g=self.num_v_heads // self.num_heads),
                (q, k),
            )
            lambda_h = repeat(lambda_h, '... h d -> ... (h g) d', g=self.num_v_heads // self.num_heads)

        beta = self.b_proj(hidden_states).sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.

        g = -self.A_log.float().exp() * F.softplus(self.a_proj(hidden_states).float() + self.dt_bias)

        # Cache packing: 4-tuple (recurrent_state, cov_state, mean_state, position_offset).
        # mean_state is None when subtract_mean=False; older 3-tuple / 2-tuple
        # caches are still accepted for backward compatibility.
        mean_state = None
        if last_state is not None and last_state['recurrent_state'] is not None:
            packed = last_state['recurrent_state']
            if isinstance(packed, tuple) and len(packed) == 4:
                recurrent_state, cov_state, mean_state, position_offset = packed
                position_offset = int(position_offset)
            elif isinstance(packed, tuple) and len(packed) == 3:
                recurrent_state, cov_state, position_offset = packed
                position_offset = int(position_offset)
            elif isinstance(packed, tuple) and len(packed) == 2:
                recurrent_state, cov_state = packed
                position_offset = 0
            else:
                recurrent_state, cov_state, position_offset = None, None, 0
        else:
            recurrent_state, cov_state, position_offset = None, None, 0
        if mode == 'chunk':
            o, recurrent_state, cov_state, mean_state = chunk_ccq_gdn(
                q=q,
                k=k,
                v=v,
                lambda_h=lambda_h,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                initial_cov_state=cov_state,
                initial_mean_state=mean_state if self.subtract_mean else None,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
                position_offset=position_offset,
                subtract_mean=self.subtract_mean,
                enable_ccq=self.enable_ccq,
            )
        elif mode == 'fused_chunk':
            o, recurrent_state, cov_state, mean_state = fused_chunk_ccq_gdn(
                q=q,
                k=k,
                v=v,
                lambda_h=lambda_h,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                initial_cov_state=cov_state,
                initial_mean_state=mean_state if self.subtract_mean else None,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
                position_offset=position_offset,
                subtract_mean=self.subtract_mean,
                enable_ccq=self.enable_ccq,
            )
        elif mode == 'fused_recurrent':
            o, recurrent_state, cov_state, mean_state = fused_recurrent_ccq_gdn(
                q=q,
                k=k,
                v=v,
                lambda_h=lambda_h,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                initial_cov_state=cov_state,
                initial_mean_state=mean_state if self.subtract_mean else None,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
                position_offset=position_offset,
                subtract_mean=self.subtract_mean,
                enable_ccq=self.enable_ccq,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        update_layer_cache(
            self,
            past_key_values,
            recurrent_state=(recurrent_state, cov_state, mean_state, position_offset + q_len),
            conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
            offset=q_len,
        )

        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values
