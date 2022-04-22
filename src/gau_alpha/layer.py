import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import get_activation

INF = 1e12


def attention_normalize(a, l, dim=-1, method="softmax"):
    """不同的注意力归一化方案
    softmax：常规/标准的指数归一化；
    squared_relu：来自 https://arxiv.org/abs/2202.10447 ；
    softmax_plus：来自 https://kexue.fm/archives/8823 。
    """
    if method == "softmax":
        return torch.softmax(a, dim=dim)
    else:
        if method == "squared_relu":
            return torch.relu(a) ** 2 / l
        elif method == "softmax_plus":
            return torch.softmax(a * torch.log(l) / np.log(512), dim=dim)
    return a


class ScaleOffset(nn.Module):
    """简单的仿射变换层（最后一维乘上gamma向量并加上beta向量）
    说明：1、具体操作为最后一维乘上gamma向量并加上beta向量；
         2、如果直接指定scale和offset，那么直接常数缩放和平移；
    """

    def __init__(
        self,
        hidden_size=768,
        scale=True,
        offset=True,
    ):
        super().__init__()
        self.scale = scale
        self.offset = offset

        if self.scale:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        if self.offset:
            self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, inputs):
        if self.scale:
            inputs = inputs * self.weight
        if self.offset:
            inputs = inputs + self.bias

        return inputs


class Norm(nn.Module):
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        variance = torch.mean(torch.square(x), dim=-1, keepdim=True)
        return x / torch.sqrt(variance + self.eps)


class GatedAttentionUnit(nn.Module):
    """门控注意力单元
    链接：https://arxiv.org/abs/2202.10447
    介绍：https://kexue.fm/archives/8934
    说明：没有加入加性相对位置编码，个人认为是不必要的；如果觉得有必要，
         可以自行通过a_bias传入。
    """

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=1536,
        attention_key_size=128,
        activation="swish",
        use_bias=False,
        normalization="softmax_plus",
        attention_scale=True,
        attention_dropout=0.1,
    ):
        super().__init__()
        self.activation = get_activation(activation)
        self.intermediate_size = intermediate_size
        self.attention_key_size = attention_key_size
        self.use_bias = use_bias
        self.normalization = normalization
        self.attention_scale = attention_scale
        self.attention_dropout = attention_dropout

        self.i_dense = nn.Linear(
            hidden_size, 2 * intermediate_size + attention_key_size, bias=self.use_bias
        )
        self.o_dense = nn.Linear(intermediate_size, hidden_size, bias=self.use_bias)

        self.q_scaleoffset = ScaleOffset(attention_key_size, offset=self.use_bias)
        self.k_scaleoffset = ScaleOffset(attention_key_size, offset=self.use_bias)

    @staticmethod
    def apply_rotary(x, sinusoidal_pos=None):
        if sinusoidal_pos is None:
            return x
        sin, cos = sinusoidal_pos
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        sinusoidal_pos=None,
        output_attentions=False,
    ):
        # 投影变换
        x = self.i_dense(hidden_states)
        u, v, qk = torch.split(
            self.activation(x),
            [self.intermediate_size, self.intermediate_size, self.attention_key_size],
            dim=-1,
        )
        q, k = self.q_scaleoffset(qk), self.k_scaleoffset(qk)

        # 加入RoPE
        q, k = self.apply_rotary(q, sinusoidal_pos), self.apply_rotary(
            k, sinusoidal_pos
        )

        # Attention
        a = torch.einsum("bmd,bnd->bmn", q, k)

        if self.attention_scale:
            a = a / self.attention_key_size ** 0.5

        if attention_mask is not None:
            if attention_mask.ndim == 2:
                attention_mask = attention_mask[:, None, :]
            a = a.masked_fill(attention_mask == 0, -INF)
            l = attention_mask.sum(-1, keepdim=True)
        else:
            l = x.shape[1]

        A = attention_normalize(a, l, dim=-1, method=self.normalization)

        A = F.dropout(A, p=self.attention_dropout, training=self.training)

        # 计算输出
        o = self.o_dense(u * torch.einsum("bmn,bnd->bmd", A, v))

        outputs = (o, A) if output_attentions else (o,)
        return outputs


class GAULayer(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        intermediate_size=1536,
        attention_key_size=128,
        activation="swish",
        use_bias=False,
        normalization="softmax_plus",
        attention_scale=True,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        eps=1e-12,
    ):
        super().__init__()
        self.gau = GatedAttentionUnit(
            hidden_size,
            intermediate_size,
            attention_key_size,
            activation,
            use_bias,
            normalization,
            attention_scale,
            attention_dropout,
        )
        self.norm = Norm(eps=eps)
        self.hidden_dropout = hidden_dropout

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        sinusoidal_pos=None,
        output_attentions=False,
    ):
        # 投影变换
        gau_output = self.gau(
            hidden_states, attention_mask, sinusoidal_pos, output_attentions
        )

        # dropout and residual
        o = F.dropout(gau_output[0], p=self.hidden_dropout, training=self.training)
        o = self.norm(hidden_states + o)

        outputs = (o,) + gau_output[1:]  # add attentions if we output them

        return outputs
