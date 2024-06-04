# Source: https://github.com/SafeAILab/EAGLE/blob/main/eagle/model/cnets.py
# Modifications: update the entire LlamaModifier
""" PyTorch LLaMA model."""

import copy
import os
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

from camel.utils.choices import mc_sim_7b_63
from camel.utils.tree import generate_tree_buffers_camel

top_k = 10


def _compression_make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
    window_size=4,
):
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len, past_key_values_length, dtype=dtype, device=device
                ),
                mask,
            ],
            dim=-1,
        )

    block_mask = (
        torch.arange(past_key_values_length + tgt_len, device=device).unsqueeze(0)
        // window_size
    )
    block_mask = block_mask != block_mask.T
    casual_block_mask = torch.logical_or(
        mask, block_mask[past_key_values_length : past_key_values_length + tgt_len, :]
    )
    mask = torch.where(casual_block_mask, torch.finfo(dtype).min, 0)
    mask = mask[None, None, :, :].expand(bsz, 1, -1, -1)
    return mask


def _speculation_make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_seen_compress_token: int = 0,
    window_size: int = 4,
    key_value_length: int = 0,
    speculation_length: int = 0,
):
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    if past_seen_compress_token < key_value_length:
        # prefill, TODO: need to optimize
        block_mask = torch.arange(tgt_len, device=device).unsqueeze(0) // window_size
        block_mask = block_mask != block_mask.T
        casual_block_mask = torch.logical_or(mask, block_mask)
        mask = torch.where(casual_block_mask, torch.finfo(dtype).min, 0)
    if speculation_length > 0:
        speculaiton_mask = torch.zeros(
            (tgt_len, speculation_length), dtype=dtype, device=device
        )
        mask = torch.cat([speculaiton_mask, mask], dim=-1)
    key_value_length += past_seen_compress_token
    if key_value_length > 0:
        kv_mask = torch.full(
            (tgt_len, key_value_length), torch.finfo(dtype).min, device=device
        )
        q_position_id = torch.arange(
            past_seen_compress_token * window_size,
            past_seen_compress_token * window_size + tgt_len,
            device=device,
        )
        kv_position_id = torch.arange(
            window_size - 1,
            key_value_length * window_size,
            step=window_size,
            device=device,
        )
        kv_mask.masked_fill_(q_position_id.view(-1, 1) > kv_position_id, 0).to(dtype)
        mask = torch.cat([kv_mask, mask], dim=-1)
    mask = mask[None, None, :, :].expand(bsz, 1, -1, -1)
    return mask


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings)
                - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim, max_position_embeddings=self.max_position_embeddings
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaCrossAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim, max_position_embeddings=self.max_position_embeddings
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        speculation_hidden_states: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if key_value_states is not None:
            # cross_attentions
            if self.config.pretraining_tp > 1:
                key_value_slicing = (
                    self.num_key_value_heads * self.head_dim
                ) // self.config.pretraining_tp
                key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
                value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

                key_states_c = [
                    F.linear(key_value_states, key_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
                key_states_c = torch.cat(key_states_c, dim=-1)

                value_states_c = [
                    F.linear(key_value_states, value_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
                value_states_c = torch.cat(value_states_c, dim=-1)

            else:
                key_states_c = self.k_proj(key_value_states)
                value_states_c = self.v_proj(key_value_states)

            key_states_c = key_states_c.view(
                bsz, -1, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)
            value_states_c = value_states_c.view(
                bsz, -1, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)

            if past_key_value is not None:
                key_states_c = torch.cat([past_key_value[0], key_states_c], dim=2)
                value_states_c = torch.cat([past_key_value[1], value_states_c], dim=2)
            past_key_value = (key_states_c, value_states_c)

        key_value_states = torch.concat(
            [speculation_hidden_states, hidden_states], dim=1
        )

        # self attention
        if self.config.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(key_value_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(key_value_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(key_value_states)
            value_states = self.v_proj(key_value_states)

        query_states = query_states.view(
            bsz, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, -1, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, -1, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        kv_seq_len = key_states.shape[-2]

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [
                    F.linear(x, gate_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ],
                dim=-1,
            )
            up_proj = torch.cat(
                [
                    F.linear(x, up_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ],
                dim=-1,
            )

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaCompressionLayer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.index = index
        if self.index != 0:
            self.input_layernorm = LlamaRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        if self.index != 0:
            hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class LlamaSpeculationLayer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.cross_attn = LlamaCrossAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.index = index
        if self.index != 0:
            self.input_layernorm = LlamaRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        speculation_hidden_states: torch.Tensor,
        key_value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        if self.index != 0:
            hidden_states = self.input_layernorm(hidden_states)

        # Cross Attention
        hidden_states, self_attn_weights, present_key_value = self.cross_attn(
            hidden_states=hidden_states,
            key_value_states=key_value_states,
            speculation_hidden_states=speculation_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class LlamaModifier(nn.Module):
    def __init__(self, config, load_embedding=False, model_path=None, bias=True):
        super().__init__()

        self.gradient_checkpointing = True
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.window_size = config.window_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.original_hidden_size, self.padding_idx
        )
        if load_embedding:
            from safetensors import safe_open
            import json

            try:
                with open(
                    os.path.join(model_path, "model.safetensors.index.json"), "r"
                ) as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                with safe_open(
                    os.path.join(model_path, emb_path), framework="pt", device="cpu"
                ) as f:
                    tensor_slice = f.get_slice("model.embed_tokens.weight")
                    vocab_size, hidden_dim = tensor_slice.get_shape()
                    tensor = tensor_slice[:, :hidden_dim].float()
            except:
                with open(
                    os.path.join(model_path, "pytorch_model.bin.index.json"), "r"
                ) as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                weights = torch.load(os.path.join(model_path, emb_path))
                tensor = weights["model.embed_tokens.weight"].float()
            self.embed_tokens.weight.data = tensor

        # self.init_tree()

        self.speculation_layer = LlamaSpeculationLayer(config, 0)
        self.compression_layer = LlamaCompressionLayer(config, 0)

        self.fc = nn.Linear(
            2 * config.original_hidden_size, config.hidden_size, bias=bias
        )
        self.fc2 = nn.Linear(config.hidden_size, config.original_hidden_size, bias=bias)
        self.act = ACT2FN[config.hidden_act]
        for param in self.embed_tokens.parameters():
            param.requires_grad = False

    def init_tree(self):
        self.tree = mc_sim_7b_63
        self.tree_buffer = generate_tree_buffers_camel(
            self.tree, self.embed_tokens.weight.device
        )

    def reset(self):
        self.tree_mask = None

    def _prepare_compression_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _compression_make_causal_mask(
                input_shape,
                # inputs_embeds.dtype,
                torch.float32,  # [MODIFIED] force to cast to float32
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
                window_size=self.window_size,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, torch.float32, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        # [MODIFIED] add tree mask
        if hasattr(self, "tree_mask") and self.tree_mask is not None:
            tree_mask = self.tree_mask
            tree_len = tree_mask.size(-1)
            combined_attention_mask[:, :, -tree_len:, -tree_len:][tree_mask == 0] = (
                torch.finfo(torch.float32).min
            )

        return combined_attention_mask

    def _prepare_speculation_cross_attention_mask(
        self,
        attention_mask,
        input_shape,
        inputs_embeds,
        past_seen_compress_token,
        key_value_length,
        speculation_length,
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            # TODO: =1 also need
            combined_attention_mask = _speculation_make_causal_mask(
                input_shape,
                # inputs_embeds.dtype,
                torch.float32,  # [MODIFIED] force to cast to float32
                device=inputs_embeds.device,
                past_seen_compress_token=past_seen_compress_token,
                window_size=self.window_size,
                key_value_length=key_value_length,
                speculation_length=speculation_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = torch.concat(
                [
                    torch.ones(
                        (
                            attention_mask.shape[0],
                            key_value_length
                            + past_seen_compress_token
                            + speculation_length,
                        ),
                        device=attention_mask.device,
                    ),
                    attention_mask,
                ],
                dim=1,
            )
            expanded_attn_mask = _expand_mask(
                attention_mask, torch.float32, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        # [MODIFIED] add tree mask
        # TODO: check whether needs to use tree mask
        if hasattr(self, "tree_mask") and self.tree_mask is not None:
            tree_mask = self.tree_mask
            tree_len = tree_mask.size(-1)
            combined_attention_mask[:, :, -tree_len:, -tree_len:][tree_mask == 0] = (
                torch.finfo(torch.float32).min
            )

        return combined_attention_mask

    def forward(
        self,
        hidden_states,
        input_ids,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        std=None,
        compression_past_key_values: Optional[
            List[torch.FloatTensor]
        ] = None,  # for speculation layer
        speculation_past_key_values: Optional[
            List[torch.FloatTensor]
        ] = None,  # for speculation layer
        speculation_hidden_states: Optional[torch.FloatTensor] = None,
        is_tree_decode: bool = False,
    ):
        batch_size, seq_len, _ = hidden_states.shape
        dtype, device = hidden_states.dtype, hidden_states.device

        with torch.no_grad():
            inputs_embeds = self.embed_tokens(input_ids)

        inputs_embeds = inputs_embeds.to(hidden_states.dtype)

        # down sample
        input_hidden_states = self.fc(torch.cat((inputs_embeds, hidden_states), dim=-1))

        past_seen_compress_tokens = (
            speculation_past_key_values[0][0].shape[2]
            if speculation_past_key_values is not None
            else 0
        )
        win_len = (
            speculation_hidden_states.shape[1]
            if speculation_hidden_states is not None
            else 0
        )

        if attention_mask is None:
            attention_mask = torch.ones(
                (
                    batch_size,
                    input_hidden_states.shape[1]
                    + win_len
                    + past_seen_compress_tokens * self.window_size,
                ),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )

        all_hidden_states = () if output_hidden_states else None
        next_decoder_cache = () if use_cache else None
        next_decoder_cache_compresssion = () if use_cache else None

        if is_tree_decode:
            # speculation
            speculation_attention_mask = attention_mask[:, -seq_len:]
            speculation_position_ids = torch.arange(
                past_seen_compress_tokens * self.window_size + win_len,
                past_seen_compress_tokens * self.window_size + win_len + seq_len,
                device=device,
                dtype=torch.long,
            )
            speculation_position_ids = speculation_position_ids[None, :].expand(
                batch_size, -1
            )
            window_hidden_states = (
                speculation_hidden_states
                if speculation_hidden_states is not None
                else torch.empty(
                    (batch_size, 0, input_hidden_states.shape[2]),
                    dtype=dtype,
                    device=device,
                )
            )
            speculation_attention_mask = self._prepare_speculation_cross_attention_mask(
                speculation_attention_mask,
                (batch_size, seq_len),
                input_hidden_states,
                past_seen_compress_tokens,
                key_value_length=0,
                speculation_length=win_len,
            )
            speculation_past_key_value = (
                speculation_past_key_values[0]
                if speculation_past_key_values is not None
                else None
            )
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(
                            *inputs, speculation_past_key_value, output_attentions
                        )

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.speculation_layer),
                    input_hidden_states,
                    window_hidden_states,
                    None,
                    speculation_attention_mask,
                    speculation_position_ids,
                    use_reentrant=False,
                )
            else:
                layer_outputs = self.speculation_layer(
                    input_hidden_states,
                    window_hidden_states,
                    None,
                    attention_mask=speculation_attention_mask,
                    position_ids=speculation_position_ids,
                    past_key_value=speculation_past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (
                    layer_outputs[2 if output_attentions else 1],
                )  # past_key_values
            speculation_hidden_states = torch.cat(
                [speculation_hidden_states, input_hidden_states], dim=1
            )
        else:
            speculation_key_value_states = None

            if speculation_past_key_values is None:
                # Prefill
                speculation_len = input_hidden_states.shape[1] % self.window_size
                compression_len = input_hidden_states.shape[1] - speculation_len
                compression_hidden_states = input_hidden_states[:, :compression_len, :]
                speculation_hidden_states = input_hidden_states[:, compression_len:, :]
                window_hidden_states = torch.empty(
                    (batch_size, 0, input_hidden_states.shape[2]),
                    dtype=dtype,
                    device=device,
                )
            else:
                # Decode
                if win_len != 0 and win_len % self.window_size == 0:
                    # overflow the window
                    compression_hidden_states = speculation_hidden_states
                    speculation_hidden_states = input_hidden_states
                    window_hidden_states = torch.empty(
                        (batch_size, 0, input_hidden_states.shape[2]),
                        dtype=dtype,
                        device=device,
                    )
                else:
                    window_hidden_states = speculation_hidden_states
                    speculation_hidden_states = torch.concat(
                        [speculation_hidden_states, input_hidden_states], dim=1
                    )
                    compression_hidden_states = torch.empty(
                        (batch_size, 0, input_hidden_states.shape[2]),
                        dtype=dtype,
                        device=device,
                    )
                compression_len = compression_hidden_states.shape[1]

            # compression
            if compression_len != 0 and compression_len % self.window_size == 0:
                # need to compress
                compression_attention_mask = attention_mask[:, :compression_len]
                compression_position_ids = torch.arange(
                    0,
                    compression_len,
                    dtype=torch.long,
                    device=device,
                )[None, :].expand(hidden_states.shape[0], -1)

                compression_attention_mask = (
                    self._prepare_compression_decoder_attention_mask(
                        compression_attention_mask,
                        (batch_size, compression_len),
                        compression_hidden_states,
                        0,
                    )
                )

                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                compression_past_key_value = (
                    compression_past_key_values[0]
                    if compression_past_key_values is not None
                    else None
                )

                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            # None for past_key_value
                            return module(*inputs, None, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(self.compression_layer),
                        compression_hidden_states,
                        compression_attention_mask,
                        compression_position_ids,
                        use_reentrant=False,
                    )
                else:
                    layer_outputs = self.compression_layer(
                        compression_hidden_states,
                        attention_mask=compression_attention_mask,
                        position_ids=compression_position_ids,
                        past_key_value=None,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )

                hidden_states = layer_outputs[0]
                if use_cache:
                    next_decoder_cache_compresssion += (
                        layer_outputs[2 if output_attentions else 1],
                    )  # past_key_values

                speculation_key_value_states = hidden_states[
                    :, (self.window_size - 1) :: self.window_size, :
                ]
            else:
                next_decoder_cache_compresssion = compression_past_key_values

            # speculation
            speculation_attention_mask = attention_mask[:, -seq_len:]
            speculation_position_ids = torch.arange(
                past_seen_compress_tokens * self.window_size,
                past_seen_compress_tokens * self.window_size + seq_len,
                device=device,
                dtype=torch.long,
            )

            speculation_position_ids = speculation_position_ids[None, :].expand(
                batch_size, -1
            )

            speculation_attention_mask = self._prepare_speculation_cross_attention_mask(
                speculation_attention_mask,
                (batch_size, seq_len),
                input_hidden_states,
                past_seen_compress_tokens,
                key_value_length=(
                    speculation_key_value_states.shape[1]
                    if speculation_key_value_states is not None
                    else 0
                ),
                speculation_length=window_hidden_states.shape[1],
            )

            speculation_past_key_value = (
                speculation_past_key_values[0]
                if speculation_past_key_values is not None
                else None
            )

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(
                            *inputs, speculation_past_key_value, output_attentions
                        )

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.speculation_layer),
                    input_hidden_states,
                    window_hidden_states,
                    speculation_key_value_states,
                    speculation_attention_mask,
                    speculation_position_ids,
                    use_reentrant=False,
                )
            else:
                layer_outputs = self.speculation_layer(
                    input_hidden_states,
                    window_hidden_states,
                    speculation_key_value_states,
                    attention_mask=speculation_attention_mask,
                    position_ids=speculation_position_ids,
                    past_key_value=speculation_past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (
                    layer_outputs[2 if output_attentions else 1],
                )  # past_key_values

        # upper sample
        hidden_states = self.fc2(hidden_states)
        if use_cache:
            return (
                hidden_states,
                next_decoder_cache,
                next_decoder_cache_compresssion,
                speculation_hidden_states,
            )

        return hidden_states

    @torch.no_grad()
    def repeat_kv(self, kv, numr):
        newkv = []
        for i in kv:
            newkv.append((i[0].repeat(numr, 1, 1, 1), i[1].repeat(numr, 1, 1, 1)))
        return tuple(newkv)

    @torch.no_grad()
    def reduce_kv(self, kv, numr):
        newkv = []
        for i in kv:
            newkv.append((i[0][:numr], i[1][:numr]))
        return tuple(newkv)

    def reset_kv(self):
        self.stable_kv = None
        self.compression_stable_kv = None
        self.window_hidden_states = None

    @torch.no_grad()
    def repeat_hidden(self, hidden_state, repeat_num):
        new_hidden = []
        for id, i in enumerate(repeat_num):
            new_hidden.append(hidden_state[:, id : id + 1].repeat(1, i, 1))
        return torch.cat(new_hidden, dim=1)

    def sample(self, logits, logits_processor, k=1, replacement=False):
        logits = logits_processor(None, logits)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        sampled_indices = torch.multinomial(probabilities, k, replacement=False)
        sampled_probs = torch.gather(probabilities, 1, sampled_indices)

        cumulative_sum = torch.cumsum(sampled_probs, dim=1)
        cumulative_sum = torch.cat(
            (
                torch.zeros(cumulative_sum.shape[0], 1, device=cumulative_sum.device),
                cumulative_sum[:, :-1],
            ),
            dim=-1,
        )

        sampled_probs = sampled_probs / (1 - cumulative_sum)
        sampled_probs[torch.isinf(sampled_probs)] = -1
        sampled_probs[torch.isnan(sampled_probs)] = -1
        sampled_probs = torch.clamp(sampled_probs, min=0.0, max=1.0)

        return sampled_indices, sampled_probs, probabilities

    @torch.no_grad()
    def topK_generate(
        self,
        hidden_states,
        input_ids,
        head,
        logits_processor,
        max_length=4,
        use_cache=True,
    ):
        input_ids = input_ids[:, 1:]
        input_ids = input_ids.to(hidden_states.device)
        ss_token, ss_prob, ss_op = [], [], []
        len_posi = input_ids.shape[1]
        self.reset()
        if use_cache:

            if hasattr(self, "stable_kv") and self.stable_kv is not None:
                kv_len = self.stable_kv[0][0].shape[2]
                win_len = self.window_hidden_states.shape[1]
                (
                    out_hidden,
                    speculation_past_key_values,
                    compression_past_key_values,
                    speculation_hidden_states,
                ) = self(
                    hidden_states,
                    input_ids=input_ids[:, kv_len * self.window_size + win_len :],
                    use_cache=True,
                    speculation_past_key_values=self.stable_kv,
                    compression_past_key_values=self.compression_stable_kv,
                    speculation_hidden_states=self.window_hidden_states,
                )
            else:
                (
                    out_hidden,
                    speculation_past_key_values,
                    compression_past_key_values,
                    speculation_hidden_states,
                ) = self(hidden_states, input_ids=input_ids, use_cache=True)
            self.stable_kv = speculation_past_key_values
            self.compression_stable_kv = compression_past_key_values
            self.window_hidden_states = speculation_hidden_states
            last_hidden = out_hidden[:, -1]
            if not self.diff_device:
                last_headout = head(last_hidden)
            else:
                if hasattr(self, "layer_device"):
                    last_headout = head(last_hidden)
                    last_headout = last_headout.to(self.layer_device)
                else:
                    last_headout = F.linear(last_hidden, self.headweight)

            for i in range(len(self.tree_buffer["tree_indices"])):
                if logits_processor is not None:
                    topk_index, topk_prob, op = self.sample(
                        last_headout,
                        logits_processor,
                        k=top_k,
                    )
                else:
                    top = torch.topk(last_headout, top_k, dim=-1)
                    topk_index, topk_prob = top.indices, top.values
                    op = None

                ss_token.append(topk_index)
                ss_prob.append(topk_prob)
                ss_op.append(op)
                # topk_index = torch.topk(last_headout, top_k, dim=-1).indices
                topk_index = topk_index.view(-1)
                select_index = topk_index[self.tree_buffer["tree_indices"][i]]
                # len_sq=select_index.shape[0]
                input_ids = select_index[None, :]
                if i == 0:
                    hidden_states = out_hidden[:, -1:]
                else:
                    hidden_states = out_hidden
                hidden_states = self.repeat_hidden(
                    hidden_states, self.tree_buffer["repeat_nums"][i]
                )
                # hidden_states = hidden_states.repeat(1,len_sq,1)
                self.tree_mask = self.tree_buffer["attn_mask"][i]
                position_ids = len_posi + self.tree_buffer["position_ids"][i]
                (
                    out_hidden,
                    speculation_past_key_values,
                    compression_past_key_values,
                    speculation_hidden_states,
                ) = self(
                    hidden_states,
                    input_ids=input_ids,
                    position_ids=position_ids,
                    use_cache=True,
                    speculation_past_key_values=speculation_past_key_values,
                    speculation_hidden_states=speculation_hidden_states,
                    compression_past_key_values=compression_past_key_values,
                    is_tree_decode=True,
                )
                len_posi += 1

                if not self.diff_device:
                    last_headout = head(out_hidden[0])
                else:
                    if hasattr(self, "layer_device"):
                        last_headout = head(out_hidden[0])
                        last_headout = last_headout.to(self.layer_device)
                    else:
                        last_headout = F.linear(out_hidden[0], self.headweight)

            if logits_processor is not None:
                topk_index, topk_prob, op = self.sample(
                    last_headout,
                    logits_processor,
                    k=top_k,
                )
            else:
                top = torch.topk(last_headout, top_k, dim=-1)
                topk_index, topk_prob = top.indices, top.values
                op = None
            ss_token.append(topk_index)
            ss_prob.append(topk_prob)
            ss_op.append(op)

        else:
            # TODO
            pass

        return (torch.cat(ss_token), torch.cat(ss_prob), ss_op)
