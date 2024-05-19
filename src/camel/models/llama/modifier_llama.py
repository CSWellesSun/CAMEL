import torch
from torch import nn

from typing import Optional, Union, Tuple, List
from transformers.models.llama import LlamaModel, LlamaConfig, LlamaPreTrainedModel
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import logging

from ...modifier_outputs import CamelModifierOutput

logger = logging.get_logger(__name__)


class LlamaCamelModifier(LlamaModel):
    def __init__(
        self,
        config: LlamaConfig,
        load_embedding=False,
        model_path="",
        combine_bias=True,
    ):
        LlamaPreTrainedModel.__init__(self, config)  # pay attention to `self`
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.window_size = config.window_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # compression_layer don't need past key values so `layer_idx` is useless
        self.compression_layer = nn.ModuleList(
            [
                LlamaDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.speculation_layer = nn.ModuleList(
            [
                LlamaDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.combine_layer = nn.Linear(
            2 * config.hidden_size, config.hidden_size, bias=combine_bias
        )

        # Initialize weights and apply final processing
        self.post_init()

        if load_embedding:
            import os
            import json
            from safetensors import safe_open

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
        for param in self.embed_tokens.parameters():
            param.requires_grad = False

    def forward(
        self,
        hidden_states: torch.FloatTensor = None,  # [bs, seq_len, hidden_size]
        input_ids: torch.LongTensor = None,  # [bs, seq_len]
        attention_mask: Optional[torch.Tensor] = None,  # [bs, seq_len]
        position_ids: Optional[torch.LongTensor] = None,  # None currently
        inputs_embeds: Optional[torch.FloatTensor] = None,  # [bs, seq_len, hidden_size]
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        speculation_past_key_values: Optional[
            Union[Cache, List[torch.FloatTensor]]
        ] = None,
        compression_past_key_values: Optional[
            Union[Cache, List[torch.FloatTensor]]
        ] = None,
    ) -> Union[Tuple, CamelModifierOutput]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = self.combine_layer(
            torch.concat([hidden_states, inputs_embeds], dim=-1)
        )

        bs, seq_len, hidden_size = hidden_states.shape
        dtype, device = hidden_states.dtype, hidden_states.device

        return_legacy_cache = False
        if use_cache and not isinstance(
            speculation_past_key_values, Cache
        ):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            speculation_past_key_values = DynamicCache.from_legacy_cache(
                speculation_past_key_values
            )

        if use_cache and not isinstance(
            compression_past_key_values, Cache
        ):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            compression_past_key_values = DynamicCache.from_legacy_cache(
                speculation_past_key_values
            )

        past_seen_tokens = compression_past_key_values.get_seq_length()
        if position_ids is None:
            position_ids = torch.arange(
                past_seen_tokens,
                past_seen_tokens + seq_len,
                dtype=torch.long,
                device=device,
            )[None, :].expand(bs, -1)

        # Compression Layer
        compression_attention_mask = self._update_compress_causal_mask(
            seq_len, hidden_states, attention_mask
        )

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        compression_hidden_states = hidden_states
        compression_position_ids = position_ids
        for decoder_layer in self.compression_layer:
            if output_hidden_states:
                all_hidden_states += (compression_hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    compression_hidden_states,
                    compression_attention_mask,
                    compression_position_ids,
                    None,
                    output_attentions,
                    False,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    compression_hidden_states,
                    attention_mask=compression_attention_mask,
                    position_ids=compression_position_ids,
                    past_key_value=None,
                    output_attentions=output_attentions,
                    use_cache=False,
                    cache_position=None,  # DynamicCache don't need
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        compression_next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            compression_next_cache = compression_next_cache.to_legacy_cache()

        # Speculation Layer
        speculation_hidden_states = hidden_states
        speculation_position_ids = position_ids
        cache_position = torch.arange(
            speculation_past_key_values.get_seq_length(),
            speculation_past_key_values.get_seq_length()
            + speculation_hidden_states.shape[1],
            device=device,
        )
        speculation_causal_mask = self._update_causal_mask(
            attention_mask=torch.gather(attention_mask, 1, speculation_position_ids),
            input_tensor=speculation_hidden_states,
            cache_position=cache_position,
            past_key_values=speculation_past_key_values,
            output_attentions=output_attentions,
        )

        for decoder_layer in self.speculation_layer:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    speculation_hidden_states,
                    speculation_causal_mask,
                    speculation_position_ids,
                    speculation_past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    speculation_hidden_states,
                    attention_mask=speculation_causal_mask,
                    position_ids=speculation_position_ids,
                    past_key_value=speculation_past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        speculation_next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            speculation_next_cache = speculation_next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    speculation_next_cache,
                    compression_next_cache,
                    all_hidden_states,
                    all_self_attns,
                ]
                if v is not None
            )
        return CamelModifierOutput(
            last_hidden_state=hidden_states,
            speculation_past_key_values=speculation_next_cache,
            compression_past_key_values=compression_next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_compress_causal_mask(self, seq_len, input_tensor, attention_mask):
        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        bs = input_tensor.shape[0]

        full_casual_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
        )
        block_mask = (
            torch.arange(seq_len, device=device).unsqueeze(0) // self.window_size
        )
        block_mask = block_mask != block_mask.T
        casual_block_mask = torch.logical_or(full_casual_mask, block_mask)
        mask = torch.where(casual_block_mask, min_dtype, 0)
        mask = mask[None, None, :, :].expand(bs, 1, -1, -1)
        if attention_mask is not None:
            # `attention_mask` of shape [bs, seq_len] indicates which token is padding,
            # so add `mask` and `attention_mask`, and we can check which position has
            # both value 0 (padding but casual), and we can mask it
            mask = mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = mask.shape[-1]
            padding_mask = (
                mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            )
            padding_mask = padding_mask == 0
            mask[:, :, :, :mask_length] = mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )
        return mask

    def generate():
        # Tree-Base Generation
        pass


if __name__ == "__main__":
    from transformers import AutoConfig

    camel_config = AutoConfig.from_pretrained(
        "../../../configs/llama_2_chat_7B_config.json"
    )
    camel_modifier = LlamaCamelModifier.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf", config=camel_config
    )
