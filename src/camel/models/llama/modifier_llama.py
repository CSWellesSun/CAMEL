import torch
from torch import nn

from typing import Optional, Union, Tuple, List
from transformers.models.llama import LlamaModel, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import logging

from ...modifier_outputs import CamelModifierOutput

logger = logging.get_logger(__name__)


class LlamaCamelModifier(LlamaModel):
    def __init__(self, config: LlamaConfig, load_embedding=False, model_path=""):
        self.window_size = config.window_size
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
        super().__init__(config)
        del self.layers
        if load_embedding:
            import os
            import json
            from safetensors import safe_open

            try:
                with open(os.path.join(model_path, "model.safetensors.index.json"), "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                with safe_open(
                    os.path.join(model_path, emb_path), framework="pt", device="cpu"
                ) as f:
                    tensor_slice = f.get_slice("model.embed_tokens.weight")
                    vocab_size, hidden_dim = tensor_slice.get_shape()
                    tensor = tensor_slice[:, :hidden_dim].float()
            except:
                with open(os.path.join(model_path, "pytorch_model.bin.index.json"), "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                weights = torch.load(os.path.join(model_path, emb_path))
                tensor = weights["model.embed_tokens.weight"].float()
            self.embed_tokens.weight.data = tensor
        for param in self.embed_tokens.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_ids: torch.LongTensor = None,  # [bs, seq_len]
        attention_mask: Optional[torch.Tensor] = None,  # [bs, seq_len]
        position_ids: Optional[torch.LongTensor] = None,  # [bs, seq_len]
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        speculation_past_key_values: Optional[
            Union[Cache, List[torch.FloatTensor]]
        ] = None,
        speculation_hidden_states: Optional[torch.FloatTensor] = None,
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

        hidden_states = inputs_embeds

        return_legacy_cache = False
        if use_cache and not isinstance(
            speculation_past_key_values, Cache
        ):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            speculation_past_key_values = DynamicCache.from_legacy_cache(
                speculation_past_key_values
            )

        if speculation_past_key_values is None:
            # Prefill
            speculation_len = hidden_states.shape[1] // self.window_size
            compression_len = hidden_states.shape[1] - speculation_len
            compression_hidden_states = hidden_states[:compression_len]
            speculation_hidden_states = hidden_states[compression_len:]
        else:
            # Decode
            if (
                speculation_hidden_states.shape[1]
                + hidden_states.shape[1] % self.window_size
                == 0
            ):
                compression_hidden_states = torch.concat(
                    [speculation_hidden_states, hidden_states]
                )
                speculation_hidden_states = torch.tensor([], dtype=inputs_embeds.dtype)
            else:
                speculation_hidden_states = torch.concat(
                    [speculation_hidden_states, hidden_states]
                )
                compression_hidden_states = torch.tensor([], dtype=inputs_embeds.dtype)

        # Compression Layer
        if compression_len != 0:
            past_seen_compress_tokens = (
                speculation_past_key_values.get_seq_length()
                if speculation_past_key_values is not None
                else 0
            )
            compression_position_ids = torch.arange(
                past_seen_compress_tokens * self.window_size,
                past_seen_compress_tokens * self.window_size + compression_len,
                dtype=torch.long,
            ).unsqueeze(0)
            compression_attention_mask = self._update_compress_causal_mask(
                compression_len
            )

            all_hidden_states = () if output_hidden_states else None
            all_self_attns = () if output_attentions else None
            next_decoder_cache = None

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
                        use_cache,
                        compression_position_ids,
                    )
                else:
                    layer_outputs = decoder_layer(
                        compression_hidden_states,
                        attention_mask=compression_attention_mask,
                        position_ids=compression_position_ids,
                        past_key_value=None,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=compression_position_ids,
                    )

                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

            hidden_states = self.norm(hidden_states)
            speculation_hidden_states = torch.concat(
                [
                    hidden_states[(self.window_size - 1) :: self.window_size],
                    speculation_hidden_states,
                ]
            )
            speculation_len += compression_len // 4

        # Speculation Layer
        if past_seen_compress_tokens != 0:
            speculation_position_ids = torch.arange(
                self.window_size - 1,
                past_seen_compress_tokens * self.window_size,
                device=inputs_embeds.device,
                step=self.window_size,
                dtype=torch.long,
            )
        else:
            speculation_position_ids = torch.LongTensor([], dtype=torch)

        cur_speculation_position_ids = torch.arange(
            speculation_position_ids.shape[0],
            speculation_position_ids.shape[0] + speculation_len,
            dtype=torch.long,
        )
        speculation_position_ids = torch.concat(
            [speculation_position_ids, cur_speculation_position_ids]
        ).unsqueeze(0)

        speculation_causal_mask = self._update_causal_mask(
            attention_mask,
            input_tensor=speculation_hidden_states,
            cache_position=cur_speculation_position_ids,
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
                    speculation_position_ids,
                )
            else:
                layer_outputs = decoder_layer(
                    speculation_hidden_states,
                    attention_mask=speculation_causal_mask,
                    position_ids=speculation_position_ids,
                    past_key_value=speculation_past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=speculation_position_ids,
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

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return CamelModifierOutput(
            last_hidden_state=hidden_states,
            speculation_past_key_values=next_cache,
            window_hidden_states=all_hidden_states,
            all_hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_compress_causal_mask(self, seq_len):
        full_casual_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1
        )
        block_mask = torch.arange(seq_len).unsqueeze(0) // self.window_size
        block_mask = block_mask != block_mask.T
        casual_block_mask = torch.logical_or(full_casual_mask, block_mask)
        mask = torch.where(casual_block_mask, float("-inf"), 0)
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
