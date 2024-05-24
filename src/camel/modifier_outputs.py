import torch
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.utils import ModelOutput


@dataclass
class CamelModifierOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    speculation_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    compression_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
