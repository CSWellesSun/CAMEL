# CAMEL

## Introduction

CAMEL(Context-Aware Modifier for Efficient Language model) is a speculative decoding method inspired by [EAGLE](https://github.com/SafeAILab/EAGLE). It compresses former input hidden states according to window size and then make speculations.

<center>
<img src="docs/arch.png" alt="architecture" width="300">
</center>

## Installation

```bash
pip install modifier
```

## Quick Start

CAMEL only supports `meta-llama/Llama-2-7b-chat-hf` currently.

```python
import torch
from camel import CamelModel

prompt = "What is artificial intelligence?"
model = CamelModel.from_pretrained(
    base_model_path="meta-llama/Llama-2-7b-chat-hf",
    modifier_path="",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = model.get_tokenizer()
input_ids = tokenizer(prompt).input_ids
output_ids = model.generate(input_ids)
output = tokenizer.decode(output_ids)
print(output)
```

## Reference

- [Medusa](https://github.com/FasterDecoding/Medusa)

- [EAGLE](https://github.com/SafeAILab/EAGLE)