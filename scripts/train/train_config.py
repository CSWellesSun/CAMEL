from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainConfig:
    datapath: str
    config_path: str
    lr: float
    bs: int
    gradient_accumulation_steps: int
    is_warmup: bool
    num_epochs: int
    num_warmup_steps: int
    total_steps: int
    p_w: float
    v_w: float
    head_w: float
    num_workers: int
    embedding: bool
    act: str
    data_noise: bool
    noise: str
    mean: float
    std: float
    residual: str
    max_len: int
    b1: float
    b2: float
    grad_clip: float
    save_freq: int
