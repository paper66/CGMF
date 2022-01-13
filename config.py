from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Config:
    # Parameter for initializing
    dataset: str = 'exchange_rate'  # exchange_rate/solar_AL/electricity/traffic
    model_string: str = "CGMF"
    horizon: int = 3
    now: str = str(datetime.now())
    seed: int = 2000
    initialize: bool = False
    device: str = "cuda:0"
    lr: float = 0.0005

    # Parameter for training
    epochs: int = 50
    batch_size: int = 16
    accumulate_step: int = 1
    clip_val: float = 0.1
    step_size: int = 5
    gamma: float = 0.8
    scheduler: str = 'steplr'
    weight_decay: float = 1e-5
    dropout: float = 0.1

    # Parameters for data loading
    filename: str = "./data/" + dataset + '.txt'
    normalize: int = 2
    train_portion: float = 0.6
    valid_portion: float = 0.2
    window: int = 96 * 2

    # Parameters for model architecture
    # Conv
    channel: int = 16
    kernel_sizes: list = (24, 48, 96)
    # SelfAttn
    d_inner: int = 8
    n_head: int = 4
    d_k: int = 3
    d_v: int = 3

    # Parameters for evaluation
    model_dir: str = 'saved_models'
    model_path = './' + model_dir + '/' + model_string + '-' + now + '.pt'
    log_dir: str = 'log'
    log_path = 'myout.file'
