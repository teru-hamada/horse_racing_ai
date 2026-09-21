from dataclasses import dataclass


@dataclass
class TrainConfig:
    epochs: int = 80
    batch_size: int = 128
    learning_rate: float = 0.001
    hidden_dim: int = 64
    dropout: float = 0.25
    patience: int = 12
    random_seed: int = 42
