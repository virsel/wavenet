from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    vocab_size: int
    n_embd: int
    n_hidden: int
    n_consecutive: int
    batch_size: int
    context_length: int


def get_default_config() -> Config:
    return Config(
        vocab_size=5001,
        n_embd=16,
        n_hidden=64,
        n_consecutive=4,
        batch_size=32,
        context_length=32,
    )
