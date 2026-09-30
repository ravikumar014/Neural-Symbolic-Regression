from dataclasses import dataclass, field
from typing import Sequence, Callable


#NSR
@dataclass
class LibraryConfig:
    funcs: Sequence[Callable]
    include_interactions: bool = True
    max_order: int = 2
    n_jobs: int = -1


@dataclass
class ModelConfig:
    hidden: int = 128
    depth: int = 2
    out_dim: int = 1
    weight_decay: float = 1e-4
    dropout: float = 0.0
    activation: str = "relu"


@dataclass
class TrainingConfig:
    lr: float = 1e-3
    batch_size: int = 128
    epochs: int = 200
    grad_clip: float = 1.0
    early_stop_patience: int = 20
    early_stop_delta: float = 1e-6
    use_amp: bool | None = None
    scheduler: str = "onecycle"
    step_size: int = 50
    gamma: float = 0.5


#PySR
@dataclass
class PySRConfig:
    niterations: int = 100
    populations: int = 20
    population_size: int = 50
    maxsize: int = 30
    maxdepth: int = 20
    unary_operators: list = field(default_factory=lambda: [
        "sin",
        "cos",
        "exp",
        "log",
        "sqrt",
    ])
    binary_operators: list = field(default_factory=lambda: [
        "+",
        "-",
        "*",
        "/",
    ])
    # Loss
    elementwise_loss: str = "loss(x, y) = (x-y)^2"

    # Parallel
    # procs: int = 0
    # multithreading: bool = True
    parallelism="serial"
    verbosity: int = 0

    # Reproducibility
    random_state: int = 42
    deterministic=True


#GPLearn
@dataclass
class GPLearnConfig:
    population_size: int = 5000
    generations: int = 20
    tournament_size: int = 20
    stopping_criteria: float = 0.01
    function_set: tuple = (
        "add",
        "sub",
        "mul",
        "div",
        "sqrt",
        "log",
        "abs",
        "neg",
        "inv",
        "sin",
        "cos",
        "tan",
    )

    #tree constraints
    init_depth: tuple = (2, 6)
    init_method: str = "half and half"
    const_range: tuple = (-5, 5)
    parsimony_coefficient: float = 0.001
    random_state: int = 42

    #reproducibility
    random_state: int = 42
    verbose: int = 0


#for ablation study
# @dataclass
# class AblationConfig:
#     name: str
#     funcs: list
#     include_interactions: bool = True
#     max_order: int = 2
#     sparse_method: str = "lasso"


#Operon
@dataclass
class OperonConfig:
    population_size: int = 1000
    generations: int = 500
    max_length: int = 50


#AIFeynman
@dataclass
class AIFeynmanConfig:
    max_degree: int = 5
    simplify: bool = True