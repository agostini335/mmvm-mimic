from dataclasses import dataclass


@dataclass
class ModelConfig:
    # mps, cuda, cpu
    device: str = "cuda"
    batch_size: int = 2
    lr: float = 5e-6
    epochs: int = 251

    latent_dim: int = 256

    resample_eval: bool = False

    # loss hyperparameters
    beta: float = 1.0

    # network architectures
    use_resnets: bool = True

    # annealing
    temp_annealing: str = "exp"

    # offline evaluation only
    offline_eval: bool = True


@dataclass
class JointModelConfig(ModelConfig):
    name: str = "joint"
    aggregation: str = "poe"


@dataclass
class MixedPriorModelConfig(ModelConfig):
    name: str = "mixedprior"
    drpm_prior: bool = False
    #
    # drpm
    n_groups: int = 2
    hard_pi: bool = True
    add_gumbel_noise: bool = False
    gamma: float = 0.0001

    # temperature annealing
    init_temp: float = 1.0
    final_temp: float = 0.5
    num_steps_annealing: int = 200000

    # weight on N(0,1) in mixed prior
    alpha_annealing: bool = True
    init_alpha_value: float = 1.0
    final_alpha_value: float = 0.0
    alpha_annealing_steps: int = 150000


@dataclass
class UnimodalModelConfig(ModelConfig):
    name: str = "unimodal"
