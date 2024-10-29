from dataclasses import dataclass, field
from typing import List
from omegaconf import MISSING

from config.DatasetConfig import DataConfig
from config.ModelConfig import ModelConfig


@dataclass
class LogConfig:
    # wandb
    wandb_entity: str = "aa335"
    # mv_mimic_mps, mv_mimic_scicore
    wandb_group: str = "test_clean"
    wandb_run_name: str = ""
    wandb_project_name: str = "mv_mimic"
    wandb_log_freq: int = 30
    wandb_offline: bool = False
    wandb_local_instance: bool = False

    # logs
    dir_logs: str = "/project/home/anandrea/Projects/mvvae/wandb"

    # logging frequencies
    downstream_logging_frequency: int = 1
    coherence_logging_frequency: int = 20000
    likelihood_logging_frequency: int = 2000000
    img_plotting_frequency: int = 1

    # debug level wandb
    debug: bool = True


@dataclass
class EvalConfig:
    # latent representation
    num_samples_train: int = 20000
    max_iteration: int = 10000
    eval_downstream_task: bool = True
    downstream_rf: bool = True
    save_encodings: bool = False

    # classifiers trained in the downstream task - only for MIMIC
    classifier_list: List[str] = field(
        default_factory=lambda: [
            "RF",
            "LR",
        ]
    )

    # metrics used for evaluation - only for MIMIC
    metric_list: List[str] = field(
        default_factory=lambda: [
            "AP",
            "AUROC",
        ]
    )
    # RF PARAMETERS
    f_n_estimators: int = 5  # Experiments are run with f_n_estimators = 5000
    f_min_samples_split: int = 5
    f_min_samples_leaf: int = 1
    f_max_features: str = "sqrt"
    f_max_depth: int = 30
    f_criterion: str = "entropy"
    f_bootstrap: bool = True

    # hyperparameter tuning
    hp_tuning: bool = False
    hp_iter: int = 20
    hp_cv: int = 4
    verbosity: int = 4
    # rf search space
    n_estimator: List[int] = field(default_factory=lambda: [500, 1000, 2000, 5000])
    max_depth: List[int] = field(default_factory=lambda: [10, 20, 30, 50, 100, 500])
    criterion: List[str] = field(default_factory=lambda: ["gini", "entropy"])
    min_samples_split: List[int] = field(default_factory=lambda: [2, 5, 10])
    min_samples_leaf: List[int] = field(default_factory=lambda: [1, 2, 4])
    max_features: List[str] = field(default_factory=lambda: ["sqrt", "log2"])
    bootstrap: List[bool] = field(default_factory=lambda: [True, False])

    # coherence
    coherence: bool = True

    # offline evaluation config
    trained_model_path: str = "./pretrained_models/epoch=250-step=806212.ckpt"
    num_enc_clf: int = (
        0  # number of encoders used to train the downstream task classifiers
    )
    modality_eval: str = (
        "frontal"  # modality to evaluate, in case of MIMIC 'frontal' or 'lateral'
    )
    before_aggregation: bool = (
        True  # take the embeddings before or after the aggregation
    )

    # offline KL
    kl_independent_trained_model_path: str = (
        "./pretrained_models/epoch=250-step=806212.ckpt"
    )
    kl_model_type: str = "mix"  # 'unimodal' or 'mix'
    num_enc_kl: int = 10000
    kl_P_frontal: bool = False


@dataclass
class MyMVWSLConfig:
    seed: int = 2
    checkpoint_metric: str = "val/loss/loss"
    # logger
    log: LogConfig = MISSING
    # dataset
    dataset: DataConfig = MISSING
    # model
    model: ModelConfig = MISSING
    # eval
    eval: EvalConfig = MISSING
