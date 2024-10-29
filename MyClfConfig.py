from dataclasses import dataclass
from dataclasses import dataclass, field
from typing import List
from omegaconf import MISSING


@dataclass
class DataConfig:
    name: str = MISSING
    num_workers: int = 8
    # num views
    num_views: int = MISSING


@dataclass
class LogConfig:
    wandb_entity: str = "aa335"
    # mv_mimic_mps, mv_mimic_scicore
    wandb_group: str = "clf"
    wandb_run_name: str = ""
    wandb_project_name: str = "mv_mimic"
    wandb_log_freq: int = 30
    wandb_offline: bool = True
    wandb_local_instance: bool = False

    # logs
    dir_logs: str = "logs"


@dataclass
class MimicCXRDataConfig(DataConfig):
    name: str = "mimic_cxr"

    # num views = 2 : lateral (LATERAL + LL) and frontal (AP + PA)
    num_views: int = 2
    dir_data: str = "/local/home/anandrea/data"
    # "/Users/ago/PycharmProjects/data/cache""
    # "/project/home/anandrea/cache/"
    dir_cache: str = "/local/home/anandrea/data/cache"
    dir_clfs_base: str = "/local/home/anandrea/data/"
    suffix_clfs: str = "mimic_clf"
    use_cache: bool = True

    # split settings
    splitting_method: str = "random"
    train_val_split: float = 0.8
    test_val_split: float = 0.5
    split_seed: int = 0
    # one_frontal_one_lateral or all_combi_no_missing
    studies_policy: str = "all_combi_no_missing"
    reduced_dataset: bool = False
    shuffle_train_dl: bool = True

    # labels
    target_list: List[str] = field(
        default_factory=lambda: [
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Edema",
            "Enlarged Cardiomediastinum",
            "Fracture",
            "Lung Lesion",
            "Lung Opacity",
            "No Finding",
            "Pleural Effusion",
            "Pleural Other",
            "Pneumonia",
            "Pneumothorax",
            "Support Devices",
        ]
    )
    n_clfs_outputs: int = 14
    num_labels: int = 14

    img_size: int = 224  # use 224
    image_channels: int = 1

    # copied from celeba - text not used
    num_layers_img: int = 5
    filter_dim_img: int = 64
    filter_dim_text: int = 64
    beta_img: float = 1.0
    beta_text: float = 1.0
    skip_connections_img_weight_a: float = 1.0
    skip_connections_img_weight_b: float = 1.0

    use_rec_weight: bool = True
    include_channels_rec_weight: bool = False

    # img settings
    img_RGB: bool = False  # set to False if you want to use grayscale images
    pre_load_images: bool = False

    # restricted dataset - ml4h experiment
    train_data_points: int = 1000
    all_data_points: bool = (
        True  # if False, the num_data_points is considered for training
    )


@dataclass
class ModelConfig:
    device: str = "cuda"
    batch_size: int = 128
    lr: float = 0.00001
    epochs: int = 250
    # 'independent','test_mix','train_mix'
    clf_type: str = "independent"


@dataclass
class MyClfConfig:
    seed: int = 2
    checkpoint_metric: str = "val/loss/loss"
    model: ModelConfig = MISSING
    log: LogConfig = MISSING
    dataset: DataConfig = MISSING
