from dataclasses import dataclass, field
from typing import List
from omegaconf import MISSING


@dataclass
class DataConfig:
    name: str = MISSING
    num_workers: int = 10
    # num views
    num_views: int = MISSING


@dataclass
class MimicCXRDataConfig(DataConfig):
    name: str = "mimic_cxr"

    # shuffle train dataloader
    shuffle_train_dl: bool = True

    # num views = 2 : lateral (LATERAL + LL) and frontal (AP + PA)
    num_views: int = 2
    dir_data: str = "/local/home/anandrea/data"
    # "/Users/ago/PycharmProjects/data/cache""
    # "/project/home/anandrea/cache/"
    dir_cache: str = "/local/home/anandrea/data/cache-test/"
    use_cache: bool = True

    # split settings
    splitting_method: str = "random"
    train_val_split: float = 0.8
    test_val_split: float = 0.5
    split_seed: int = 0
    # one_frontal_one_lateral / all_combi_no_missing / all_combi_missing
    studies_policy: str = "all_combi_no_missing"
    reduced_dataset: bool = True  # if True, use only 10% of the dataset - for debugging

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

    img_size: int = 224
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
