import os, sys
import json

import numpy as np
import pandas as pd
import torch
from torchvision import transforms
import PIL.Image as Image

from utils.MimicCXRDataset import MimicCXRDatasetBase, MimicCXRDatasetBaseAllCombi
from utils.MimicCXRSplitter import MimicCXRSplitter
import dask.array as da

transform = transforms.Compose([transforms.ToTensor()])


def get_dataset(cfg):
    if cfg.dataset.name.startswith("mimic_cxr"):
        ds = get_dataset_mimic_cxr(cfg)
    else:
        raise NotImplementedError
    return ds


def get_transform_mimic_cxr(cfg):
    # TODO fix
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize(
                size=(cfg.dataset.img_size, cfg.dataset.img_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
        ]
    )
    return transform


def get_dataset_mimic_cxr(cfg):
    data_path = os.path.join(
        cfg.dataset.dir_cache,
        f"seed-{cfg.seed}_trainval-"
        f"{cfg.dataset.train_val_split}_testval-"
        f"{cfg.dataset.test_val_split}_sp-"
        f"{cfg.dataset.studies_policy}",
    )

    if not os.path.exists(data_path) or not cfg.dataset.use_cache:
        raise Exception("build cache dir with MimicCXRSplitter first")

    transform = get_transform_mimic_cxr(cfg)

    if cfg.dataset.studies_policy == "one_frontal_one_lateral":
        # d_train
        frontal_metadata_train = pd.read_csv(
            os.path.join(data_path, "FRONTAL_train_metadata.csv")
        )
        lateral_metadata_train = pd.read_csv(
            os.path.join(data_path, "LATERAL_train_metadata.csv")
        )
        img_frontal_train = da.from_npy_stack(
            os.path.join(data_path, "FRONTAL_train_images.npy")
        )
        img_lateral_train = da.from_npy_stack(
            os.path.join(data_path, "LATERAL_train_images.npy")
        )
        d_train = MimicCXRDatasetBase(
            cfg,
            frontal_metadata_train,
            lateral_metadata_train,
            images_frontal=img_frontal_train,
            images_lateral=img_lateral_train,
            transform=transform,
        )
        # d_eval TODO adjust transform for evaluation dataset
        frontal_metadata_val = pd.read_csv(
            os.path.join(data_path, "FRONTAL_val_metadata.csv")
        )
        lateral_metadata_val = pd.read_csv(
            os.path.join(data_path, "LATERAL_val_metadata.csv")
        )
        img_frontal_val = da.from_npy_stack(
            os.path.join(data_path, "FRONTAL_val_images.npy")
        )
        img_lateral_val = da.from_npy_stack(
            os.path.join(data_path, "LATERAL_val_images.npy")
        )
        d_eval = MimicCXRDatasetBase(
            cfg,
            frontal_metadata_val,
            lateral_metadata_val,
            images_frontal=img_frontal_val,
            images_lateral=img_lateral_val,
            transform=transform,
        )

    if (
        cfg.dataset.studies_policy == "all_combi_no_missing"
        or cfg.dataset.studies_policy == "all_combi_missing"
    ):

        # load full numpy arrays - data will be split online
        ap_img = da.from_npy_stack(
            os.path.join(cfg.dataset.dir_data, "mimic-dask", "AP")
        )
        pa_img = da.from_npy_stack(
            os.path.join(cfg.dataset.dir_data, "mimic-dask", "PA")
        )
        lat_img = da.from_npy_stack(
            os.path.join(cfg.dataset.dir_data, "mimic-dask", "LATERAL")
        )
        ll_img = da.from_npy_stack(
            os.path.join(cfg.dataset.dir_data, "mimic-dask", "LL")
        )

        # load into memory if pre_load_images is True
        if cfg.dataset.pre_load_images:
            ap_img = ap_img.compute()
            pa_img = pa_img.compute()
            lat_img = lat_img.compute()
            ll_img = ll_img.compute()

        # load metadata
        if cfg.dataset.reduced_dataset:
            metadata = pd.read_csv(
                os.path.join(
                    data_path, cfg.dataset.studies_policy + "_metadata_splits.csv"
                )
            )[0:10000]
        else:
            metadata = pd.read_csv(
                os.path.join(
                    data_path, cfg.dataset.studies_policy + "_metadata_splits.csv"
                )
            )

        # split metadata
        metadata_train = metadata[metadata["split"] == "train"].copy()
        metadata_val = metadata[metadata["split"] == "val"].copy()
        metadata_test = metadata[metadata["split"] == "test"].copy()

        # check metadata file loaded is consistent and splits are disjoint
        assert (
            len(
                set(metadata_train["subject_id"]).intersection(
                    set(metadata_val["subject_id"])
                )
            )
            == 0
        )
        assert (
            len(
                set(metadata_train["subject_id"]).intersection(
                    set(metadata_test["subject_id"])
                )
            )
            == 0
        )
        assert (
            len(
                set(metadata_val["subject_id"]).intersection(
                    set(metadata_test["subject_id"])
                )
            )
            == 0
        )
        assert (
            len(
                set(metadata_train["study_id"]).intersection(
                    set(metadata_val["study_id"])
                )
            )
            == 0
        )
        assert (
            len(
                set(metadata_train["study_id"]).intersection(
                    set(metadata_test["study_id"])
                )
            )
            == 0
        )
        assert (
            len(
                set(metadata_val["study_id"]).intersection(
                    set(metadata_test["study_id"])
                )
            )
            == 0
        )

        # random filtering the training dataset if needed for the experiment (ml4h clf experiment)
        if "all_data_points" in cfg.dataset and "train_data_points" in cfg.dataset:
            if not cfg.dataset.all_data_points:
                print(
                    "Filtering training dataset to ",
                    cfg.dataset.train_data_points,
                    " training data points",
                )
                metadata_train = metadata_train.sample(
                    n=cfg.dataset.train_data_points, random_state=cfg.seed
                )
            else:
                print(
                    "RND FILTER NOT APPLIED: ",
                    len(metadata_train),
                    " training data points",
                )
        else:
            print(
                "RND FILTER NOT APPLIED: ", len(metadata_train), " training data points"
            )

        # d_train
        d_train = MimicCXRDatasetBaseAllCombi(
            cfg=cfg,
            metadata=metadata_train,
            AP_images=ap_img,
            PA_images=pa_img,
            LL_images=ll_img,
            LATERAL_images=lat_img,
            transform=transform,
        )
        # d_eval
        d_eval = MimicCXRDatasetBaseAllCombi(
            cfg=cfg,
            metadata=metadata_val,
            AP_images=ap_img,
            PA_images=pa_img,
            LL_images=ll_img,
            LATERAL_images=lat_img,
            transform=transform,
        )

    train_loader = torch.utils.data.DataLoader(
        d_train,
        batch_size=cfg.model.batch_size,
        shuffle=cfg.dataset.shuffle_train_dl,
        num_workers=cfg.dataset.num_workers,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        d_eval,
        batch_size=cfg.model.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        drop_last=True,
    )
    return train_loader, d_train, val_loader, d_eval
