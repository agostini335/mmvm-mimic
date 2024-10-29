import os
import json
from typing import List
from collections import Counter, OrderedDict
from collections import defaultdict
import pandas as pd
from torch.utils.data import Dataset
import PIL.Image as Image
import torch


class MimicCXRDatasetBase(Dataset):
    """Custom Dataset for loading Mimic CXR images - Base class"""

    def __init__(
            self,
            cfg,
            frontal_metadata,
            lateral_metadata,
            images_frontal,
            images_lateral,
            transform=None,
    ):
        self.cfg = cfg
        self.images_frontal = images_frontal
        self.images_lateral = images_lateral
        self.transform = transform

        # load in memory images_frontal and images_lateral
        if cfg.dataset.pre_load_images:
            self.images_lateral = self.images_lateral.compute()
            self.images_frontal = self.images_frontal.compute()

        self.frontal_metadata = frontal_metadata
        self.lateral_metadata = lateral_metadata
        self.label_names = cfg.dataset.target_list
        self.labels = self._create_labels()
        self.img_RGB = cfg.dataset.img_RGB
        if self.img_RGB:
            self.mod = "RGB"
        else:
            self.mod = "L"

    def __getitem__(self, index):
        if self.cfg.dataset.studies_policy == "one_frontal_one_lateral":
            if self.cfg.dataset.pre_load_images:
                image_frontal = Image.fromarray(self.images_frontal[index]).convert(
                    self.mod
                )
                image_lateral = Image.fromarray(self.images_lateral[index]).convert(
                    self.mod
                )
            else:
                image_frontal = Image.fromarray(
                    self.images_frontal[index].compute()
                ).convert(self.mod)
                image_lateral = Image.fromarray(
                    self.images_lateral[index].compute()
                ).convert(self.mod)
        else:
            raise NotImplementedError

        if self.transform:
            image_frontal = self.transform(image_frontal)
            image_lateral = self.transform(image_lateral)

        label_values = self.labels.iloc[index]
        label = torch.tensor(label_values.values.astype(int)).float()
        sample = {"frontal": image_frontal, "lateral": image_lateral}

        return sample, label

    def __len__(self):
        return len(self.labels)

    def _create_labels(self):
        # assert frontal_metadata and lateral_metadata have the same study_id
        # assert that study_id lists are the same
        for i, row in self.frontal_metadata.iterrows():
            assert row["study_id"] == self.lateral_metadata.iloc[i]["study_id"]
        labels = self.frontal_metadata[self.label_names].copy()
        labels = labels.fillna(0)
        labels.replace(-1, 0, inplace=True)
        assert all((labels == 0) | (labels == 1))
        return labels


class MimicCXRDatasetBaseAllCombi(Dataset):
    """Custom Dataset for loading Mimic CXR images - Base class for all_combi_no_missing and all_combi_missing policy"""

    def __init__(
            self,
            cfg,
            metadata,
            AP_images,
            PA_images,
            LL_images,
            LATERAL_images,
            transform=None,
    ):
        self.cfg = cfg
        self.AP_images = AP_images
        self.PA_images = PA_images
        self.LL_images = LL_images
        self.LATERAL_images = LATERAL_images
        self.transform = transform
        self.metadata = metadata
        self.label_names = cfg.dataset.target_list
        self.labels = self._create_labels()
        self.img_RGB = cfg.dataset.img_RGB
        if self.img_RGB:
            self.mod = "RGB"
        else:
            self.mod = "L"

    def __getitem__(self, index):
        if self.cfg.dataset.studies_policy == "all_combi_no_missing":
            # identify view_position from metadata
            frontal_view = self.metadata.iloc[index]["view_pos_frontal"]
            lateral_view = self.metadata.iloc[index]["view_pos_lateral"]

            if frontal_view == "AP":
                frontal_image_ref = self.AP_images[self.metadata.iloc[index]["idx_frontal"]]
            if frontal_view == "PA":
                frontal_image_ref = self.PA_images[self.metadata.iloc[index]["idx_frontal"]]
            if lateral_view == "LL":
                lateral_image_ref = self.LL_images[self.metadata.iloc[index]["idx_lateral"]]
            if lateral_view == "LATERAL":
                lateral_image_ref = self.LATERAL_images[
                    self.metadata.iloc[index]["idx_lateral"]
                ]

            if self.cfg.dataset.pre_load_images:
                image_frontal = Image.fromarray(frontal_image_ref).convert(self.mod)
                image_lateral = Image.fromarray(lateral_image_ref).convert(self.mod)
            else:
                image_frontal = Image.fromarray(frontal_image_ref.compute()).convert(
                    self.mod
                )
                image_lateral = Image.fromarray(lateral_image_ref.compute()).convert(
                    self.mod
                )

            if self.transform:
                image_frontal = self.transform(image_frontal)
                image_lateral = self.transform(image_lateral)

            label_values = self.labels.iloc[index]
            label = torch.tensor(label_values.values.astype(int)).float()
            sample = {"frontal": image_frontal, "lateral": image_lateral}
            return sample, label

        elif self.cfg.dataset.studies_policy == "all_combi_missing":
            frontal_image_ref = None
            lateral_image_ref = None

            # identify view_position from metadata
            frontal_view = self.metadata.iloc[index]["view_pos_frontal"]
            lateral_view = self.metadata.iloc[index]["view_pos_lateral"]

            if frontal_view == "AP":
                frontal_image_ref = self.AP_images[self.metadata.iloc[index]["idx_frontal"]]
            if frontal_view == "PA":
                frontal_image_ref = self.PA_images[self.metadata.iloc[index]["idx_frontal"]]
            if lateral_view == "LL":
                lateral_image_ref = self.LL_images[self.metadata.iloc[index]["idx_lateral"]]
            if lateral_view == "LATERAL":
                lateral_image_ref = self.LATERAL_images[
                    self.metadata.iloc[index]["idx_lateral"]
                ]

            if self.cfg.dataset.pre_load_images:
                if frontal_image_ref is not None:
                    image_frontal = Image.fromarray(frontal_image_ref).convert(self.mod)
                if lateral_image_ref is not None:
                    image_lateral = Image.fromarray(lateral_image_ref).convert(self.mod)
            else:
                if frontal_image_ref is not None:
                    image_frontal = Image.fromarray(frontal_image_ref.compute()).convert(
                        self.mod
                    )
                if lateral_image_ref is not None:
                    image_lateral = Image.fromarray(lateral_image_ref.compute()).convert(
                        self.mod
                    )

            if self.transform:
                if frontal_image_ref is not None:
                    image_frontal = self.transform(image_frontal)
                if lateral_image_ref is not None:
                    image_lateral = self.transform(image_lateral)

            label_values = self.labels.iloc[index]
            label = torch.tensor(label_values.values.astype(int)).float()

            sample = {"frontal": torch.zeros([1, self.cfg.dataset.img_size, self.cfg.dataset.img_size]),
                      "lateral": torch.zeros([1, self.cfg.dataset.img_size, self.cfg.dataset.img_size])}
            avail_mod = {"frontal": False, "lateral": False}

            if frontal_image_ref is not None:
                sample["frontal"] = image_frontal
                avail_mod["frontal"] = True
            if lateral_image_ref is not None:
                sample["lateral"] = image_lateral
                avail_mod["lateral"] = True

            return sample, label, avail_mod

    def __len__(self):
        return len(self.labels)

    def _create_labels(self):
        labels = self.metadata[self.label_names].copy()
        labels = labels.fillna(0)
        labels.replace(-1, 0, inplace=True)
        assert all((labels == 0) | (labels == 1))
        return labels
