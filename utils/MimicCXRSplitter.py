import json
import os

import numpy as np
import pandas as pd
import yaml
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import dask.array as da

"""
Implemented Policies:
1. all_combi_no_missing
2. all_combi_missing
3. one_frontal_one_lateral
"""


class MimicCXRSplitter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.modalities = ["AP", "PA", "LATERAL", "LL"]

        # filename metadata
        filename_ap_meta = os.path.join(
            self.cfg.dataset.dir_data, "AP", "meta_data.csv"
        )
        filename_lateral_meta = os.path.join(
            self.cfg.dataset.dir_data, "LATERAL", "meta_data.csv"
        )
        filename_pa_meta = os.path.join(
            self.cfg.dataset.dir_data, "PA", "meta_data.csv"
        )
        filename_ll_meta = os.path.join(
            self.cfg.dataset.dir_data, "LL", "meta_data.csv"
        )

        # filename images
        filename_ap_img = os.path.join(cfg.dataset.dir_data, "AP", "files_224.npy")
        filename_lateral_img = os.path.join(
            cfg.dataset.dir_data, "LATERAL", "files_224.npy"
        )
        filename_pa_img = os.path.join(cfg.dataset.dir_data, "PA", "files_224.npy")
        filename_ll_img = os.path.join(cfg.dataset.dir_data, "LL", "files_224.npy")

        # load matrices if needed - some policies does not require additional matrices
        if (
            self.cfg.dataset.studies_policy != "all_combi_no_missing"
            and self.cfg.dataset.studies_policy != "all_combi_missing"
        ):
            self.img_mat_ap = np.load(filename_ap_img, mmap_mode="r")
            self.img_mat_pa = np.load(filename_pa_img, mmap_mode="r")
            self.img_mat_ll = np.load(filename_ll_img, mmap_mode="r")
            self.img_mat_lateral = np.load(filename_lateral_img, mmap_mode="r")

        # filename general metadata
        filename_general_meta = os.path.join(
            self.cfg.dataset.dir_data, "mimic-cxr-2.0.0-metadata.csv"
        )

        # load metadata dfs
        self.df_ap = pd.read_csv(filename_ap_meta)
        self.df_pa = pd.read_csv(filename_pa_meta)
        self.df_ll = pd.read_csv(filename_ll_meta)
        self.df_lateral = pd.read_csv(filename_lateral_meta)
        self.df_general = pd.read_csv(filename_general_meta)

        # prepare metadata dfs mod
        self.df_general_mod = self.df_general.copy()
        self.df_ap_mod = self.df_ap.copy()
        self.df_pa_mod = self.df_pa.copy()
        self.df_ll_mod = self.df_ll.copy()
        self.df_lateral_mod = self.df_lateral.copy()

        # prepare final metadata dfs
        self.frontal_metadata_df_out = {"train": None, "val": None, "test": {None}}
        self.lateral_metadata_df_out = {"train": None, "val": None, "test": {None}}

        # prepare final image matrices
        self.frontal_image_mat_out = {"train": None, "val": None, "test": {None}}
        self.lateral_image_mat_out = {"train": None, "val": None, "test": {None}}

        # load pre-computed metadata for policy all_combi_no_missing
        if self.cfg.dataset.studies_policy == "all_combi_no_missing":
            self.all_combi_no_missing_metadata = pd.read_csv(
                os.path.join(
                    self.cfg.dataset.dir_data, "all_combi_no_missing_metadata.csv"
                )
            )

        if self.cfg.dataset.studies_policy == "all_combi_missing":
            self.all_combi_missing_metadata = pd.read_csv(
                os.path.join(
                    self.cfg.dataset.dir_data, "all_combi_missing_metadata.csv"
                )
            )

        # create splits by subject_id
        self._get_subject_splits()
        # get relevant studies
        self._get_relevant_studies()
        # create data points
        self._create_data_points()
        # save matrices and dfs to disk to a cache folder
        self._save_to_disk()
        print("MimicCXRSplitter -> Done")

    def _get_subject_splits(self):
        modalities = self.modalities

        if not self.cfg.dataset.splitting_method == "random":
            raise NotImplementedError(
                "Only random splitting is supported at the moment"
            )

        dfs_meta = {
            "AP": self.df_ap,
            "PA": self.df_pa,
            "LL": self.df_ll,
            "LATERAL": self.df_lateral,
        }

        # print metadata info
        for m in modalities:
            print(f"Number of metadata rows for {m}: ", len(dfs_meta[m]))
            print(
                f"Number of unique dicom_ids for {m}: ",
                len(dfs_meta[m]["dicom_id"].unique()),
            )
            print(
                f"Number of unique study_ids for {m}: ",
                len(dfs_meta[m]["study_id"].unique()),
            )
            print(
                f"Number of unique subject_ids for {m}: ",
                len(dfs_meta[m]["subject_id"].unique()),
            )
            print(" ")

        # get list of subjects
        self.list_of_subjects = set(
            self.df_pa["subject_id"].tolist()
            + self.df_ap["subject_id"].tolist()
            + self.df_ll["subject_id"].tolist()
            + self.df_lateral["subject_id"].tolist()
        )
        print(
            "Number of total subject_ids in considered modalities: ",
            len(self.list_of_subjects),
        )
        print(
            "Number of total subject_ids in the original metadata: ",
            len(self.df_general["subject_id"].unique()),
        )
        print(" ")

        self.list_of_subjects = list(
            set(
                self.df_pa["subject_id"].tolist()
                + self.df_ap["subject_id"].tolist()
                + self.df_ll["subject_id"].tolist()
                + self.df_lateral["subject_id"].tolist()
            )
        )
        # split all -> train - (test, val)
        train_subjects, test_val_subjects = train_test_split(
            self.list_of_subjects,
            train_size=self.cfg.dataset.train_val_split,
            shuffle=True,
            random_state=self.cfg.dataset.split_seed,
        )

        # split (test, val) -> test - val
        test_subjects, val_subjects = train_test_split(
            test_val_subjects,
            test_size=self.cfg.dataset.test_val_split,
            shuffle=True,
            random_state=self.cfg.dataset.split_seed,
        )

        subject_splits = {
            "train": train_subjects,
            "val": val_subjects,
            "test": test_subjects,
        }
        print("Number of subject_ids in the training set: ", len(train_subjects))
        print("Number of subject_ids in the val set: ", len(val_subjects))
        print("Number of subject_ids in the test set: ", len(test_subjects))
        # assert that the splits are disjoint
        assert len(set(train_subjects).intersection(set(val_subjects))) == 0
        assert len(set(train_subjects).intersection(set(test_subjects))) == 0
        assert len(set(val_subjects).intersection(set(test_subjects))) == 0
        self.subject_splits = subject_splits

    def _get_relevant_studies(self):
        if self.cfg.dataset.studies_policy == "all_combi_no_missing":
            print("relevant studies are precomputed")
            return
        if self.cfg.dataset.studies_policy == "all_combi_missing":
            print("relevant studies are precomputed")
            return
        self.list_of_study_ids = list(
            set(
                self.df_pa["study_id"].tolist()
                + self.df_ap["study_id"].tolist()
                + self.df_ll["study_id"].tolist()
                + self.df_lateral["study_id"].tolist()
            )
        )

        print(
            "\nNumber of total study_ids in considered modalities: ",
            len(self.list_of_study_ids),
        )
        print(
            "Number of total study_ids in the original metadata: ",
            len(self.df_general["study_id"].unique()),
        )

        # iterate over rows of df and calculate modality
        modality_col = []
        for i, row in self.df_general_mod.iterrows():
            if row.ViewPosition == "AP" or row.ViewPosition == "PA":
                modality_col.append("FRONTAL")
            elif row.ViewPosition == "LL" or row.ViewPosition == "LATERAL":
                modality_col.append("LATERAL")
            else:
                modality_col.append("OTHER")
        self.df_general_mod["modality"] = modality_col

        if not self.cfg.dataset.studies_policy == "one_frontal_one_lateral":
            raise NotImplementedError(
                "Only one_frontal_one_lateral - all_combi_no_missing - all_combi_missing policies are supported at the moment"
            )

        if self.cfg.dataset.studies_policy == "one_frontal_one_lateral":
            # list of study with exactly one frontal and one lateral view
            img_comb_per_study = (
                self.df_general_mod.groupby("study_id")["modality"]
                .apply(list)
                .reset_index()
            )
            relevant_studies = []
            for i, row in img_comb_per_study.iterrows():
                if (
                    row["modality"].count("FRONTAL") == 1
                    and row["modality"].count("LATERAL") == 1
                ):
                    relevant_studies.append(row["study_id"])
            print(
                "GENERAL METADATA - Number of relevant studies one_frontal_one_lateral: ",
                len(relevant_studies),
            )

            # exclude studies not available in the considered modalities
            studies_frontal = set(
                self.df_ap_mod["study_id"].tolist()
                + self.df_pa_mod["study_id"].tolist()
            )
            studies_lateral = set(
                self.df_lateral_mod["study_id"].tolist()
                + self.df_ll_mod["study_id"].tolist()
            )

            relevant_studies = list(
                set(relevant_studies).intersection(set(studies_frontal))
            )
            relevant_studies = list(
                set(relevant_studies).intersection(set(studies_lateral))
            )

            print(
                "POSTPROCESSED - Number of relevant studies one_frontal_one_lateral in considered modalities: ",
                len(relevant_studies),
            )
            self.relevant_studies = relevant_studies

            assert len(
                set(self.relevant_studies).intersection(
                    set(
                        self.df_ap_mod["study_id"].tolist()
                        + self.df_pa_mod["study_id"].tolist()
                    )
                )
            ) == len(set(self.relevant_studies))

            assert len(
                set(self.relevant_studies).intersection(
                    set(
                        self.df_lateral_mod["study_id"].tolist()
                        + self.df_ll_mod["study_id"].tolist()
                    )
                )
            ) == len(set(self.relevant_studies))

    def _create_data_points(self):
        dfs_meta_mod = {
            "AP": self.df_ap_mod,
            "PA": self.df_pa_mod,
            "LL": self.df_ll_mod,
            "LATERAL": self.df_lateral_mod,
        }

        if (
            self.cfg.dataset.studies_policy != "one_frontal_one_lateral"
            and self.cfg.dataset.studies_policy != "all_combi_no_missing"
        ) and self.cfg.dataset.studies_policy != "all_combi_missing":
            raise NotImplementedError("policy not supported at the moment")

        if self.cfg.dataset.studies_policy == "all_combi_no_missing":
            split_list = []
            for i, row in tqdm(self.all_combi_no_missing_metadata.iterrows()):
                if row["subject_id"] in self.subject_splits["train"]:
                    split_list.append("train")
                elif row["subject_id"] in self.subject_splits["val"]:
                    split_list.append("val")
                elif row["subject_id"] in self.subject_splits["test"]:
                    split_list.append("test")
                else:
                    raise ValueError("Subject not in any split")
            self.all_combi_no_missing_metadata["split"] = split_list
            return

        if self.cfg.dataset.studies_policy == "all_combi_missing":
            split_list = []
            for i, row in tqdm(self.all_combi_missing_metadata.iterrows()):
                if row["subject_id"] in self.subject_splits["train"]:
                    split_list.append("train")
                elif row["subject_id"] in self.subject_splits["val"]:
                    split_list.append("val")
                elif row["subject_id"] in self.subject_splits["test"]:
                    split_list.append("test")
                else:
                    raise ValueError("Subject not in any split")
            self.all_combi_missing_metadata["split"] = split_list
            return

        if self.cfg.dataset.studies_policy == "one_frontal_one_lateral":
            # for each view position, mark each row with the relevant split (train, val, test, EXCLUDED)
            for df in dfs_meta_mod:
                split_list = []
                # mark each row with the relevant split
                for i, row in tqdm(dfs_meta_mod[df].iterrows()):
                    if row["study_id"] in self.relevant_studies:
                        if row["subject_id"] in self.subject_splits["train"]:
                            split_list.append("train")
                        elif row["subject_id"] in self.subject_splits["val"]:
                            split_list.append("val")
                        elif row["subject_id"] in self.subject_splits["test"]:
                            split_list.append("test")
                        else:
                            raise ValueError("Subject not in any split")
                    else:
                        split_list.append("EXCLUDED")
                # add split column to the metadat
                dfs_meta_mod[df]["split"] = split_list
                print("ViewPosition: ", df)
                print(dfs_meta_mod[df][["subject_id", "study_id", "split"]].head(10))

            # create image matrices for each modality and for each split
            for split in self.subject_splits:
                # FRONTAL MODALITY - AP and PA
                # img
                frontal_img_mat = da.concatenate(
                    [self.img_mat_ap, self.img_mat_pa], axis=0
                )
                # metadata
                frontal_meta_data = pd.concat(
                    [self.df_ap_mod, self.df_pa_mod]
                ).reset_index()
                # filter relevant split
                frontal_meta_data = frontal_meta_data[
                    frontal_meta_data["split"] == split
                ]
                frontal_img_mat = frontal_img_mat[frontal_meta_data.index]
                frontal_meta_data = frontal_meta_data.reset_index(drop=True)

                # LATERAL MODALITY - LATERAL and LL
                # img
                lateral_img_mat = da.concatenate(
                    [self.img_mat_lateral, self.img_mat_ll], axis=0
                )
                # metadata
                lateral_meta_data = pd.concat(
                    [self.df_lateral_mod, self.df_ll_mod]
                ).reset_index()
                # filter relevant split
                lateral_meta_data = lateral_meta_data[
                    lateral_meta_data["split"] == split
                ]
                lateral_img_mat = lateral_img_mat[lateral_meta_data.index]
                lateral_meta_data = lateral_meta_data.reset_index(drop=True)

                # sort the data by study_id so that the order is the same for all the modalities
                frontal_meta_data = frontal_meta_data.sort_values("study_id")
                lateral_meta_data = lateral_meta_data.sort_values("study_id")
                frontal_img_mat = frontal_img_mat[frontal_meta_data.index]
                lateral_img_mat = lateral_img_mat[lateral_meta_data.index]
                frontal_meta_data = frontal_meta_data.reset_index(drop=True)
                lateral_meta_data = lateral_meta_data.reset_index(drop=True)

                # assert that study_id lists are the same
                for i, row in frontal_meta_data.iterrows():
                    assert row["study_id"] == lateral_meta_data.iloc[i]["study_id"]
                assert len(frontal_img_mat) == len(frontal_meta_data)
                assert len(lateral_img_mat) == len(lateral_meta_data)

                self.frontal_metadata_df_out[split] = frontal_meta_data
                self.lateral_metadata_df_out[split] = lateral_meta_data
                self.frontal_image_mat_out[split] = frontal_img_mat
                self.lateral_image_mat_out[split] = lateral_img_mat

    def _save_to_disk(self):
        path = os.path.join(
            self.cfg.dataset.dir_cache,
            f"seed-{self.cfg.seed}_trainval-"
            f"{self.cfg.dataset.train_val_split}_testval-"
            f"{self.cfg.dataset.test_val_split}_sp-"
            f"{self.cfg.dataset.studies_policy}",
        )

        if not os.path.exists(path):
            os.makedirs(path)
        else:
            print("Directory already exists - overwriting files")

        for split in self.subject_splits:
            if self.cfg.dataset.studies_policy == "one_frontal_one_lateral":
                # save metadata
                self.frontal_metadata_df_out[split].to_csv(
                    os.path.join(path, f"FRONTAL_{split}_metadata.csv")
                )
                self.lateral_metadata_df_out[split].to_csv(
                    os.path.join(path, f"LATERAL_{split}_metadata.csv")
                )
                # save image matrices
                da.to_npy_stack(
                    os.path.join(path, f"FRONTAL_{split}_images.npy"),
                    self.frontal_image_mat_out[split],
                )
                da.to_npy_stack(
                    os.path.join(path, f"LATERAL_{split}_images.npy"),
                    self.lateral_image_mat_out[split],
                )
            if self.cfg.dataset.studies_policy == "all_combi_no_missing":
                # save metadata
                self.all_combi_no_missing_metadata.to_csv(
                    os.path.join(path, f"all_combi_no_missing_metadata_splits.csv")
                )
                # save a yaml file to disk
                with open(os.path.join(path, "config_recap.yaml"), "w") as outfile:
                    yaml.dump(OmegaConf.to_yaml(self.cfg), outfile)
            if self.cfg.dataset.studies_policy == "all_combi_missing":
                # save metadata
                self.all_combi_missing_metadata.to_csv(
                    os.path.join(path, f"all_combi_missing_metadata_splits.csv")
                )
                # save a yaml file to disk
                with open(os.path.join(path, "config_recap.yaml"), "w") as outfile:
                    yaml.dump(OmegaConf.to_yaml(self.cfg), outfile)
        print("Saved to disk")
