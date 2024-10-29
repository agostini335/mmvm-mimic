import pytorch_lightning as pl
import os
import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
from sklearn.metrics import roc_auc_score, average_precision_score

from networks.NetworkImgClfMimic import ClfImg
import statistics


class ClfMimicCXR(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()

        self.clfs = nn.ModuleList(
            [ClfImg(cfg).to(cfg.model.device), ClfImg(cfg).to(cfg.model.device)]
        )
        self.cfg = cfg
        self.loss = nn.CrossEntropyLoss()
        self.validation_step_outputs = []
        self.training_step_outputs = []

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        out = self.forward(self.cfg, batch)
        # loss, mean_ap = self.compute_loss("train", batch, out)
        imgs, labels = batch
        preds, losses = out
        loss = losses.mean(dim=1).mean(dim=0)
        self.log("train/loss/loss", loss)
        return loss

    def on_validation_epoch_start(self):
        self.val_preds = {}
        self.val_labels = {}
        for m in range(self.cfg.dataset.num_views):
            self.val_preds[m] = {}
            self.val_labels[m] = {}
            for k in range(self.cfg.dataset.num_labels):
                self.val_preds[m][k] = []
                self.val_labels[m][k] = []

    def validation_step(self, batch, batch_idx):
        out = self.forward(self.cfg, batch)
        imgs, labels = batch
        preds, losses = out
        loss = losses.mean(dim=1).mean(dim=0)

        n_labels = labels.shape[1]
        aurocs = []

        for m in range(self.cfg.dataset.num_views):
            loss_m = losses[:, m, :].mean(dim=0)
            pred_m = preds[:, m, :]
            auroc_m = torch.zeros(n_labels)

            for k in range(0, n_labels):
                self.val_preds[m][k].extend(pred_m[:, k].detach().cpu().numpy())
                self.val_labels[m][k].extend(labels[:, k].cpu())
                try:
                    auroc_m[k] = roc_auc_score(
                        labels[:, k].cpu(), pred_m[:, k].detach().cpu().numpy()
                    )
                    self.log("val/error", torch.zeros(1))
                    self.log(
                        "val/auroc/v"
                        + str(m)
                        + "/"
                        + str(self.cfg.dataset.target_list[k]),
                        auroc_m[k],
                    )
                except:
                    self.log("val/error", torch.ones(1))
                    print("Error in auroc calculation")

            aurocs.append(auroc_m.mean())
            self.log("val/loss/v" + str(m), loss_m)
            self.log("val/auroc/v" + str(m), auroc_m.mean())
        mean_auroc = torch.tensor(aurocs).mean()
        self.log("val/loss/loss", loss)
        self.log("val/loss/mean_auroc", mean_auroc)
        self.validation_step_outputs.append(mean_auroc)

        return loss

    def on_validation_epoch_end(self):
        aurocs = []
        for m in range(self.cfg.dataset.num_views):
            for k in range(self.cfg.dataset.num_labels):
                all_preds = self.val_preds[m][k]
                all_labels = self.val_labels[m][k]

        for m in range(self.cfg.dataset.num_views):
            auroc_m = torch.zeros(self.cfg.dataset.num_labels)
            for k in range(0, self.cfg.dataset.num_labels):
                all_preds = self.val_preds[m][k]
                all_labels = self.val_labels[m][k]
                try:
                    auroc_m[k] = roc_auc_score(all_labels, all_preds)
                    self.log("val/macro_error", torch.zeros(1))
                    self.log(
                        "val/macro_auroc/v"
                        + str(m)
                        + "/"
                        + str(self.cfg.dataset.target_list[k]),
                        auroc_m[k],
                    )
                except:
                    self.log("val/macro_error", torch.ones(1))
                    print("ERROR MACRO AUROC CALCULATION")

            aurocs.append(auroc_m.mean())

            self.log("val/macro_auroc/v" + str(m), auroc_m.mean())
        mean_auroc = torch.tensor(aurocs).mean()
        self.log("val/loss/macro_mean_auroc", mean_auroc)
        self.validation_step_outputs.clear()  # free memory

    def forward(self, cfg, batch):
        data, labels = batch
        n_labels = labels.shape[1]
        preds = torch.zeros(
            (cfg.model.batch_size, cfg.dataset.num_views, n_labels),
            device=cfg.model.device,
        )
        losses = torch.zeros(
            (cfg.model.batch_size, cfg.dataset.num_views, 1), device=cfg.model.device
        )

        for m, m_key in enumerate(data.keys()):
            if m_key == "frontal":
                assert m == 0
            if m_key == "lateral":
                assert m == 1
            print("m", m, "m_key", m_key)
            print(data.keys())
            m_val = data[m_key]
            pred_m = self.clfs[m](m_val)
            preds[:, m, :] = pred_m
            # save_preds_to_file(pred_m, labels, m, m_key)
            np.save(f"preds.npy", preds.detach().cpu().numpy())
            loss_m = self.loss(pred_m, labels)
            losses[:, m, :] = loss_m
        return preds, losses

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.model.lr,
        )
        return {
            "optimizer": optimizer,
        }


class ClfMimicCXRTestMixed(pl.LightningModule):
    # Supervised-Ensemble model: Only at test time the scores are averaged
    def __init__(self, cfg):
        super().__init__()

        self.clfs = nn.ModuleList(
            [ClfImg(cfg).to(cfg.model.device), ClfImg(cfg).to(cfg.model.device)]
        )
        self.cfg = cfg
        self.loss = nn.CrossEntropyLoss()
        self.validation_step_outputs = []
        self.training_step_outputs = []
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        out = self.forward(self.cfg, batch)
        # loss, mean_ap = self.compute_loss("train", batch, out)
        imgs, labels = batch
        preds, losses = out
        loss = losses.mean(dim=1).mean(dim=0)
        self.log("train/loss/loss", loss)
        return loss

    def on_validation_epoch_start(self):
        self.val_preds = {}
        self.val_labels = {}
        for m in range(self.cfg.dataset.num_views):
            self.val_preds[m] = {}
            self.val_labels[m] = {}
            for k in range(self.cfg.dataset.num_labels):
                self.val_preds[m][k] = []
                self.val_labels[m][k] = []

    def validation_step(self, batch, batch_idx):
        out = self.forward(self.cfg, batch)
        imgs, labels = batch
        preds, losses = out
        loss = losses.mean(dim=1).mean(dim=0)
        self.log("val/loss/loss", loss)
        n_labels = labels.shape[1]

        for m in range(self.cfg.dataset.num_views):
            pred_m = preds[:, m, :]
            for k in range(0, n_labels):
                self.val_preds[m][k].extend(pred_m[:, k].detach().cpu().numpy())
                self.val_labels[m][k].extend(labels[:, k].cpu())

        return loss

    def on_validation_epoch_end(self):
        aurocs = []
        mix_aurocs = []

        # mixed F and L
        for k in range(self.cfg.dataset.num_labels):
            mod_specific_preds = (
                []
            )  # list of lists -> each list is the predictions of a view
            mixed_labels = []
            for m in range(self.cfg.dataset.num_views):
                mod_specific_preds.append(self.val_preds[m][k])
                mixed_labels.append(self.val_labels[m][k])

            # Assert that the labels are the same across views for the current label 'k'
            # assert mixed_labels[0] == mixed_labels[1] - commented for performance

            m_labels = mixed_labels[0]
            # mix the scores
            mix_preds = [
                statistics.mean(x)
                for x in zip(mod_specific_preds[0], mod_specific_preds[1])
            ]

            try:
                mix_aurocs.append(roc_auc_score(m_labels, mix_preds))
                self.log(
                    "val/auroc_ensemble/" + str(self.cfg.dataset.target_list[k]) + "/",
                    mix_aurocs[k],
                )
            except:
                raise ValueError("ERROR MIX MACRO AUROC CALCULATION")

        print("MEAN AUROC ENSEMBLE", np.array(mix_aurocs).mean())
        self.log("val/mean_auroc_ensemble", np.array(mix_aurocs).mean())

        # Separated
        for m in range(self.cfg.dataset.num_views):
            auroc_m = torch.zeros(self.cfg.dataset.num_labels)
            for k in range(0, self.cfg.dataset.num_labels):
                all_preds = self.val_preds[m][k]
                all_labels = self.val_labels[m][k]
                try:
                    auroc_m[k] = roc_auc_score(all_labels, all_preds)
                    self.log(
                        "val/auroc_independent/v"
                        + str(m)
                        + "/"
                        + str(self.cfg.dataset.target_list[k]),
                        auroc_m[k],
                    )
                except:
                    raise ValueError("ERROR MACRO AUROC CALCULATION")

            aurocs.append(auroc_m.mean())

            self.log("val/auroc_independent/v" + str(m) + "/mean/", auroc_m.mean())
        self.log("val/mean_auroc_independent", np.array(aurocs).mean())
        self.validation_step_outputs.clear()  # free memory

    def forward(self, cfg, batch):
        data, labels = batch
        n_labels = labels.shape[1]
        preds = torch.zeros(
            (cfg.model.batch_size, cfg.dataset.num_views, n_labels),
            device=cfg.model.device,
        )
        losses = torch.zeros(
            (cfg.model.batch_size, cfg.dataset.num_views, 1), device=cfg.model.device
        )

        for m, m_key in enumerate(data.keys()):
            if m_key == "frontal":
                assert m == 0
            if m_key == "lateral":
                assert m == 1
            print("m", m, "m_key", m_key)
            print(data.keys())
            m_val = data[m_key]
            pred_m = self.clfs[m](m_val)
            preds[:, m, :] = pred_m
            np.save(f"preds.npy", preds.detach().cpu().numpy())
            loss_m = self.loss(pred_m, labels)
            losses[:, m, :] = loss_m
        return preds, losses

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.model.lr,
        )
        return {
            "optimizer": optimizer,
        }


class ClfMimicCXRTrainMixed(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()

        self.clfs = nn.ModuleList(
            [ClfImg(cfg).to(cfg.model.device), ClfImg(cfg).to(cfg.model.device)]
        )
        self.cfg = cfg
        self.loss = nn.CrossEntropyLoss()
        self.validation_step_outputs = []
        self.training_step_outputs = []
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        out = self.forward(self.cfg, batch)
        preds, loss = out
        self.log("train/loss/loss", loss)
        return loss

    def on_validation_epoch_start(self):
        self.val_preds = {}
        self.val_labels = {}
        for k in range(self.cfg.dataset.num_labels):
            self.val_preds[k] = []
            self.val_labels[k] = []

    def validation_step(self, batch, batch_idx):
        out = self.forward(self.cfg, batch)
        imgs, labels = batch
        preds, loss = out
        self.log("val/loss/loss", loss)
        n_labels = self.cfg.dataset.num_labels
        for k in range(0, n_labels):
            self.val_preds[k].extend(preds[:, k].detach().cpu().numpy())
            self.val_labels[k].extend(labels[:, k].cpu())
        return loss

    def on_validation_epoch_end(self):
        mix_aurocs = []
        for k in range(self.cfg.dataset.num_labels):

            preds = self.val_preds[k]
            labels = self.val_labels[k]
            try:
                mix_aurocs.append(roc_auc_score(labels, preds))
                self.log(
                    "val/auroc_multimodal/"
                    + str(self.cfg.dataset.target_list[k])
                    + "/",
                    mix_aurocs[k],
                )
            except:
                raise ValueError("ERROR MIX MACRO AUROC CALCULATION")

        print("MEAN AUROC MULTIMODAL", np.array(mix_aurocs).mean())
        self.log("val/mean_auroc_multimodal", np.array(mix_aurocs).mean())

    def forward(self, cfg, batch):
        data, labels = batch
        n_labels = labels.shape[1]
        preds = torch.zeros(
            (cfg.model.batch_size, cfg.dataset.num_views, n_labels),
            device=cfg.model.device,
        )
        for m, m_key in enumerate(data.keys()):
            if m_key == "frontal":
                assert m == 0
            if m_key == "lateral":
                assert m == 1

            m_val = data[m_key]
            pred_m = self.clfs[m](m_val)
            preds[:, m, :] = pred_m

        # avg_preds
        preds = torch.mean(preds, dim=1)
        loss = self.loss(preds, labels)
        return preds, loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.model.lr,
        )
        return {
            "optimizer": optimizer,
        }
