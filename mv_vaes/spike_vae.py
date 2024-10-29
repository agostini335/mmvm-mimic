import os
import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb

from torchvision.utils import make_grid
import torchvision.transforms.functional as F

from networks.NetworksRatsspike import Encoder, Decoder
import matplotlib.pyplot as plt

class SPIKEVAE(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.original_dims = [92, 79, 104, 49, 46]
        self.encoders = nn.ModuleList(
            [
                Encoder(cfg.model.latent_dim, self.original_dims[m]).to(cfg.model.device)
                for m in range(cfg.dataset.num_views)
            ]
        )
        self.decoders = nn.ModuleList(
            [
                Decoder(cfg.model.latent_dim, self.original_dims[m]).to(cfg.model.device)
                for m in range(cfg.dataset.num_views)
            ]
        )

        self.validation_step_outputs = []
        self.training_step_outputs = []

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        # print(self.encoders[0](batch[0]["m0"])[0].shape)
        # print(torch.Tensor((np.random.rand((256 * 92))>.5).astype(float).reshape((256, 92))))
        # print(self.encoders[0](torch.Tensor((np.random.rand((256 * 92))>.5).astype(float).reshape((256, 92))).to("cuda:0"))[0][:5])
        # print(self.encoders[0](batch[0]["m0"])[0][:5])
        # print(self.encoders[0].parameters())
        out = self.forward(batch, resample=True)
        loss = self.compute_loss("train", batch, out)
        bs = self.cfg.model.batch_size
        if len(self.training_step_outputs) * bs < self.cfg.eval.num_samples_train:
            self.training_step_outputs.append([out, batch])
        return loss

    def validation_step(self, batch, batch_idx):
        # print(batch[0]["m0"].shape, batch[0]["m0"][:5,:5])
        # print(self.encoders[0](batch[0]["m0"])[0][:5])
        # print(self.encoders[0].parameters())
        out = self.forward(batch, resample=self.cfg.model.resample_eval)
        # print(out[1][0][1].shape, out[1][0][0].shape, out[1][0][0][:5])
        loss = self.compute_loss("val", batch, out)

        if self.cfg.eval.coherence:
            acc_coh = self.evaluate_conditional_generation(out, batch)
        else:
            acc_coh = None
        self.validation_step_outputs.append([out, batch, acc_coh])
        self.log_additional_values(out)
        return loss


    def on_validation_epoch_end(self):
        # print(self.encoders[0].state_dict())#.get('mu.bias'), self.encoders[0].state_dict().get('logvar.bias'))
        enc_mu_val = {str(m): [] for m in range(self.cfg.dataset.num_views)}
        enc_lv_val = {str(m): [] for m in range(self.cfg.dataset.num_views)}
        labels_val = []

        for idx, val_out in enumerate(self.validation_step_outputs):
            out, batch, acc_coh = val_out
            imgs, labels = batch
            dists_out = out[1]
            for m in range(self.cfg.dataset.num_views):
                mu_m, lv_m = dists_out[m]
                enc_mus_m = enc_mu_val[str(m)]
                enc_lvs_m = enc_lv_val[str(m)]
                enc_mus_m.append(mu_m)
                enc_lvs_m.append(lv_m)
                enc_mu_val[str(m)] = enc_mus_m
                enc_lv_val[str(m)] = enc_lvs_m
            labels_val.append(labels)
        self.log_additional_values_val()
        self.validation_step_outputs.clear()  # free memory

        for m in range(self.cfg.dataset.num_views):
            enc_mu_m_val = enc_mu_val[str(m)]
            enc_mu_m_val = torch.cat(enc_mu_m_val, dim=0)
            enc_mu_val[str(m)] = enc_mu_m_val
            enc_lv_m_val = enc_lv_val[str(m)]
            enc_lv_m_val = torch.cat(enc_lv_m_val, dim=0)
            enc_lv_val[str(m)] = enc_lv_m_val
        labels_val = torch.cat(labels_val, dim=0)

        if self.cfg.eval.eval_downstream_task:
            colors = ['deepskyblue', 'tan', 'mediumseagreen', 'purple', 'CORAL']
            # all the views/rats
            plt.figure(figsize=(10,6))
            plt.clf()
            for m in range(self.cfg.dataset.num_views):
                enc_mu_m_val = enc_mu_val[str(m)]
                plt.scatter(enc_mu_m_val.cpu().numpy()[:, 0],
                            enc_mu_m_val.cpu().numpy()[:, 1],
                            s=.5,
                            c=[colors[odor] for odor in labels_val.cpu().numpy().astype(int)])
            plot_title = "scatter_spike_rats_"+"beta"+str(self.cfg.model.beta)+"stdnormweight"+str(self.cfg.model.stdnormweight)
            plt.title(plot_title)
            plot_name = plot_title+".png"
            plt.savefig(self.cfg.log.dir_logs+"/figs/"+plot_name)
            self.logger.log_image(key="scatter rats", images=[self.cfg.log.dir_logs+"/figs/"+plot_name])
            
            # each view/rat
            plt.clf()
            for m in range(self.cfg.dataset.num_views):
                enc_mu_m_val = enc_mu_val[str(m)]
                plt.clf()
                plt.scatter(enc_mu_m_val.cpu().numpy()[:, 0],
                            enc_mu_m_val.cpu().numpy()[:, 1],
                            s=.5,
                            c=[colors[odor] for odor in labels_val.cpu().numpy().astype(int)])
                plot_title = "scatter_spike_rat_"+str(m)+"beta"+str(self.cfg.model.beta)+"stdnormweight"+str(self.cfg.model.stdnormweight)
                plt.title(plot_title)
                plot_name = plot_title+".png"
                plt.savefig(self.cfg.log.dir_logs+"/figs/"+plot_name)
                self.logger.log_image(key="scatter rat " + str(m), images=[self.cfg.log.dir_logs+"/figs/"+plot_name])

    def kl_div_z(self, dist):
        mu, lv = dist
        prior_mu = torch.zeros_like(mu)
        prior_lv = torch.zeros_like(lv)
        prior_d = torch.distributions.normal.Normal(prior_mu, prior_lv.exp() + 1e-6)
        d1 = torch.distributions.normal.Normal(mu, lv.exp() + 1e-6)
        kld = torch.distributions.kl.kl_divergence(d1, prior_d).sum(dim=-1)
        return kld

    def kl_div_z_two_dists(self, dist1, dist2):
        mu1, lv1 = dist1
        mu2, lv2 = dist2
        d1 = torch.distributions.normal.Normal(mu1, lv1.exp() + 1e-6)
        d2 = torch.distributions.normal.Normal(mu2, lv2.exp() + 1e-6)
        kld = torch.distributions.kl.kl_divergence(d1, d2).sum(dim=-1)
        return kld

    def compute_rec_loss(self, imgs, imgs_rec):
        rec_loss_all = []
        # output probability x_m
        for m in range(self.cfg.dataset.num_views):
            img_gt_m = imgs["m" + str(m)]
            logits_m = imgs_rec["m" + str(m)]
            # BCE loss with logits
            bce = nn.BCEWithLogitsLoss(reduction='none')
            rec_loss_m = bce(input=logits_m[0], target=img_gt_m).sum(-1)
            rec_loss_all.append(rec_loss_m.unsqueeze(1))
        rec_loss_avg = torch.cat(rec_loss_all, dim=1).sum(dim=1)
        return rec_loss_avg


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.model.lr,
        )
        return {
            "optimizer": optimizer,
        }

    def aggregate_latents(self, mus, lvs):
        batch_size, num_views, num_latents = mus.shape
        mu_agg = (mus.sum(dim=1) / float(num_views)).squeeze(1)
        lv_agg = (lvs.exp().sum(dim=1) / float(num_views)).log().squeeze(1)
        return mu_agg, lv_agg

    def reparametrize(self, mu, log_sigma):
        """
        Reparametrized sampling from gaussian
        """
        dist = torch.distributions.normal.Normal(mu, log_sigma.exp() + 1e-6)
        return dist.rsample()
