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

from networks.ConvNetworksPolyMNIST import Encoder, Decoder
from networks.ConvNetworksPolyMNIST import EncoderDist
from networks.ConvNetworksPolyMNIST import ResnetEncoderDist
from utils.eval import generate_samples

from drpm.mv_drpm import MVDRPM
from drpm.pl import PL

from mv_vaes.mv_vae import MVVAE


class MVDRPMVAE(MVVAE):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.model.learn_const_dist_params:
            # log omega parameters
            self.log_omega = nn.Parameter(
                torch.randn(
                    size=(1, cfg.model.n_groups),
                    device=cfg.model.device,
                )
            )
            self.log_scores = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.randn(
                            size=(1, cfg.model.latent_dim),
                            device=cfg.model.device,
                        )
                    )
                    for m in range(cfg.dataset.num_views)
                ]
            )
        elif cfg.model.encoders_rpm:
            if cfg.model.use_resnets:
                self.encoders_rpm = nn.ModuleList(
                    [
                        ResnetEncoderDist(cfg).to(cfg.model.device)
                        for m in range(cfg.dataset.num_views)
                    ]
                )
            else:
                self.encoders_rpm = nn.ModuleList(
                    [
                        EncoderDist(cfg.model.latent_dim, cfg.model.n_groups).to(
                            cfg.model.device
                        )
                        for m in range(cfg.dataset.num_views)
                    ]
                )
        else:
            # log scores fc
            self.estimate_log_scores = nn.ModuleList(
                [
                    nn.Linear(cfg.model.latent_dim, cfg.model.latent_dim).to(
                        cfg.model.device
                    )
                    for m in range(cfg.dataset.num_views)
                ]
            )
            # log omega fc
            self.estimate_log_omega = nn.Linear(
                cfg.model.latent_dim, cfg.model.n_groups
            ).to(cfg.model.device)
        self.relu = torch.nn.ReLU()

        # Initialize random partition module
        self.mw_drpm = MWDRPM(n_cluster=cfg.model.n_groups, device=cfg.model.device)
        self.save_hyperparameters()

    def log_additional_values(self, out):
        log = {}
        dist_params = out[5]
        log_omega, log_scores = dist_params
        log["parameters/log_omega"] = wandb.Histogram(log_omega.detach().cpu())
        log["parameters/log_scores"] = wandb.Histogram(log_scores.detach().cpu())
        self.logger.log_metrics(log)

    def log_additional_values_val(self):
        n_hat_val = []
        for idx, val_out in enumerate(self.validation_step_outputs):
            out, batch = val_out
            n_hat_batch = torch.cat(out[4], dim=1)
            n_hat_val.append(n_hat_batch)
        # number of shared generative factors
        n_hat_val = torch.cat(n_hat_val, dim=0)
        n_hat_avg = n_hat_val.mean(dim=0)
        for k in range(0, n_hat_avg.shape[0]):
            self.log("val/n_factors/n_" + str(k).zfill(2), n_hat_avg[k])

    def forward(self, batch, resample):
        images = batch[0]
        labels = batch[1]

        mus = []
        lvs = []
        log_scores = []
        log_omegas = []
        # encode views
        for m in range(0, self.cfg.dataset.num_views):
            img_m = images["m" + str(m)]
            mu_m, lv_m = self.encoders[m](img_m)
            mus.append(mu_m.unsqueeze(1))
            lvs.append(lv_m.unsqueeze(1))
            if self.cfg.model.learn_const_dist_params:
                log_scores_m = self.log_scores[m]
            elif self.cfg.model.encoders_rpm:
                log_omega_m, log_scores_m = self.encoders_rpm[m](img_m)
                log_omegas.append(log_omega_m.unsqueeze(1))
            else:
                log_scores_m = self.estimate_log_scores[m](mu_m)
            log_scores.append(log_scores_m.unsqueeze(1))
        mus = torch.cat(mus, dim=1)
        lvs = torch.cat(lvs, dim=1)
        log_scores = torch.cat(log_scores, dim=1)

        mu_agg = torch.mean(mus, dim=1).squeeze(1)
        if self.cfg.model.learn_const_dist_params:
            log_omega = self.log_omega
        elif self.cfg.model.encoders_rpm:
            log_omega = torch.mean(torch.cat(log_omegas, dim=1), dim=1)
        else:
            log_omega = self.estimate_log_omega(mu_agg)

        hard_pi = self.hparams.cfg.model.hard_pi
        if resample:
            gs_noise = self.hparams.cfg.model.add_gumbel_noise
        else:
            gs_noise = False
        tau = self.compute_current_temperature()
        self.log("temperature", tau)
        (
            permutations,
            rpms,
            ohts,
            ohts_filled_shifted,
            n_hat_integer,
            log_p_n,
            log_p_pi,
        ) = self.mw_drpm(
            log_scores,
            log_omega,
            g_noise=gs_noise,
            hard_pi=hard_pi,
            temperature=tau,
        )
        log_p_rpms = [log_p_n, log_p_pi]
        rpm_dist_params = [log_omega, log_scores]
        # generate shared representation
        mus_shared = []
        lvs_shared = []
        for m in range(0, self.cfg.dataset.num_views):
            mu_in_m, lv_in_m = mus[:, m, :], lvs[:, m, :]
            rpm_perms_m = permutations[:, m, :, :, :]
            # we define the first subset as the shared one
            perm_m_shared = rpm_perms_m[:, 0, :, :]
            mu_m_in_shared = (perm_m_shared @ mu_in_m.unsqueeze(-1)).squeeze(-1)
            lv_m_in_shared = (perm_m_shared @ lv_in_m.unsqueeze(-1)).squeeze(-1)
            mus_shared.append(mu_m_in_shared.unsqueeze(1))
            lvs_shared.append(lv_m_in_shared.unsqueeze(1))
        mus_shared = torch.cat(mus_shared, dim=1)
        lvs_shared = torch.cat(lvs_shared, dim=1)
        mu_shared, lv_shared = self.aggregate_latents(mus_shared, lvs_shared)

        # decode views
        mus_out = []
        lvs_out = []
        dists_out = []
        imgs_rec = {}
        for m in range(0, self.cfg.dataset.num_views):
            mu_m_in, lv_m_in = mus[:, m, :], lvs[:, m, :]
            rpm_perms_m = permutations[:, m, :, :, :]
            perm_m_shared = rpm_perms_m[:, 0, :, :]
            perm_m_shared_T = torch.transpose(perm_m_shared, dim0=-2, dim1=-1)
            mu_m_shared_out = (perm_m_shared_T @ mu_shared.unsqueeze(-1)).squeeze(-1)
            lv_m_shared_out = (perm_m_shared_T @ lv_shared.unsqueeze(-1)).squeeze(-1)

            perm_m_ind = rpm_perms_m[:, 1, :, :]
            part_m_ind = perm_m_ind.sum(dim=-2)
            mu_m_ind_out = part_m_ind * mu_m_in
            lv_m_ind_out = part_m_ind * lv_m_in

            mu_m_out = mu_m_shared_out + mu_m_ind_out
            lv_m_out = lv_m_shared_out + lv_m_ind_out

            mus_out.append(mu_m_out)
            lvs_out.append(lv_m_out)
            z_m = self.reparametrize(mu_m_out, lv_m_out)
            img_hat_m = self.decoders[m](z_m)
            imgs_rec["m" + str(m)] = img_hat_m

            dist_out_m = [mu_m_out, lv_m_out]
            dists_out.append(dist_out_m)

        return (
            imgs_rec,
            dists_out,
            log_p_rpms,
            ohts,
            n_hat_integer,
            rpm_dist_params,
            rpms,
            permutations,
            ohts_filled_shifted,
        )

    def kl_div_rpm(self, log_p_n, log_scores, n_hat):
        ## Create integer number from mvhg sample to compute Pi_Y
        n_hat_integer_cat = (
            torch.cat(n_hat, dim=1)
            .squeeze(-1)
            .float()
            .to(self.hparams.cfg.model.device)
        )
        n_hat_integer_cat = n_hat_integer_cat * torch.arange(
            self.hparams.cfg.model.latent_dim + 1
        ).unsqueeze(0).to(self.hparams.cfg.model.device)
        n_hat_integer_cat = n_hat_integer_cat.sum(dim=-1).to(
            self.hparams.cfg.model.device
        )
        n_hat_integer = [
            n_hat_integer_cat[:, i].reshape(-1, 1, 1)
            for i in range(n_hat_integer_cat.shape[-1])
        ]
        n_hat_integer_concat = torch.cat(n_hat_integer, dim=-1)

        ## Compute mvhg prior probability
        prior_log_omega = torch.zeros(
            (self.hparams.cfg.model.batch_size, self.hparams.cfg.model.n_groups),
            device=self.hparams.cfg.model.device,
        )
        prior_log_p_n = self.mw_drpm.get_log_prob_mvhg(
            prior_log_omega.repeat(1, 1),
            n_hat_integer,
            n_hat,
        )
        log_num_Y_permutations = (
            (self.relu(n_hat_integer_concat) + 1.0).lgamma().squeeze(1).sum(dim=-1)
        )
        kl_div_n = torch.relu((log_p_n + log_num_Y_permutations - prior_log_p_n))

        ## Get maximum prior probability of permutations by ignoring gumbel noise in PL sampling
        sort_prior = PL(
            torch.zeros_like(log_scores[:, 0, :]),
            tau=self.compute_current_temperature(),
            g_noise=False,
        )
        max_prior_log_p_Ps = sort_prior.log_prob(sort_prior.rsample([1]))

        kl_div_pi = []
        ## Get maximum probability of permutations by ignoring gumbel noise in PL sampling
        for m in range(self.cfg.dataset.num_views):
            log_scores_m = log_scores[:, m, :]
            sort_post_m = PL(
                log_scores_m,
                tau=self.compute_current_temperature(),
                g_noise=False,
            )
            max_log_p_Ps_m = sort_post_m.log_prob(sort_post_m.rsample([1]))
            kl_div_pi_m = max_log_p_Ps_m - max_prior_log_p_Ps
            kl_div_pi.append(kl_div_pi_m.unsqueeze(-1))
        kl_div_pi = torch.cat(kl_div_pi, dim=-1).sum(dim=-1)
        return kl_div_n, kl_div_pi

    def compute_loss(self, str_set, batch, forward_out):
        imgs, labels = batch
        imgs_rec = forward_out[0]
        dists_out = forward_out[1]
        log_p_rpms = forward_out[2]
        n_ohts = forward_out[3]
        rpm_dist_params = forward_out[5]

        log_omega, log_scores = rpm_dist_params

        ## compute kl div of latent distributions given partitioning
        klds = []
        for m in range(self.cfg.dataset.num_views):
            dist_m = dists_out[m]
            kld_m = self.kl_div_z(dist_m)
            klds.append(kld_m.unsqueeze(1))
        klds_sum = torch.cat(klds, dim=1).sum(dim=1)

        ## compute reconstruction loss/ conditional log-likelihood out data
        ## given latents
        loss_rec = self.compute_rec_loss(imgs, imgs_rec)

        ## Compute kl div of partition
        log_p_n, log_p_pi_n_M = log_p_rpms
        kl_div_n, kl_div_pi = self.kl_div_rpm(log_p_n, log_scores, n_ohts)

        ## weighting of individual loss terms
        beta = self.hparams.cfg.model.beta
        gamma = self.hparams.cfg.model.gamma
        delta = self.hparams.cfg.model.delta
        Y_kl_div = (gamma * kl_div_n + delta * kl_div_pi).mean(dim=0)
        loss_mv_vae = (loss_rec + beta * klds_sum).mean(dim=0)
        total_loss = loss_mv_vae + Y_kl_div

        # logging
        self.log(str_set + "/loss/klds_avg", klds_sum.mean(dim=0))
        self.log(str_set + "/loss/loss_rec", loss_rec.mean(dim=0))
        self.log(str_set + "/kl_divs/n", kl_div_n.mean(dim=0))
        self.log(str_set + "/kl_divs/pi", kl_div_pi.mean(dim=0))
        self.log(str_set + "/loss/Y", Y_kl_div)
        self.log(str_set + "/loss/mv_vae", loss_mv_vae)
        self.log(str_set + "/loss/loss", total_loss)
        return total_loss

    def compute_current_temperature(self):
        """
        Compute temperature based on current step
        -> exponential temperature annealing
        """
        final_temp = self.hparams.cfg.model.final_temp
        init_temp = self.hparams.cfg.model.init_temp
        num_steps_annealing = self.hparams.cfg.model.num_steps_annealing
        rate = (math.log(final_temp) - math.log(init_temp)) / float(num_steps_annealing)
        curr_temp = max(init_temp * math.exp(rate * self.global_step), final_temp)
        return curr_temp
