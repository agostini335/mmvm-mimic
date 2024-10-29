import torch
from torch import nn

from mv_vaes.mv_vae import MVVAE


class MVMixedPriorVAE(MVVAE):
    def __init__(self, cfg):
        super().__init__(cfg)

        if cfg.model.drpm_prior:
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
            # Initialize random partition module
            self.mw_drpm = MVDRPM(n_cluster=cfg.model.n_groups, device=cfg.model.device)

        self.save_hyperparameters()

    def log_additional_values(self, out):
        pass

    def log_additional_values_val(self):
        pass

    def forward(self, batch, resample):
        data = batch[0]
        labels = batch[1]

        dists_enc_out = {}
        dists_out = {}
        mods_rec = {}
        for m, key in enumerate(data.keys()):
            # encode views: img_m -> z_m
            mod_m = data[key]
            mu_m, lv_m = self.encoders[m](mod_m)
            dists_enc_out[key] = [mu_m, lv_m]
            z_m = self.reparametrize(mu_m, lv_m)

            # decode views: z_m -> img_hat_m
            mod_hat_m = self.decoders[m](z_m)
            mods_rec[key] = mod_hat_m

            dist_out_m = [mu_m, lv_m]
            dists_out[key] = dist_out_m
        return (mods_rec, dists_out, dists_enc_out)

    def compute_log_prob_Y(self, log_omega, log_scores, n_hat):
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
        prior_log_p_n = self.mw_drpm.get_log_prob_mvhg(
            log_omega.repeat(1, 1),
            n_hat_integer,
            n_hat,
        )

        prior_log_p_pis = []
        for m in range(self.cfg.dataset.num_views):
            sort_prior = PL(
                log_scores[:, m, :],
                tau=self.compute_current_temperature(),
                g_noise=False,
            )
            prior_log_p_Ps = sort_prior.log_prob(sort_prior.rsample([1]))
            prior_log_p_pis.append(prior_log_p_Ps.unsqueeze(-1))
        prior_log_p_pis = torch.cat(prior_log_p_pis, dim=-1).sum(dim=-1)
        log_p_prior = prior_log_p_n + prior_log_p_pis
        return log_p_prior

    def compute_loss(self, str_set, batch, forward_out):
        imgs, labels = batch
        imgs_rec = forward_out[0]
        dists_out = forward_out[1]

        # kl divergence of latent distribution
        if self.cfg.model.drpm_prior:
            log_scores = []
            log_omegas = []
            mus = []
            lvs = []
            # encode views
            for m, key in enumerate(self.modality_names):
                mu_m, lv_m = dists_out[key]
                mus.append(mu_m.unsqueeze(1))
                lvs.append(lv_m.unsqueeze(1))
                log_scores_m = self.log_scores[m]
                log_scores.append(log_scores_m.unsqueeze(1))
            mus = torch.cat(mus, dim=1)
            lvs = torch.cat(lvs, dim=1)
            log_scores = torch.cat(log_scores, dim=1)
            log_omega = self.log_omega

            hard_pi = self.hparams.cfg.model.hard_pi
            gs_noise = self.hparams.cfg.model.add_gumbel_noise
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
            for k in range(self.cfg.model.n_groups):
                self.log(str_set + "/n_factors/n_" + str(k).zfill(2), n_hat_integer[k])
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
            # generate prior distributions
            priors = []
            prior_mu_std = torch.zeros_like(mus[:, 0, :])
            prior_lv_std = torch.zeros_like(mus[:, 0, :])
            for m, key in enumerate(self.modality_names):
                mu_m_in, lv_m_in = mus[:, m, :], lvs[:, m, :]
                rpm_perms_m = permutations[:, m, :, :, :]
                perm_m_shared = rpm_perms_m[:, 0, :, :]
                perm_m_shared_T = torch.transpose(perm_m_shared, dim0=-2, dim1=-1)
                mu_m_shared_out = (perm_m_shared_T @ mu_shared.unsqueeze(-1)).squeeze(
                    -1
                )
                lv_m_shared_out = (perm_m_shared_T @ lv_shared.unsqueeze(-1)).squeeze(
                    -1
                )

                perm_m_ind = rpm_perms_m[:, 1, :, :]
                part_m_ind = perm_m_ind.sum(dim=-2)
                mu_m_ind_out = part_m_ind * mu_m_in
                lv_m_ind_out = part_m_ind * lv_m_in

                prior_m_mu = mu_m_shared_out + prior_mu_std
                prior_m_lv = lv_m_shared_out + prior_lv_std
                priors.append([prior_m_mu, prior_m_lv])
        else:
            priors = dists_out

        if self.cfg.model.alpha_annealing:
            init_temp = self.cfg.model.init_alpha_value
            final_temp = self.cfg.model.final_alpha_value
            annealing_steps = self.cfg.model.alpha_annealing_steps
            alpha_weight = self.compute_current_temperature(
                init_temp, final_temp, annealing_steps
            )
        else:
            alpha_weight = self.cfg.model.final_alpha_value
        self.log("alpha annealing", alpha_weight)
        klds = []
        for m, key in enumerate(self.modality_names):
            dist_m = dists_out[key]
            for m_tilde, key_tilde in enumerate(self.modality_names):
                dist_m_tilde = priors[key_tilde]
                kld_m_m_tilde = self.kl_div_z_two_dists(dist_m, dist_m_tilde)
                # KL(q_m | q_m_tilde) * (1-alpha)
                klds.append(kld_m_m_tilde.unsqueeze(1) * (1.0 - alpha_weight))
            # add N(0,1) as a component
            kld_m = self.kl_div_z(dist_m)
            # KL(q_m | N(0,1)) * alpha * M
            klds.append(kld_m.unsqueeze(1) * alpha_weight * self.cfg.dataset.num_views)
        # SUM_{m}:( alpha * KL(q_m|N(0,1)) + (1-alpha)/M * SUM_{m_tilde}:KL(q_m|q_m_tilde) )
        # when alpha = 0: mixedprior
        # when alpha = 1: unimodal
        # when alpha = 1/(M+1): mixedpriorstdnorm
        klds_sum = torch.cat(klds, dim=1).sum(dim=1) / self.cfg.dataset.num_views

        if self.cfg.model.drpm_prior:
            log_p_Y = self.compute_log_prob_Y(log_omega, log_scores, ohts)

        ## compute reconstruction loss/ conditional log-likelihood out data
        ## given latents
        loss_rec, loss_rec_mods, loss_rec_mods_weighted = self.compute_rec_loss(
            imgs, imgs_rec
        )
        for m, key in enumerate(self.modality_names):
            self.log(
                f"{str_set}/loss/weighted_rec_loss_{key}",
                loss_rec_mods_weighted[key],
            )
            self.log(
                f"{str_set}/loss/rec_loss_{key}",
                loss_rec_mods[key],
            )

        beta = self.cfg.model.beta
        loss_mv_vae = (loss_rec + beta * klds_sum).mean(dim=0)
        if self.cfg.model.drpm_prior:
            gamma = self.cfg.model.gamma * (1.0 - self.compute_current_temperature())
            loss_mv_vae -= gamma * log_p_Y.mean(dim=0)
            self.log(str_set + "/loss/log_p_Y", log_p_Y.mean(dim=0))
        total_loss = loss_mv_vae
        # logging
        self.log(str_set + "/loss/klds_avg", klds_sum.mean(dim=0))
        self.log(str_set + "/loss/loss_rec", loss_rec.mean(dim=0))
        self.log(str_set + "/loss/mv_vae", loss_mv_vae)
        self.log(str_set + "/loss/loss", total_loss)
        return total_loss, loss_rec
