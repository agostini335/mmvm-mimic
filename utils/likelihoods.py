
import numpy as np
import math
import torch

LOG2PI = float(np.log(2.0 * math.pi))


def get_latent_samples(model, latents, n_imp_samples):
    mu, lv = latents
    mu_rep = mu.unsqueeze(0).repeat(n_imp_samples, 1, 1)
    lv_rep = lv.unsqueeze(0).repeat(n_imp_samples, 1, 1)
    z = model.reparametrize(mu_rep, lv_rep)
    embs = {'mu': mu_rep, 'lv': lv_rep, 'z': z}
    return embs

def log_mean_exp(x, dim=1):
    """
    log(1/k * sum(exp(x))): this normalizes x.
    @param x: PyTorch.Tensor
              samples from gaussian
    @param dim: integer (default: 1)
                which dimension to take the mean over
    @return: PyTorch.Tensor
             mean of x
    """
    m = torch.max(x, dim=dim, keepdim=True)[0]
    return m + torch.log(torch.mean(torch.exp(x - m),
                         dim=dim, keepdim=True))


def gaussian_log_pdf(x, mu, logvar):
    """
    Log-likelihood of data given ~N(mu, exp(logvar))
    @param x: samples from gaussian
    @param mu: mean of distribution
    @param logvar: log variance of distribution
    @return log_pdf: PyTorch.Tensor
                     log-likelihood
    """
    global LOG2PI
    log_pdf = -0.5 * LOG2PI - logvar / 2. - torch.pow(x - mu, 2) / (2. * torch.exp(logvar))
    return torch.sum(log_pdf, dim=1)


def unit_gaussian_log_pdf(x):
    """
    Log-likelihood of data given ~N(0, 1)
    @param x: PyTorch.Tensor
              samples from gaussian
    @return log_pdf: PyTorch.Tensor
                     log-likelihood
    """
    global LOG2PI
    log_pdf = -0.5 * LOG2PI - math.log(1.) / 2. - torch.pow(x, 2) / 2.
    return torch.sum(log_pdf, dim=1)


def log_marginal_estimate(cfg, n_samples, likelihood, image, embs):
    batch_size = cfg.model.batch_size;

    d_shape = image.shape;
    if len(d_shape) == 3:
        image = image.unsqueeze(0).repeat(n_samples, 1, 1, 1);
        image = image.view(batch_size*n_samples, d_shape[-2], d_shape[-1])
    elif len(d_shape) == 4:
        image = image.unsqueeze(0).repeat(n_samples, 1, 1, 1, 1);
        image = image.view(batch_size*n_samples, d_shape[-3], d_shape[-2],
                           d_shape[-1])
    
    z = embs['z']
    mu = embs['mu']
    logvar = embs['lv']
    log_p_x_given_z_2d = likelihood.log_prob(image).view(batch_size*n_samples,
                                                        -1).sum(dim=1)
    log_q_z_given_x_2d = gaussian_log_pdf(z, mu, logvar)
    log_p_z_2d = unit_gaussian_log_pdf(z)
    log_weight_2d = log_p_x_given_z_2d + log_p_z_2d - log_q_z_given_x_2d;
    log_weight = log_weight_2d.view(batch_size, n_samples)
    # need to compute normalization constant for weights
    # i.e. log ( mean ( exp ( log_weights ) ) )
    log_p = log_mean_exp(log_weight, dim=1)
    return torch.mean(log_p)


def log_joint_estimate(cfg, n_samples, likelihoods, data, embs):
    batch_size = cfg.model.batch_size;
    z = embs['z']
    mu = embs['mu'];
    logvar = embs['lv'];

    num_views = cfg.dataset.num_views
    log_px_zs = torch.zeros(num_views, batch_size * n_samples);
    log_px_zs = log_px_zs.to(cfg.model.device);
    for k, key in enumerate(data.keys()):
        batch_d = data[key]
        d_shape = batch_d.shape;
        if len(d_shape) == 3:
            batch_d = batch_d.unsqueeze(0).repeat(n_samples, 1, 1, 1);
            batch_d = batch_d.view(batch_size*n_samples, d_shape[-2], d_shape[-1])
        elif len(d_shape) == 4:
            batch_d = batch_d.unsqueeze(0).repeat(n_samples, 1, 1, 1, 1);
            batch_d = batch_d.view(batch_size*n_samples, d_shape[-3], d_shape[-2],
                               d_shape[-1])
        lhood = likelihoods[key]
        log_p_x_given_z_2d = lhood.log_prob(batch_d).view(batch_size * n_samples, -1).sum(dim=1);
        log_px_zs[k] = log_p_x_given_z_2d;

    # compute components of likelihood estimate
    log_joint_zs_2d = log_px_zs.sum(0)  # sum over modalities
    log_p_z_2d = unit_gaussian_log_pdf(z)
    log_q_z_given_x_2d = gaussian_log_pdf(z, mu, logvar)
    log_weight_2d = log_joint_zs_2d + log_p_z_2d - log_q_z_given_x_2d
    log_weight = log_weight_2d.view(batch_size, n_samples)
    log_p = log_mean_exp(log_weight, dim=1)
    return torch.mean(log_p)

def generate_sufficient_statistics(mvvae, dists):
    mods_gen = {}
    for m, key in enumerate(mvvae.modality_names):
        emb_m = dists[key]
        mu_m, lv_m, z_m = emb_m["mu"], emb_m["lv"], emb_m["z"] 
        mods_gen_m = {}
        for m_tilde, key_tilde in enumerate(mvvae.modality_names):
            if key_tilde == "text":
                cg_m_m_tilde = torch.distributions.one_hot_categorical.OneHotCategorical(
                    *mvvae.decoders[m_tilde](z_m),
                    validate_args=False
                )
            else:
                cg_m_m_tilde = torch.distributions.laplace.Laplace(
                    *mvvae.decoders[m_tilde](z_m)
                )
            mods_gen_m[key_tilde] = cg_m_m_tilde
        mods_gen[key] = mods_gen_m
    return mods_gen

#at the moment: only marginals and joint
# everything is based on latents of modalities (not subsets)
def calc_log_likelihood_batch(cfg, model, model_out, batch, num_imp_samples=10):
    data, labels = batch
    latents = model_out[1]
    embs_lin = {}
    for m, key in enumerate(model.modality_names):
        dist_m = latents[key]
        n_total_samples = dist_m[0].shape[0]*num_imp_samples;
        emb_mod = get_latent_samples(model, dist_m, num_imp_samples);
        emb_mod_lin = {'mu': emb_mod['mu'].view(n_total_samples, -1),
                       'lv': emb_mod['lv'].view(n_total_samples, -1),
                       'z': emb_mod['z'].view(n_total_samples, -1)}
        embs_lin[key] = emb_mod_lin
    gen = generate_sufficient_statistics(model, embs_lin);

    lls = {}
    for m, key in enumerate(model.modality_names):
        ll_m = {}
        for m_tilde, key_tilde in enumerate(model.modality_names):
            ll_m_m_tilde = log_marginal_estimate(cfg,
                                        num_imp_samples,
                                        gen[key][key_tilde],
                                        data[key_tilde],
                                        embs_lin[key])
            ll_m[key_tilde] = ll_m_m_tilde
        ll_joint = log_joint_estimate(cfg,
                                      num_imp_samples,
                                      gen[key],
                                      data,
                                      embs_lin[key])
        ll_m['joint'] = ll_joint;
        lls[key] = ll_m
    return lls


def estimate_likelihoods(cfg, model, data):
    model = exp.mm_vae;
    mods = exp.modalities;
    bs_normal = exp.flags.batch_size;
    exp.flags.batch_size = 64;
    d_loader = DataLoader(exp.dataset_test,
                          batch_size=exp.flags.batch_size,
                          shuffle=True,
                          num_workers=8, drop_last=True);

    subsets = exp.subsets;
    lhoods = dict()
    for k, s_key in enumerate(subsets.keys()):
        if s_key != '':
            lhoods[s_key] = dict();
            for m, m_key in enumerate(mods.keys()):
                lhoods[s_key][m_key] = [];
            lhoods[s_key]['joint'] = [];

    for iteration, batch in enumerate(d_loader):
        batch_d = batch[0];
        for m, m_key in enumerate(mods.keys()):
            batch_d[m_key] = batch_d[m_key].to(exp.flags.device);

        latents = model.inference(batch_d);
        for k, s_key in enumerate(subsets.keys()): 
            if s_key != '':
                subset = subsets[s_key];
                ll_batch = calc_log_likelihood_batch(exp, latents,
                                                     s_key, subset,
                                                     batch_d,
                                                     num_imp_samples=12)
                for l, m_key in enumerate(ll_batch.keys()):
                    lhoods[s_key][m_key].append(ll_batch[m_key].item());

    for k, s_key in enumerate(lhoods.keys()):
        lh_subset = lhoods[s_key];
        for l, m_key in enumerate(lh_subset.keys()):
            mean_val = np.mean(np.array(lh_subset[m_key]))
            lhoods[s_key][m_key] = mean_val;
    exp.flags.batch_size = bs_normal;
    return lhoods;
