import sys
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RandomizedSearchCV
import torch


def train_clfs_mimic(encodings, labels, clfs, cfg=None, logger=None):
    # check classifier availability
    for clf in clfs:
        if clf != "RF" and clf != "LR":
            raise NotImplementedError("Only RF and LR are supported")

    n_labels = labels.shape[1]

    # initialize clf_dict dict with clf as key
    # example:
    # clf_dict['RF'] return a list of Random Forest classifiers trained on each label
    # clf_dict['RF'][0] return a Random Forest trained on label nr 0

    clfs_dict = {}
    for clf in clfs:
        clfs_dict[clf] = []

    # train classifiers
    for clf in clfs:
        for k in range(0, n_labels):
            print(k)
            if logger is not None:  # used in offline eval only
                logger.log({f"k": k})
            if clf == "LR":
                clfs_dict[clf].append(
                    LogisticRegression(max_iter=10000).fit(
                        encodings.cpu(), labels[:, k].cpu()
                    )
                )
            if clf == "RF":
                if cfg is None or not cfg.eval.hp_tuning:
                    clfs_dict[clf].append(
                        RandomForestClassifier(
                            n_estimators=cfg.eval.f_n_estimators,
                            min_samples_split=cfg.eval.f_min_samples_split,
                            min_samples_leaf=cfg.eval.f_min_samples_leaf,
                            max_features=cfg.eval.f_max_features,
                            max_depth=cfg.eval.f_max_depth,
                            criterion=cfg.eval.f_criterion,
                            bootstrap=cfg.eval.f_bootstrap,
                            n_jobs=22,
                        ).fit(encodings.cpu(), labels[:, k].cpu())
                    )
                else:
                    # Random Forest with cv tuned hyperparameters
                    best_params = hyperparameter_tuning_rf(
                        encodings.cpu(), labels[:, k].cpu(), cfg
                    )
                    rcf_tuned = RandomForestClassifier(
                        n_jobs=-1, random_state=cfg.seed, **best_params
                    )
                    clfs_dict[clf].append(
                        rcf_tuned.fit(encodings.cpu(), labels[:, k].cpu())
                    )

    return clfs_dict


def hyperparameter_tuning_rf(encodings, labels, cfg):
    print("start HPTuning")
    print(cfg.eval.n_estimator)
    rfc_search_space = {
        "n_estimators": np.array(cfg.eval.n_estimator),
        "max_depth": np.array(cfg.eval.max_depth),
        "min_samples_split": np.array(cfg.eval.min_samples_split),
        "min_samples_leaf": np.array(cfg.eval.min_samples_leaf),
        "max_features": np.array(cfg.eval.max_features),
        "bootstrap": np.array(cfg.eval.bootstrap),
        "criterion": np.array(cfg.eval.criterion),
    }

    rfc = RandomForestClassifier(random_state=cfg.seed)

    random_search = RandomizedSearchCV(
        estimator=rfc,
        param_distributions=rfc_search_space,
        n_iter=cfg.eval.hp_iter,
        cv=cfg.eval.hp_cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=cfg.eval.verbosity,
    )
    random_search.fit(encodings, labels)
    best_params = random_search.best_params_

    # todo log best_params
    # todo log time
    print(best_params)
    return best_params


def eval_clfs_mimic(clfs_dict, encodings, labels, metrics):
    # check metrics availability
    for metric in metrics:
        if metric != "AP" and metric != "AUROC":
            raise NotImplementedError(metric)

    n_labels = labels.shape[1]
    clfs_scores = {}

    for clf_name, trained_clfs in clfs_dict.items():

        # initialize scores as a dict with metrics as keys
        # example: scores['AUROC'][0] return the auroc score for label nr 0
        scores = {}
        for metric in metrics:
            scores[metric] = torch.zeros(n_labels)

        for k in range(0, n_labels):
            clf_k = trained_clfs[k]
            y_pred_k = clf_k.predict(encodings.cpu())
            if "AP" in metrics:
                scores["AP"][k] = average_precision_score(labels[:, k].cpu(), y_pred_k)
            if "AUROC" in metrics:
                scores["AUROC"][k] = roc_auc_score(
                    labels[:, k].cpu(), clf_k.predict_proba(encodings.cpu())[:, 1]
                )
        clfs_scores[clf_name] = scores

    return clfs_scores


def generate_samples(decoders, rep):
    imgs_gen = []
    for dec in decoders:
        img_gen = dec(rep)
        imgs_gen.append(img_gen[0])
    return imgs_gen


def conditional_generation(mvvae, dists):
    imgs_gen = []
    for idx, dist in enumerate(dists):
        mu, lv = dist
        imgs_gen_dist = []
        for m in range(len(mvvae.decoders)):
            z_out = mvvae.reparametrize(mu, lv)
            cond_gen_m = mvvae.decoders[m](z_out)[0]
            imgs_gen_dist.append(cond_gen_m)
        imgs_gen.append(imgs_gen_dist)
    return imgs_gen


def load_modality_clfs(cfg):
    if cfg.dataset.name.startswith("mimic_cxr"):
        raise NotImplementedError
    else:
        print("dataset does not exist..exit")
        sys.exit()
    return model


def load_modality_clfs_mimic(cfg):
    raise NotImplementedError


def calc_coherence_acc(cfg, clf, imgs, labels):
    out_clf = clf(cfg, [imgs, labels])
    preds = out_clf[0]
    return preds


def from_preds_to_acc(preds, labels, modality_names):
    n_views = len(modality_names)
    accs = torch.zeros((n_views, n_views, 1))
    for m, m_key in enumerate(modality_names):
        for m_tilde, m_tilde_key in enumerate(modality_names):
            preds_m_mtilde = preds[:, m, m_tilde, :]
            acc_m_mtilde = accuracy_score(
                labels.cpu(),
                np.argmax(preds_m_mtilde.cpu().numpy(), axis=1).astype(int),
            )
            accs[m, m_tilde, 0] = acc_m_mtilde
    return accs


def from_preds_to_ap(preds, labels, modality_names):
    n_views = len(modality_names)
    n_labels = labels.shape[1]
    aps = torch.zeros((n_views, n_views, n_labels))
    for m, m_key in enumerate(modality_names):
        for m_tilde, m_tilde_key in enumerate(modality_names):
            preds_m_mtilde = preds[:, m, m_tilde, :]
            for k in range(0, n_labels):
                ap_m_mtilde_k = average_precision_score(
                    labels[:, k].cpu(), preds_m_mtilde[:, k].detach().cpu().numpy()
                )
                aps[m, m_tilde, k] = ap_m_mtilde_k
    return aps


def calc_coherence_ap(cfg, clf, mods, labels):
    out_clf = clf(cfg, [mods, labels])
    preds = out_clf[0]
    return preds
