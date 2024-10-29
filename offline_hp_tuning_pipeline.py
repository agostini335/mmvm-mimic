import numpy as np
import pandas as pd
import os
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from hydra.core.config_store import ConfigStore
import datetime
from config.MyMVWSLConfig import MyMVWSLConfig
from config.MyMVWSLConfig import LogConfig
from config.ModelConfig import JointModelConfig
from config.ModelConfig import MixedPriorModelConfig
from config.ModelConfig import UnimodalModelConfig
from config.DatasetConfig import MimicCXRDataConfig
from config.MyMVWSLConfig import EvalConfig
from hydra import compose, initialize
from omegaconf import OmegaConf

cs = ConfigStore.instance()

cs.store(group="log", name="log", node=LogConfig)
cs.store(group="model", name="joint", node=JointModelConfig)
cs.store(group="model", name="mixedprior", node=MixedPriorModelConfig)
cs.store(group="model", name="unimodal", node=UnimodalModelConfig)
cs.store(group="eval", name="eval", node=EvalConfig)
cs.store(group="dataset", name="Mimic_cxr", node=MimicCXRDataConfig)
cs.store(name="base_config", node=MyMVWSLConfig)


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

    rfc = RandomForestClassifier(random_state=cfg.seed, n_jobs=5)

    random_search = RandomizedSearchCV(
        estimator=rfc,
        param_distributions=rfc_search_space,
        n_iter=cfg.eval.hp_iter,
        cv=cfg.eval.hp_cv,
        scoring="roc_auc",
        verbose=cfg.eval.verbosity,
        random_state=cfg.seed,
    )
    random_search.fit(encodings, labels)
    best_params = random_search.best_params_
    cv_results = random_search.cv_results_
    print(best_params)
    print(cv_results)
    return best_params, cv_results


#############################################################################################
# check results are the same with different order
models = [
    {
        "model_name": "mix_frontal_s2",
        "encs": "encodings_scicore/encodings_scicore_seed_2/seed_2_archive/encodings_frontal_239_train_2024-05-18 10:28:04_mixedprior.npy",
        "labs": "encodings_scicore/encodings_scicore_seed_2/seed_2_archive/labels_239_train_2024-05-18 10:53:48.npy",
    },
    {
        "model_name": "independent_frontal_s2",
        "encs": "encodings_scicore/encodings_scicore_seed_2/seed_2_archive/encodings_frontal_239_train_2024-05-18 10:53:48_unimodal.npy",
        "labs": "encodings_scicore/encodings_scicore_seed_2/seed_2_archive/labels_239_train_2024-05-18 10:19:17.npy",
    },
    {
        "model_name": "poe_frontal_s2",
        "encs": "encodings_scicore/encodings_scicore_seed_2/seed_2_archive/encodings_frontal_239_train_2024-05-18 10:19:17_joint.npy",
        "labs": "encodings_scicore/encodings_scicore_seed_2/seed_2_archive/labels_239_train_2024-05-18 10:28:04.npy",
    },
    {
        "model_name": "moe_frontal_s2",
        "encs": "encodings_scicore/encodings_scicore_seed_2/MOE/encodings_frontal_239_train_2024-06-09 06:54:15_joint.npy",
        "labs": "encodings_scicore/encodings_scicore_seed_2/MOE/labels_239_train_2024-06-09 06:54:15.npy",
    },
    {
        "model_name": "avg_frontal_s2",
        "encs": "encodings_scicore/encodings_scicore_seed_2/AVG/encodings_frontal_239_train_2024-06-13 11:24:47_joint.npy",
        "labs": "encodings_scicore/encodings_scicore_seed_2/AVG/labels_239_train_2024-06-13 11:24:46.npy",
    },
]


def run_hptuning_pipeline(model_name, encs, labs):
    directory = "hpt_" + model_name
    if not os.path.exists(directory):
        os.makedirs(directory)

    encodings = np.load(encs)
    labels = np.load(labs)

    with initialize(version_base=None, config_path="conf"):
        cfg = compose(config_name="config")
        print(OmegaConf.to_yaml(cfg))

    # current date and time
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    # save best_params to a file as pandas dataframe
    bp, cv_results = hyperparameter_tuning_rf(encodings, labels, cfg)
    bp = pd.DataFrame(bp, index=[0])
    bp.to_csv(f"{directory}/best_params_{timestamp}.csv", index=False)

    # save cv results
    cv_results = pd.DataFrame(cv_results)
    cv_results.to_csv(
        f"{directory}/cv_results_{timestamp}_{model_name}.csv", index=False
    )

    # save cfg to a file
    with open(f"{directory}/cfg_{timestamp}_{model_name}.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # save enc/lab name used
    with open(
        f"{directory}/enc_lab_name_{timestamp}_{model_name}.yaml", "w"
    ) as text_file:
        text_file.write(encs)
        text_file.write("\n")
        text_file.write(labs)


# check labels
label_reference = models[0]["labs"]
for model_dict in models:
    assert np.array_equal(np.load(label_reference), np.load(model_dict["labs"]))

# check combi params is the same

for model_dict in models:
    print(model_dict["model_name"])
    run_hptuning_pipeline(
        model_dict["model_name"], model_dict["encs"], model_dict["labs"]
    )
