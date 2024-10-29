import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from utils.MimicCXRSplitter import MimicCXRSplitter
from config.MyMVWSLConfig import MyMVWSLConfig
from config.MyMVWSLConfig import LogConfig
from config.ModelConfig import JointModelConfig
from config.ModelConfig import MixedPriorModelConfig
from config.ModelConfig import UnimodalModelConfig
from config.DatasetConfig import MimicCXRDataConfig
from config.MyMVWSLConfig import EvalConfig

cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.

cs.store(group="log", name="log", node=LogConfig)
cs.store(group="model", name="joint", node=JointModelConfig)
cs.store(group="model", name="mixedprior", node=MixedPriorModelConfig)
cs.store(group="model", name="unimodal", node=UnimodalModelConfig)
cs.store(group="eval", name="eval", node=EvalConfig)
cs.store(group="dataset", name="Mimic_cxr", node=MimicCXRDataConfig)
# cs.store(group="dataset", name="dataset", node=DataConfig)
cs.store(name="base_config", node=MyMVWSLConfig)

from hydra import compose, initialize
from omegaconf import OmegaConf

for seed in range(3):
    with initialize(version_base=None, config_path="conf"):
        cfg = compose(
            config_name="config",
            overrides=["dataset.studies_policy='all_combi_missing'"],
        )
        cfg.seed = seed
        print(OmegaConf.to_yaml(cfg))
        print(cfg.dataset.studies_policy)
        print("OVERRIDES APPLIED")
        mimic_cxr_splitter = MimicCXRSplitter(cfg)
