import torch
import torch.nn as nn

from networks.FeatureExtractorImgMimic import FeatureExtractorImg
from networks.FeatureCompressorMimic import LinearFeatureCompressor
from networks.DataGeneratorImgMimic import DataGeneratorImg


class MimicEncoderImg(nn.Module):
    def __init__(self, cfg):
        super(MimicEncoderImg, self).__init__()
        self.feature_extractor = FeatureExtractorImg(
            cfg,
            a=cfg.dataset.skip_connections_img_weight_a,
            b=cfg.dataset.skip_connections_img_weight_b,
        ).to(cfg.model.device)
        self.feature_compressor = LinearFeatureCompressor(
            cfg.dataset.num_layers_img * cfg.dataset.filter_dim_img,
            0,
            cfg.model.latent_dim,
        ).to(cfg.model.device)

    def forward(self, x_img):
        h_img = self.feature_extractor(x_img)
        mu, logvar = self.feature_compressor(h_img)
        return mu, logvar


class MimicDecoderImg(nn.Module):
    def __init__(self, cfg):
        super(MimicDecoderImg, self).__init__()
        self.feature_generator = nn.Linear(
            cfg.model.latent_dim,
            cfg.dataset.num_layers_img
            * cfg.dataset.filter_dim_img,  # questo va fissato a 1024 con filter_dim_img=128
            bias=True,
        ).to(cfg.model.device)
        self.img_generator = DataGeneratorImg(
            cfg,
            a=cfg.dataset.skip_connections_img_weight_a,
            b=cfg.dataset.skip_connections_img_weight_b,
        ).to(cfg.model.device)

    def forward(self, z):
        img_feat_hat = self.feature_generator(z)
        img_feat_hat = img_feat_hat.view(
            img_feat_hat.size(0), img_feat_hat.size(1), 1, 1
        )
        img_hat = self.img_generator(img_feat_hat)
        return img_hat, torch.tensor(0.75).to(z.device)
