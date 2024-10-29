import torch
import torch.nn as nn

from networks.FeatureExtractorImgMimic import FeatureExtractorImg


class ClfImg(nn.Module):
    def __init__(self, cfg):
        super(ClfImg, self).__init__()

        self.feature_extractor = FeatureExtractorImg(cfg,
                                                     a=cfg.dataset.skip_connections_img_weight_a,
                                                     b=cfg.dataset.skip_connections_img_weight_b)
        self.dropout = nn.Dropout(p=0.5, inplace=False)
        self.linear = nn.Linear(in_features=cfg.dataset.num_layers_img * cfg.dataset.filter_dim_img,
                                out_features=len(cfg.dataset.target_list), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_img):
        h = self.feature_extractor(x_img)
        h = self.dropout(h)
        h = h.view(h.size(0), -1)
        h = self.linear(h)
        out = self.sigmoid(h)
        return out

    def get_activations(self, x_img):
        h = self.feature_extractor(x_img)
        return h
