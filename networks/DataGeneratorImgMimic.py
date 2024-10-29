import sys

import torch
import torch.nn as nn

from networks.ResidualBlocksMimic import ResidualBlock2dTransposeConv


def make_res_block_data_generator(in_channels, out_channels, kernelsize, stride, padding, o_padding, dilation,
                                  a_val=1.0, b_val=1.0):
    upsample = None
    if (kernelsize != 1 and stride != 1) or (in_channels != out_channels):
        upsample = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels,
                                                    kernel_size=kernelsize,
                                                    stride=stride,
                                                    padding=padding,
                                                    dilation=dilation,
                                                    output_padding=o_padding),
                                 nn.BatchNorm2d(out_channels))
    layers = []
    print("MAKE RES BLOCK IN", in_channels)
    print("MAKE RES BLOCK OUT", out_channels)
    print("MAKE RES BLOCK upsample", upsample)
    print("\n")
    print("\n")
    layers.append(ResidualBlock2dTransposeConv(in_channels, out_channels,
                                               kernelsize=kernelsize,
                                               stride=stride,
                                               padding=padding,
                                               dilation=dilation,
                                               o_padding=o_padding,
                                               upsample=upsample,
                                               a=a_val, b=b_val))
    return nn.Sequential(*layers)


class DataGeneratorImg(nn.Module):
    def __init__(self, cfg, a=2.0, b=0.3):
        super(DataGeneratorImg, self).__init__()
        self.cfg = cfg




        modules = [
            make_res_block_data_generator(5 * self.cfg.dataset.filter_dim_img,
                                          4 * self.cfg.dataset.filter_dim_img,
                                          kernelsize=4,
                                          stride=1,
                                          padding=0,
                                          dilation=1,
                                          o_padding=0,
                                          a_val=a,
                                          b_val=b,
                                          ),

            make_res_block_data_generator(4 * self.cfg.dataset.filter_dim_img,
                                          3 * self.cfg.dataset.filter_dim_img,
                                          kernelsize=4, stride=2,
                                          padding=1, dilation=1,
                                          o_padding=0, a_val=a,
                                          b_val=b),
            make_res_block_data_generator(3 * self.cfg.dataset.filter_dim_img,
                                          2 * self.cfg.dataset.filter_dim_img,
                                          kernelsize=4, stride=2,
                                          padding=1, dilation=1,
                                          o_padding=0, a_val=a,
                                          b_val=b),
            make_res_block_data_generator(2 * self.cfg.dataset.filter_dim_img,
                                          1 * self.cfg.dataset.filter_dim_img,
                                          kernelsize=4, stride=2,
                                          padding=1, dilation=1,
                                          o_padding=0, a_val=a,
                                          b_val=b)]

        if self.cfg.dataset.img_size == 128:
            modules.append(make_res_block_data_generator(1 * self.cfg.dataset.filter_dim_img,
                                                         1 * self.cfg.dataset.filter_dim_img,
                                                         kernelsize=4, stride=2,
                                                         padding=1, dilation=1,
                                                         o_padding=0, a_val=a,
                                                         b_val=b))
        if self.cfg.dataset.img_size == 256:
            modules.append(make_res_block_data_generator(1 * self.cfg.dataset.filter_dim_img,
                                                         1 * self.cfg.dataset.filter_dim_img,
                                                         kernelsize=4, stride=2,
                                                         padding=1, dilation=1,
                                                         o_padding=0, a_val=a,
                                                         b_val=b))
            modules.append(make_res_block_data_generator(1 * self.cfg.dataset.filter_dim_img,
                                                         1 * self.cfg.dataset.filter_dim_img,
                                                         kernelsize=4, stride=2,
                                                         padding=1, dilation=1,
                                                         o_padding=0, a_val=a,
                                                         b_val=b))



        if not self.cfg.dataset.img_size == 224:
            modules.append(nn.ConvTranspose2d(self.cfg.dataset.img_size,
                                              self.cfg.dataset.image_channels,
                                              kernel_size=3,
                                              stride=2,
                                              padding=1,
                                              dilation=1,
                                              output_padding=1))
        else:
            # todo refactor hin=256, hout=224
            print("224 case")
            modules.append(make_res_block_data_generator(1 * self.cfg.dataset.filter_dim_img,
                                                         1 * self.cfg.dataset.filter_dim_img,
                                                         kernelsize=4, stride=2,
                                                         padding=1, dilation=1,
                                                         o_padding=0, a_val=a,
                                                         b_val=b))
            modules.append(make_res_block_data_generator(1 * self.cfg.dataset.filter_dim_img,
                                                         1 * self.cfg.dataset.filter_dim_img,
                                                         kernelsize=4, stride=2,
                                                         padding=1, dilation=1,
                                                         o_padding=0, a_val=a,
                                                         b_val=b))

            modules.append(nn.ConvTranspose2d(in_channels=64,
                                              out_channels=self.cfg.dataset.image_channels,
                                              kernel_size=3,
                                              stride=2,
                                              padding=17,
                                              dilation=1,
                                              output_padding=1,))

        self.generator = nn.Sequential(*modules)

        print("Data Generator:")
        print(self.generator)

    def forward(self, feats):
        #print("Feats shape", feats.shape)
        d = feats
        #d = self.generator(d)
        for l in self.generator:
            d = l(d)
            #print(d.shape)
        return d
