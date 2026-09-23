# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MCC: https://github.com/facebookresearch/MCC
# Point-E: https://github.com/openai/point-e
# RIN: https://arxiv.org/pdf/2212.11972
# This code includes the implementation of our default two-stream model.
# Our default two-stream implementation is based on RIN and MCC,
# Other backbone in the two-stream family such as PerceiverIO will also work.
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from timm.models.vision_transformer import PatchEmbed, Block
from utils import get_2d_sincos_pos_embed
from modules import Denoiser_backbone
from perceiver_pytorch import Perceiver


class PointCloudConditioning(nn.Module):
    latent_dim = 512

    def __init__(self):
        super().__init__()
        self.model = Perceiver(
            input_channels=6,          
            input_axis=1,              
            num_freq_bands=6,
            max_freq=10.,
            depth=6,
            num_latents=256,
            latent_dim=512,
            cross_heads=1,
            latent_heads=8,
            cross_dim_head=64,
            latent_dim_head=64,
            num_classes=1,              # classifier output is unused; embeddings are returned
            attn_dropout=0.,
            ff_dropout=0.,
            weight_tie_layers=False,
            fourier_encode_data=False,
            self_per_cross_attn=2,
        )
    def forward(self, points):
        
        """
        points: [B, N, 6]
                XYZ + normals
        """

        latents = self.model(
            points,
            return_embeddings=True
        )

        return latents

class TwoStreamDenoiser(nn.Module):
    '''
    Point diffusion model using point-cloud conditioning with the Two Stream backbone
    '''
    def __init__(
        self,
        num_points: int = 1024,
        num_latents: int = 256,
        cond_drop_prob: float = 0.1,
        input_channels: int = 3,
        output_channels: int = 3,
        latent_dim: int = 768,
        num_blocks: int = 6,
        num_compute_layers: int = 4,
        **kwargs,
    ):
        super().__init__()
        # define encoder
        self.point_cloud_conditioning = PointCloudConditioning()
        # define backbone
        self.denoiser_backbone = Denoiser_backbone(input_channels=input_channels, output_channels=output_channels, 
                                      num_x=num_points, num_z=num_latents, z_dim=latent_dim, 
                                      num_blocks=num_blocks, num_compute_layers=num_compute_layers)
        self.cond_embed = nn.Sequential(
            nn.LayerNorm(
                normalized_shape=(self.point_cloud_conditioning.latent_dim,)
            ),
            nn.Linear(self.point_cloud_conditioning.latent_dim, self.denoiser_backbone.z_dim),
        )
        self.cond_drop_prob = cond_drop_prob
        self.num_points = num_points

    def cached_model_kwargs(self, model_kwargs):
        with torch.no_grad():
            cond_dict = {}
            embeddings = self.point_cloud_conditioning(model_kwargs["point_cloud"])
            cond_dict["embeddings"] = embeddings
            if "prev_latent" in model_kwargs:
                cond_dict["prev_latent"] = model_kwargs["prev_latent"]
            return cond_dict

    def forward(
        self,
        x,
        t,
        point_cloud=None,
        embeddings=None,
        prev_latent=None,
    ):
        """
        Forward pass through the model.

        Parameters:
        x: Tensor of shape [B, C, N_points], raw input point cloud.
        t: Tensor of shape [B], time step.
        point_cloud (Tensor, optional): A batch of point clouds with XYZ and normals,
                           shaped [B, N, 6].
        embeddings (Tensor, optional): A batch of conditional latent (avoid duplicate 
                                        computation of MCC encoder in diffusion inference)
        prev_latent (Tensor, optional): Self-conditioning latent.

        Returns:
        x_denoised: Tensor of shape [B, C, N_points], denoised point cloud/noise.
        """
        assert point_cloud is not None or embeddings is not None, "must specify point_cloud or embeddings"
        assert point_cloud is None or embeddings is None, "cannot specify both point_cloud and embeddings"
        assert x.shape[-1] == self.num_points

        # get the condition vectors with the point cloud encoder
        if point_cloud is not None:
            cond_vec = self.point_cloud_conditioning(point_cloud)
        else:
            cond_vec = embeddings
        # condition dropout
        if self.training:
            mask = torch.rand(size=[len(x)]) >= self.cond_drop_prob
            cond_vec = cond_vec * mask[:, None, None].to(cond_vec)
        cond_vec = self.cond_embed(cond_vec)

        # denoiser forward
        x_denoised, latent = self.denoiser_backbone(x.permute(0, 2, 1).contiguous(), t, cond_vec, prev_latent=prev_latent)
        x_denoised = x_denoised.permute(0, 2, 1).contiguous()
        return x_denoised, latent