import hydra
from omegaconf import DictConfig
from torch import nn
import torch


class PointMLPResNetEncoder(nn.Module):
    def __init__(
        self,
        point_mlp_encoder: DictConfig,
        resnet_encoder: DictConfig,
    ):
        super().__init__()

        self.point_mlp_encoder = hydra.utils.instantiate(point_mlp_encoder)
        self.resnet_encoder = hydra.utils.instantiate(resnet_encoder)

        self.align_mlp = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
        )

    def forward(self, obs_dict, lang_cond=None):

        pc_emb = self.point_mlp_encoder(obs_dict["point_cloud"])
        resnet_emb = self.resnet_encoder(obs_dict, lang_cond)

        resnet_emb = self.align_mlp(resnet_emb)

        return torch.cat([resnet_emb, pc_emb], dim=1)
