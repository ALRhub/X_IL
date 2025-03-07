import torch
import torch.nn as nn

from omegaconf import DictConfig
import hydra


class BC_Policy(nn.Module):

    def __init__(
            self,
            backbones: DictConfig,
            device: str = 'cpu',
    ):

        super(BC_Policy, self).__init__()

        self.model = hydra.utils.instantiate(backbones).to(device)

    def forward(self, perceptual_emb, latent_goal):

        # shape of perceptural_emb is torch.Size([64, 1, 256])
        # make prediction
        pred = self.model(
            perceptual_emb,
            latent_goal
        )

        return pred

    def get_params(self):
        return self.parameters()
