import logging
import os

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
import hydra
from typing import Optional
from agents.base_agent import BaseAgent
from agents.models.beso.utils.lr_schedulers.tri_stage_scheduler import TriStageLRScheduler

log = logging.getLogger(__name__)


class BC_Agent(BaseAgent):
    def __init__(
        self,
        model: DictConfig,
        obs_encoders: DictConfig,
        language_encoders: DictConfig,
        optimization: DictConfig,
        obs_seq_len: int,
        act_seq_len: int,
        cam_names: list[str],
        if_robot_states: bool = False,
        if_film_condition: bool = False,
        device: str = "cpu",
        state_dim: int = 7,
        latent_dim: int = 64,
    ):
        super().__init__(
            model=model,
            obs_encoders=obs_encoders,
            language_encoders=language_encoders,
            device=device,
            state_dim=state_dim,
            latent_dim=latent_dim,
            obs_seq_len=obs_seq_len,
            act_seq_len=act_seq_len,
            cam_names=cam_names
        )

        # self.img_encoder = hydra.utils.instantiate(obs_encoders).to(device)
        # self.model = hydra.utils.instantiate(model).to(device)
        # self.state_emb = nn.Linear(state_dim, latent_dim)

        self.if_robot_states = if_robot_states
        self.if_film_condition = if_film_condition

        self.eval_model_name = "eval_best_bc.pth"
        self.last_model_name = "last_bc.pth"

        self.optimizer_config = optimization
        self.use_lr_scheduler = False

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.optimizer_config, params=self.parameters()
        )
        return optimizer

    # def configure_optimizers(self):
    #     """
    #     Initialize optimizers and learning rate schedulers based on model configuration.
    #     """
    #     # Configuration for models using transformer weight decay
    #     '''optim_groups = self.action_decoder.model.inner_model.get_optim_groups(
    #         weight_decay=self.optimizer_config.transformer_weight_decay
    #     )'''
    #     optim_groups = [
    #         {"params": self.model.model.parameters(), "weight_decay": self.optimizer_config.transformer_weight_decay},
    #     ]
    #
    #     optim_groups.extend([
    #         {"params": self.img_encoder.parameters(), "weight_decay": self.optimizer_config.transformer_weight_decay},
    #     ])
    #
    #     optimizer = torch.optim.AdamW(optim_groups, lr=self.optimizer_config.learning_rate,
    #                                   betas=self.optimizer_config.betas)
    #
    #     # Optionally initialize the scheduler
    #     if self.use_lr_scheduler:
    #         lr_configs = OmegaConf.create(self.lr_scheduler)
    #         scheduler = TriStageLRScheduler(optimizer, lr_configs)
    #
    #         return optimizer, scheduler
    #         # lr_scheduler = {
    #         #     "scheduler": scheduler,
    #         #     "interval": 'step',
    #         #     "frequency": 1,
    #         # }
    #         # return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
    #     else:
    #         return optimizer

    def forward(self, obs_dict, actions=None):

        # with torch.no_grad():
        perceptual_emb, latent_goal = self.compute_input_embeddings(obs_dict)

        # shape of perceptural_emb is torch.Size([64, 1, 256])
        # make prediction
        pred = self.model(
            perceptual_emb,
            latent_goal
        )

        if self.training and actions is not None:
            loss = F.mse_loss(pred, actions)

            return loss

        return pred

