import logging
import os

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig
import hydra
from typing import Optional
from agents.base_agent import BaseAgent

log = logging.getLogger(__name__)


class DDPM_Agent(BaseAgent):
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

        self.if_robot_states = if_robot_states
        self.if_film_condition = if_film_condition

        self.eval_model_name = "eval_best_bc.pth"
        self.last_model_name = "last_bc.pth"

        self.optimizer_config = optimization
        self.use_lr_scheduler = False

    def set_scaler(self, scaler):
        self.scaler = scaler
        self.min_action = torch.from_numpy(self.scaler.y_bounds[0, :]).to(self.device)
        self.max_action = torch.from_numpy(self.scaler.y_bounds[1, :]).to(self.device)

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.optimizer_config, params=self.parameters()
        )
        return optimizer

    def forward(self, obs_dict, actions=None):
        # make prediction
        perceptual_emb, latent_goal = self.compute_input_embeddings(obs_dict)

        if self.training and actions is not None:
            loss = self.model(perceptual_emb, latent_goal, action=actions, if_train=True)
            return loss
        else:
            predicted_action = self.model(perceptual_emb, latent_goal, if_train=False)

            return predicted_action