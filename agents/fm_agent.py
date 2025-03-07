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


class FM_Agent(BaseAgent):
    def __init__(
        self,
        model: DictConfig,
        obs_encoders: DictConfig,
        language_encoders: DictConfig,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        action_dim,
        obs_seq_len: int,
        act_seq_len: int,
        cam_names: list[str],
        use_lr_scheduler: bool = True,
        if_robot_states: bool = False,
        if_film_condition: bool = False,
        device: str = "cpu",
        state_dim: int = 7,
        latent_dim: int = 64,
        sampling_steps: int = 50,
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

        self.action_dim = action_dim

        self.if_robot_states = if_robot_states
        self.if_film_condition = if_film_condition

        self.eval_model_name = "eval_best_bc.pth"
        self.last_model_name = "last_bc.pth"

        self.optimizer_config = optimizer
        self.lr_scheduler = lr_scheduler

        self.use_lr_scheduler = use_lr_scheduler

        self.sampling_steps = sampling_steps

    def configure_optimizers(self):
        """
        Initialize optimizers and learning rate schedulers based on model configuration.
        """
        optim_groups = [
            {"params": self.model.model.parameters(), "weight_decay": self.optimizer_config.transformer_weight_decay},
        ]

        optim_groups.extend([
            {"params": self.img_encoder.parameters(), "weight_decay": self.optimizer_config.transformer_weight_decay},
        ])

        # optim_groups.extend([
        #     {"params": self.clip_proj.parameters(), "weight_decay": self.optimizer_config.obs_encoder_weight_decay},
        #     {"params": self.logit_scale, "weight_decay": self.optimizer_config.obs_encoder_weight_decay},
        # ])

        optimizer = torch.optim.AdamW(optim_groups, lr=self.optimizer_config.learning_rate,
                                      betas=self.optimizer_config.betas)

        # Optionally initialize the scheduler
        if self.use_lr_scheduler:
            lr_configs = OmegaConf.create(self.lr_scheduler)
            scheduler = TriStageLRScheduler(optimizer, lr_configs)

            return optimizer, scheduler
            # lr_scheduler = {
            #     "scheduler": scheduler,
            #     "interval": 'step',
            #     "frequency": 1,
            # }
            # return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        else:
            return optimizer

    def forward(self, obs_dict, actions=None):

        # with torch.no_grad():
        perceptual_emb, latent_goal = self.compute_input_embeddings(obs_dict)

        if self.training and actions is not None:
            loss, _ = self.model(actions, perceptual_emb, latent_goal)

            return loss

        noise_action = torch.randn((len(perceptual_emb), self.act_seq_len, self.action_dim), device=self.device)
        pred_act_seq = self.model.sample(noise_action, perceptual_emb, latent_goal, sample_steps=self.sampling_steps)

        return pred_act_seq

