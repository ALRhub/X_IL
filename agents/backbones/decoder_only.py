import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import einops
import math
from typing import Optional
from torch.nn import functional as F
import logging

from agents.utils.time_embedding import BESO_TimeEmbedding, RF_TimeEmbedding, DDPM_TimeEmbedding

logger = logging.getLogger(__name__)


# Non-diffusion based decoder-only model
class Dec_only(nn.Module):
    def __init__(
            self,
            encoder: DictConfig,
            state_dim: int,
            action_dim: int,
            goal_dim: int,
            device: str,
            goal_conditioned: bool,
            embed_dim: int,
            embed_pdrob: float,
            goal_seq_len: int,
            obs_seq_len: int,
            action_seq_len: int,
            linear_output: bool = False,
    ):
        super().__init__()

        self.encoder = hydra.utils.instantiate(encoder)

        self.device = device

        # mainly used for language condition or goal image condition
        self.goal_conditioned = goal_conditioned
        if not goal_conditioned:
            goal_seq_len = 0

        # the seq_size is the number of tokens in the input sequence
        self.seq_size = goal_seq_len + obs_seq_len + action_seq_len

        # linear embedding for the state
        self.tok_emb = nn.Linear(state_dim, embed_dim)

        # linear embedding for the goal
        self.goal_emb = nn.Linear(goal_dim, embed_dim)

        # position embedding
        self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_size, embed_dim))
        self.drop = nn.Dropout(embed_pdrob)
        self.drop.to(self.device)

        # get an action embedding
        self.query_embed = nn.Embedding(action_seq_len, embed_dim)

        self.action_dim = action_dim
        self.obs_dim = state_dim
        self.embed_dim = embed_dim

        self.goal_seq_len = goal_seq_len
        self.obs_seq_len = obs_seq_len
        self.action_seq_len = action_seq_len

        # action pred module
        if linear_output:
            self.action_pred = nn.Linear(embed_dim, action_dim)
        else:
            self.action_pred = nn.Sequential(
                nn.Linear(embed_dim, 100),
                nn.GELU(),
                nn.Linear(100, self.action_dim)
            )
        self.action_pred.to(self.device)

        self.apply(self._init_weights)

        # logger.info(
        #     "number of parameters: %e", sum(p.numel() for p in self.parameters())
        # )
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
            self,
            states,
            goals=None,
    ):

        if len(states.size()) != 3:
            states = states.unsqueeze(0)

        b, t, dim = states.size()

        if self.goal_conditioned:
            goal_embed = self.goal_emb(goals)
            goal_x = self.drop(goal_embed + self.pos_emb[:, :self.goal_seq_len, :])

        state_embed = self.tok_emb(states)
        state_x = self.drop(state_embed + self.pos_emb[:, self.goal_seq_len:(self.goal_seq_len + t), :])

        action_seq = self.query_embed.weight.unsqueeze(0).repeat(b, 1, 1)

        if self.goal_conditioned:
            input_seq = torch.cat([goal_x, state_x, action_seq], dim=1)
        else:
            input_seq = torch.cat([state_x, action_seq], dim=1)

        encoder_output = self.encoder(input_seq)

        pred_actions = self.action_pred(encoder_output[:, -self.action_seq_len:, :])

        return pred_actions


# Diffusion based decoder-only model, we need time embedding and noisy antions inputs here
class Noise_Dec_only(nn.Module):
    def __init__(
            self,
            encoder: DictConfig,
            state_dim: int,
            action_dim: int,
            goal_dim: int,
            device: str,
            goal_conditioned: bool,
            embed_dim: int,
            embed_pdrob: float,
            goal_seq_len: int,
            obs_seq_len: int,
            action_seq_len: int,
            linear_output: bool = False,
            use_ada_conditioning: bool = False,
            diffusion_type: str = "beso", # ddpm, beso or rf
            use_pos_emb: bool = True
    ):
        super().__init__()

        self.encoder = hydra.utils.instantiate(encoder)

        self.device = device

        # mainly used for language condition or goal image condition
        self.goal_conditioned = goal_conditioned
        if not goal_conditioned:
            goal_seq_len = 0

        # the seq_size is the number of tokens in the input sequence
        self.seq_size = goal_seq_len + obs_seq_len + action_seq_len

        # linear embedding for the state
        self.tok_emb = nn.Linear(state_dim, embed_dim)

        # linear embedding for the goal
        self.goal_emb = nn.Linear(goal_dim, embed_dim)

        # linear embedding for the action
        self.action_emb = nn.Linear(action_dim, embed_dim)

        self.diffusion_type = diffusion_type

        if diffusion_type == "beso":
            self.sigma_emb = BESO_TimeEmbedding(embed_dim)
        elif diffusion_type == "rf":
            self.sigma_emb = RF_TimeEmbedding(embed_dim)
        elif diffusion_type == "ddpm":
            self.sigma_emb = DDPM_TimeEmbedding(embed_dim)
        else:
            raise ValueError(f"Diffusion type {diffusion_type} is not supported")

        self.use_pos_emb = use_pos_emb
        if use_pos_emb:
            # position embedding
            self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_size, embed_dim))

        self.drop = nn.Dropout(embed_pdrob)
        self.drop.to(self.device)

        self.action_dim = action_dim
        self.obs_dim = state_dim
        self.embed_dim = embed_dim

        self.goal_seq_len = goal_seq_len
        self.obs_seq_len = obs_seq_len
        self.action_seq_len = action_seq_len

        self.use_ada_conditioning = use_ada_conditioning

        # action pred module
        if linear_output:
            self.action_pred = nn.Linear(embed_dim, action_dim)
        else:
            self.action_pred = nn.Sequential(
                nn.Linear(embed_dim, 100),
                nn.GELU(),
                nn.Linear(100, self.action_dim)
            )
        self.action_pred.to(self.device)

        self.apply(self._init_weights)

        # logger.info(
        #     "number of parameters: %e", sum(p.numel() for p in self.parameters())
        # )
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
            self,
            states,
            actions,
            goals,
            sigma
    ):

        if len(states.size()) != 3:
            states = states.unsqueeze(0)

        # t for the states does not mean the time, but the number of inputs tokens
        b, t, dim = states.size()
        _, t_a, _ = actions.size()

        if self.goal_conditioned:
            goal_embed = self.goal_emb(goals)
            if self.use_pos_emb:
                goal_embed += self.pos_emb[:, :self.goal_seq_len, :]
            goal_x = self.drop(goal_embed)

        state_embed = self.tok_emb(states)
        if self.use_pos_emb:
            state_embed += self.pos_emb[:, self.goal_seq_len:(self.goal_seq_len + t), :]
        state_x = self.drop(state_embed)

        action_embed = self.action_emb(actions)
        if self.use_pos_emb:
            action_embed += self.pos_emb[:, (self.goal_seq_len + t):(self.goal_seq_len + t + t_a), :]
        action_x = self.drop(action_embed)

        emb_t = self.sigma_emb(sigma)

        if self.goal_conditioned:
            input_seq = torch.cat([emb_t, goal_x, state_x, action_x], dim=1)
        else:
            input_seq = torch.cat([emb_t, state_x, action_x], dim=1)

        if self.use_ada_conditioning:
            encoder_output = self.encoder(input_seq, emb_t)
        else:
            encoder_output = self.encoder(input_seq)

        pred_actions = self.action_pred(encoder_output[:, -self.action_seq_len:, :])

        return pred_actions
