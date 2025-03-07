from matplotlib.backend_bases import MouseEvent
from typing import Optional, Union
import torch
import torch.nn as nn
import numpy as np
import torch.functional as F
from omegaconf import DictConfig
import hydra

from .utils import (cosine_beta_schedule,
                    linear_beta_schedule,
                    vp_beta_schedule,
                    extract,
                    Losses
                    )


# code adapted from https://github.com/twitter/diffusion-rl/blob/master/agents/diffusion.py
class Diffusion(nn.Module):

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            inner_model: DictConfig,
            beta_schedule: str,
            n_timesteps: int,
            loss_type: str,
            predict_epsilon=True,
            device: str = 'cuda',
            diffusion_x: bool = False,
            diffusion_x_M: int = 32,
    ) -> None:
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bounds = None
        # chose your beta style
        if beta_schedule == 'linear':
            self.betas = linear_beta_schedule(n_timesteps).to(self.device)
        elif beta_schedule == 'cosine':
            self.betas = cosine_beta_schedule(n_timesteps).to(self.device)
        elif beta_schedule == 'vp':
            # beta max: 10 beta min: 0.1
            self.betas = vp_beta_schedule(n_timesteps).to(self.device)
        
        self.model = hydra.utils.instantiate(inner_model)
        self.n_timesteps = n_timesteps
        self.predict_epsilon = predict_epsilon

        # define alpha stuff
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0).to(self.device)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device), self.alphas_cumprod[:-1]])
        # required for forward diffusion q( x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod).to(self.device)
        self.log_one_minus_alphas_cumprod = torch.log(1. - self.alphas_cumprod).to(self.device)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod).to(self.device)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1).to(self.device)
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod).to(
            self.device)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20)).to(self.device)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod).to(
            self.device)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (
                    1. - self.alphas_cumprod).to(self.device)

        # either l1 or l2
        self.loss_fn = Losses[loss_type]()

        self.diffusion_x = diffusion_x
        self.diffusion_x_M = diffusion_x_M

    def diffusion_loss(self,
        perceptual_emb: torch.Tensor,
        actions: torch.Tensor,
        latent_goal: torch.Tensor = None,
        weights = 1.0
    ) -> torch.Tensor:
        batch_size = len(actions)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=actions.device).long()
        loss = self._p_losses(actions, perceptual_emb, latent_goal, t, weights)
        return loss
    
    def forward(self, 
        perceptual_emb: torch.Tensor, 
        latent_goal: Optional[torch.Tensor] = None, 
        action: Optional[torch.Tensor] = None,
        if_train: bool = False
    ) -> Union[torch.Tensor, float]:
        """
        Forward pass of the diffusion model.
        
        Args:
            perceptual_emb: Perceptual embeddings of the current state
            latent_goal: Optional latent goal representation
            action: Optional action for training
            if_train: Whether in training mode
            
        Returns:
            Either the diffusion loss during training or generated action sequence during inference
        """
        if if_train:
            return self.diffusion_loss(perceptual_emb, action, latent_goal)
        return self.denoise_actions(perceptual_emb=perceptual_emb, latent_goal=latent_goal, inference=True)
    
    def denoise_actions(
        self, 
        perceptual_emb: torch.Tensor, 
        latent_goal: torch.Tensor, 
        inference: bool = False,
        extra_args: dict = {}
    ):
        """
        Main Method to generate actions conditioned on the batch of state inputs

        :param state: the current state observation to conditon the diffusion model

        :return: x_{0} the predicted actions from the diffusion model
        """
        batch_size = perceptual_emb.shape[0]
        if len(perceptual_emb.shape) == 3:
            shape = (batch_size, self.model.action_seq_len, self.action_dim)
        else:
            shape = (batch_size, self.action_dim)
        action = self._p_sample_loop(perceptual_emb, latent_goal, shape)
        return action
    
    def _p_losses(self, x_start: torch.Tensor, state: torch.Tensor, goal: torch.Tensor, t: torch.Tensor, weights=1.0):
        """
        Computes the training loss of the diffusion model given a batch of data. At every
        training sample of the batch we generate noisy samples at different timesteps t
        and let the diffusion model predict the initial sample from the generated noisy one.
        The loss is computed by comparing the denoised sample from the diffusion model against
        the initial sample.

        :param x_start: the inital action samples
        :param state:   the current state observation batch
        :param t:       the batch of chosen timesteps
        :param weights: parameter to weight the individual losses

        :return loss: the total loss of the model given the input batch
        """
        noise = torch.randn_like(x_start)
        x_noisy = self._q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.model(state, x_noisy, goal, t)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise, weights)
        else:
            loss = self.loss_fn(x_recon, x_start, weights)
        
        return loss

    def _q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise=None):
        """
        Main Method to sample the forward diffusion start with random noise and get
        the required values for the desired noisy sample at q(x_{t} | x_{0})
        at timestep t. The method is used for training only.

        :param x_start: the initial dataset batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.

        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample
    
    def _p_sample_loop(self, state, goal, shape, verbose=False, return_diffusion=False):
        """
        Main loop for generating samples using the learned model and the inverse diffusion step

        :param state: the current state observation
        :param shape: the shape of the action samples [B, D]
        :param return diffusion: bool, if the complete diffusion chain should be returned or not

        :return: either the predicted x_0 sample or a list with [x_{t-1}, .., x_{0}]
        """
        batch_size = shape[0]
        x = torch.randn(shape, device=self.device)
        if return_diffusion:
            diffusion = [x]

        # start the inverse diffusion process to generate the action conditioned on the noise
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            x = self._p_sample(x, timesteps, state, goal, grad=False)
            # if we want to return the complete chain add thee together
            if return_diffusion:
                diffusion.append(x)

        if self.diffusion_x:
            # The sampling process runs as normal for T denoising timesteps. The denoising timestep is then fixed,
            # t = 0, and extra denoising iterations continue to run for M timesteps. The intuition behind this
            # is that samples continue to be moved toward higher-likelihood regions for longer.
            # https://openreview.net/pdf?id=Pv1GPQzRrC8

            timesteps = torch.full((batch_size,), 0, device=self.device, dtype=torch.long)
            for m in range(0, self.diffusion_x_M):
                x = self._p_sample(x, timesteps, state, goal, grad=False)
                if return_diffusion:
                    diffusion.append(x)

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x
    
    def _p_sample(self, x: torch.Tensor, t: torch.Tensor, s: torch.Tensor, g: torch.Tensor, grad: bool = True):
        """
        Generated a denoised sample x_{t-1} given the trained model and noisy sample x_{t}

        :param x:  noisy input action
        :param t:  batch of timesteps
        :param s:  the current state observations batch
        :param grad:  bool, if the gradient should be computed

        :return:    torch.Tensor x_{t-1}
        """
        b, *_ = x.shape
        model_mean, _, model_log_variance = self._p_mean_variance(x=x, t=t, s=s, g=g, grad=grad)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    
    def _p_mean_variance(self, x: torch.Tensor, t: torch.Tensor, s: torch.Tensor, g: torch.Tensor, grad: bool = True):
        '''
        Predicts the denoised x_{t-1} sample given the current diffusion model

        :param x:  noisy input action
        :param t:  batch of timesteps
        :param s:  the current state observations batch
        :param grad:  bool, if the gradient should be computed

        :return:
            - the model mean prediction
            - the model log variance prediction
            - the posterior variance
        '''
        
        with torch.set_grad_enabled(grad):
            x_recon = self._predict_start_from_noise(x, t=t, noise=self.model(s, x, g, t))
        
        model_mean, posterior_variance, posterior_log_variance = self._q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance
    
    def _predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        '''
        Converts model output into prediction of x₀.
        If predict_epsilon=True:
            - model outputs predicted noise (ϵ)
            - we convert that noise into x₀ using the formula
        If predict_epsilon=False:
            - model directly predicts x₀
            - we return that prediction as-is
        '''
        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise 

    def _q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        """
        Computes the posterior mean and variance of the diffusion step at timestep t

        q( x_{t-1} | x_t, x_0)
        """
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)

        return posterior_mean, posterior_variance, posterior_log_variance_clipped
   