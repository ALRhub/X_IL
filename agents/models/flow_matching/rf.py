import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig


class RF(nn.Module):
    def __init__(
            self,
            backbones: DictConfig,
            ln=False,
            device: str = 'cpu',
    ):
        super(RF, self).__init__()

        self.model = hydra.utils.instantiate(backbones).to(device)
        self.ln = ln

    def forward(self, x, state, latent_goal):
        b = x.size(0)
        if self.ln:
            nt = torch.randn((b,)).to(x.device)
            t = torch.sigmoid(nt)
        else:
            t = torch.rand((b,)).to(x.device)
        texp = t.view([b, *([1] * len(x.shape[1:]))])
        z1 = torch.randn_like(x)
        zt = (1 - texp) * x + texp * z1

        # for fitting the architecture inputs
        # vtheta = self.model(zt, t, cond)
        vtheta = self.model(states=state, actions=zt, goals=latent_goal, sigma=t)

        batchwise_mse = ((z1 - x - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))
        tlist = batchwise_mse.detach().cpu().reshape(-1).tolist()
        ttloss = [(tv, tloss) for tv, tloss in zip(t, tlist)]
        return batchwise_mse.mean(), ttloss

    @torch.no_grad()
    def sample(self, z, state, latent_goal, null_cond=None, sample_steps=50, cfg=2.0):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]
        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device)

            # vc = self.model(z, t, cond)
            vc = self.model(states=state, actions=z, goals=latent_goal, sigma=t)

            # if null_cond is not None:
            #     vu = self.model(z, t, null_cond)
            #     vc = vu + cfg * (vc - vu)

            z = z - dt * vc
            images.append(z)
        return images[-1]