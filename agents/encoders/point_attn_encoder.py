import torch
from torch import nn
from agents.backbones.transformer.blocks import TransformerEncoder


class PointAttnEncoder(nn.Module):
    def __init__(
        self,
        use_pc_color: bool,
        out_channels: int,
        n_layers: int = 4,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.use_pc_color = use_pc_color
        self.in_channels = 6 if use_pc_color else 3

        block_channel = [64, 128, out_channels]

        self.mlp = nn.Sequential(
            nn.Linear(self.in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layer_norm else nn.Identity(),
            nn.ReLU()
        )

        # self.point_emb = nn.Linear(self.in_channels, out_channels)
        self.attn = TransformerEncoder(
            embed_dim=out_channels,
            n_heads=8,
            n_layers=n_layers,
            attn_pdrop=0.3,
            resid_pdrop=0.1,
            mlp_pdrop=0.1,
            bias=False
        )

        self.point_out_token = nn.Embedding(1, out_channels)

    def forward(self, x):
        assert (
            x.shape[-1] == self.in_channels
        ), f"Use pc color: {self.use_pc_color}, but input channels: {x.shape[-1]}"

        b, t, dim = x.size()

        x = self.mlp(x)
        point_token = self.point_out_token.weight.unsqueeze(0).repeat(b, 1, 1)

        x = torch.cat([x, point_token], dim=1)

        x = self.attn(x)[:, -1:, :]

        return x
