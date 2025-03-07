import math
import einops
import torch
from torch import nn
from torchvision.transforms import Resize, Normalize, CenterCrop, InterpolationMode, Compose


class DINOImgEncoder(nn.Module):
    def __init__(self, model_name: str, freeze_backbone: bool, device: str, camera_names: list[str] = None):
        super().__init__()

        self.dino_version, backbone = model_name.split("_")
        self.img_encoder = torch.hub.load(
            f"facebookresearch/{self.dino_version}:main", model_name
        ).to(device)

        if freeze_backbone:
            for param in self.img_encoder.parameters():
                param.requires_grad = False

        self.camera_names = camera_names

        self.img_preprocessor = Compose(
            [
                Resize(256, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(224),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @torch.no_grad()
    def forward(self, obs_dict):
        x = [obs_dict[f"{camera}_image"] for camera in self.camera_names]
        x = torch.cat(x, dim=0)

        x = self.img_preprocessor(x)

        if self.dino_version == "dinov2":
            emb = self.img_encoder(x)
        else:
            emb = self.img_encoder.get_intermediate_layers(x)[0][
                :, 1:
            ]  # remove the [CLS] token

        emb = einops.rearrange(emb, "(b n) d -> b n d", n=len(self.camera_names))

        return emb


if __name__ == "__main__":
    img_encoder = DINOImgEncoder("dinov2_vits14", freeze_backbone=True, device="cuda")
    x = torch.randn(3, 3, 128, 128).to("cuda")
    emb = img_encoder(x)
    print(emb.shape)