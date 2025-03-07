import torch
from torch import nn
import einops
from torchvision.transforms import Compose
from agents.models.beso.models.networks.clip import available_models, load_clip


class CLIPImgEncoder(nn.Module):
    def __init__(self, model_name: str, freeze_backbone: bool, device: str, camera_names: list[str] = None):
        super().__init__()

        self.clip_model, clip_transforms = load_clip(model_name, device=device)

        if model_name.startswith("ViT"):
            self.img_preprocessor = Compose(
                [
                    clip_transforms.transforms[0],  # Resize 224
                    clip_transforms.transforms[1],  # CenterCrop 224
                    clip_transforms.transforms[4],  # Normalize
                ]
            )
        elif model_name.startswith("RN"):
            self.img_preprocessor = clip_transforms.transforms[-1]  # Normalize
        else:
            raise ValueError(
                f"Model {model_name} not supported. Available models: ${available_models()}"
            )

        self.camera_names = camera_names

        if freeze_backbone:
            for param in self.clip_model.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def forward(self, obs_dict):
        x = [obs_dict[f"{camera}_image"] for camera in self.camera_names]

        x = torch.stack(x, dim=1)
        x = einops.rearrange(x, "b n c h w -> (b n) c h w")

        x = self.img_preprocessor(x)
        emb = self.clip_model.encode_image(x)

        emb = einops.rearrange(emb, "(b n) d -> b n d", n=len(self.camera_names))

        return emb.float()


if __name__ == "__main__":
    model = CLIPImgEncoder("ViT-B/32", True, "cuda")
    img = torch.randn(3, 3, 128, 128, device="cuda")

    print(model(img).shape)