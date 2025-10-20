import torch
import torch.nn as nn
import torchvision


class ImageEncoder(nn.Module):
    """
    ResNet50-based image encoder that outputs a fixed-size embedding.
    """
    def __init__(self, output_dim: int, pretrained: bool = True, freeze: bool = False):
        super().__init__()
        try:
            # Newer torchvision API
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = torchvision.models.resnet50(weights=weights)
        except AttributeError:
            # Fallback for older torchvision
            backbone = torchvision.models.resnet50(pretrained=pretrained)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.mapper = nn.Linear(in_features, output_dim)

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)  # [B, 2048]
        emb = self.mapper(feats)  # [B, output_dim]
        return emb
