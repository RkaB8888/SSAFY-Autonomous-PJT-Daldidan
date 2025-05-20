import timm
import torch.nn as nn
import torch

class FusionModel(nn.Module):
    def __init__(self, manual_feature_dim, output_dim=1):
        super().__init__()
        self.cnn = timm.create_model('efficientnet_b0', pretrained=True)
        self.cnn.classifier = nn.Identity()

        for name, param in self.cnn.named_parameters():
            if "blocks.5" in name or "blocks.6" in name or "conv_head" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        cnn_output_dim = self.cnn.num_features

        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_output_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        self.manual_proj = nn.Sequential(
            nn.Linear(manual_feature_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 + 128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, image, manual_features):
        cnn_features = self.cnn(image)
        cnn_emb = self.cnn_proj(cnn_features)
        manual_emb = self.manual_proj(manual_features)
        combined = torch.cat([cnn_emb, manual_emb], dim=1)
        return self.fc(combined)
