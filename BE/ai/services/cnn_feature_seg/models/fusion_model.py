import timm
import torch.nn as nn
import torch

class FusionModel(nn.Module):
    def __init__(self, manual_feature_dim, output_dim=1):
        super().__init__()
        self.cnn = timm.create_model('efficientnet_b0', pretrained=True)
        cnn_output_dim = self.cnn.classifier.in_features
        self.cnn.classifier = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(cnn_output_dim + manual_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, image, manual_features):
        cnn_features = self.cnn(image)
        combined = torch.cat([cnn_features, manual_features], dim=1)
        output = self.fc(combined)
        return output
