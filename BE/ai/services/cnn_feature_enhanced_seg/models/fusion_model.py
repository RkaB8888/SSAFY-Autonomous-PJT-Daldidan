import timm
import torch.nn as nn
import torch

# 기존 낮은 6차원적 feature적용용
# class FusionModel(nn.Module):
#     def __init__(self, manual_feature_dim, output_dim=1):
#         super().__init__()
#         self.cnn = timm.create_model('efficientnet_b0', pretrained=True)
#         cnn_output_dim = self.cnn.classifier.in_features
#         self.cnn.classifier = nn.Identity()

#         self.fc = nn.Sequential(
#             nn.Linear(cnn_output_dim + manual_feature_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, output_dim)
#         )

#     def forward(self, image, manual_features):
#         cnn_features = self.cnn(image)
#         combined = torch.cat([cnn_features, manual_features], dim=1)
#         output = self.fc(combined)
#         return output


# manual feature는 항상 64차원으로 투영됨
# CNN output = 1280차원
# 🔧 결과: 1280 + 64 = 1344차원

class FusionModel(nn.Module):
    def __init__(self, manual_feature_dim, output_dim=1):
        super().__init__()
        self.cnn = timm.create_model('efficientnet_b0', pretrained=True)
        cnn_output_dim = self.cnn.classifier.in_features
        self.cnn.classifier = nn.Identity()

        # 🔥 manual feature 강화 layer
        self.manual_proj = nn.Sequential(
            nn.Linear(manual_feature_dim, 64),
            nn.ReLU()
        )

        # Fusion + 회귀 head
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_dim + 64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, image, manual_features):
        cnn_features = self.cnn(image)  # (batch, 1280)
        manual_out = self.manual_proj(manual_features)  # (batch, 64)
        combined = torch.cat([cnn_features, manual_out], dim=1)
        return self.fc(combined).squeeze()