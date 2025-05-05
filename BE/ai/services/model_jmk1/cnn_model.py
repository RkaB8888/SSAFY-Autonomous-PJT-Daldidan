# models/cnn_model.py
import torch.nn as nn
import torch

class AppleSugarRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # 3x224x224 → 16x112x112
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 32x56x56
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))                # 32x1x1
        )
        self.flatten = nn.Flatten()
        
        # CNN feature + color(3) + texture(4) → 32 + 3 + 4 = 39
        self.fc = nn.Sequential(
            nn.Linear(32 + 3 + 4, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 회귀
        )

    # 버전1
    def forward(self, x, color_feat, texture_feat):
        cnn_feat = self.cnn(x)
        cnn_feat = self.flatten(cnn_feat)  # (batch, 32)
        
        combined = torch.cat([cnn_feat, color_feat, texture_feat], dim=1)
        return self.fc(combined)
