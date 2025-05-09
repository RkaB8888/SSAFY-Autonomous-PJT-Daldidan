import timm
import torch.nn as nn
import torch

# class FusionModel(nn.Module):
#     def __init__(self, manual_feature_dim, output_dim=1):
#         super(FusionModel, self).__init__()
        
#         # EfficientNet-B0 backbone
#         self.cnn = timm.create_model('efficientnet_b0', pretrained=True)
#         cnn_output_dim = self.cnn.classifier.in_features  # EfficientNet-B0 output dim
#         self.cnn.classifier = nn.Identity()  # classifier 제거 (feature extractor로 사용)
        
#         # fusion layer (CNN + manual feature)
#         self.fc = nn.Sequential(
#             nn.Linear(cnn_output_dim + manual_feature_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, output_dim)
#         )
    
#     def forward(self, image, manual_features):
#         cnn_features = self.cnn(image)  # (batch, cnn_output_dim)
#         combined = torch.cat([cnn_features, manual_features], dim=1)  # (batch, cnn_output_dim + manual_feature_dim)
#         output = self.fc(combined)
#         return output
class FusionModel(nn.Module):
    def __init__(self, manual_feature_dim, output_dim=1):
        super().__init__()
        # CNN layer 삭제
        self.fc1 = nn.Linear(manual_feature_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, manual_features):
        print(f"[DEBUG] input manual_features: {manual_features}")  # ✅ 입력값 출력
        x = self.fc1(manual_features)
        print(f"[DEBUG] after fc1: {x}")  # ✅ fc1 결과 출력
        x = self.relu(x)
        print(f"[DEBUG] after relu: {x}")  # ✅ ReLU 결과 출력
        output = self.fc2(x)
        print(f"[DEBUG] output: {output}")  # ✅ 최종 출력 출력
        return output

