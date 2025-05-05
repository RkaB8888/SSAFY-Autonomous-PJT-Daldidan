# predict.py

#버전1
import torch
from PIL import Image
import torchvision.transforms as transforms
from models.cnn_model import AppleSugarRegressor

# Load model
model = AppleSugarRegressor()
model.load_state_dict(torch.load("apple_model.pth"))
model.eval()

# Preprocess
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # add batch dim
    with torch.no_grad():
        output = model(image)
    return output.item()

if __name__ == "__main__":
    img_path = r""
    prediction = predict_image(img_path)
    print(f"Predicted Sugar Content: {prediction:.2f} Brix")


