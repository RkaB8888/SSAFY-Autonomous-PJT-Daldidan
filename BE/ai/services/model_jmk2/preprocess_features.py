from sklearn.preprocessing import StandardScaler
import numpy as np

features = np.load("/home/j-k12e206/jmk/S12P31E206/BE/ai/services/model_jmk2/meme/features.npy")
labels = np.load("/home/j-k12e206/jmk/S12P31E206/BE/ai/services/model_jmk2/meme/labels.npy")

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

np.save("/home/j-k12e206/jmk/S12P31E206/BE/ai/services/model_jmk2/meme/features_scaled.npy", features_scaled)
print("✅ features_scaled 저장 완료")
