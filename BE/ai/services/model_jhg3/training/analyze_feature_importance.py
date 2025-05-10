# services/model_jhg3/training/analyze_feature_importance.py
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from services.model_jhg3.config import MODEL_SAVE_PATH

BASE_DIR = Path(__file__).resolve().parent

# 1) 모델+메타 읽기
bundle = joblib.load(MODEL_SAVE_PATH)
model = bundle["model"]
selector = bundle["selector"]
feature_names = bundle["feature_names"]  # 우리가 저장해 둔 실제 이름들

# 2) selector 적용 후 실제 사용 피처 이름만 골라내기
mask = selector.get_support()
used_names = [n for n, m in zip(feature_names, mask) if m]

# 3) 중요도 DataFrame
df = pd.DataFrame(
    {"feature": used_names, "importance": model.feature_importances_}
).sort_values("importance", ascending=False)

# 4) 상위 30개 시각화
plt.figure(figsize=(8, 10))
topk = df.head(30)
plt.barh(topk["feature"][::-1], topk["importance"][::-1])
plt.title("Top 30 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()

out_png = BASE_DIR / "feature_importance_top30_named.png"
plt.savefig(out_png)
print("✅ 저장:", out_png)

# 5) 전체 CSV로도
csv_out = BASE_DIR / "feature_importance_all_named.csv"
df.to_csv(csv_out, index=False)
print("✅ CSV 저장:", csv_out)
