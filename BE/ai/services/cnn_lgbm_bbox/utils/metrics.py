# services/cnn_lgbm_bbox/utils/metrics.py
"""
유틸: 회귀 모델 평가 지표 계산
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """
    회귀 예측 성능 평가 지표를 계산합니다.

    Args:
        y_true (np.ndarray): 실제 타깃 값 배열
        y_pred (np.ndarray): 예측된 타깃 값 배열

    Returns:
        tuple: (MAE, RMSE, R2)
    """
    # 평균 절대 오차
    mae = mean_absolute_error(y_true, y_pred)
    # RMSE = sqrt(MSE)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # 결정 계수 (R^2)
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2
