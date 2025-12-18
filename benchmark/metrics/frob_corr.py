import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from ...dataset import Dataset

import warnings
warnings.filterwarnings('ignore')


def frob_corr_metric(synthetic_data, real_data) -> float:
    """
    Вычисляет Frobenius norm между корреляционными матрицами.
    
    Args:
        synthetic_data: Dataset с синтетическими данными
        real_data: Dataset с реальными данными
        
    Returns:
        float: Значение LF метрики (Frobenius norm)
    """
    # Получаем только числовые признаки
    real_numerical = real_data.get_numerical_features()
    synth_numerical = synthetic_data.get_numerical_features()
    
    # Находим общие числовые признаки
    common_numerical = [f for f in synth_numerical if f in real_numerical]
    
    # Для регрессии добавляем таргет в список общих признаков
    include_target = False
    if real_data.info.task_type == 'regression':
        real_target = real_data.get_target()
        synth_target = synthetic_data.get_target()
        
        # Проверяем, что таргет числовой
        if pd.api.types.is_numeric_dtype(real_target) and pd.api.types.is_numeric_dtype(synth_target):
            include_target = True
            common_numerical.append(real_data.info.target_name)
    
    if len(common_numerical) < 2:
        return float('inf')
    
    # Получаем данные
    if include_target:
        real_df = real_data.get_data()[common_numerical]
        synth_df = synthetic_data.get_data()[common_numerical]
    else:
        real_df = real_data.get_X(registered_only=True)[common_numerical]
        synth_df = synthetic_data.get_X(registered_only=True)[common_numerical]
    
    # Вычисляем корреляционные матрицы
    real_corr = real_df.corr()
    synth_corr = synth_df.corr()
    
    # Заполняем NaN нулями
    real_corr = real_corr.fillna(0)
    synth_corr = synth_corr.fillna(0)
    
    # Вычисляем Frobenius norm
    return np.linalg.norm(real_corr.values - synth_corr.values, ord='fro')