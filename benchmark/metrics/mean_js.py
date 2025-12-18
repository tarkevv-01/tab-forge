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


def mean_js_metric(synthetic_data, real_data, include_target: bool = True) -> float:
    """
    Вычисляет среднее Jensen-Shannon Divergence по всем зарегистрированным признакам.
    
    Args:
        synthetic_data: Dataset с синтетическими данными
        real_data: Dataset с реальными данными
        include_target: Включать ли целевую переменную в расчет
        
    Returns:
        float: Среднее значение Jensen-Shannon Divergence
    """
    # Получаем список всех зарегистрированных признаков
    real_features = real_data.get_registered_features()
    synth_features = synthetic_data.get_registered_features()
    
    # Находим общие признаки
    common_features = [f for f in synth_features if f in real_features]
    
    if not common_features:
        return 1.0  # Максимальное расхождение
    
    # Получаем данные
    real_df = real_data.get_X(registered_only=True)[common_features]
    synth_df = synthetic_data.get_X(registered_only=True)[common_features]
    
    # Добавляем таргет если нужно
    columns_to_compare = common_features.copy()
    if include_target:
        target_name = real_data.info.target_name
        real_df[target_name] = real_data.get_target()
        synth_df[target_name] = synthetic_data.get_target()
        columns_to_compare.append(target_name)
    
    js_distances = []
    
    for col in columns_to_compare:
        real_col = real_df[col].dropna()
        synth_col = synth_df[col].dropna()
        
        if len(real_col) == 0 or len(synth_col) == 0:
            continue
        
        # Для числовых данных создаем гистограммы
        if pd.api.types.is_numeric_dtype(real_col):
            # Определяем общий диапазон для гистограмм
            min_val = min(real_col.min(), synth_col.min())
            max_val = max(real_col.max(), synth_col.max())
            
            # Создаем bins
            bins = np.linspace(min_val, max_val, 50)
            
            # Создаем гистограммы
            real_hist, _ = np.histogram(real_col, bins=bins, density=True)
            synth_hist, _ = np.histogram(synth_col, bins=bins, density=True)
            
            # Нормализуем для создания вероятностных распределений
            real_hist = real_hist / real_hist.sum() if real_hist.sum() > 0 else real_hist
            synth_hist = synth_hist / synth_hist.sum() if synth_hist.sum() > 0 else synth_hist
            
        else:
            # Для категориальных данных
            real_counts = real_col.value_counts(normalize=True)
            synth_counts = synth_col.value_counts(normalize=True)
            
            # Создаем общий индекс
            all_categories = set(real_counts.index) | set(synth_counts.index)
            
            real_hist = np.array([real_counts.get(cat, 0) for cat in all_categories])
            synth_hist = np.array([synth_counts.get(cat, 0) for cat in all_categories])
        
        # Добавляем небольшую константу для избежания деления на ноль
        real_hist = real_hist + 1e-10
        synth_hist = synth_hist + 1e-10
        
        # Вычисляем Jensen-Shannon divergence
        js_distance = jensenshannon(real_hist, synth_hist)
        
        if not np.isnan(js_distance):
            js_distances.append(js_distance)
    
    return np.mean(js_distances) if js_distances else 1.0