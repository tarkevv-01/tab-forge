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


def rmse_metric(synthetic_data, real_data, model: str = 'xgboost') -> float:
    """
    Вычисляет RMSE между моделями, обученными на разных датасетах.
    
    Args:
        synthetic_data: Dataset с синтетическими данными
        real_data: Dataset с реальными данными
        model: Тип модели ('xgboost', 'linear', 'linearregression')
        
    Returns:
        float: Значение RMSE метрики
    """
    # Проверяем типы задач
    if real_data.info.task_type != 'regression':
        raise ValueError("RMSE метрика поддерживается только для regression задач")
    
    if synthetic_data.info.task_type != real_data.info.task_type:
        raise ValueError("Типы задач в датасетах не совпадают")
    
    # Получаем зарегистрированные признаки
    X_train = synthetic_data.get_X(registered_only=True)
    y_train = synthetic_data.get_target()
    
    X_test = real_data.get_X(registered_only=True)
    y_test = real_data.get_target()
    
    # Проверяем, что признаки совпадают
    if not set(X_train.columns).issubset(set(X_test.columns)):
        raise ValueError("Признаки в синтетических данных не совпадают с реальными")
    
    # Используем только общие признаки в правильном порядке
    common_features = [f for f in X_train.columns if f in X_test.columns]
    X_train = X_train[common_features]
    X_test = X_test[common_features]
    
    # Приводим категориальные признаки к pandas category
    cat_features = synthetic_data.get_categorical_features()

    # Оставляем только те, что реально используются
    cat_features = [f for f in cat_features if f in common_features]

    for col in cat_features:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')
        
    if len(common_features) == 0:
        raise ValueError("Нет общих признаков для обучения модели")
    
    # Удаляем строки с пропусками
    train_mask = ~(X_train.isnull().any(axis=1) | y_train.isnull())
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    
    test_mask = ~(X_test.isnull().any(axis=1) | y_test.isnull())
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]
    
    if len(X_train) == 0 or len(X_test) == 0:
        return float('inf')
    
    # Выбираем модель
    model_lower = model.lower()
    if model_lower in ['xgboost', 'xgb']:
        estimator = XGBRegressor(random_state=42, verbosity=0, enable_categorical=True)
    elif model_lower in ['linear', 'linearregression']:
        # LinearRegression: One-Hot
        X_train = pd.get_dummies(X_train, columns=cat_features, drop_first=True)
        X_test = pd.get_dummies(X_test, columns=cat_features, drop_first=True)

        # выравниваем признаки
        X_train, X_test = X_train.align(
            X_test,
            join='left',
            axis=1,
            fill_value=0
        )
        
        # Модель
        estimator = LinearRegression()
    else:
        raise ValueError(f"Неизвестная модель: {model}. Доступны: 'xgboost', 'linear'")
    
    # Обучаем модель
    estimator.fit(X_train, y_train)
    
    # Предсказываем на реальных данных
    y_pred = estimator.predict(X_test)
    
    # Вычисляем RMSE
    return np.sqrt(mean_squared_error(y_test, y_pred))