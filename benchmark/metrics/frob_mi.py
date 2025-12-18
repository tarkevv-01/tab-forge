import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from xgboost import XGBRegressor


def lf_metric(synthetic_data, real_data) -> float:
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
    
    # Стандартизируем данные
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Выбираем модель
    model_lower = model.lower()
    if model_lower in ['xgboost', 'xgb']:
        estimator = XGBRegressor(random_state=42, verbosity=0)
    elif model_lower in ['linear', 'linearregression']:
        estimator = LinearRegression()
    else:
        raise ValueError(f"Неизвестная модель: {model}. Доступны: 'xgboost', 'linear'")
    
    # Обучаем модель
    estimator.fit(X_train_scaled, y_train)
    
    # Предсказываем на реальных данных
    y_pred = estimator.predict(X_test_scaled)
    
    # Вычисляем RMSE
    return np.sqrt(mean_squared_error(y_test, y_pred))


def jensen_shannon_metric(synthetic_data, real_data, include_target: bool = True) -> float:
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


def _entropy(labels):
    """Вычисляет энтропию"""
    if len(labels) == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))


def _joint_entropy(labels1, labels2):
    """Вычисляет совместную энтропию"""
    if len(labels1) == 0:
        return 0.0
    joint = np.column_stack((labels1, labels2))
    joint_view = joint.view([('', joint.dtype)] * joint.shape[1])
    _, counts = np.unique(joint_view, return_counts=True)
    probs = counts / len(labels1)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))


def _mutual_info_score_custom(x, y):
    """Вычисляет взаимную информацию"""
    hx = _entropy(x)
    hy = _entropy(y)
    hxy = _joint_entropy(x, y)
    return hx + hy - hxy


def frob_mi_metric(synthetic_data, real_data, n_bins: int = 25, 
                     method: str = 'quantile', include_target: bool = True) -> float:
    """
    Вычисляет Frobenius norm между матрицами взаимной информации (MI).
    
    Args:
        synthetic_data: Dataset с синтетическими данными
        real_data: Dataset с реальными данными
        n_bins: Количество бинов для дискретизации числовых признаков
        method: Метод дискретизации ('uniform' или 'quantile')
        include_target: Включать ли целевую переменную в расчет
        
    Returns:
        float: Значение MI matrix метрики (Frobenius norm)
    """
    # Получаем зарегистрированные признаки
    real_features = real_data.get_registered_features()
    synth_features = synthetic_data.get_registered_features()
    
    # Находим общие признаки
    common_features = [f for f in synth_features if f in real_features]
    
    if not common_features:
        return float('inf')
    
    # Получаем данные
    real_df = real_data.get_X(registered_only=True)[common_features].copy()
    synth_df = synthetic_data.get_X(registered_only=True)[common_features].copy()
    
    # Добавляем таргет если нужно
    if include_target:
        target_name = real_data.info.target_name
        real_df[target_name] = real_data.get_target()
        synth_df[target_name] = synthetic_data.get_target()
    
    # Удаляем строки с NaN
    real_df = real_df.dropna()
    synth_df = synth_df.dropna()
    
    if len(real_df) == 0 or len(synth_df) == 0:
        return float('inf')
    
    # Определяем числовые и категориальные колонки
    numerical_cols = real_data.get_numerical_features()
    categorical_cols = real_data.get_categorical_features()
    
    # Если таргет включен, проверяем его тип
    if include_target:
        target_name = real_data.info.target_name
        if pd.api.types.is_numeric_dtype(real_df[target_name]):
            numerical_cols = numerical_cols + [target_name]
        else:
            categorical_cols = categorical_cols + [target_name]
    
    # Фильтруем только те колонки, которые есть в real_df
    numerical_cols = [col for col in numerical_cols if col in real_df.columns]
    categorical_cols = [col for col in categorical_cols if col in real_df.columns]
    
    # Дискретизируем только числовые признаки
    real_processed = real_df.copy()
    synth_processed = synth_df.copy()
    
    if numerical_cols:
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=method)
        real_processed[numerical_cols] = discretizer.fit_transform(real_df[numerical_cols]).astype(int)
        synth_processed[numerical_cols] = discretizer.fit_transform(synth_df[numerical_cols]).astype(int)
    
    # Категориальные признаки кодируем в числа (LabelEncoder)
    if categorical_cols:
        from sklearn.preprocessing import LabelEncoder
        for col in categorical_cols:
            le = LabelEncoder()
            # Объединяем все уникальные значения из обоих датасетов
            combined = pd.concat([real_df[col], synth_df[col]]).astype(str)
            le.fit(combined)
            real_processed[col] = le.transform(real_df[col].astype(str))
            synth_processed[col] = le.transform(synth_df[col].astype(str))
    
    # Конвертируем в numpy
    real_binned = real_processed.values.astype(int)
    synth_binned = synth_processed.values.astype(int)
    
    n_features = real_processed.shape[1]
    
    # MI матрица для реальных данных
    mi_matrix_real = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(n_features):
            mi_matrix_real[i, j] = _mutual_info_score_custom(real_binned[:, i], real_binned[:, j])
    
    # MI матрица для синтетических данных
    mi_matrix_synth = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(n_features):
            mi_matrix_synth[i, j] = _mutual_info_score_custom(synth_binned[:, i], synth_binned[:, j])
    
    # Норма Фробениуса
    return np.linalg.norm(mi_matrix_real - mi_matrix_synth, ord='fro')