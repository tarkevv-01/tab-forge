from typing import List
from sklearn.model_selection import KFold, StratifiedKFold
import pandas as pd
import numpy as np
from .dataset import Dataset, Optional

def merge_datasets(datasets: List['Dataset'], reset_index: bool = True) -> 'Dataset':
        """
        Объединить несколько датасетов в один
        
        Parameters
        ----------
        datasets : List[Dataset]
            Список датасетов для объединения
        reset_index : bool, default=True
            Сбросить ли индекс после объединения
            
        Returns
        -------
        Dataset
            Новый объединенный датасет
            
        Raises
        ------
        ValueError
            Если датасеты несовместимы (разные таргеты, task_type или признаки)
            
        Examples
        --------
        >>> train, test = dataset.train_test_split(test_size=0.2)
        >>> merged = Dataset.merge([train, test])
        >>> print(f"Размер объединенного датасета: {len(merged)}")
        """
        if not datasets:
            raise ValueError("Список датасетов пуст")
        
        if len(datasets) == 1:
            return datasets[0]
        
        # Проверка совместимости датасетов
        first_ds = datasets[0]
        for i, ds in enumerate(datasets[1:], 1):
            if ds._target != first_ds._target:
                raise ValueError(
                    f"Датасет {i} имеет другую целевую переменную: "
                    f"'{ds._target}' != '{first_ds._target}'"
                )
            
            if ds._task_type != first_ds._task_type:
                raise ValueError(
                    f"Датасет {i} имеет другой task_type: "
                    f"'{ds._task_type}' != '{first_ds._task_type}'"
                )
            
            # Проверка совпадения колонок
            if set(ds._data.columns) != set(first_ds._data.columns):
                raise ValueError(
                    f"Датасет {i} имеет другой набор колонок"
                )
        
        # Объединение данных
        merged_data = pd.concat(
            [ds._data for ds in datasets],
            ignore_index=reset_index
        )
        
        # Объединение списков признаков (уникальные значения в порядке появления)
        seen_numerical = set()
        merged_numerical = []
        for ds in datasets:
            for feat in ds._numerical_features:
                if feat not in seen_numerical:
                    merged_numerical.append(feat)
                    seen_numerical.add(feat)
        
        seen_categorical = set()
        merged_categorical = []
        for ds in datasets:
            for feat in ds._categorical_features:
                if feat not in seen_categorical:
                    merged_categorical.append(feat)
                    seen_categorical.add(feat)
        
        # Создание нового датасета
        merged_dataset = Dataset(
            data=merged_data,
            target=first_ds._target,
            task_type=first_ds._task_type,
            numerical_features=merged_numerical,
            categorical_features=merged_categorical
        )
        
        return merged_dataset

def split_folds(
        dataset: Dataset,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = None,
        stratified: bool = False
    ) -> List['Dataset']:
    """
    Разбить датасет на заданное количество фолдов
    
    Parameters
    ----------
    dataset: Dataset
    n_splits : int, default=5
        Количество фолдов
    shuffle : bool, default=True
        Перемешивать ли данные перед разбиением
    random_state : int, optional
        Seed для воспроизводимости результатов
    stratified : bool, default=False
        Использовать ли стратификацию (только для классификации)
        
    Returns
    -------
    List[Dataset]
        Список из n_splits Dataset объектов
        
    Examples
    --------
    >>> folds = dataset.split_folds(n_splits=5, random_state=42)
    >>> print(f"Создано {len(folds)} фолдов")
    >>> print(f"Размер первого фолда: {len(folds[0])}")
    """
    if n_splits < 2:
        raise ValueError(f"n_splits должен быть >= 2, получено: {n_splits}")
    
    if n_splits > len(dataset._data):
        raise ValueError(f"n_splits ({n_splits}) не может быть больше количества образцов ({len(dataset._data)})")
    
    # Получаем данные (с перемешиванием или без)
    data = dataset._data.copy()
    if shuffle:
        data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Стратификация
    if stratified:
        if dataset._task_type != "classification":
            raise ValueError("stratified=True поддерживается только для classification")
        
        # Используем StratifiedKFold для получения индексов
        skf = StratifiedKFold(n_splits=n_splits, shuffle=False, random_state=None)
        fold_indices = []
        for _, test_idx in skf.split(data, data[dataset._target]):
            fold_indices.append(test_idx)
    else:
        # Простое разбиение на равные части
        fold_size = len(data) // n_splits
        remainder = len(data) % n_splits
        
        fold_indices = []
        start = 0
        for i in range(n_splits):
            # Распределяем остаток по первым фолдам
            current_fold_size = fold_size + (1 if i < remainder else 0)
            end = start + current_fold_size
            fold_indices.append(np.arange(start, end))
            start = end
    
    # Создание фолдов
    folds = []
    for indices in fold_indices:
        fold_data = data.iloc[indices].reset_index(drop=True)
        
        fold_dataset = Dataset(
            data=fold_data,
            target=dataset._target,
            task_type=dataset._task_type,
            numerical_features=dataset._numerical_features.copy(),
            categorical_features=dataset._categorical_features.copy()
        )
        folds.append(fold_dataset)
    
    return folds