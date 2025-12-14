from dataclasses import dataclass
from typing import List, Optional, Literal, Dict, Set
import pandas as pd
import numpy as np


@dataclass
class DatasetInfo:
    """Информация о датасете"""
    n_samples: int
    n_features: int
    n_numerical: int
    n_categorical: int
    n_registered: int
    n_unregistered: int
    task_type: Literal['regression', 'classification']
    target_name: str
    numerical_features: List[str]
    categorical_features: List[str]
    unregistered_features: List[str]
    registered_features: List[str]
    all_features: List[str]
    
    def __repr__(self):
        return f"""
Dataset Information:
==================
Samples: {self.n_samples}
Total features: {self.n_features}
Task type: {self.task_type}
Target: {self.target_name}

Feature breakdown:
  - Numerical: {self.n_numerical}
  - Categorical: {self.n_categorical}
  - Registered: {self.n_registered}
  - Unregistered: {self.n_unregistered}
"""


class Dataset:
    """
    Класс для управления табличными данными в AutoML модуле
    
    Parameters
    ----------
    data : pd.DataFrame
        Датафрейм с данными
    target : str
        Название целевой переменной
    task_type : {'regression', 'classification'}
        Тип задачи
    numerical_features : List[str], optional
        Список числовых признаков
    categorical_features : List[str], optional
        Список категориальных признаков
    
    Attributes
    ----------
    info : DatasetInfo
        Объект с полной информацией о датасете
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        target: str,
        task_type: Literal['regression', 'classification'],
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None
    ):
        self._data = data.copy()
        self._target = target
        self._task_type = task_type
        self._numerical_features = numerical_features or []
        self._categorical_features = categorical_features or []
        self._input_order_features = [col for col in data.columns if col != target]
        
        # Валидация
        self._validate()
        
        # Создание info объекта
        self._create_info()
    
    def _validate(self):
        """Валидация данных при инициализации"""
        errors = []
        
        # Проверка наличия данных
        if self._data.empty:
            errors.append("DataFrame пустой!")
        
        # Проверка наличия таргета
        if self._target not in self._data.columns:
            errors.append(f"Целевая переменная '{self._target}' не найдена в данных")
        
        # Проверка task_type
        if self._task_type not in ['regression', 'classification']:
            errors.append(f"task_type должен быть 'regression' или 'classification', получено: {self._task_type}")
        
        # Проверка числовых признаков
        invalid_numerical = [f for f in self._numerical_features if f not in self._data.columns]
        if invalid_numerical:
            errors.append(f"Числовые признаки не найдены в данных: {invalid_numerical}")
        
        # Проверка категориальных признаков
        invalid_categorical = [f for f in self._categorical_features if f not in self._data.columns]
        if invalid_categorical:
            errors.append(f"Категориальные признаки не найдены в данных: {invalid_categorical}")
        
        # Проверка пересечения числовых и категориальных
        intersection = set(self._numerical_features) & set(self._categorical_features)
        if intersection:
            errors.append(f"Признаки не могут быть одновременно числовыми и категориальными: {intersection}")
        
        # Проверка таргета в признаках
        if self._target in self._numerical_features:
            errors.append(f"Целевая переменная '{self._target}' не должна быть в числовых признаках")
        if self._target in self._categorical_features:
            errors.append(f"Целевая переменная '{self._target}' не должна быть в категориальных признаках")
        
        # Если есть ошибки, выбросить исключение
        if errors:
            raise ValueError("Ошибки валидации:\n- " + "\n- ".join(errors))
    
    def _create_info(self):
        """Создание объекта info с информацией о датасете"""
        all_features = [col for col in self._data.columns if col != self._target]
        registered = [f for f in self._input_order_features if (f in self._numerical_features) or (f in self._categorical_features)] 
        unregistered = [f for f in all_features if f not in registered]
        
        self.info = DatasetInfo(
            n_samples=len(self._data),
            n_features=len(all_features),
            n_numerical=len(self._numerical_features),
            n_categorical=len(self._categorical_features),
            n_registered=len(registered),
            n_unregistered=len(unregistered),
            task_type=self._task_type,
            target_name=self._target,
            numerical_features=self._numerical_features.copy(),
            categorical_features=self._categorical_features.copy(),
            unregistered_features=unregistered,
            registered_features=list(registered),
            all_features=all_features
        )
    
    def get_unregistered_features(self) -> List[str]:
        """Получить список незарегистрированных признаков"""
        return self.info.unregistered_features.copy()
    
    def get_registered_features(self) -> List[str]:
        """Получить список зарегистрированных признаков"""
        return self.info.registered_features.copy()
    
    def get_numerical_features(self) -> List[str]:
        """Получить список числовых признаков"""
        return self.info.numerical_features.copy()
    
    def get_categorical_features(self) -> List[str]:
        """Получить список категориальных признаков"""
        return self.info.categorical_features.copy()
    
    def get_X(self, registered_only: bool = False) -> pd.DataFrame:
        """
        Получить признаки
        
        Parameters
        ----------
        registered_only : bool, default=False
            Если True, вернуть только зарегистрированные признаки
        """
        if registered_only:
            return self._data[self.info.registered_features].copy()
        return self._data.drop(columns=[self._target]).copy()
    
    def get_target(self) -> pd.Series:
        """Получить целевую переменную"""
        return self._data[self._target].copy()
    
    def get_data(self) -> pd.DataFrame:
        """Получить полный датафрейм"""
        return self._data.copy()
    
    def get_registered_data(self) -> pd.DataFrame:
        """
        Получить датафрейм только с зарегистрированными признаками и таргетом
        
        Returns
        -------
        pd.DataFrame
            Датафрейм, содержащий только зарегистрированные признаки и целевую переменную
        """
        columns = self.info.registered_features + [self._target]
        return self._data[columns].copy()
    
    def register_features(
        self, 
        numerical: Optional[List[str]] = None,
        categorical: Optional[List[str]] = None
    ):
        """
        Зарегистрировать дополнительные признаки
        
        Parameters
        ----------
        numerical : List[str], optional
            Дополнительные числовые признаки
        categorical : List[str], optional
            Дополнительные категориальные признаки
        """
        if numerical:
            self._numerical_features.extend(numerical)
        if categorical:
            self._categorical_features.extend(categorical)
        
        # Переваливация и обновление info
        self._validate()
        self._create_info()
    
    def summary(self) -> Dict:
        """Получить краткую сводку о датасете"""
        return {
            'shape': self._data.shape,
            'task_type': self._task_type,
            'target': self._target,
            'n_numerical': self.info.n_numerical,
            'n_categorical': self.info.n_categorical,
            'n_unregistered': self.info.n_unregistered,
            'missing_values': self._data.isnull().sum().to_dict()
        }
    
    def __repr__(self):
        return f"Dataset(samples={self.info.n_samples}, features={self.info.n_features}, task='{self._task_type}')"
    
    def __len__(self):
        return len(self._data)
