from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd


class BaseGenerativeModel(ABC):
    """
    Абстрактный базовый класс, определяющий единый интерфейс 
    для всех генеративных моделей табличных данных.
    """
    
    def __init__(self, **kwargs):
        """
        Инициализация модели с заданными гиперпараметрами.
        
        Args:
            **kwargs: Гиперпараметры модели
        """
        self.hyperparameters = kwargs.copy()
        self.is_fitted = False
        self.feature_columns = None
        self.target_column = None
        self.num_columns = None
        self.cat_columns = None
        self._model = None
        
    @abstractmethod
    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """
        Обучение модели на предоставленных данных.
        
        Args:
            data: Табличные данные для обучения
            **kwargs: Дополнительные параметры обучения
        """
        pass
    
    @abstractmethod
    def generate(self, n_samples: int) -> pd.DataFrame:
        """
        Генерация синтетических данных.
        
        Args:
            n_samples: Количество генерируемых образцов
            
        Returns:
            DataFrame с синтетическими данными
            
        Raises:
            RuntimeError: Если модель не обучена
        """
        pass
    
    def set_hyperparameters(self, hyperparameters: Dict[str, Any]) -> None:
        """
        Обновление гиперпараметров модели.
        
        Args:
            hyperparameters: Словарь новых гиперпараметров
        """
        self.hyperparameters.update(hyperparameters)
        # Если модель уже была обучена, сбрасываем флаг
        # так как гиперпараметры изменились
        if self.is_fitted:
            print("Предупреждение: Гиперпараметры изменены. Требуется переобучение модели.")
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """
        Получение текущих гиперпараметров модели.
        
        Returns:
            Словарь с текущими гиперпараметрами
        """
        return self.hyperparameters.copy()
    
    def is_model_fitted(self) -> bool:
        """
        Проверка, обучена ли модель.
        
        Returns:
            True если модель обучена, False иначе
        """
        return self.is_fitted
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Получение информации о модели.
        
        Returns:
            Словарь с информацией о модели
        """
        return {
            'model_class': self.__class__.__name__,
            'is_fitted': self.is_fitted,
            'hyperparameters': self.hyperparameters,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'num_columns': self.num_columns,
            'cat_columns': self.cat_columns
        }
    
    def _validate_fitted(self) -> None:
        """
        Проверка того, что модель обучена.
        
        Raises:
            RuntimeError: Если модель не обучена
        """
        if not self.is_fitted:
            raise RuntimeError(
                f"Модель {self.__class__.__name__} не обучена. "
                "Вызовите метод fit() перед генерацией данных."
            )
    
    
    
    def __repr__(self) -> str:
        """Строковое представление модели."""
        return (
            f"{self.__class__.__name__}("
            f"fitted={self.is_fitted}, "
            f"hyperparameters={len(self.hyperparameters)} params"
            ")"
        )
    
    def __str__(self) -> str:
        """Удобочитаемое строковое представление модели."""
        status = "обучена" if self.is_fitted else "не обучена"
        return f"{self.__class__.__name__} ({status})"