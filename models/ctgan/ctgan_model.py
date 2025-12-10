from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from .base_model import BaseGenerativeModel
from sdv.single_table import CTGANSynthesizer as CTGAN
from sdv.metadata import SingleTableMetadata

class CTGANModel(BaseGenerativeModel):
    """
    Обертка для модели CTGAN из библиотеки SDV.
    
    CTGAN (Conditional Tabular GAN) - это генеративно-состязательная сеть,
    специально разработанная для генерации табличных данных.
    """
    
    def __init__(self, **kwargs):
        """
        Инициализация CTGAN модели.
        
        Args:
            **kwargs: Гиперпараметры модели
        """
        super().__init__(**kwargs)
        
        # Установка значений по умолчанию
        default_params = {
            'discriminator_lr': 2e-4,
            'generator_lr': 2e-4,
            'batch_size': 500,
            'embedding_dim': 128,
            'generator_dim': [256, 256],
            'discriminator_dim': [256, 256],
            'generator_decay': 1e-6,
            'discriminator_decay': 1e-6,
            'discriminator_steps': 1,
            'log_frequency': True,
            'transformation_num_type': 'CDF',
            'transformation_cat_type': 'OHE',
            'pac': 10,
            'epochs': 300,
            'verbose': False
        }

        # Обновляем значения по умолчанию переданными параметрами
        for key, value in default_params.items():
            if key not in self.hyperparameters:
                self.hyperparameters[key] = value
        
        self._synthesizer = None
        self._metadata = None


    def fit(self, data: pd.DataFrame, 
            target_column: Optional[str] = None,
            num_columns: Optional[List[str]] = None,
            cat_columns: Optional[List[str]] = None,
            **kwargs) -> None:
        """
        Обучение CTGAN модели на предоставленных данных.
        
        Args:
            data: Табличные данные для обучения
            target_column: Название целевой колонки
            num_columns: Список числовых колонок
            cat_columns: Список категориальных колонок
            **kwargs: Дополнительные параметры обучения
        """
        
        # Создаем метаданные для SDV
        self._metadata = SingleTableMetadata()
        self._metadata.detect_from_dataframe(data)
        
        # Обновляем метаданные с информацией о типах колонок
        if num_columns:
            for col in num_columns:
                if col in data.columns:
                    self._metadata.update_column(col, sdtype='numerical')
        
        if cat_columns:
            for col in cat_columns:
                if col in data.columns:
                    self._metadata.update_column(col, sdtype='categorical')
        
        # Создаем и настраиваем синтезатор
        ctgan_params = self._prepare_ctgan_params()
        
        try:
            self._synthesizer = CTGAN(
                metadata=self._metadata,
                discriminator_lr=ctgan_params['discriminator_lr'],
                generator_lr=ctgan_params['generator_lr'],
                batch_size=ctgan_params['batch_size'],
                embedding_dim=ctgan_params['embedding_dim'],
                discriminator_dim=ctgan_params['discriminator_dim'],
                generator_dim=ctgan_params['generator_dim'],
                generator_decay=ctgan_params['generator_decay'],
                discriminator_decay=ctgan_params['discriminator_decay'],
                discriminator_steps=ctgan_params['discriminator_steps'],
                log_frequency=ctgan_params['log_frequency'],
                pac=ctgan_params['pac'],
                epochs=ctgan_params['epochs']
            )
            
            # Обучаем модель
            self._synthesizer.fit(data)
            
            self.is_fitted = True

            
        except Exception as e:
            raise RuntimeError(f"Ошибка при обучении CTGAN: {str(e)}")


    def generate(self, n_samples: int) -> pd.DataFrame:
        """
        Генерация синтетических данных.
        
        Args:
            n_samples: Количество генерируемых образцов
            
        Returns:
            DataFrame с синтетическими данными
        """
        
        try:
            synthetic_data = self._synthesizer.sample(num_rows=n_samples)
            return synthetic_data
            
        except Exception as e:
            raise RuntimeError(f"Ошибка при генерации данных CTGAN: {str(e)}")
    
    
    def _prepare_ctgan_params(self) -> Dict[str, Any]:
        """
        Подготовка параметров для CTGAN синтезатора.
        
        Returns:
            Словарь параметров для CTGANSynthesizer
        """
        # Параметры, которые напрямую передаются в CTGAN
        ctgan_params = {}
        
        # Соответствие наших параметров параметрам SDV
        param_mapping = {
            'epochs': 'epochs',
            'pac': 'pac',
            'batch_size': 'batch_size',
            'discriminator_lr': 'discriminator_lr',
            'generator_lr': 'generator_lr',
            'discriminator_decay': 'discriminator_decay',
            'generator_decay': 'generator_decay',
            'embedding_dim': 'embedding_dim',
            'generator_dim': 'generator_dim',
            'discriminator_dim': 'discriminator_dim',
            'discriminator_steps': 'discriminator_steps',
            'log_frequency': 'log_frequency',
            'verbose': 'verbose'
        }
        
        for our_param, sdv_param in param_mapping.items():
            if our_param in self.hyperparameters:
                ctgan_params[sdv_param] = self.hyperparameters[our_param]
        
        return ctgan_params
    
    

if __name__ == '__main__':
    ctgan = CTGANModel()

    print(ctgan.is_model_fitted())
    print(ctgan.get_hyperparameters())
