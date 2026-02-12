from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from ..base_model import BaseGenerativeModel
from .wgan_gp import WGANSynthesizer
from ...dataset import Dataset


class WGANGPSynthesizer(BaseGenerativeModel):
    """
    Обертка для модели WGAN-GP.
    """
    
    def __init__(self, **kwargs):
        """
        Инициализация WGAN-GP модели.
        
        Args:
            **kwargs: Гиперпараметры модели
        """
        super().__init__(**kwargs)
        
        # Установка значений по умолчанию
        default_params = {
                 'generator_dim': (256, 256),
                 'discriminator_dim': (256, 256),
                 'embedding_dim': 64,
                 'discriminator_lr': 0.0001,
                 'generator_lr': 0.0001,
                 'batch_size': 64,
                 'gp_weight': 1.0,
                 'critic_iterations': 3,
                 'epochs': 300
        }

        # Обновляем значения по умолчанию переданными параметрами
        for key, value in default_params.items():
            if key not in self.hyperparameters:
                self.hyperparameters[key] = value
        
        self._synthesizer = None
        self._metadata = None

    def fit(self, dataset: Dataset, **kwargs) -> None:
        try:
            self._num_columns = dataset.get_numerical_features()
            self._cat_columns = dataset.get_categorical_features()
            self._target_column = dataset.summary()['target']
            self._order_features = dataset.get_registered_data().columns.tolist()
            self._task_type = dataset.summary()['task_type']
            # Подготовка параметров для WGANSynthesizer
            wgan_gp_params = self.hyperparameters
            
            
            self._synthesizer = WGANSynthesizer(
                learning_rate_D=wgan_gp_params['discriminator_lr'],
                learning_rate_G=wgan_gp_params['generator_lr'],
                batch_size=wgan_gp_params['batch_size'],
                emb_dim=wgan_gp_params['embedding_dim'],
                discriminator_dim=wgan_gp_params['discriminator_dim'],
                generator_dim=wgan_gp_params['generator_dim'],
                critic_iterations=wgan_gp_params['critic_iterations'],
                epochs=wgan_gp_params['epochs']
            )
            
            # Обучаем модель
            self._synthesizer.fit(dataset.get_registered_data(), target_column=self._target_column)
            
            self.is_fitted = True

            
        except Exception as e:
            raise RuntimeError(f"Ошибка при обучении WGAN-GP: {str(e)}")


    def generate(self, n_samples: int) -> pd.DataFrame:
        """
        Генерация синтетических данных.
        
        Args:
            n_samples: Количество генерируемых образцов
            
        Returns:
            DataFrame с синтетическими данными
        """
        
        try:
            synthetic_data = self._synthesizer.generate(n_samples)
            return synthetic_data
            
        except Exception as e:
            raise RuntimeError(f"Ошибка при генерации данных WGAN-GP: {str(e)}")
    
    
    def structed_generate(self, n_samples: int) -> Dataset:
        """
        Генерация синтетических данных, но с сохранением информации с помощью класса Dataset.
        
        Args:
            n_samples: Количество генерируемых образцов
            
        Returns:
            Dataset с синтетическими данными
        """
        try:
            synthetic_data = self._synthesizer.generate(n_samples)
            dataset = Dataset(data=synthetic_data,
                              target=self._target_column,
                              task_type=self._task_type,
                              numerical_features=self._num_columns,
                              categorical_features=self._cat_columns)
            
            return dataset
            
        except Exception as e:
            raise RuntimeError(f"Ошибка при генерации данных WGAN-GP: {str(e)}")
        
        
    def get_losses(self):
        return self._synthesizer._loss_values
    


if __name__ == '__main__':
    wgan_gp = WGANGPSynthesizer()

    print(wgan_gp.is_model_fitted())
    print(wgan_gp.get_hyperparameters())