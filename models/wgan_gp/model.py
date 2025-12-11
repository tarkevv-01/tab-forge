from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from ..base_model import BaseGenerativeModel
from .wgan_gp import WGANSynthesizer



class WGANGPModel(BaseGenerativeModel):
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

    def fit(self, data: pd.DataFrame, 
            target_column: Optional[str] = None,
            num_columns: Optional[List[str]] = None,
            cat_columns: Optional[List[str]] = None,
            **kwargs) -> None:
       
        
        try:
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
            self._synthesizer.fit(data, target_column=target_column)
            
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
    
    
    def get_losses(self):
        return self._synthesizer._loss_values
    


if __name__ == '__main__':
    wgan_gp = WGANGPModel()

    print(wgan_gp.is_model_fitted())
    print(wgan_gp.get_hyperparameters())