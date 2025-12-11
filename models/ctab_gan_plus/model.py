from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from ..base_model import BaseGenerativeModel
from .ctab_gan_plus import CTABGANSynthesizer



class CTABGANPlusModel(BaseGenerativeModel):
    """
    Обертка для модели CTAB-GAN-PLUS.
    """
    
    def __init__(self, **kwargs):
        """
        Инициализация CTAB-GAN-PLUS модели.
        
        Args:
            **kwargs: Гиперпараметры модели
        """
        super().__init__(**kwargs)
        
        # Установка значений по умолчанию
        default_params = {
            'lr': 0.0002,
            'random_dim': 128,
            'critic_iterations': 3,
            'batch_size': 64,
            'class_dim': [256, 256],
            'l2scale': 1e-4,
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
            # Подготовка параметров для CTABGANSynthesize
            ctab_gan_params = self.hyperparameters
            
            
            self._synthesizer = CTABGANSynthesizer(
                lr=ctab_gan_params['lr'],
                random_dim=ctab_gan_params['random_dim'],
                batch_size=ctab_gan_params['batch_size'],
                critic_steps=ctab_gan_params['critic_iterations'],
                class_dim=ctab_gan_params['class_dim'],
                l2scale=ctab_gan_params['l2scale'],
                epochs=ctab_gan_params['epochs']
            )
            
            # Обучаем модель
            self._synthesizer.fit(
                train_data=data,
                categorical=cat_columns,
                general=num_columns,
                type=kwargs['type_task']
            )
            
            self.is_fitted = True

            
        except Exception as e:
            raise RuntimeError(f"Ошибка при обучении CTAB-GAN-PLUS: {str(e)}")


    def generate(self, n_samples: int) -> pd.DataFrame:
        """
        Генерация синтетических данных.
        
        Args:
            n_samples: Количество генерируемых образцов
            
        Returns:
            DataFrame с синтетическими данными
        """
        
        try:
            synthetic_data = self._synthesizer.sample(n_samples)
            return synthetic_data
            
        except Exception as e:
            raise RuntimeError(f"Ошибка при генерации данных CTAB-GAN-PLUS: {str(e)}")
    
    
    def get_losses(self):
        df_loss_values = self._synthesizer.loss_df
        return df_loss_values.to_dict(orient='list')
    


if __name__ == '__main__':
    ctab_gan_plus = CTABGANPlusModel()

    print(ctab_gan_plus.is_model_fitted())
    print(ctab_gan_plus.get_hyperparameters())