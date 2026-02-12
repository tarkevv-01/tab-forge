from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from ..base_model import BaseGenerativeModel
from .ctab_gan_plus import CTABGANSynthesizer
from ...dataset import Dataset



class CTABGANPlusSynthesizer(BaseGenerativeModel):
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

    def fit(self, dataset: Dataset, **kwargs) -> None:

        if self.is_fitted:
            return
        
        try:
            #Извлекаем стрктуру данных
            self._num_columns = dataset.get_numerical_features()
            self._cat_columns = dataset.get_categorical_features()
            self._target_column = dataset.summary()['target']
            self._order_features = dataset.get_registered_data().columns.tolist()
            self._task_type = dataset.summary()['task_type']
            
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
                train_data=dataset.get_registered_data(),
                categorical=self._cat_columns,
                general=self._num_columns,
                #из-за специфики, нужно увеличить первый симов
                type={self._task_type.capitalize(): self._target_column} 
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
            return pd.DataFrame(synthetic_data, columns=self._order_features)
            
        except Exception as e:
            raise RuntimeError(f"Ошибка при генерации данных CTAB-GAN-PLUS: {str(e)}")
    
    def structed_generate(self, n_samples: int) -> Dataset:
        """
        Генерация синтетических данных, но с сохранением информации с помощью класса Dataset.
        
        Args:
            n_samples: Количество генерируемых образцов
            
        Returns:
            Dataset с синтетическими данными
        """
        
        try:
            synthetic_data = self._synthesizer.sample(n_samples)
            gan_df = pd.DataFrame(synthetic_data, columns=self._order_features)
            
            dataset = Dataset(data=gan_df,
                              target=self._target_column,
                              task_type=self._task_type,
                              numerical_features=self._num_columns,
                              categorical_features=self._cat_columns)
            
            return dataset
        
        except Exception as e:
            raise RuntimeError(f"Ошибка при генерации данных CTAB-GAN-PLUS: {str(e)}")
    
    def get_losses(self):
        df = self._synthesizer.loss_df.rename(columns={
            'epoch': 'Epoch',
            'generator_loss': 'Generator Loss',
            'discriminator_loss': 'Discriminator Loss'
        })
        return df.to_dict(orient='list')

    


if __name__ == '__main__':
    ctab_gan_plus = CTABGANPlusSynthesizer()

    print(ctab_gan_plus.is_model_fitted())
    print(ctab_gan_plus.get_hyperparameters())