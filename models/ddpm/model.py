from typing import Dict, Any
import pandas as pd
from ..base_model import BaseGenerativeModel
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader
from ...dataset import Dataset


class DDPMSynthesizer(BaseGenerativeModel):
    """
    Обёртка для модели DDPM из библиотеки Synthcity.

    DDPM (Denoising Diffusion Probabilistic Model) — диффузионная модель
    для генерации табличных данных.

    Поддерживаемые гиперпараметры (все опциональны, дефолты — от Synthcity):
        batch_size      : int   — размер батча
        lr              : float — learning rate
        num_timesteps   : int   — количество шагов диффузии (100, 250, 500, 1000)
        model_type      : str   — тип внутренней модели, по умолчанию 'mlp'
        scheduler       : str   — расписание шума ('linear' или 'cosine')

        # Параметры архитектуры MLP (передаются через model_params):
        n_layers_hidden : int   — количество скрытых слоёв
        n_units_hidden  : int   — ширина скрытых слоёв
        dropout         : float — dropout

    Примечание: n_layers_hidden, n_units_hidden, dropout упакованы в
    model_params автоматически внутри _prepare_ddpm_params().
    """

    # Верхнеуровневые параметры Synthcity DDPM
    _DDPM_TOP_PARAMS = {
        'batch_size',
        'lr',
        'num_timesteps',
        'model_type',
        'scheduler',
    }

    # Параметры архитектуры, которые уходят в model_params
    _DDPM_MODEL_PARAMS = {
        'n_layers_hidden',
        'n_units_hidden',
        'dropout',
    }

    def __init__(self, **kwargs):
        """
        Инициализация DDPM модели.

        Args:
            **kwargs: Гиперпараметры модели. Непереданные параметры примут
                      значения по умолчанию из Synthcity.
        """
        super().__init__(**kwargs)
        self._synthesizer = None

    def fit(self, dataset: Dataset, **kwargs) -> None:
        """
        Обучение DDPM модели на предоставленных данных.

        Args:
            dataset : Dataset — обучающий датасет
            **kwargs: Дополнительные параметры (не используются)
        """
        self._num_columns = dataset.get_numerical_features()
        self._cat_columns = dataset.get_categorical_features()
        self._target_column = dataset.summary()['target']
        self._task_type = dataset.summary()['task_type']

        loader = GenericDataLoader(
            dataset.get_registered_data(),
            target_column=self._target_column,
        )

        top_params, model_params = self._prepare_ddpm_params()

        try:
            self._synthesizer = Plugins().get(
                "ddpm",
                **top_params,
                **({"model_params": model_params} if model_params else {}),
            )
            self._synthesizer.fit(loader)
            self.is_fitted = True

        except Exception as e:
            raise RuntimeError(f"Ошибка при обучении DDPM: {str(e)}")

    def generate(self, n_samples: int) -> pd.DataFrame:
        """
        Генерация синтетических данных.

        Args:
            n_samples : int — количество генерируемых образцов

        Returns:
            pd.DataFrame с синтетическими данными
        """
        if not self.is_fitted:
            raise RuntimeError("Модель не обучена. Сначала вызовите fit().")

        try:
            return self._synthesizer.generate(count=n_samples).dataframe()

        except Exception as e:
            raise RuntimeError(f"Ошибка при генерации данных DDPM: {str(e)}")

    def structed_generate(self, n_samples: int) -> Dataset:
        """
        Генерация синтетических данных с сохранением метаинформации
        через класс Dataset.

        Args:
            n_samples : int — количество генерируемых образцов

        Returns:
            Dataset с синтетическими данными
        """
        if not self.is_fitted:
            raise RuntimeError("Модель не обучена. Сначала вызовите fit().")

        try:
            synthetic_data = self.generate(n_samples)

            return Dataset(
                data=synthetic_data,
                target=self._target_column,
                task_type=self._task_type,
                numerical_features=self._num_columns,
                categorical_features=self._cat_columns,
            )

        except Exception as e:
            raise RuntimeError(f"Ошибка при структурированной генерации DDPM: {str(e)}")

    def get_losses(self) -> pd.DataFrame:
        """
        Возвращает историю лоссов обучения в виде DataFrame.

        Returns:
            pd.DataFrame с колонкой 'loss' и индексом по шагам
        """
        if not self.is_fitted:
            raise RuntimeError("Модель не обучена. Сначала вызовите fit().")

        try:
            return self._synthesizer.loss_history

        except Exception as e:
            raise RuntimeError(f"Не удалось получить лоссы DDPM: {str(e)}")

    def _prepare_ddpm_params(self):
        """
        Разделяет self.hyperparameters на два словаря:
          - top_params   : верхнеуровневые параметры Plugins().get("ddpm", ...)
          - model_params : параметры архитектуры MLP для ключа model_params

        Непереданные параметры примут дефолтные значения Synthcity.

        Returns:
            tuple[dict, dict]
        """
        top_params = {
            key: value
            for key, value in self.hyperparameters.items()
            if key in self._DDPM_TOP_PARAMS
        }

        # model_type по умолчанию 'mlp', если не задан явно
        if 'model_type' not in top_params:
            top_params['model_type'] = 'mlp'

        model_params = {
            key: value
            for key, value in self.hyperparameters.items()
            if key in self._DDPM_MODEL_PARAMS
        }

        return top_params, model_params

