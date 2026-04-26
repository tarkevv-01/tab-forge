from typing import Dict, Any
import pandas as pd
from ..base_model import BaseGenerativeModel
from sdv.single_table import TVAESynthesizer as TVAE
from sdv.metadata import SingleTableMetadata
from ...dataset import Dataset


class TVAESynthesizer(BaseGenerativeModel):
    """
    Обёртка для модели TVAE из библиотеки SDV.

    TVAE (Tabular Variational AutoEncoder) — вариационный автоэнкодер,
    специально разработанный для генерации табличных данных.

    Поддерживаемые гиперпараметры (все опциональны, дефолты — от SDV):
        epochs          : int         — количество эпох обучения
        batch_size      : int         — размер батча
        embedding_dim   : int         — размерность латентного пространства
        compress_dims   : tuple[int]  — размеры скрытых слоёв энкодера, напр. (256, 256)
        decompress_dims : tuple[int]  — размеры скрытых слоёв декодера, напр. (256, 256)
        l2scale         : float       — коэффициент L2-регуляризации
        loss_factor     : float       — вес reconstruction loss относительно KL-дивергенции
    """

    # Параметры, которые напрямую принимает SDV TVAESynthesizer
    _TVAE_PARAMS = {
        'epochs',
        'batch_size',
        'embedding_dim',
        'compress_dims',
        'decompress_dims',
        'l2scale',
        'loss_factor',
    }

    def __init__(self, **kwargs):
        """
        Инициализация TVAE модели.

        Args:
            **kwargs: Гиперпараметры модели. Непереданные параметры примут
                      значения по умолчанию из SDV.
        """
        super().__init__(**kwargs)
        self._synthesizer = None
        self._metadata = None

    def fit(self, dataset: Dataset, **kwargs) -> None:
        """
        Обучение TVAE модели на предоставленных данных.

        Args:
            dataset : Dataset — обучающий датасет
            **kwargs: Дополнительные параметры (не используются)
        """
        # Извлекаем структуру датасета
        self._num_columns = dataset.get_numerical_features()
        self._cat_columns = dataset.get_categorical_features()
        self._target_column = dataset.summary()['target']
        self._task_type = dataset.summary()['task_type']
        self._order_features = dataset.get_registered_data().columns.tolist()

        # Создаём и заполняем метаданные — идентично CTGAN
        self._metadata = SingleTableMetadata()
        self._metadata.detect_from_dataframe(dataset.get_registered_data())

        if self._num_columns:
            for col in self._num_columns:
                if col in self._order_features:
                    self._metadata.update_column(col, sdtype='numerical')

        if self._cat_columns:
            for col in self._cat_columns:
                if col in self._order_features:
                    self._metadata.update_column(col, sdtype='categorical')

        if self._task_type == 'classification' and self._target_column:
            self._metadata.update_column(self._target_column, sdtype='categorical')
        else:
            self._metadata.update_column(self._target_column, sdtype='numerical')

        # Передаём только явно заданные пользователем параметры.
        # Остальные примут дефолтные значения SDV.
        tvae_params = self._prepare_tvae_params()

        try:
            self._synthesizer = TVAE(
                metadata=self._metadata,
                **tvae_params
            )
            self._synthesizer.fit(dataset.get_registered_data())
            self.is_fitted = True

        except Exception as e:
            raise RuntimeError(f"Ошибка при обучении TVAE: {str(e)}")

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
            return self._synthesizer.sample(num_rows=n_samples)

        except Exception as e:
            raise RuntimeError(f"Ошибка при генерации данных TVAE: {str(e)}")

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
            synthetic_data = self._synthesizer.sample(num_rows=n_samples)

            return Dataset(
                data=synthetic_data,
                target=self._target_column,
                task_type=self._task_type,
                numerical_features=self._num_columns,
                categorical_features=self._cat_columns,
            )

        except Exception as e:
            raise RuntimeError(f"Ошибка при структурированной генерации TVAE: {str(e)}")

    def get_losses(self) -> pd.DataFrame:
        """
        Возвращает историю лоссов обучения в виде DataFrame.

        Returns:
            pd.DataFrame с колонками ['Epoch', 'Loss']
        """
        if not self.is_fitted:
            raise RuntimeError("Модель не обучена. Сначала вызовите fit().")

        try:
            return self._synthesizer.get_loss_values()

        except Exception as e:
            raise RuntimeError(f"Не удалось получить лоссы TVAE: {str(e)}")

    def _prepare_tvae_params(self) -> Dict[str, Any]:
        """
        Фильтрует self.hyperparameters, оставляя только параметры,
        которые принимает SDV TVAE. Непереданные параметры
        примут дефолтные значения самой библиотеки.

        Returns:
            Словарь параметров для TVAESynthesizer(...)
        """
        return {
            key: value
            for key, value in self.hyperparameters.items()
            if key in self._TVAE_PARAMS
        }


if __name__ == '__main__':
    tvae = TVAESynthesizer(epochs=400, batch_size=500, compress_dims=(256, 256))
    print(tvae.is_model_fitted())
    print(tvae.get_hyperparameters())