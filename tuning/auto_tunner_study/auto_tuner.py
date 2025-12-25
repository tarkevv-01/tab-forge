# FOUNDED BY OPTUNA DEVELOPMENT TEAM & ITMO

import optuna
import torch
from typing import Callable, Optional, Dict, Any, Literal
from tab_forge.benchmark import Benchmark
from ...dataset.functions import merge_datasets, split_folds
from .config import MODEL_CONFIGS
import optuna.logging

optuna.logging.set_verbosity(optuna.logging.WARNING)


class AutoTuningStudy:
    """
    Автоматический тюнинг синтезаторов табличных данных
    
    Parameters
    ----------
    model_class : class
        Класс модели (например, CTGANSynthesizer)
    get_params : Callable
        Функция для выбора параметров пользователем: get_params(trial) -> dict
    cv : int
        Количество фолдов для кросс-валидации
    sampler : optuna.samplers.BaseSampler
        Сэмплер для optuna
    search_space_mode : {'manual', 'extended'}
        Режим поиска: 'manual' - только параметры пользователя,
        'extended' - + автоматические параметры из лучшего пространства
    benchmark : Benchmark, optional
        Бенчмарк для оценки. Если None, используется default для модели
    direction : str, default='minimize'
        Направление оптимизации ('minimize' или 'maximize')
    **study_kwargs
        Дополнительные параметры для optuna.create_study
    
    Examples
    --------
    >>> def my_params(trial):
    ...     return {
    ...         'epochs': trial.suggest_int('epochs', 200, 500),
    ...         'generator_lr': trial.suggest_float('generator_lr', 1e-4, 1e-3)
    ...     }
    >>> 
    >>> study = AutoTuningStudy(
    ...     model_class=CTGANSynthesizer,
    ...     get_params=my_params,
    ...     cv=3,
    ...     sampler=TPESampler(),
    ...     search_space_mode='extended'
    ... )
    >>> 
    >>> study.optimize(dataset, n_trials=50)
    """
    
    def __init__(
        self,
        model_class,
        get_params: Callable = lambda trial: {},
        cv: int = 5,
        sampler = optuna.samplers.RandomSampler(),
        search_space_mode: Literal['manual', 'extended'] = 'manual',
        benchmark: Optional[Benchmark] = None,
        direction: str = 'minimize',
        **study_kwargs
    ):
        self.model_class = model_class
        self.model_name = model_class.__name__
        self.get_params = get_params
        self.cv = cv
        self.search_space_mode = search_space_mode
        self.benchmark = benchmark
        
        # Получаем конфиг модели
        if self.model_name not in MODEL_CONFIGS:
            raise ValueError(
                f"Модель {self.model_name} не поддерживается. "
                f"Доступные: {list(MODEL_CONFIGS.keys())}"
            )
        
        self.model_config = MODEL_CONFIGS[self.model_name]
        
        # Если бенчмарк не передан, используем default
        if self.benchmark is None:
            self.benchmark = Benchmark(self.model_config['default_benchmark'])
        
        # Создаем study
        self.study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            **study_kwargs
        )
    
    def _objective(self, trial, dataset, verbose):
        """Внутренний objective для optuna"""
        
        # 1. Получаем параметры от пользователя
        user_params = self.get_params(trial)
        
        # 2. Если extended mode, добавляем дополнительные параметры
        params = user_params.copy()
        if self.search_space_mode == 'extended':
            extended_fn = self.model_config['extended_space_fn']
            extended_params = extended_fn(trial, dataset, user_params)
            params.update(extended_params)
        
        if verbose:
            print(f"Trial {trial.number} with params: {params}")
        # 3. Кросс-валидация
        folds = split_folds(dataset=dataset, n_splits=self.cv, shuffle=True, random_state=42)
        
        # Словарь для накопления scores по каждой метрике
        metric_scores = {}
        
        for i, val_fold in enumerate(folds):
            # Собираем train из остальных фолдов
            train_folds = [f for j, f in enumerate(folds) if j != i]
            train_dataset = merge_datasets(train_folds)
            
            # Обучаем модель
            model = self.model_class(**params)
            model.fit(train_dataset)
            
            # Генерируем синтетику
            synth_dataset = model.structed_generate(len(val_fold))
            
            # Оцениваем
            benchmark_result = self.benchmark.fit(synth_dataset, val_fold)
            
            # Достаем метрики из результата
            if isinstance(benchmark_result.metrics, dict):
                # Если пользователь передал словарь метрик
                for metric_name, metric_value in benchmark_result.metrics.items():
                    if metric_name not in metric_scores:
                        metric_scores[metric_name] = []
                    metric_scores[metric_name].append(metric_value)
            else:
                # Если пользователь передал список метрик
                for idx, metric_value in enumerate(benchmark_result.metrics):
                    metric_name = f"metric_{idx}"
                    if metric_name not in metric_scores:
                        metric_scores[metric_name] = []
                    metric_scores[metric_name].append(metric_value)
        
        if verbose:
            print(f"Scores per fold: {metric_scores}")
        # Считаем среднее по фолдам для каждой метрики
        metric_means = []
        for metric_name, scores in metric_scores.items():
            metric_means.append(sum(scores) / len(scores))
        
        if verbose:
            print(f'Trial {trial.number} finish with value: {sum(metric_means) / len(metric_means)}')
            
        # Возвращаем среднее средних всех метрик
        return sum(metric_means) / len(metric_means)
    
    def optimize(self, dataset, n_trials: int, verbose: bool = False, **optimize_kwargs):
        """
        Запуск оптимизации
        
        Parameters
        ----------
        dataset : Dataset
            Датасет для обучения
        n_trials : int
            Количество trials
        verbose : bool
            Выводить ли подробности
        **optimize_kwargs
            Дополнительные параметры для study.optimize
        
        Returns
        -------
        optuna.Study
            Объект study после оптимизации
        """
        objective = lambda trial: self._objective(trial, dataset, verbose)
        self.study.optimize(objective, n_trials=n_trials, **optimize_kwargs)
        return self.study
    
    @property
    def best_params(self):
        """Лучшие найденные параметры"""
        return self.study.best_params
    
    @property
    def best_value(self):
        """Лучшее значение метрики"""
        return self.study.best_value
    
    @property
    def best_trial(self):
        """Лучший trial"""
        return self.study.best_trial
