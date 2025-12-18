from typing import List, Dict, Tuple, Callable, Union, Any
from dataclasses import dataclass
import importlib


@dataclass
class BenchmarkResult:
    """Результат выполнения бенчмарка"""
    metrics: Union[List[float], Dict[str, float]]
    
    def __repr__(self):
        if isinstance(self.metrics, dict):
            lines = ["Benchmark Results:"]
            lines.append("=" * 40)
            for name, value in self.metrics.items():
                lines.append(f"  {name:25s}: {value:.6f}")
            return "\n".join(lines)
        else:
            return f"Benchmark Results: {self.metrics}"


class Benchmark:
    """
    Класс для оценки качества синтетических данных различными метриками
    
    Parameters
    ----------
    metrics_spec : Union[List[Tuple], Dict[str, Tuple]]
        Спецификация метрик в виде:
        - Списка кортежей: [('r2', {...}), ('rmse', {...})]
        - Словаря: {'r2_xgb': ('r2', {...}), 'rmse_linear': ('rmse', {...})}
        Каждый кортеж содержит (название_метрики_или_функция, словарь_аргументов)
    
    Examples
    --------
    >>> bench = Benchmark([
    ...     ('r2', {'model': 'xgboost'}),
    ...     ('rmse', {'model': 'linearregression'})
    ... ])
    
    >>> bench = Benchmark({
    ...     'r2_xgb': ('r2', {'model': 'xgboost'}),
    ...     'r2_linear': ('r2', {'model': 'linearregression'}),
    ...     'my_metric': (my_func, {'k': 25})
    ... })
    """
    
    def __init__(self, metrics_spec: Union[List[Tuple], Dict[str, Tuple]]):
        self._is_named = isinstance(metrics_spec, dict)
        self._metrics_spec = metrics_spec
        self._parsed_metrics = self._parse_metrics(metrics_spec)
    
    def _parse_metrics(self, metrics_spec: Union[List[Tuple], Dict[str, Tuple]]) -> List[Tuple[str, Callable, Dict]]:
        """
        Парсит спецификацию метрик и возвращает список (имя, функция, аргументы)
        """
        parsed = []
        
        if isinstance(metrics_spec, dict):
            # Именованный словарь
            for name, (metric, kwargs) in metrics_spec.items():
                func = self._resolve_metric(metric)
                parsed.append((name, func, kwargs or {}))
        else:
            # Список кортежей
            for i, (metric, kwargs) in enumerate(metrics_spec):
                func = self._resolve_metric(metric)
                name = metric if isinstance(metric, str) else f"metric_{i}"
                parsed.append((name, func, kwargs or {}))
        
        return parsed
    
    def _resolve_metric(self, metric: Union[str, Callable]) -> Callable:
        """
        Преобразует название метрики в функцию или возвращает переданную функцию
        """
        if callable(metric):
            return metric
        
        # Пытаемся импортировать из metrics.py
        try:
            from .metrics import (
                r2_metric, frob_corr_metric, mean_js_metric,
                frob_mi_metric, rmse_metric
            )
            
            metric_map = {
                'r2': r2_metric,
                'frob_corr': frob_corr_metric,
                'js_mean': mean_js_metric,
                'frob_mi': frob_mi_metric,
                'rmse': rmse_metric
            }
            
            if metric in metric_map:
                return metric_map[metric]
            else:
                raise ValueError(f"Неизвестная метрика: {metric}")
        except ImportError:
            raise ImportError(f"Не удалось импортировать метрику '{metric}' из metrics.py")
    
    def evaluate(self, synthetic_data, real_data) -> BenchmarkResult:
        """
        Вычисляет все метрики для синтетических и реальных данных
        
        Parameters
        ----------
        synthetic_data : Dataset
            Синтетические данные
        real_data : Dataset
            Реальные данные
        
        Returns
        -------
        BenchmarkResult
            Объект с результатами всех метрик
        """
        results = {}
        results_list = []
        
        for name, func, kwargs in self._parsed_metrics:
            # Вызываем метрику с синтетическими и реальными данными
            result = func(synthetic_data, real_data, **kwargs)
            
            if self._is_named:
                results[name] = result
            else:
                results_list.append(result)
        
        metrics_output = results if self._is_named else results_list
        return BenchmarkResult(metrics=metrics_output)
    
    def fit(self, synthetic_data, real_data) -> BenchmarkResult:
        """
        Алиас для evaluate() для совместимости
        """
        return self.evaluate(synthetic_data, real_data)
    
    def __repr__(self):
        n_metrics = len(self._parsed_metrics)
        metric_names = [name for name, _, _ in self._parsed_metrics]
        return f"Benchmark(n_metrics={n_metrics}, metrics={metric_names})"