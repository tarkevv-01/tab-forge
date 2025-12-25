from .extended_fun import _suggest_ctgan_extended

MODEL_CONFIGS = {
    'CTGANSynthesizer': {
        'default_benchmark': {
            'rmse_xgboost': ('rmse', {'model': 'xgboost'})
        },
        'extended_space_fn': lambda trial, dataset, user_params: _suggest_ctgan_extended(trial, dataset, user_params)
    },
    # Легко добавить другие модели
}