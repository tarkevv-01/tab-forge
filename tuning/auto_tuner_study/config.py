from .extended_fun import _suggest_ctgan_extended, _suggest_wgan_gp_extended, _suggest_gan_mfs_extended, _suggest_ctab_gan_plus_extended

MODEL_CONFIGS = {
    'CTGANSynthesizer': {
        'default_benchmark': {
            'rmse_xgboost': ('rmse', {'model': 'xgboost'})
        },
        'direction': 'minimize',
        'extended_space_fn': lambda trial, dataset, user_params: _suggest_ctgan_extended(trial, dataset, user_params)
    },
    'WGANGPSynthesizer': {
        'default_benchmark': {
            'js_metric': ('js_mean', {})
        },
        'direction': 'minimize',
        'extended_space_fn': lambda trial, dataset, user_params: _suggest_wgan_gp_extended(trial, dataset, user_params)
    },
    'GANMFSSynthesizer': {
        'default_benchmark': {
            'r2_xgboost': ('r2', {'model': 'xgboost'})
        },
        'direction': 'maximize',
        'extended_space_fn': lambda trial, dataset, user_params: _suggest_gan_mfs_extended(trial, dataset, user_params)
    },
    'CTABGANPlusSynthesizer': {
        'default_benchmark': {
            'rmse_xgboost': ('rmse', {'model': 'xgboost'})
        },
        'direction': 'minimize',
        'extended_space_fn': lambda trial, dataset, user_params: _suggest_ctab_gan_plus_extended(trial, dataset, user_params)
    },
}