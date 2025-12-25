import torch

def _suggest_ctgan_extended(trial, dataset, user_params):
    """Расширенное пространство для CTGAN (параметры, которые пользователь не задал)"""
    params = {}
    
    # Только если пользователь не задал эти параметры
    if 'generator_layers' not in user_params:
        gen_n_layers = trial.suggest_int('generator_layers', 1, 4)
        if 'gen_first_layer_size' not in user_params:
            gen_first_layer_size = trial.suggest_int('gen_first_layer_size', 50, 150)
            params['generator_dim'] = [gen_first_layer_size] * gen_n_layers
    
    if 'discriminator_layers' not in user_params:
        dis_n_layers = trial.suggest_int('discriminator_layers', 1, 4)
        if 'dis_first_layer_size' not in user_params:
            dis_first_layer_size = trial.suggest_int('dis_first_layer_size', 50, 150)
            params['discriminator_dim'] = [dis_first_layer_size] * dis_n_layers
    
    if 'batch_size' not in user_params:
        num_samples = len(dataset)
        left_bs = max(384, int(num_samples // 20))
        right_bs = max(1024, int(num_samples // 10))
        batch_size = trial.suggest_int('batch_size', left_bs, right_bs)
        params['batch_size'] = int(round(batch_size / 10) * 10)
    
    if 'discriminator_lr' not in user_params:
        params['discriminator_lr'] = trial.suggest_categorical('discriminator_lr', [1e-4, 2e-4, 1e-3])
    
    if 'generator_lr' not in user_params:
        params['generator_lr'] = trial.suggest_categorical('generator_lr', [1e-4, 2e-4, 1e-3])
    
    if 'discriminator_steps' not in user_params:
        params['discriminator_steps'] = trial.suggest_int('discriminator_steps', 1, 3)
    
    if 'generator_decay' not in user_params:
        params['generator_decay'] = trial.suggest_categorical('generator_decay', [1e-4, 1e-3])
    
    if 'discriminator_decay' not in user_params:
        params['discriminator_decay'] = trial.suggest_categorical('discriminator_decay', [1e-4, 1e-3])
    
    if 'epochs' not in user_params:
        params['epochs'] = trial.suggest_int('epochs', 100, 1000, step=100)
    
    if 'cuda' not in user_params:
        params['cuda'] = torch.cuda.is_available()
    
    return params