import torch

# CTGAN
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




# WGAN-GP
def _suggest_wgan_gp_extended(trial, dataset, user_params):
    """Расширенное пространство для WGAN-GP (параметры, которые пользователь не задал)"""
    params = {}
    
    # Только если пользователь не задал эти параметры
    if 'generator_layers' not in user_params:
        gen_n_layers = trial.suggest_categorical('generator_layers', [1, 2, 3, 4])
        if 'gen_layer_size' not in user_params:
            gen_layer_size = trial.suggest_int('gen_layer_size', 50, 150)
            params['generator_dim'] = (gen_layer_size,) * gen_n_layers
    
    if 'discriminator_layers' not in user_params:
        dis_n_layers = trial.suggest_categorical('discriminator_layers', [1, 2, 3, 4])
        if 'dis_layer_size' not in user_params:
            dis_layer_size = trial.suggest_int('dis_layer_size', 50, 150)
            params['discriminator_dim'] = (dis_layer_size,) * dis_n_layers
    
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
    
    if 'embedding_dim' not in user_params:
        params['embedding_dim'] = trial.suggest_categorical('embedding_dim', [128])
    
    if 'gp_weight' not in user_params:
        params['gp_weight'] = trial.suggest_float('gp_weight', 1.0, 10.0)
    
    if 'critic_iterations' not in user_params:
        params['critic_iterations'] = trial.suggest_categorical('critic_iterations', [1, 2, 3])
    
    if 'epochs' not in user_params:
        params['epochs'] = trial.suggest_int('epochs', 100, 1000, step=100)
    
    return params



# GAN-MFS
def _suggest_gan_mfs_extended(trial, dataset, user_params):
    """Расширенное пространство для GAN-MFS (параметры, которые пользователь не задал)"""
    params = {}
    
    # Только если пользователь не задал эти параметры
    if 'generator_layers' not in user_params:
        gen_n_layers = trial.suggest_categorical('generator_layers', [1, 2, 3, 4])
        if 'gen_layer_size' not in user_params:
            gen_layer_size = trial.suggest_int('gen_layer_size', 50, 150)
            params['generator_dim'] = (gen_layer_size,) * gen_n_layers
    
    if 'discriminator_layers' not in user_params:
        dis_n_layers = trial.suggest_categorical('discriminator_layers', [1, 2, 3, 4])
        if 'dis_layer_size' not in user_params:
            dis_layer_size = trial.suggest_int('dis_layer_size', 50, 150)
            params['discriminator_dim'] = (dis_layer_size,) * dis_n_layers
    
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
    
    if 'embedding_dim' not in user_params:
        params['embedding_dim'] = trial.suggest_categorical('embedding_dim', [128])
    
    if 'mfs_lambda' not in user_params:
        params['mfs_lambda'] = trial.suggest_float('mfs_lambda', 0.1, 1.5)
    
    if 'subset_mfs' not in user_params:
        params['subset_mfs'] = ['mean', 'var']
    
    if 'sample_number' not in user_params:
        params['sample_number'] = trial.suggest_categorical('sample_number', [3])
    
    if 'sample_frac' not in user_params:
        params['sample_frac'] = trial.suggest_float('sample_frac', 0.3, 0.5)
    
    if 'gp_weight' not in user_params:
        params['gp_weight'] = trial.suggest_float('gp_weight', 1.0, 10.0)
    
    if 'critic_iterations' not in user_params:
        params['critic_iterations'] = trial.suggest_categorical('critic_iterations', [1, 2, 3])
    
    if 'epochs' not in user_params:
        params['epochs'] = trial.suggest_int('epochs', 100, 1000, step=100)
    
    return params



#CTAB-GAN+
def _suggest_ctab_gan_plus_extended(trial, dataset, user_params):
    """Расширенное пространство для CTAB-GAN-PLUS (параметры, которые пользователь не задал)"""
    params = {}
    
    # Только если пользователь не задал эти параметры
    if 'cls_n_layers' not in user_params:
        c_n_layers = trial.suggest_int('cls_n_layers', 1, 4)
        if 'cls_size_layer' not in user_params:
            c_size_layer = trial.suggest_categorical('cls_size_layer', [64, 128, 256])
            params['class_dim'] = [c_size_layer] * c_n_layers
    
    if 'batch_size' not in user_params:
        num_samples = len(dataset)
        left_bs = max(384, int(num_samples // 20))
        right_bs = max(1024, int(num_samples // 10))
        batch_size = trial.suggest_int('batch_size', left_bs, right_bs)
        params['batch_size'] = int(round(batch_size / 10) * 10)
    
    if 'lr' not in user_params:
        params['lr'] = trial.suggest_categorical('lr', [1e-4, 2e-4, 1e-3])
    
    if 'random_dim' not in user_params:
        params['random_dim'] = trial.suggest_categorical('random_dim', [64, 128, 256, 512])
    
    if 'critic_iterations' not in user_params:
        params['critic_iterations'] = trial.suggest_int('critic_iterations', 1, 3)
    
    if 'l2scale' not in user_params:
        params['l2scale'] = trial.suggest_categorical('l2scale', [1e-4, 1e-3])
    
    if 'epochs' not in user_params:
        params['epochs'] = trial.suggest_int('epochs', 100, 1000, step=100)
    
    return params

#TVAE
def _suggest_tvae_extended(trial, dataset, user_params):
    """Расширенное пространство для TVAE (параметры, которые пользователь не задал)"""
    params = {}

    if 'compress_dims' not in user_params:
        encoder_depth = trial.suggest_categorical('encoder_depth', [2, 3, 4])
        encoder_dim = trial.suggest_categorical('encoder_dim', [256, 512])
        params['compress_dims'] = tuple([encoder_dim] * encoder_depth)

    if 'decompress_dims' not in user_params:
        decoder_depth = trial.suggest_categorical('decoder_depth', [2, 4])
        decoder_dim = trial.suggest_categorical('decoder_dim', [256, 512])
        params['decompress_dims'] = tuple([decoder_dim] * decoder_depth)

    if 'batch_size' not in user_params:
        num_samples = len(dataset)
        min_batch = max(10, min(256, num_samples // 20))
        max_batch = max(min_batch, min(1024, num_samples // 10))
        batch_size = trial.suggest_int('batch_size', min_batch, max_batch)
        params['batch_size'] = max(10, int(round(batch_size / 10) * 10))

    if 'embedding_dim' not in user_params:
        params['embedding_dim'] = trial.suggest_categorical('embedding_dim', [16, 32, 64])

    if 'l2scale' not in user_params:
        params['l2scale'] = trial.suggest_float('l2scale', 1e-5, 6.3e-5, log=True)

    if 'loss_factor' not in user_params:
        params['loss_factor'] = trial.suggest_categorical('loss_factor', [2, 3])

    if 'epochs' not in user_params:
        params['epochs'] = trial.suggest_int('epochs', 100, 1000, step=100)

    return params


#DDPM
def _suggest_ddpm_extended(trial, dataset, user_params):
    """Расширенное пространство для DDPM (параметры, которые пользователь не задал)"""
    params = {}

    if 'batch_size' not in user_params:
        num_samples = len(dataset)
        bs_min = max(64, num_samples // 50)
        bs_max = min(4096, num_samples // 4)
        # Защита: bs_max должен быть >= bs_min
        bs_max = max(bs_min, bs_max)
        batch_size = trial.suggest_int('batch_size', bs_min, bs_max, step=64)
        params['batch_size'] = batch_size

    if 'lr' not in user_params:
        params['lr'] = trial.suggest_float('lr', 1e-5, 2e-3, log=True)

    if 'num_timesteps' not in user_params:
        params['num_timesteps'] = trial.suggest_categorical('num_timesteps', [100, 250, 500, 1000])

    if 'scheduler' not in user_params:
        params['scheduler'] = trial.suggest_categorical('scheduler', ['linear', 'cosine'])

    if 'n_layers_hidden' not in user_params:
        params['n_layers_hidden'] = trial.suggest_int('n_layers_hidden', 2, 6)

    if 'n_units_hidden' not in user_params:
        params['n_units_hidden'] = trial.suggest_categorical('n_units_hidden', [256, 512, 1024])

    if 'dropout' not in user_params:
        params['dropout'] = trial.suggest_float('dropout', 0.0, 0.2, step=0.05)

    return params