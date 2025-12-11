import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn

# ===== MODELS =====
class Residual(nn.Module):
    """Residual layer for the CTGAN."""
    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = nn.Linear(i, o)
        self.bn = nn.BatchNorm1d(o)
        self.relu = nn.ReLU()
    
    def forward(self, input_):
        """Apply the Residual layer to the `input_`."""
        out = self.fc(input_)
        # out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)


class Generator(nn.Module):
    """Generator for the CTGAN."""
    def __init__(self, embedding_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        self.latent_dim = embedding_dim
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(nn.Linear(dim, data_dim))
        self.seq = nn.Sequential(*seq)
    
    def forward(self, input_):
        """Apply the Generator to the `input_`."""
        data = self.seq(input_)
        return data
    
    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.latent_dim))


class Discriminator(nn.Module):
    """Discriminator for the CTGAN."""
    def __init__(self, data_dim, discriminator_dim):
        super(Discriminator, self).__init__()
        seq = []
        self.data_dim = data_dim
        dim = data_dim
        for item in list(discriminator_dim):
            seq += [nn.Linear(dim, item), nn.LeakyReLU(0.2)]
            dim = item
        seq += [nn.Linear(dim, 1)]
        self.seq = nn.Sequential(*seq)
    
    def forward(self, input_):
        """Apply the Discriminator to the `input_`."""
        return self.seq(input_.view(-1, self.data_dim))

# ===== MFE TO TORCH =====
from typing import Optional, Union

class MFEToTorch:
    device = torch.device("cpu")

    @property
    def feature_methods(self):
        return {
            "cor": self.ft_cor_torch,
            "cov": self.ft_cov_torch,
            "eigenvalues": self.ft_eigenvals,
            "iq_range": self.ft_iq_range,
            "gravity": self.ft_gravity_torch,
            "kurtosis": self.ft_kurtosis,
            "skewness": self.ft_skewness,
            "mad": self.ft_mad,
            "max": self.ft_max,
            "min": self.ft_min,
            "mean": self.ft_mean,
            "median": self.ft_median,
            "range": self.ft_range,
            "sd": self.ft_std,
            "var": self.ft_var,
            "sparsity": self.ft_sparsity,
        }

    def change_device(self, device):
        self.device = device

    @staticmethod
    def cov(tensor, rowvar=True, bias=False):
        """Estimate a covariance matrix (np.cov)"""
        tensor = tensor if rowvar else tensor.transpose(-1, -2)
        tensor = tensor - tensor.mean(dim=-1, keepdim=True)
        factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
        return factor * tensor @ tensor.transpose(-1, -2).conj()

    def corrcoef(self, tensor, rowvar=True):
        """Get Pearson product-moment correlation coefficients (np.corrcoef)"""
        covariance = self.cov(tensor, rowvar=rowvar)
        variance = covariance.diagonal(0, -1, -2)
        if variance.is_complex():
            variance = variance.real
        stddev = variance.sqrt()
        covariance /= stddev.unsqueeze(-1)
        covariance /= stddev.unsqueeze(-2)
        if covariance.is_complex():
            covariance.real.clip_(-1, 1)
            covariance.imag.clip_(-1, 1)
        else:
            covariance.clip_(-1, 1)
        return covariance

    def ft_cor_torch(self, N: torch.Tensor) -> torch.Tensor:
        corr_mat = self.corrcoef(N, rowvar=False)
        res_num_rows, _ = corr_mat.shape
        tril_indices = torch.tril_indices(res_num_rows, res_num_rows, offset=-1)
        inf_triang_vals = corr_mat[tril_indices[0], tril_indices[1]]
        return torch.abs(inf_triang_vals)

    def ft_cov_torch(self, N: torch.Tensor) -> torch.Tensor:
        cov_mat = self.cov(N, rowvar=False)
        res_num_rows = cov_mat.shape[0]
        tril_indices = torch.tril_indices(res_num_rows, res_num_rows, offset=-1)
        inf_triang_vals = cov_mat[tril_indices[0], tril_indices[1]]
        return torch.abs(inf_triang_vals)

    def ft_eigenvals(self, x: torch.Tensor) -> torch.Tensor:
        centered = x - x.mean(dim=0, keepdim=True)
        covs = self.cov(centered, rowvar=False)
        return torch.linalg.eigvalsh(covs)

    @staticmethod
    def ft_iq_range(X: torch.Tensor) -> torch.Tensor:
        q75, q25 = torch.quantile(X, 0.75, dim=0), torch.quantile(X, 0.25, dim=0)
        iqr = q75 - q25
        return iqr

    @staticmethod
    def ft_kurtosis(x: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(x)
        diffs = x - mean
        var = torch.mean(torch.pow(diffs, 2.0))
        std = torch.pow(var, 0.5)
        zscores = diffs / std
        kurtoses = torch.mean(torch.pow(zscores, 4.0)) - 3.0
        return kurtoses

    @staticmethod
    def ft_skewness(x: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(x)
        diffs = x - mean
        var = torch.mean(torch.pow(diffs, 2.0))
        std = torch.pow(var, 0.5)
        zscores = diffs / std
        skews = torch.mean(torch.pow(zscores, 3.0))
        return skews

    @staticmethod
    def ft_mad(x: torch.Tensor, factor: float = 1.4826) -> torch.Tensor:
        m = x.median(dim=0, keepdim=True).values
        ama = torch.abs(x - m)
        mama = ama.median(dim=0).values
        return mama / (1 / factor)

    @staticmethod
    def ft_mean(N: torch.Tensor) -> torch.Tensor:
        return N.mean(dim=0)

    @staticmethod
    def ft_max(N: torch.Tensor) -> torch.Tensor:
        return N.max(dim=0, keepdim=False).values

    @staticmethod
    def ft_median(N: torch.Tensor) -> torch.Tensor:
        return N.median(dim=0).values

    @staticmethod
    def ft_min(N: torch.Tensor) -> torch.Tensor:
        return N.min(dim=0).values

    @staticmethod
    def ft_var(N):
        return torch.var(N, dim=0)

    @staticmethod
    def ft_std(N):
        return torch.std(N, dim=0)

    @staticmethod
    def ft_range(N: torch.Tensor) -> torch.Tensor:
        return N.max(dim=0).values - N.min(dim=0).values

    def ft_sparsity(self, N: torch.Tensor) -> torch.Tensor:
        ans = torch.tensor([attr.size(0) / torch.unique(attr).size(0) for attr in N.T])
        num_inst = N.size(0)
        norm_factor = 1.0 / (num_inst - 1.0)
        result = (ans - 1.0) * norm_factor
        return result.to(self.device)

    @staticmethod
    def ft_gravity_torch(N: torch.Tensor, y: torch.Tensor, norm_ord: Union[int, float] = 2,
                        classes: Optional[torch.Tensor] = None,
                        class_freqs: Optional[torch.Tensor] = None,
                        cls_inds: Optional[torch.Tensor] = None):
        if classes is None or class_freqs is None:
            classes, class_freqs = torch.unique(y, return_counts=True)
        ind_cls_maj = torch.argmax(class_freqs)
        class_maj = classes[ind_cls_maj]
        remaining_classes = torch.cat((classes[:ind_cls_maj], classes[ind_cls_maj + 1:]))
        remaining_freqs = torch.cat((class_freqs[:ind_cls_maj], class_freqs[ind_cls_maj + 1:]))
        ind_cls_min = torch.argmin(remaining_freqs)
        
        if cls_inds is not None:
            insts_cls_maj = N[cls_inds[ind_cls_maj]]
            if ind_cls_min >= ind_cls_maj:
                ind_cls_min += 1
            insts_cls_min = N[cls_inds[ind_cls_min]]
        else:
            class_min = remaining_classes[ind_cls_min]
            insts_cls_maj = N[y == class_maj]
            insts_cls_min = N[y == class_min]
        
        center_maj = insts_cls_maj.mean(dim=0)
        center_min = insts_cls_min.mean(dim=0)
        gravity = torch.norm(center_maj - center_min, p=norm_ord)
        return gravity

    def pad_only(self, tensor, target_len):
        if tensor.shape[0] < target_len:
            padding = torch.zeros(target_len - tensor.shape[0]).to(self.device)
            return torch.cat([tensor, padding])
        return tensor

    def get_mfs(self, X, y=None, subset=None):
        if subset is None:
            subset = ["mean", "var"]

        mfs = []
        for name in subset:
            if name not in self.feature_methods:
                raise ValueError(f"Unsupported meta-feature: '{name}'")

            if name == "gravity":
                if y is None:
                    raise ValueError("Meta-feature 'gravity' requires `y`.")
                res = self.feature_methods[name](X, y)
                res = torch.tile(res, (X.shape[-1],))
            else:
                res = self.feature_methods[name](X)

            mfs.append(res)
        
        shapes = [i.shape.numel() for i in mfs]
        mfs = [self.pad_only(mf, max(shapes)) for mf in mfs]
        return torch.stack(mfs)

# ===== DATASET =====
class ToyDataset(Dataset):
    def __init__(self, dataset):
        self.X = dataset

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        return self.X[idx]

# ===== UTILITY FUNCTIONS =====
def create_variates(X, sample_number=10, sample_frac=0.5):
    """Create random variates from dataset for MFS calculation"""
    variates = []
    n_samples = int(len(X) * sample_frac)
    
    for _ in range(sample_number):
        indices = torch.randperm(len(X))[:n_samples]
        variate = X[indices]
        variates.append(variate)
    
    return variates

# ===== SIMPLIFIED TRAINER =====
from torch.autograd import grad as torch_grad
import ot

class TrainerModified:
    def __init__(self, generator, discriminator, gen_optimizer, dis_optimizer, 
                 device, batch_size, mfs_lambda, subset_mfs, sample_number,
                 critic_iterations=1, gp_weight=10, aim_track=None, 
                 gen_model_name="wGAN_mfs", target_mfs=0):
        
        self.G = generator
        self.D = discriminator
        self.G_opt = gen_optimizer
        self.D_opt = dis_optimizer
        self.device = device
        self.batch_size = batch_size
        self.mfs_lambda = mfs_lambda
        self.subset_mfs = subset_mfs
        self.sample_number = sample_number
        self.critic_iterations = critic_iterations
        self.gp_weight = gp_weight
        self.aim_track = aim_track
        self.target_mfs = target_mfs if target_mfs else {"other_mfs": 0}
        
        # Initialize MFS manager
        self.mfs_manager = MFEToTorch()
        self.mfs_manager.change_device(device)
        
        # Move models to device
        self.G.to(device)
        self.D.to(device)
        
        # Loss tracking
        self.G_loss = 0
        self.D_loss = 0
        self.mfs_loss = 0
        self.GP_grad_norm = 0

    def sample_generator(self, num_samples):
        latent_samples = self.G.sample_latent(num_samples).to(self.device)
        generated_data = self.G(latent_samples)
        return generated_data

    def calculate_mfs_torch(self, X: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        return self.mfs_manager.get_mfs(X, y, subset=self.subset_mfs).to(self.device)

    @staticmethod
    def reshape_mfs_from_variates(mfs_from_variates: list):
        stacked = torch.stack(mfs_from_variates)
        reshaped = stacked.transpose(0, 1)
        return reshaped

    def wasserstein_distance_2d(self, x1, x2):
        batch_size = x1.shape[0]
        ab = torch.ones(batch_size) / batch_size
        ab = ab.to(self.device)
        M = ot.dist(x1, x2)
        return ot.emd2(ab, ab, M)

    def wasserstein_loss_mfs(self, mfs1, mfs2, average=True):
        n_features = mfs1.shape[0]
        wsds = []
        for first, second in zip(mfs1, mfs2):
            wsd = self.wasserstein_distance_2d(first, second)
            wsds.append(wsd)
        
        if average:
            return sum(wsds) / n_features
        else:
            return torch.stack(wsds).to(self.device)

    def _gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1).to(self.device)
        alpha = alpha.expand_as(real_data)
        
        interpolated = alpha * real_data + (1 - alpha) * generated_data
        interpolated.requires_grad_(True)
        
        prob_interpolated = self.D(interpolated)
        grad_outputs = torch.ones_like(prob_interpolated)
        gradients = torch_grad(outputs=prob_interpolated,
                              inputs=interpolated,
                              grad_outputs=grad_outputs,
                              create_graph=True,
                              retain_graph=True,
                              only_inputs=True)[0]
        
        gradients = gradients.view(batch_size, -1)
        gradients_norm = gradients.norm(2, dim=1)
        self.GP_grad_norm = gradients_norm.mean().item()
        
        gp = self.gp_weight * ((gradients_norm - 1) ** 2).mean()
        return gp

    def _critic_train_iteration(self, data):
        self.D_opt.zero_grad()
        self.D.train()
        
        data = data.to(self.device)
        batch_size = data.size(0)
        
        with torch.no_grad():
            generated_data = self.sample_generator(batch_size)
            generated_data = generated_data.to(self.device)
        
        generated_data = generated_data.detach()
        
        d_real = self.D(data)
        d_generated = self.D(generated_data)
        
        gradient_penalty = self._gradient_penalty(data, generated_data)
        d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
        
        d_loss.backward()
        self.D_opt.step()
        
        self.D_loss = d_loss

    def _generator_train_iteration(self, data):
        """
        Улучшенная версия с поддержкой scalar mfs_lambda для всех мета-признаков
        """
        self.G_opt.zero_grad()
        
        batch_size = data.size(0)
        generated_variates = []
        
        for _ in range(self.sample_number):
            generated_data = self.sample_generator(batch_size)
            generated_data.requires_grad_(True)
            generated_data.retain_grad()
            generated_variates.append(generated_data)
        
        d_generated = self.D(generated_variates[0])
        
        fake_mfs = [self.calculate_mfs_torch(X) for X in generated_variates]
        fake_mfs = self.reshape_mfs_from_variates(fake_mfs)
        
        # Улучшенная обработка mfs_lambda
        if isinstance(self.mfs_lambda, (int, float)):
            # Если scalar - применяем ко всем мета-признакам одинаково
            mfs_dist = self.wasserstein_loss_mfs(fake_mfs, self.target_mfs["other_mfs"], average=True)
            loss_mfs = self.mfs_lambda * mfs_dist
            
        elif isinstance(self.mfs_lambda, list):
            # Если список - используем индивидуальные веса
            mfs_lambda = torch.Tensor(self.mfs_lambda).to(self.device)
            mfs_dist = self.wasserstein_loss_mfs(fake_mfs, self.target_mfs["other_mfs"], average=False)
            
            # Проверяем размерности и подгоняем их
            num_mfs = len(self.subset_mfs)  # количество мета-признаков
            num_lambdas = len(self.mfs_lambda)  # количество весов
            
            if num_lambdas == num_mfs:
                # Размерности совпадают - используем как есть
                loss_mfs = mfs_lambda @ mfs_dist
            elif num_lambdas < num_mfs:
                # Весов меньше чем признаков - дублируем последний вес
                extended_lambda = torch.cat([
                    mfs_lambda, 
                    mfs_lambda[-1].repeat(num_mfs - num_lambdas)
                ]).to(self.device)
                loss_mfs = extended_lambda @ mfs_dist
            else:
                # Весов больше чем признаков - обрезаем
                truncated_lambda = mfs_lambda[:num_mfs]
                loss_mfs = truncated_lambda @ mfs_dist
                
        else:
            raise TypeError("mfs_lambda must be either a number (int/float) or a list")
        
        g_loss = -d_generated.mean() + loss_mfs
        g_loss.backward()
        self.G_opt.step()
        
        self.G_loss = g_loss
        self.mfs_loss = loss_mfs

    def _train_epoch(self, data_loader):
        data_iter = iter(data_loader)
        num_batches = len(data_loader)
        
        for _ in range(num_batches):
            # Critic updates
            for _ in range(self.critic_iterations):
                try:
                    data = next(data_iter)
                except StopIteration:
                    data_iter = iter(data_loader)
                    data = next(data_iter)
                data = data.to(self.device)
                self._critic_train_iteration(data)
                
            
            # Generator update
            try:
                data = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                data = next(data_iter)
            data = data.to(self.device)
            self._generator_train_iteration(data)

    def train(self, data_loader, epochs, plot_freq=100):
        
        self.G_history = []
        self.D_history = []
        self.mfs_history = []
        for epoch in range(epochs):
            self._train_epoch(data_loader)
            
            self.G_history.append(self.G_loss.item())
            self.D_history.append(self.D_loss.item())
            self.mfs_history.append(self.mfs_loss.item())
            # if epoch % plot_freq == 0:
            #     print(f"Epoch {epoch}: G_loss={self.G_loss:.4f}, D_loss={self.D_loss:.4f}, MFS_loss={self.mfs_loss:.4f}")

    def get_losses(self):
        return {
            "Epoch": list(range(len(self.G_history))), 
            "Generator Loss": self.G_history,
            "Discriminator Loss": self.D_history,
            "MFS Loss": self.mfs_history,
            "GP_grad_norm": self.GP_grad_norm
        }
        

# ===== MAIN SYNTHESIZER CLASS =====
class GANMFSSynthesizer:
    def __init__(self, 
                 generator_dim=(64, 128, 64),
                 discriminator_dim=(64, 128, 64),
                 emb_dim=32,
                 learning_rate_G=0.0001,
                 learning_rate_D=0.0001,
                 batch_size=64,
                 mfs_lambda=[0.1, 2.0],
                 subset_mfs=["mean", "var"],
                 sample_number=10,
                 sample_frac=0.5,
                 gp_weight=10,
                 critic_iterations=1,
                 epochs=300):
        
        self.params = {
            'generator_dim': generator_dim,
            'discriminator_dim': discriminator_dim,
            'emb_dim': emb_dim,
            'learning_rate_G': learning_rate_G,
            'learning_rate_D': learning_rate_D,
            'batch_size': batch_size,
            'mfs_lambda': mfs_lambda,
            'subset_mfs': subset_mfs,
            'sample_number': sample_number,
            'sample_frac': sample_frac,
            'gp_weight': gp_weight,
            'critic_iterations': critic_iterations,
            'epochs': epochs
        }
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.trainer = None
        self.is_trained = False
        
    def fit(self, df, epochs=None, target_column=None, test_size=0.3):
        """
        Обучить GAN на датасете
        
        Args:
            df: pandas DataFrame с данными
            target_column: название колонки с таргетом (если есть)
            epochs: количество эпох обучения
            test_size: размер тестовой выборки
        """
        
        # Подготовка данных
        if target_column:
            X = df.drop(columns=[target_column]).values
            y = df[target_column].values
            
            # Тут раньше было азделение на train/test
            X_train, y_train = X, y
            
            # Объединяем X и y для обучения GAN
            train_data = np.hstack([X_train, y_train.reshape(-1, 1)])
        else:
            train_data = df.values
        
        # Сохраняем информацию о данных
        self.data_dim = train_data.shape[1]
        self.columns = df.columns.tolist()
        
        # Инициализация моделей
        self.generator = Generator(
            embedding_dim=self.params['emb_dim'],
            generator_dim=self.params['generator_dim'],
            data_dim=self.data_dim
        ).to(self.device)
        
        self.discriminator = Discriminator(
            data_dim=self.data_dim,
            discriminator_dim=self.params['discriminator_dim']
        ).to(self.device)
        
        # Оптимизаторы
        G_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=self.params['learning_rate_G'],
            betas=(0.5, 0.9)
        )
        
        D_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=self.params['learning_rate_D'],
            betas=(0.5, 0.9)
        )
        
        # Инициализация тренера с MFS
        self.trainer = TrainerModified(
            generator=self.generator,
            discriminator=self.discriminator,
            gen_optimizer=G_optimizer,
            dis_optimizer=D_optimizer,
            device=self.device,
            batch_size=self.params['batch_size'],
            mfs_lambda=self.params['mfs_lambda'],
            subset_mfs=self.params['subset_mfs'],
            sample_number=self.params['sample_number'],
            critic_iterations=self.params['critic_iterations'],
            gp_weight=self.params['gp_weight'],
            aim_track=None,
            gen_model_name="wGAN_mfs"
        )
        
        # Подготовка MFS вариантов для обучения
        X_tensor = torch.tensor(train_data, dtype=torch.float)
        
        variates = create_variates(
            X_tensor,
            sample_number=self.params['sample_number'],
            sample_frac=self.params['sample_frac']
        )
        
        mfs_distr = [self.trainer.calculate_mfs_torch(X_sample) for X_sample in variates]
        mfs_distr = self.trainer.reshape_mfs_from_variates(mfs_distr)
        
        target_features = {
            "persistent_diagram": None,
            "other_mfs": mfs_distr
        }
        
        self.trainer.target_mfs = target_features
        
        # Создание DataLoader
        dataset = ToyDataset(X_tensor)
        data_loader = DataLoader(
            dataset, 
            batch_size=self.params['batch_size'], 
            shuffle=True, 
            num_workers=0
        )
        
        # Обучение
        if epochs is not None:
            self.params['epochs'] = epochs
        self.trainer.train(data_loader, self.params['epochs'], plot_freq=100)
        
        self._loss_values = self.trainer.get_losses()

        self.is_trained = True
        
    def generate(self, n_samples):
        """
        Генерировать синтетические данные
        
        Args:
            n_samples: количество образцов для генерации
            
        Returns:
            pandas DataFrame с синтетическими данными
        """
        if not self.is_trained:
            raise ValueError("Модель должна быть обучена перед генерацией данных!")
        
        # Генерация данных
        synthetic_data = self.trainer.sample_generator(n_samples).cpu().detach().numpy()
        
        # Создание DataFrame
        synthetic_df = pd.DataFrame(synthetic_data, columns=self.columns)
        
        return synthetic_df