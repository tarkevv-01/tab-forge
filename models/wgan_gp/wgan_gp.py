import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.autograd import grad as torch_grad

# ===== MODELS =====
class Residual(nn.Module):
    """Residual layer for the WGAN-GP."""
    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = nn.Linear(i, o)
        self.relu = nn.ReLU()
    
    def forward(self, input_):
        """Apply the Residual layer to the `input_`."""
        out = self.fc(input_)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)


class Generator(nn.Module):
    """Generator for the WGAN-GP."""
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
    """Discriminator (Critic) for the WGAN-GP."""
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

# ===== TRAINER =====
class WGANGPTrainer:
    def __init__(self, generator, discriminator, gen_optimizer, dis_optimizer, 
                 device, batch_size, critic_iterations=5, gp_weight=10):
        
        self.G = generator
        self.D = discriminator
        self.G_opt = gen_optimizer
        self.D_opt = dis_optimizer
        self.device = device
        self.batch_size = batch_size
        self.critic_iterations = critic_iterations
        self.gp_weight = gp_weight
        
        # Move models to device
        self.G.to(device)
        self.D.to(device)
        
        # Loss tracking
        self.G_loss = 0
        self.D_loss = 0

    def sample_generator(self, num_samples):
        latent_samples = self.G.sample_latent(num_samples).to(self.device)
        generated_data = self.G(latent_samples)
        return generated_data

    def _gradient_penalty(self, real_data, generated_data):
        """Compute gradient penalty for Lipschitz constraint."""
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
        gp = self.gp_weight * ((gradients_norm - 1) ** 2).mean()
        return gp

    def _critic_train_iteration(self, data):
        """Train discriminator for one iteration."""
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
        """Train generator for one iteration."""
        self.G_opt.zero_grad()
        
        batch_size = data.size(0)
        generated_data = self.sample_generator(batch_size)
        
        d_generated = self.D(generated_data)
        g_loss = -d_generated.mean()
        
        g_loss.backward()
        self.G_opt.step()
        
        self.G_loss = g_loss

    def _train_epoch(self, data_loader):
        """Train for one epoch."""
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
        """Train the WGAN-GP."""
        self.G_history = []
        self.D_history = []
        
        for epoch in range(epochs):
            self._train_epoch(data_loader)
            
            self.G_history.append(self.G_loss.item())
            self.D_history.append(self.D_loss.item())
            
            # if epoch % plot_freq == 0:
            #     print(f"Epoch {epoch}: G_loss={self.G_loss:.4f}, D_loss={self.D_loss:.4f}")

    def get_losses(self):
        return {
            "Epoch": list(range(len(self.G_history))), 
            "Generator Loss": self.G_history,
            "Discriminator Loss": self.D_history,
        }

# ===== MAIN SYNTHESIZER CLASS =====
class WGANSynthesizer:
    def __init__(self, 
                 generator_dim=(64, 128, 64),
                 discriminator_dim=(64, 128, 64),
                 emb_dim=32,
                 learning_rate_G=0.0001,
                 learning_rate_D=0.0001,
                 batch_size=64,
                 gp_weight=10,
                 critic_iterations=5,
                 epochs=300):
        
        self.params = {
            'generator_dim': generator_dim,
            'discriminator_dim': discriminator_dim,
            'emb_dim': emb_dim,
            'learning_rate_G': learning_rate_G,
            'learning_rate_D': learning_rate_D,
            'batch_size': batch_size,
            'gp_weight': gp_weight,
            'critic_iterations': critic_iterations,
            'epochs': epochs
        }
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.trainer = None
        self.is_trained = False
        
    def fit(self, df, epochs=None, target_column=None, test_size=0.3):
        """
        Train WGAN-GP on dataset
        
        Args:
            df: pandas DataFrame with data
            target_column: name of target column (optional)
            epochs: number of training epochs
            test_size: size of test split
        """
        
        # Data preparation
        if target_column:
            X = df.drop(columns=[target_column]).values
            y = df[target_column].values
            
            # Train/test split
            X_train, y_train = X, y
            
            # Combine X and y for training
            train_data = np.hstack([X_train, y_train.reshape(-1, 1)])
        else:
            train_data = df.values
        
        # Save data info
        self.data_dim = train_data.shape[1]
        self.columns = df.columns.tolist()
        
        # Initialize models
        self.generator = Generator(
            embedding_dim=self.params['emb_dim'],
            generator_dim=self.params['generator_dim'],
            data_dim=self.data_dim
        ).to(self.device)
        
        self.discriminator = Discriminator(
            data_dim=self.data_dim,
            discriminator_dim=self.params['discriminator_dim']
        ).to(self.device)
        
        # Optimizers
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
        
        # Initialize trainer
        self.trainer = WGANGPTrainer(
            generator=self.generator,
            discriminator=self.discriminator,
            gen_optimizer=G_optimizer,
            dis_optimizer=D_optimizer,
            device=self.device,
            batch_size=self.params['batch_size'],
            critic_iterations=self.params['critic_iterations'],
            gp_weight=self.params['gp_weight']
        )
        
        # Create DataLoader
        X_tensor = torch.tensor(train_data, dtype=torch.float)
        dataset = ToyDataset(X_tensor)
        data_loader = DataLoader(
            dataset, 
            batch_size=self.params['batch_size'], 
            shuffle=True, 
            num_workers=0
        )
        
        # Train
        if epochs is not None:
            self.params['epochs'] = epochs
        self.trainer.train(data_loader, self.params['epochs'], plot_freq=100)
        
        self._loss_values = self.trainer.get_losses()
        self.is_trained = True
        
    def generate(self, n_samples):
        """
        Generate synthetic data
        
        Args:
            n_samples: number of samples to generate
            
        Returns:
            pandas DataFrame with synthetic data
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before generating data!")
        
        # Generate data
        synthetic_data = self.trainer.sample_generator(n_samples).cpu().detach().numpy()
        
        # Create DataFrame
        synthetic_df = pd.DataFrame(synthetic_data, columns=self.columns)
        
        return synthetic_df