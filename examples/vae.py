import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import init


class VAE_model(nn.Module):
    def __init__(self, input_dim, inter_dim, inter_dim1, latent_dim, desired_sample_size, training_samples, batch_size,
                 epochs, lr=0.001):

        super(VAE_model, self).__init__()
        self.input_dim = input_dim
        self.inter_dim = inter_dim
        self.inter_dim1 = inter_dim1
        self.latent_dim = latent_dim
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.training_samples = training_samples
        self.desired_sample_size = desired_sample_size
        # Encoder
        self.fc1 = nn.Linear(input_dim, inter_dim)
        self.fc2 = nn.Linear(inter_dim, inter_dim1)
        self.fc3 = nn.Linear(inter_dim1, latent_dim * 2)
        self.fc_mean = nn.Linear(latent_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim * 2, latent_dim)
        # Decoder
        self.fc4 = nn.Linear(latent_dim, inter_dim1)
        self.fc4_1 = nn.Linear(inter_dim1, inter_dim)
        self.fc5 = nn.Linear(inter_dim, input_dim)
        # Initialize weights
        self._init_weights()
        # Create optimizer，Adam
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')  # He初始化
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)

    def encode(self, x):
        h1 = F.leaky_relu(self.fc1(x))
        h2 = F.leaky_relu(self.fc2(h1))
        h3 = F.leaky_relu(self.fc3(h2))
        return self.fc_mean(h3), self.fc_logvar(h3)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        h4 = F.leaky_relu(self.fc4(z))
        h4_1 = F.leaky_relu(self.fc4_1(h4))
        h5 = F.leaky_relu(self.fc5(h4_1))
        return h5

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar

    def loss_function(self, recon_x, x, mean, logvar,epoch):
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        beta = min(1.0, epoch / 10)  # KL Annealing
        return recon_loss + beta * kl_loss

    def train(self, initial_samples):
        # Convert initial samples to tensors
        initial_samples = torch.from_numpy(initial_samples).float()
        # Create data loader
        dataset = TensorDataset(initial_samples)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        best_loss = float('inf')
        best_parameters = None

        # Train VAE
        for epoch in range(self.epochs):
            for batch_idx, (x,) in enumerate(dataloader):
                self.optimizer.zero_grad()
                recon_x, mean, logvar = self.forward(x)
                loss = self.loss_function(recon_x, x, mean, logvar,epoch)
                loss.backward()
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)  # 指定最大范数为1.0
                self.optimizer.step()

            # Check if current loss is lower than best loss
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_parameters = self.state_dict()

            print(f'Epoch {epoch}, Loss: {loss.item()}')

        # Use best parameters for further processing
        self.load_state_dict(best_parameters)

    def generate_samples(self, desired_sample_size):
        # Sample from standard normal distribution
        z = torch.randn(desired_sample_size, self.latent_dim)
        # Decode samples
        new_samples = self.decode(z)
        return new_samples.detach().numpy()
