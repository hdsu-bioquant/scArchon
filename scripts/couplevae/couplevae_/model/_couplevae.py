import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from .util import load_h5ad_to_dataloader

class Encoderc(nn.Module):
    def __init__(self, input_dim, z_dim, dropout_rate):
        super(Encoderc, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_dim, 800),
                                 nn.BatchNorm1d(800, momentum=0.01, eps=0.001),
                                  nn.LayerNorm(800, elementwise_affine=False),
                                 nn.LeakyReLU(),
                                 nn.Dropout(p=dropout_rate))
        self.fc2 = nn.Sequential(nn.Linear(800, 800),
                                 nn.BatchNorm1d(800, momentum=0.01, eps=0.001),
                                  nn.LayerNorm(800, elementwise_affine=False),
                                 nn.LeakyReLU(),
                                 nn.Dropout(p=dropout_rate))
        # self.fc3 = nn.Sequential(nn.Linear(512, 256),
        #                          nn.BatchNorm1d(256),
        #                          nn.ReLU())
        # self.fc_mean = nn.Sequential(nn.Linear(800, z_dim),
        #                              nn.ReLU())
        # self.fc_log_var = nn.Sequential(nn.Linear(800, z_dim),
        #                              nn.ReLU())
        self.fc_mean = nn.Linear(800, z_dim)
        self.fc_log_var = nn.Linear(800, z_dim)
        
    def forward(self, x):
        h = self.fc1(x)
        h = self.fc2(h)
        # h = self.fc3(h)
        mean = self.fc_mean(h)
        log_var = self.fc_log_var(h)
        return mean, log_var
    
class Encoderp(nn.Module):
    def __init__(self, input_dim, z_dim, dropout_rate):
        super(Encoderp, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_dim, 800),
                                 nn.BatchNorm1d(800, momentum=0.01, eps=0.001),
                                  nn.LayerNorm(800, elementwise_affine=False),
                                 nn.LeakyReLU(),
                                 nn.Dropout(p=dropout_rate))
        self.fc2 = nn.Sequential(nn.Linear(800, 800),
                                 nn.BatchNorm1d(800, momentum=0.01, eps=0.001),
                                  nn.LayerNorm(800, elementwise_affine=False),
                                 nn.LeakyReLU(),
                                 nn.Dropout(p=dropout_rate))
        # self.fc3 = nn.Sequential(nn.Linear(512, 256),
        #                          nn.BatchNorm1d(256),
        #                          nn.ReLU())
        # self.fc_mean = nn.Sequential(nn.Linear(800, z_dim),
        #                              nn.ReLU())
        # self.fc_log_var = nn.Sequential(nn.Linear(800, z_dim),
        #                              nn.ReLU())
        self.fc_mean = nn.Linear(800, z_dim)
        self.fc_log_var = nn.Linear(800, z_dim)
        
    def forward(self, x):
        h = self.fc1(x)
        h = self.fc2(h)
        # h = self.fc3(h)
        mean = self.fc_mean(h)
        log_var = self.fc_log_var(h)
        return mean, log_var
    
class Decoderc(nn.Module):
    def __init__(self, z_dim, output_dim, dropout_rate):
        super(Decoderc, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(z_dim, 800),
                                 # nn.BatchNorm1d(800, momentum=0.01, eps=0.001),
                                 # nn.LayerNorm(800, elementwise_affine=False),
                                 nn.LeakyReLU(),
                                 nn.Dropout(p=dropout_rate))
        self.fc2 = nn.Sequential(nn.Linear(800, 800),
                                 # nn.BatchNorm1d(800, momentum=0.01, eps=0.001),
                                 # nn.LayerNorm(800, elementwise_affine=False),
                                 nn.LeakyReLU(),
                                 nn.Dropout(p=dropout_rate))
        self.fc3 = nn.Sequential(nn.Linear(800, output_dim),
                                  nn.ReLU())
        # self.fc3 = nn.Linear(800, output_dim)
        
    def forward(self, z):
        h = self.fc1(z)
        h = self.fc2(h)
        h = self.fc3(h)
        return h

class Decoderp(nn.Module):
    def __init__(self, z_dim, output_dim, dropout_rate):
        super(Decoderp, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(z_dim, 800),
                                 # nn.BatchNorm1d(800, momentum=0.01, eps=0.001),
                                 # nn.LayerNorm(800, elementwise_affine=False),
                                 nn.LeakyReLU(),
                                 nn.Dropout(p=dropout_rate))
        self.fc2 = nn.Sequential(nn.Linear(800, 800),
                                 # nn.BatchNorm1d(800, momentum=0.01, eps=0.001),
                                 # nn.LayerNorm(800, elementwise_affine=False),
                                 nn.LeakyReLU(),
                                 nn.Dropout(p=dropout_rate))
        self.fc3 = nn.Sequential(nn.Linear(800, output_dim),
                                  nn.ReLU())
        # self.fc3 = nn.Linear(800, output_dim)
        
    def forward(self, z):
        h = self.fc1(z)
        h = self.fc2(h)
        h = self.fc3(h)
        return h


class Decodercp(nn.Module):
    def __init__(self, z_dim, output_dim, dropout_rate):
        super(Decodercp, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(z_dim, 800),
                                 # nn.BatchNorm1d(800, momentum=0.01, eps=0.001),
                                 # nn.LayerNorm(800, elementwise_affine=False),
                                 nn.LeakyReLU(),
                                 nn.Dropout(p=dropout_rate))
        self.fc2 = nn.Sequential(nn.Linear(800, 800),
                                 # nn.BatchNorm1d(800, momentum=0.01, eps=0.001),
                                 # nn.LayerNorm(800, elementwise_affine=False),
                                 nn.LeakyReLU(),
                                 nn.Dropout(p=dropout_rate))
        self.fc3 = nn.Sequential(nn.Linear(800, output_dim),
                                  nn.ReLU())
        # self.fc3 = nn.Linear(800, output_dim)
        
    def forward(self, z):
        h = self.fc1(z)
        h = self.fc2(h)
        h = self.fc3(h)
        return h


class Decoderpc(nn.Module):
    def __init__(self, z_dim, output_dim, dropout_rate):
        super(Decoderpc, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(z_dim, 800),
                                 # nn.BatchNorm1d(800, momentum=0.01, eps=0.001),
                                 # nn.LayerNorm(800, elementwise_affine=False),
                                 nn.LeakyReLU(),
                                 nn.Dropout(p=dropout_rate))
        self.fc2 = nn.Sequential(nn.Linear(800, 800),
                                 # nn.BatchNorm1d(800, momentum=0.01, eps=0.001),
                                 # nn.LayerNorm(800, elementwise_affine=False),
                                 nn.LeakyReLU(),
                                 nn.Dropout(p=dropout_rate))
        self.fc3 = nn.Sequential(nn.Linear(800, output_dim),
                                  nn.ReLU())
        # self.fc3 = nn.Linear(800, output_dim)
        
    def forward(self, z):
        h = self.fc1(z)
        h = self.fc2(h)
        h = self.fc3(h)
        return h


class Couplerc(nn.Module):
    def __init__(self, z_dim):
        super(Couplerc, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(z_dim, z_dim),
                                 nn.BatchNorm1d(z_dim),
                                 nn.ReLU())
        self.fc_mean = nn.Linear(z_dim, z_dim)
        self.fc_log_var = nn.Linear(z_dim, z_dim)
        
    def forward(self, z):
        h = self.fc1(z)
        mean = self.fc_mean(h)
        log_var = self.fc_log_var(h)
        return mean, log_var
    
class Couplerp(nn.Module):
    def __init__(self, z_dim):
        super(Couplerp, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(z_dim, z_dim),
                                 nn.BatchNorm1d(z_dim),
                                 nn.ReLU())
        self.fc_mean = nn.Linear(z_dim, z_dim)
        self.fc_log_var = nn.Linear(z_dim, z_dim)
        
    def forward(self, z):
        h = self.fc1(z)
        mean = self.fc_mean(h)
        log_var = self.fc_log_var(h)
        return mean, log_var

class VAE(nn.Module):
    def __init__(self, x_dim, z_dim=16, learning_rate=0.001, dropout_rate = 0.2, alpha=0.01, beta=1):
        super(VAE, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.alpha = alpha
        self.beta = beta
        
        self.encoder_c = Encoderc(x_dim, z_dim, dropout_rate)
        self.encoder_p = Encoderp(x_dim, z_dim, dropout_rate)
        
        self.decoder_c = Decoderc(z_dim, x_dim, dropout_rate)
        self.decoder_p = Decoderp(z_dim, x_dim, dropout_rate)
        self.decoder_cp = Decoderp(z_dim, x_dim, dropout_rate)
        self.decoder_pc = Decoderc(z_dim, x_dim, dropout_rate)
        
        self.coupler_c = Couplerc(z_dim)
        self.coupler_p = Couplerp(z_dim)
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def sample_z(self, mean, log_var):
        eps = torch.randn(mean.size(0), self.z_dim).to(mean.device)
        return mean + torch.exp(log_var / 2) * eps

    def forward(self, x_0, x_1):
        mu_0, log_var_0 = self.encoder_c(x_0)
        mu_1, log_var_1 = self.encoder_p(x_1)
        
        z_mean_c = self.sample_z(mu_0, log_var_0)
        z_mean_p = self.sample_z(mu_1, log_var_1)
        
        mu_p, log_var_p = self.coupler_c(z_mean_c)
        mu_c, log_var_c = self.coupler_p(z_mean_p)
        
        z_mean_1 = self.sample_z(mu_p, log_var_p)
        z_mean_0 = self.sample_z(mu_c, log_var_c)
        
        x_hat_0 = self.decoder_c(z_mean_c)
        x_hat_1 = self.decoder_p(z_mean_p)
        x_hat_cp = self.decoder_cp(z_mean_1)
        x_hat_pc = self.decoder_pc(z_mean_0)
        
        return x_hat_0, x_hat_1, x_hat_cp, x_hat_pc, z_mean_c, z_mean_p, z_mean_1, z_mean_0, mu_0, log_var_0, mu_1, log_var_1, mu_p, log_var_p, mu_c, log_var_c

    def loss_function(self, x_0, x_1, x_hat_0, x_hat_1, x_hat_cp, x_hat_pc, z_mean_c, z_mean_p, z_mean_1, z_mean_0, mu_0, log_var_0, mu_1, log_var_1, mu_p, log_var_p, mu_c, log_var_c):
        kl_loss0 = 0.25 * torch.sum(torch.exp(log_var_0) + mu_0**2 - 1 - log_var_0, dim=1)
        recon_loss0 = 0.25 * torch.sum((x_0 - x_hat_0)**2, dim=1)
        trans_loss0 = 0.25 * torch.sum((x_0 - x_hat_pc)**2, dim=1)
        coupl_loss0 = 0.25 * torch.sum((mu_c - mu_0)**2, dim=1)
        
        kl_loss1 = 0.25 * torch.sum(torch.exp(log_var_1) + mu_1**2 - 1 - log_var_1, dim=1)
        recon_loss1 = 0.25 * torch.sum((x_1 - x_hat_1)**2, dim=1)
        trans_loss1 = 0.25 * torch.sum((x_1 - x_hat_cp)**2, dim=1)
        coupl_loss1 = 0.25 * torch.sum((mu_p - mu_1)**2, dim=1)
        
        kl_loss = kl_loss0 + kl_loss1
        recon_loss = recon_loss0 + recon_loss1
        trans_loss = trans_loss0 + trans_loss1
        coupl_loss = coupl_loss0 + coupl_loss1
        
        vae_loss = torch.mean(recon_loss + trans_loss + self.alpha * kl_loss + self.beta * coupl_loss)
        
        return vae_loss
    
    def to_latent(self, data):
        self.eval()
        adata = torch.tensor(data.X).float().to('cuda')
        
        
        dataset = TensorDataset(adata)
        dataloader = DataLoader(dataset, batch_size=data.X.shape[0], shuffle=True)
        with torch.no_grad():
            for data in dataloader:
                mu, log_var = self.encoder_c(data)
                latent = self.sample_z(mu, log_var)
                mu_p, log_var_p = self.coupler_c(latent)
                latent = self.sample_z(mu_p, log_var_p)    
        return latent
        
    def reconstruct(self, data):
        self.eval()
        
        with torch.no_grad():
            latent = self.to_latent(data)
            reconstruct = self.decoder_cp(latent)
        return reconstruct
        
    def predict(self, adata_c, adata_p):
        self.eval()
        train_data_c = torch.tensor(adata_c.X).float().to('cuda')
        train_data_p = torch.tensor(adata_p.X).float().to('cuda')
        
        dataset = TensorDataset(train_data_c, train_data_p)
        dataloader = DataLoader(dataset, batch_size=adata_c.X.shape[0], shuffle=True)
        
        with torch.no_grad():
            for data,data1 in dataloader:
                  
                x_hat_0, x_hat_1, x_hat_cp, x_hat_pc, z_mean_c, z_mean_p, z_mean_1, z_mean_0, mu_0, log_var_0, mu_1, log_var_1, mu_p, log_var_p, mu_c, log_var_c = self(data, data1)
            
                pred = x_hat_cp.cpu().numpy()
        return pred
        

    def train_vae(self, adata_c, adata_p, batch_size=128, n_epochs=3, save_path=None, device='cuda'):
        self.train()
        self.to(device)
        train_data_c = torch.tensor(adata_c.X).float().to(device)
        train_data_p = torch.tensor(adata_p.X).float().to(device)
        
        dataset = TensorDataset(train_data_c, train_data_p)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        losses=[]
        
        for epoch in range(n_epochs):
            for data, data1 in dataloader:
                self.optimizer.zero_grad()
                x_hat_0, x_hat_1, x_hat_cp, x_hat_pc, z_mean_c, z_mean_p, z_mean_1, z_mean_0, mu_0, log_var_0, mu_1, log_var_1, mu_p, log_var_p, mu_c, log_var_c = self(data, data1)
                loss = self.loss_function(data, data1, x_hat_0, x_hat_1, x_hat_cp, x_hat_pc, z_mean_c, z_mean_p, z_mean_1, z_mean_0, mu_0, log_var_0, mu_1, log_var_1, mu_p, log_var_p, mu_c, log_var_c)
                loss.backward()
                self.optimizer.step()
            print(f"Epoch {epoch + 1}: Train VAE Loss: {loss.item()}")
            losses.append(loss.item())
        
        
        if save_path:
            torch.save(self.state_dict(), save_path)
            print(f"Model saved at {save_path}")

    def load_model(self, load_path):
        self.load_state_dict(torch.load(load_path))
        print(f"Model loaded from {load_path}")

