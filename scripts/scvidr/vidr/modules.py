import torch
from torch import nn
from torch.distributions import Normal

class VIDREncoder(nn.Module):
    def __init__(
        self,
        input_dim: int, 
        latent_dim: int,
        hidden_dim: int,
        n_hidden_layers: int,
        momentum: float = 0.01,
        eps: float = 0.001,
        dropout_rate: float = 0.2,
        reparam_eps: float = 1e-4,
        nonlin = nn.LeakyReLU,
    ):
        super(VIDREncoder, self).__init__()
        self.eps = reparam_eps
        self.fclayers = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim, momentum=momentum, eps=eps),
                    nonlin(),
                    nn.Dropout(p = dropout_rate),
        )
    
        #Encoder
        #Include Input Module
        modules = [nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim, momentum=momentum, eps=eps),
                nonlin(),
                nn.Dropout(p = dropout_rate),
                )]

        #Add hidden fully connected layers
        for i in range(n_hidden_layers-1):
                modules.append(self.fclayers)

        self.encoder = nn.Sequential(*modules)

        self.mean = nn.Linear(hidden_dim, latent_dim)

        self.log_var = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, inputs):
        #encode
        results = self.encoder(inputs)
        mean = self.mean(results)
        log_var = self.log_var(results)
        
        var = torch.exp(log_var) +  self.eps
        
        #reparameterize
        latent_rep = Normal(mean, var.sqrt()).rsample()
        
        return mean, var, latent_rep

class VIDRDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int, 
        latent_dim: int,
        hidden_dim: int,
        n_hidden_layers: int,
        momentum: float = 0.01,
        eps: float = 0.001,
        dropout_rate: float = 0.2,
        nonlin = nn.LeakyReLU,
    ):
        super(VIDRDecoder, self).__init__()
        self.fclayers = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim, momentum=momentum, eps=eps),
                    nonlin(),
                    nn.Dropout(p = dropout_rate),
        )
    
        #Encoder
        #Include Input Module
        modules = [nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim, momentum=momentum, eps=eps),
                nonlin(),
                nn.Dropout(p = dropout_rate),
                )]

        #Add hidden fully connected layers
        for i in range(n_hidden_layers-1):
                modules.append(self.fclayers)
        
        modules.append(nn.Linear(hidden_dim, input_dim))
        self.decoder = nn.Sequential(*modules)
    
    def forward(self, latent_rep):
        #decode
        x_hat = self.decoder(latent_rep)
        return x_hat
