import torch.nn as nn
from torch.nn.utils import spectral_norm


# spectral normalization
class Discriminator(nn.Module):
    def __init__(self, n_features, min_hidden_size, out_dim):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(spectral_norm(nn.Linear(n_features, min_hidden_size * 2)),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(spectral_norm(nn.Linear(min_hidden_size * 2, min_hidden_size)),
                                    nn.ReLU())
        self.out = nn.Sequential(nn.Linear(min_hidden_size, out_dim))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.out(out)
        return out
