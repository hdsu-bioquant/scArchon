import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim, min_hidden_size, n_features):
        super(Generator, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(z_dim, min_hidden_size),
                                    nn.BatchNorm1d(min_hidden_size),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(min_hidden_size, min_hidden_size * 2),
                                    nn.BatchNorm1d(min_hidden_size * 2),
                                    nn.ReLU())
        self.out = nn.Sequential(nn.Linear(min_hidden_size * 2, n_features),
                                 nn.BatchNorm1d(n_features),
                                 nn.ReLU())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.out(out)
        return out
