import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, n_features, min_hidden_size, z_dim):
        super(Encoder, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(n_features, min_hidden_size * 2),
                                    nn.BatchNorm1d(min_hidden_size * 2),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(min_hidden_size * 2, min_hidden_size),
                                    nn.BatchNorm1d(min_hidden_size),
                                    nn.ReLU())
        self.out = nn.Sequential(nn.Linear(min_hidden_size, z_dim))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.out(out)
        return out


class Encoder_AC_layer(nn.Module):
    def __init__(self, n_features, min_hidden_size, z_dim):
        super(Encoder_AC_layer, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(n_features, min_hidden_size * 2),
                                    nn.LayerNorm(min_hidden_size * 2),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(min_hidden_size * 2, min_hidden_size),
                                    nn.LayerNorm(min_hidden_size),
                                    nn.ReLU())
        self.out = nn.Sequential(nn.Linear(min_hidden_size, z_dim))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.out(out)
        return out