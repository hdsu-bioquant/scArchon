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


class Discriminator_AC(nn.Module):
    def __init__(self, n_features, min_hidden_size, out_dim, n_classes=0):
        super(Discriminator_AC, self).__init__()
        self.layer1 = nn.Sequential(spectral_norm(nn.Linear(n_features, min_hidden_size * 2)),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(spectral_norm(nn.Linear(min_hidden_size * 2, min_hidden_size)),
                                    nn.ReLU())
        self.out = nn.Sequential(nn.Linear(min_hidden_size, out_dim), nn.Sigmoid())

        if n_classes > 0:
            self.classify = nn.Sequential(nn.Linear(min_hidden_size, out_features=n_classes),
                                            nn.Softmax(dim=1))

    def forward(self, x, y):
        #if y is not None:
        #    x = torch.cat(tensors=(x, y), dim=1)
        h = self.layer1(x)
        h = self.layer2(h)
        if y is not None:
            classify = self.classify(h)
        out = self.out(h)
        return out, classify