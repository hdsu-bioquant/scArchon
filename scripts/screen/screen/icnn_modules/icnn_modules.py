import torch
import torch.nn as nn

def get_activation(activation):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'elu':
        return nn.ELU()
    elif activation == 'leaky_relu':
        return nn.LeakyReLU(0.2)
    elif activation == 'celu':
        return nn.CELU()
    elif activation == 'selu':
        return nn.SELU()
    elif activation == 'tanh':
        return nn.Tanh()
    else:
        raise NotImplementedError('activation [%s] is not found' % activation)

class ConvexLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(ConvexLinear, self).__init__(*kargs, **kwargs)
        if not hasattr(self.weight, 'be_positive'):
            self.weight.be_positive = 1.0

    def forward(self, input):
        out = nn.functional.linear(input, self.weight, self.bias)
        return out

class Simple_Feedforward_3Layer_ICNN_LastInp_Quadratic(nn.Module):

    def __init__(self, input_dim, hidden_dim, activation):

        super(Simple_Feedforward_3Layer_ICNN_LastInp_Quadratic, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation

        self.fc1_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.activ_1 = get_activation(self.activation)

        self.fc2_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.fc2_convex = ConvexLinear(self.hidden_dim, self.hidden_dim, bias=False)
        self.activ_2 = get_activation(self.activation)

        self.fc3_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.fc3_convex = ConvexLinear(self.hidden_dim, self.hidden_dim, bias=False)
        self.activ_3 = get_activation(self.activation)

        self.last_convex = ConvexLinear(self.hidden_dim, 1, bias=False)
        self.last_linear = nn.Linear(self.input_dim, 1, bias=True)

    def forward(self, input):
        
        x = self.activ_1(self.fc1_normal(input)).pow(2)
        x = self.activ_2(self.fc2_convex(x).add(self.fc2_normal(input)))
        x = self.activ_3(self.fc3_convex(x).add(self.fc3_normal(input)))
        x = self.last_convex(x).add(self.last_linear(input).pow(2))

        return x



class Simple_Feedforward_3Layer_ICNN_LastFull_Quadratic(nn.Module):

    def __init__(self, input_dim, hidden_dim, activation):

        super(Simple_Feedforward_3Layer_ICNN_LastFull_Quadratic, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
    
        self.fc1_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.activ_1 = get_activation(self.activation)

        self.fc2_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.fc2_convex = ConvexLinear(self.hidden_dim, self.hidden_dim, bias=False)
        self.activ_2 = get_activation(self.activation)

        self.fc3_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.fc3_convex = ConvexLinear(self.hidden_dim, self.hidden_dim, bias=False)
        self.activ_3 = get_activation(self.activation)

        self.last_convex = ConvexLinear(self.hidden_dim, 1, bias=False)
        self.last_linear = nn.Linear(self.input_dim, 1, bias=True)

    def forward(self, input):

        x = self.activ_1(self.fc1_normal(input)).pow(2)
        x = self.activ_2(self.fc2_convex(x).add(self.fc2_normal(input)))
        x = self.activ_3(self.fc3_convex(x).add(self.fc3_normal(input)))
        x = self.last_convex(x).add(self.last_linear(input)).pow(2)

        return x