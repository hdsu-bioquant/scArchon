import scanpy as sc
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from scipy import sparse
import pandas as pd
from torch import Tensor
from torch.distributions import kl_divergence as kl
import argparse
from torch.autograd import Variable
from icnn_modules.icnn_modules import *
from scipy.stats import truncnorm
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class VAE(nn.Module):
    def __init__(self, input_dim, inter_dim, latent_dim,noise_rate):
        super(VAE, self).__init__()
        
        self.noise_dropout=nn.Dropout(noise_rate)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, inter_dim),
            nn.ReLU(),
            nn.Linear(inter_dim, latent_dim * 2),
        )

        self.decoder =  nn.Sequential(
            nn.Linear(latent_dim, inter_dim),
            nn.ReLU(),
            nn.Linear(inter_dim, input_dim),
            nn.ReLU(),
        )

    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def forward(self, x):
        
        x=self.noise_dropout(x)

        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)
        z = self.reparameterise(mu, logvar)
        recon_x = self.decoder(z)

        return recon_x, mu, logvar
    
def to_recon (model,adata):
        if sparse.issparse(adata.X):
            tensor = Tensor(adata.X.A)
        else:
            tensor = Tensor(adata.X)
 
        tensor = tensor.cuda()
        latent = model.encoder(tensor)
        
        mu, logvar = latent.chunk(2, dim=1)
        z = model.reparameterise(mu, logvar)
        recon = model.decoder(z)
        
        recon_adata = sc.AnnData(X=recon.cpu().detach().numpy(), obs=adata.obs.copy())
        recon_adata.obs['recon_label']='recon'
        
        Ori_adata=sc.AnnData(X=adata.X, obs=adata.obs.copy())
        Ori_adata.obs['recon_label']='ori'
        
        return recon_adata,Ori_adata

def to_latent (model,adata):
        if sparse.issparse(adata.X):
            tensor = Tensor(adata.X.A)
        else:
            tensor = Tensor(adata.X)
 
        tensor = tensor.cuda()
        latent = model.encoder(tensor)
        
        mu, logvar = latent.chunk(2, dim=1)
        z = model.reparameterise(mu, logvar)
        
        latent_adata = sc.AnnData(X=z.cpu().detach().numpy(), obs=adata.obs.copy())
        return latent_adata

def balancer(adata,cell_type_key):
    class_names = np.unique(adata.obs[cell_type_key])
    class_pop = {}
    for cls in class_names:
        class_pop[cls] = adata[adata.obs[cell_type_key] == cls].shape[0]
    max_number = np.max(list(class_pop.values()))
    index_all = []
    for cls in class_names:
        class_index = np.array(adata.obs[cell_type_key] == cls)
        index_cls = np.nonzero(class_index)[0]
        index_cls_r = index_cls[np.random.choice(len(index_cls), max_number)]
        index_all.append(index_cls_r)

    balanced_data = adata[np.concatenate(index_all)].copy()
    return balanced_data

def get_latent (adata,model,ctrl_key=None,stim_key=None,cell_type_key=None,condition_key=None):
    
    adata=to_latent(model,adata)
    ctrl_x = adata[adata.obs[condition_key] == ctrl_key, :]
    stim_x = adata[adata.obs[condition_key] == stim_key, :]
    ctrl_x = balancer(ctrl_x, cell_type_key)
    stim_x = balancer(stim_x, cell_type_key)
    
    eq = min(ctrl_x.X.shape[0], stim_x.X.shape[0])
    cd_ind = np.random.choice(range(ctrl_x.shape[0]), size=eq, replace=False)
    stim_ind = np.random.choice(range(stim_x.shape[0]), size=eq, replace=False)
    ctrl_adata = ctrl_x[cd_ind, :]
    stim_adata = stim_x[stim_ind, :]

    latent_ctrl = ctrl_adata.X
    latent_stim = stim_adata.X

    return latent_ctrl,latent_stim

def compute_constraint_loss(list_of_params):
    
    loss_val = 0
    for p in list_of_params:
        loss_val += torch.relu(-p).pow(2).sum()
    return loss_val

def compute_optimal_transport_map(y, convex_g):

    g_of_y = convex_g(y).sum()
    grad_g_of_y = torch.autograd.grad(g_of_y, y, create_graph=True)[0]

    return grad_g_of_y

def truncated_normal(size, threshold=1):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    return values

def main():
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('-in', '--input_file_path', type=str, default=None, help='the file path to read dataset')
    parser.add_argument('-ou', '--output_file_path', type=str, default=None, help='the file path to save results')
    parser.add_argument('--label', type=str, default=None, help='the cell type to predict')
    parser.add_argument('--condition_key', type=str, default='condition', help='Key for condition in dataset')
    parser.add_argument('--cell_type_key', type=str, default='cell_type', help='Key for cell type in dataset')
    parser.add_argument('--ctrl_key', type=str, default='control', help='Key for control condition')
    parser.add_argument('--stim_key', type=str, default='stimulated', help='Key for stimulated condition')
    parser.add_argument('--latent_dim', type=int, default=100, help='Dimension of the low-dimensional latent variable in the VAE model')
    parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
    parser.add_argument('--epochs', type=int, default=40, metavar='S',help='number of ot epochs')
    parser.add_argument('--full_quadratic', type=bool, default=False, help='if the last layer is full quadratic or not')
    parser.add_argument('--activation', type=str, default='leaky_relu', help='which activation to use for')
    parser.add_argument('--optimizer', type=str, default='Adam', help='which optimizer to use')
    
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    args.lr_schedule = 2 if args.batch_size == 64 else 4
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    #Load data
    train = sc.read(args.input_file_path)
    cell_label = args.label
    batch_size=args.batch_size
    train_new = train[~((train.obs[args.cell_type_key] == "{}".format(cell_label)) &
                        (train.obs[args.condition_key] == args.stim_key))]
    if sparse.issparse(train_new.X):
        train_new_pd = pd.DataFrame(data=train_new.X.A, index=train_new.obs_names,columns=train_new.var_names)
    else:
        train_new_pd = pd.DataFrame(data=train_new.X, index=train_new.obs_names,columns=train_new.var_names)
    train_new_tensor = Tensor(np.array(train_new_pd))
    train_new_tensor = train_new_tensor.cuda() 
    train_loader = torch.utils.data.DataLoader(dataset=train_new_tensor,batch_size=batch_size,shuffle=True,drop_last=True)
    input_dim = train_new_tensor.shape[1]

    #Train vae model
    latent_dim = args.latent_dim
    inter_dim = 800 
    noise_rate=0.15
    epochs = 200
    kl_weight=0.00000001
    kl_loss = lambda mu, logvar: 0.5 * torch.sum(torch.exp(logvar) + torch.pow(mu, 2) - 1. - logvar)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE(input_dim, inter_dim, latent_dim,noise_rate)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in tqdm(range(epochs)):
        print(f"Epoch {epoch}")
        model.train()
        train_loss = 0.
        train_num = len(train_loader.dataset)

        for idx, x in enumerate(train_loader):
            batch = batch_size
            recon_x, mu, logvar = model(x)
            recon_criterion = nn.MSELoss()
            recon = recon_criterion(recon_x, x)
            
            kl = kl_loss(mu, logvar)
            loss =recon + kl * kl_weight
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    #Get latent data
    latent_adata=to_latent(model,train_new)
    ctrl_latent,stim_latent=get_latent(adata=train_new,model=model,ctrl_key=args.ctrl_key,stim_key=args.stim_key,cell_type_key=args.cell_type_key,condition_key=args.condition_key)
    ctrl_latent_tensor = Tensor(ctrl_latent)
    stim_latent_tensor = Tensor(stim_latent)
    train_loader_source = torch.utils.data.DataLoader(ctrl_latent_tensor, batch_size=args.batch_size, shuffle=True, **kwargs)
    train_loader = torch.utils.data.DataLoader(stim_latent_tensor, batch_size=args.batch_size, shuffle=True, **kwargs)
    train_source = iter(train_loader_source)

    #Building OT model 
    NUM_NEURON=1024
    N_GENERATOR_ITERS=16
    LR=1e-4
    LAMBDA_CVX=0.1
    LAMBDA_MEAN=0.0
    
    if args.full_quadratic:
        convex_f = Simple_Feedforward_3Layer_ICNN_LastFull_Quadratic(args.latent_dim, NUM_NEURON, args.activation)
        convex_g = Simple_Feedforward_3Layer_ICNN_LastFull_Quadratic(args.latent_dim, NUM_NEURON, args.activation)
    else:
        convex_f = Simple_Feedforward_3Layer_ICNN_LastInp_Quadratic(args.latent_dim, NUM_NEURON, args.activation)
        convex_g = Simple_Feedforward_3Layer_ICNN_LastInp_Quadratic(args.latent_dim, NUM_NEURON, args.activation)
        
    for param_tensor in convex_f.state_dict():
        print(param_tensor, "\t", convex_f.state_dict()[param_tensor].size())

    f_positive_params = []
    for p in list(convex_f.parameters()):
        if hasattr(p, 'be_positive'):
            f_positive_params.append(p)        
        p.data = torch.from_numpy(truncated_normal(p.shape, threshold=1./np.sqrt(p.shape[1] if len(p.shape)>1 else p.shape[0]))).float()
        
    g_positive_params = []
    for p in list(convex_g.parameters()):
        if hasattr(p, 'be_positive'):
            g_positive_params.append(p)       
        p.data = torch.from_numpy(truncated_normal(p.shape, threshold=1./np.sqrt(p.shape[1] if len(p.shape)>1 else p.shape[0]))).float()
       
    if args.cuda:
        convex_f.cuda()
        convex_g.cuda()

    num_parameters = sum([l.nelement() for l in convex_f.parameters()])
    f_positive_constraint_loss = compute_constraint_loss(f_positive_params)
    g_positive_constraint_loss = compute_constraint_loss(g_positive_params)

    if args.optimizer == 'SGD':
        optimizer_f = optim.SGD(convex_f.parameters(), lr=LR, momentum=0.0)
        optimizer_g = optim.SGD(convex_g.parameters(), lr=LR, momentum=0.0)
    if args.optimizer == 'Adam':
        optimizer_f = optim.Adam(convex_f.parameters(), lr=LR, betas=(0.5, 0.99), weight_decay=1e-5)
        optimizer_g = optim.Adam(convex_g.parameters(), lr=LR, betas=(0.5, 0.99), weight_decay=1e-5)

    def train(epoch):
        convex_f.train()
        convex_g.train()
        w_2_loss_value_epoch = 0
        g_OT_loss_value_epoch = 0
        g_Constraint_loss_value_epoch = 0       
        train_loader_source = torch.utils.data.DataLoader(ctrl_latent_tensor, batch_size=args.batch_size, shuffle=True, **kwargs)
        train_source = iter(train_loader_source)
        for batch_idx, real_data in enumerate(train_loader):
            if args.cuda:
                real_data = real_data.cuda()
            real_data = Variable(real_data)

            y = next(train_source)
            y = Variable(y, requires_grad= True)

            if args.cuda:
                y = y.cuda()

            optimizer_f.zero_grad()
            optimizer_g.zero_grad()

            g_OT_loss_val_batch = 0
            g_Constraint_loss_val_batch = 0

            for inner_iter in range(1, N_GENERATOR_ITERS+1):

                optimizer_g.zero_grad()
                g_of_y = convex_g(y).sum()
                grad_g_of_y = torch.autograd.grad(g_of_y, y, create_graph=True)[0]
                f_grad_g_y = convex_f(grad_g_of_y).mean()
                loss_g = f_grad_g_y - torch.dot(grad_g_of_y.reshape(-1), y.reshape(-1)) / y.size(0)
                g_OT_loss_val_batch += loss_g.item()

                if LAMBDA_MEAN > 0:
                    mean_difference_loss = LAMBDA_MEAN * (real_data.mean(0) - grad_g_of_y.mean(0)).pow(2).sum()
                    variance_difference_loss = LAMBDA_MEAN * (real_data.std(0) - grad_g_of_y.std(0)).pow(2).sum()
                    loss_g += mean_difference_loss + variance_difference_loss

                loss_g.backward()

                if LAMBDA_CVX > 0:
                    g_positive_constraint_loss = LAMBDA_CVX*compute_constraint_loss(g_positive_params)
                    g_Constraint_loss_val_batch += g_positive_constraint_loss
                    g_positive_constraint_loss.backward()            

                optimizer_g.step()

                if LAMBDA_CVX == 0:
                    for p in g_positive_params:
                        p.data.copy_(torch.relu(p.data))               
                if inner_iter != N_GENERATOR_ITERS:
                    optimizer_f.zero_grad()

            g_OT_loss_val_batch /= N_GENERATOR_ITERS
            g_Constraint_loss_val_batch /= N_GENERATOR_ITERS
    
            for p in list(convex_f.parameters()):
                p.grad.copy_(-p.grad)

            remaining_f_loss = convex_f(real_data).mean()
            remaining_f_loss.backward()
            optimizer_f.step()

            for p in f_positive_params:
                p.data.copy_(torch.relu(p.data))

            w_2_loss_value_batch = g_OT_loss_val_batch - remaining_f_loss.item() + 0.5*real_data.pow(2).sum(dim=1).mean().item() + 0.5*y.pow(2).sum(dim=1).mean().item()
            w_2_loss_value_epoch += w_2_loss_value_batch

            g_OT_loss_value_epoch += g_OT_loss_val_batch
            g_Constraint_loss_value_epoch += g_Constraint_loss_val_batch
        
        w_2_loss_value_epoch /= len(train_loader)
        g_OT_loss_value_epoch/= len(train_loader)
        g_Constraint_loss_value_epoch /= len(train_loader)

        return w_2_loss_value_epoch, g_OT_loss_value_epoch, g_Constraint_loss_value_epoch
    
    #Train OT model
    total_w_2_epoch_loss_list = []
    total_g_OT_epoch_loss_list = []
    total_g_Constraint_epoch_loss_list = []

    for epoch in tqdm(range(1, args.epochs + 1), desc="OT Training Progress"):

        w_2_loss_value_epoch, g_OT_loss_value_epoch, g_Constraint_loss_value_epoch = train(epoch)

        total_w_2_epoch_loss_list.append(w_2_loss_value_epoch)
        total_g_OT_epoch_loss_list.append(g_OT_loss_value_epoch)
        total_g_Constraint_epoch_loss_list.append(g_Constraint_loss_value_epoch)

        if epoch % args.lr_schedule == 0:
         
            optimizer_g.param_groups[0]['lr'] = optimizer_g.param_groups[0]['lr'] * 0.5
            optimizer_f.param_groups[0]['lr'] = optimizer_f.param_groups[0]['lr'] * 0.5

    #Predict and save
    train= sc.read(args.input_file_path)
    control_=train[((train.obs[args.cell_type_key]=='{}'.format(cell_label))&(train.obs[args.condition_key]==args.ctrl_key))]
    real_=train[((train.obs[args.cell_type_key]=='{}'.format(cell_label))&(train.obs[args.condition_key]==args.stim_key))]

    ctrl_latent=latent_adata[(latent_adata.obs[args.condition_key] == args.ctrl_key)]
    ctrl_latent=ctrl_latent[(ctrl_latent.obs[args.cell_type_key]=='{}'.format(cell_label))]
    ctrl_latent_tensor = Tensor(ctrl_latent.X)
    ctrl_latent_tensor = Variable(ctrl_latent_tensor,requires_grad=True)
    if args.cuda:
        ctrl_latent_tensor=ctrl_latent_tensor.cuda()
    
    g_of_ctrl_latent = convex_g(ctrl_latent_tensor).sum()
    pred_latent = torch.autograd.grad(g_of_ctrl_latent ,ctrl_latent_tensor, create_graph=True)[0]
    pred_latent= pred_latent.cpu().detach().numpy()
    pred_latent_tonsor=Tensor(pred_latent).cuda()       
    pred_data= model.decoder(pred_latent_tonsor)           
    pred_ = sc.AnnData(X=pred_data.cpu().detach().numpy(), obs=control_.obs.copy(),var=control_.var.copy())

    control_.obs[args.condition_key]='{}_Ctrl'.format(cell_label)
    real_.obs[args.condition_key]='{}_Real'.format(cell_label)
    pred_.obs[args.condition_key]='{}_SCREEN'.format(cell_label)
    reconstructed_ = control_.concatenate(pred_, real_)
    reconstructed_.write_h5ad(args.output_file_path+'/SCREEN_{}.h5ad'.format(cell_label))

if __name__ == "__main__":
    main()