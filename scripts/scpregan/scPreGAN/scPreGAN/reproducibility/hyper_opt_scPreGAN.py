from __future__ import print_function
import os
from pathlib import Path
from functools import partial
import random
import json
import numpy as np
import torch
import torch.nn as nn
import multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.cuda import is_available as cuda_is_available
from torch import Tensor, FloatTensor
from torch.utils.data import random_split
from torch import autograd
from torch import mean, exp, unique, cat, isnan
from torch import norm as torch_norm

import torch.nn.functional as F
import scanpy as sc
import anndata
from scipy import sparse
from ray import tune
from ray.tune import CLIReporter
from util import load_anndata

from model.Discriminator import Discriminator
from model.Generator import Generator
from model.Encoder import Encoder

import warnings

warnings.filterwarnings('ignore')


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform(m.weight, 1e-2)
        m.bias.data.fill_(0.01)


def create_model(n_features, z_dim, min_hidden_size, use_cuda, use_sn):
    if use_sn:
        D_A = Discriminator(n_features=n_features, min_hidden_size=min_hidden_size, out_dim=1)
        D_B = Discriminator(n_features=n_features, min_hidden_size=min_hidden_size, out_dim=1)
    else:
        D_A = Discriminator(n_features=n_features, min_hidden_size=min_hidden_size, out_dim=1)
        D_B = Discriminator(n_features=n_features, min_hidden_size=min_hidden_size, out_dim=1)

    G_A = Generator(z_dim=z_dim, min_hidden_size=min_hidden_size, n_features=n_features)
    G_B = Generator(z_dim=z_dim, min_hidden_size=min_hidden_size, n_features=n_features)
    E = Encoder(n_features=n_features, min_hidden_size=min_hidden_size, z_dim=z_dim)

    # print("Encoder model:")
    # print(E)
    init_weights(E)
    # print("GeneratorA model:")
    # print(G_A)
    init_weights(G_A)
    init_weights(G_B)
    # print("disc_model:")
    # print(D_A)
    init_weights(D_A)
    init_weights(D_B)

    if use_cuda and torch.cuda.is_available():
        D_A = D_A.cuda()
        D_B = D_B.cuda()
        G_A = G_A.cuda()
        G_B = G_B.cuda()
        E = E.cuda()

    return E, G_A, G_B, D_A, D_B


def calc_gradient_penalty(netD, real_data, fake_data, batch_size, use_cuda, lambta, use_wgan_div, k=2, p=6):
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    if use_wgan_div:
        gradient_penalty = torch.pow(gradients.norm(2, dim=1), p).mean() * k
    else:
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambta
    return gradient_penalty


def train_BranchGAN(config, opt):
    if opt['manual_seed'] is None:
        opt['manual_seed'] = random.randint(1, 10000)
    print("Random Seed: ", opt['manual_seed'])
    random.seed(opt['manual_seed'])
    torch.manual_seed(opt['manual_seed'])
    if opt['cuda']:
        torch.cuda.manual_seed_all(opt['manual_seed'])
    A_pd, A_celltype_ohe_pd, B_pd, B_celltype_ohe_pd = load_anndata(path=opt['dataPath'],
                                                                    condition_key=opt['condition_key'],
                                                                    condition=opt['condition'],
                                                                    cell_type_key=opt['cell_type_key']
                                                                    )
    A_tensor = Tensor(np.array(A_pd))
    B_tensor = Tensor(np.array(B_pd))

    if opt['cuda'] and torch.cuda.is_available():
        A_tensor = A_tensor.cuda()
        B_tensor = B_tensor.cuda()

    A_Dataset = torch.utils.data.TensorDataset(A_tensor)
    B_Dataset = torch.utils.data.TensorDataset(B_tensor)

    A_pd_val, A_celltype_ohe_pd_val, B_pd_val, B_celltype_ohe_pd_val = load_anndata(path=opt['valid_dataPath'],
                                                                                    condition_key=opt['condition_key'],
                                                                                    condition=opt['condition'],
                                                                                    cell_type_key=opt['cell_type_key']
                                                                                    )
    print(f"use validation dataset, lenth of A: {A_pd_val.shape}, lenth of B: {B_pd_val.shape}")
    A_tensor_val = Tensor(np.array(A_pd_val))
    B_tensor_val = Tensor(np.array(B_pd_val))

    if opt['cuda'] and torch.cuda.is_available():
        A_tensor_val = A_tensor_val.cuda()
        B_tensor_val = B_tensor_val.cuda()

    A_Dataset_val = torch.utils.data.TensorDataset(A_tensor_val)
    B_Dataset_val = torch.utils.data.TensorDataset(B_tensor_val)

    A_train_loader = torch.utils.data.DataLoader(dataset=A_Dataset,
                                                 batch_size=int(config['batch_size']),
                                                 shuffle=True,
                                                 drop_last=True)

    B_train_loader = torch.utils.data.DataLoader(dataset=B_Dataset,
                                                 batch_size=int(config['batch_size']),
                                                 shuffle=True,
                                                 drop_last=True)
    A_valid_loader = torch.utils.data.DataLoader(dataset=A_Dataset_val,
                                                 batch_size=int(config['batch_size']),
                                                 shuffle=True,
                                                 drop_last=True)
    B_valid_loader = torch.utils.data.DataLoader(dataset=B_Dataset_val,
                                                 batch_size=int(config['batch_size']),
                                                 shuffle=True,
                                                 drop_last=True)

    opt['n_features'] = A_pd.shape[1]
    A_train_loader_it = iter(A_train_loader)
    B_train_loader_it = iter(B_train_loader)

    E, G_A, G_B, D_A, D_B = create_model(n_features=opt['n_features'],
                                         z_dim=config['z_dim'],
                                         min_hidden_size=config['min_hidden_size'],
                                         use_cuda=opt['cuda'], use_sn=opt['use_sn'])

    recon_criterion = nn.MSELoss()
    encoding_criterion = nn.MSELoss()

    optimizerD_A = torch.optim.Adam(D_A.parameters(), lr=config['lr_disc'], betas=(0.5, 0.9))
    optimizerD_B = torch.optim.Adam(D_B.parameters(), lr=config['lr_disc'], betas=(0.5, 0.9))
    optimizerG_A = torch.optim.Adam(G_A.parameters(), lr=config['lr_g'], betas=(0.5, 0.9))
    optimizerG_B = torch.optim.Adam(G_B.parameters(), lr=config['lr_g'], betas=(0.5, 0.9))
    optimizerE = torch.optim.Adam(E.parameters(), lr=config['lr_e'])

    ones = torch.ones(config['batch_size'], 1)
    print('ones type:', type(ones))
    zeros = torch.zeros(config['batch_size'], 1)

    if opt['cuda'] and cuda_is_available():
        ones = ones.cuda()
        zeros = zeros.cuda()

    D_A.train()
    D_B.train()
    G_A.train()
    G_B.train()
    E.train()

    D_A_loss = 0.0
    D_B_loss = 0.0

    iteration = 0
    for iteration in range(1, config['niter'] + 1):
        if iteration % 10000 == 0:
            for param_group in optimizerD_A.param_groups:
                param_group['lr'] = param_group['lr'] * 0.9
            for param_group in optimizerD_B.param_groups:
                param_group['lr'] = param_group['lr'] * 0.9
            for param_group in optimizerG_A.param_groups:
                param_group['lr'] = param_group['lr'] * 0.9
            for param_group in optimizerG_B.param_groups:
                param_group['lr'] = param_group['lr'] * 0.9
            for param_group in optimizerE.param_groups:
                param_group['lr'] = param_group['lr'] * 0.9

        for count in range(0, 5):
            try:
                real_A = A_train_loader_it.next()[0]
                real_B = B_train_loader_it.next()[0]
            except StopIteration:
                A_train_loader_it, B_train_loader_it = iter(A_train_loader), iter(B_train_loader)
                real_A = A_train_loader_it.next()[0]
                real_B = B_train_loader_it.next()[0]

            if (opt['cuda']) and cuda_is_available():
                real_A = real_A.cuda()
                real_B = real_B.cuda()

            D_A.zero_grad()
            D_B.zero_grad()

            out_A = D_A(real_A)
            out_B = D_B(real_B)

            real_A_z = E(real_A)
            AB = G_B(real_A_z)
            real_B_z = E(real_B)
            BA = G_A(real_B_z)

            out_BA = D_A(BA.detach())
            out_AB = D_B(AB.detach())

            D_A_gradient_penalty = calc_gradient_penalty(D_A, real_A.detach(), BA.detach(),
                                                         batch_size=config['batch_size'], use_cuda=opt['cuda'],
                                                         lambta=config['lambta_gp'], use_wgan_div=opt['use_wgan_div'])
            D_B_gradient_penalty = calc_gradient_penalty(D_B, real_B.detach(), AB.detach(),
                                                         batch_size=config['batch_size'], use_cuda=opt['cuda'],
                                                         lambta=config['lambta_gp'], use_wgan_div=opt['use_wgan_div'])

            if opt['gan_loss'] == 'vanilla':
                D_A_real_loss = F.binary_cross_entropy_with_logits(out_A, ones)
                D_B_real_loss = F.binary_cross_entropy_with_logits(out_B, ones)
                D_A_fake_loss = F.binary_cross_entropy_with_logits(out_BA, zeros)
                D_B_fake_loss = F.binary_cross_entropy_with_logits(out_AB, zeros)
                D_A_loss = D_A_real_loss + D_A_fake_loss
                D_B_loss = D_B_real_loss + D_B_fake_loss
            elif opt['gan_loss'] == 'lsgan':
                D_A_real_loss = F.mse_loss(out_A, ones)
                D_B_real_loss = F.mse_loss(out_B, ones)
                D_A_fake_loss = F.mse_loss(out_BA, zeros)
                D_B_fake_loss = F.mse_loss(out_AB, zeros)
                D_A_loss = D_A_real_loss + D_A_fake_loss
                D_B_loss = D_B_real_loss + D_B_fake_loss
            elif opt['gan_loss'] == 'wgan':
                D_A_real_loss = -torch.mean(out_A)
                D_B_real_loss = -torch.mean(out_B)
                D_A_fake_loss = torch.mean(out_BA)
                D_B_fake_loss = torch.mean(out_AB)
                D_A_loss = D_A_real_loss + D_A_fake_loss + D_A_gradient_penalty
                D_B_loss = D_B_real_loss + D_B_fake_loss + D_B_gradient_penalty
            else:
                NotImplementedError("not implement loss")

            D_A_loss.backward()
            D_B_loss.backward()
            optimizerD_A.step()
            optimizerD_B.step()

        try:
            real_A = A_train_loader_it.next()[0]
            real_B = B_train_loader_it.next()[0]
        except StopIteration:
            A_train_loader_it, B_train_loader_it = iter(A_train_loader), iter(B_train_loader)
            real_A = A_train_loader_it.next()[0]
            real_B = B_train_loader_it.next()[0]

        if (opt['cuda']) and cuda_is_available():
            real_A = real_A.cuda()
            real_B = real_B.cuda()

        G_A.zero_grad()
        G_B.zero_grad()
        E.zero_grad()

        real_A_z = E(real_A)
        AA = G_A(real_A_z)
        AB = G_B(real_A_z)

        AA_z = E(AA)
        AB_z = E(AB)
        ABA = G_A(AB_z)

        real_B_z = E(real_B)
        BA = G_A(real_B_z)
        BB = G_B(real_B_z)
        BA_z = E(BA)
        BB_z = E(BB)
        BAB = G_B(BA_z)

        out_AA = D_A(AA)
        out_AB = D_B(AB)
        out_BA = D_A(BA)
        out_BB = D_B(BB)
        out_ABA = D_A(ABA)
        out_BAB = D_B(BAB)

        if opt['gan_loss'] == 'vanilla':
            G_AA_adv_loss = F.binary_cross_entropy_with_logits(out_AA, ones)
            G_BA_adv_loss = F.binary_cross_entropy_with_logits(out_BA, ones)
            G_ABA_adv_loss = F.binary_cross_entropy_with_logits(out_ABA, ones)

            G_BB_adv_loss = F.binary_cross_entropy_with_logits(out_BB, ones)
            G_AB_adv_loss = F.binary_cross_entropy_with_logits(out_AB, ones)
            G_BAB_adv_loss = F.binary_cross_entropy_with_logits(out_BAB, ones)
        elif opt['gan_loss'] == 'lsgan':
            G_AA_adv_loss = F.mse_loss(out_AA, ones)
            G_BA_adv_loss = F.mse_loss(out_BA, ones)
            G_ABA_adv_loss = F.mse_loss(out_ABA, ones)

            G_BB_adv_loss = F.mse_loss(out_BB, ones)
            G_AB_adv_loss = F.mse_loss(out_AB, ones)
            G_BAB_adv_loss = F.mse_loss(out_BAB, ones)
        elif opt['gan_loss'] == 'wgan':
            G_AA_adv_loss = -torch.mean(out_AA)
            G_BA_adv_loss = -torch.mean(out_BA)
            G_ABA_adv_loss = -torch.mean(out_ABA)

            G_BB_adv_loss = -torch.mean(out_BB)
            G_AB_adv_loss = -torch.mean(out_AB)
            G_BAB_adv_loss = -torch.mean(out_BAB)
        else:
            NotImplementedError("not implement loss")
        G_A_adv_loss = G_AA_adv_loss + G_BA_adv_loss + G_ABA_adv_loss
        G_B_adv_loss = G_BB_adv_loss + G_AB_adv_loss + G_BAB_adv_loss
        adv_loss = (G_A_adv_loss + G_B_adv_loss) * config['lambda_adv']

        # reconstruction loss
        l_rec_AA = recon_criterion(AA, real_A)
        l_rec_BB = recon_criterion(BB, real_B)

        recon_loss = (l_rec_AA + l_rec_BB) * config['lambda_recon']

        # encoding loss
        tmp_real_A_z = real_A_z.detach()
        tmp_real_B_z = real_B_z.detach()
        l_encoding_AA = encoding_criterion(AA_z, tmp_real_A_z)
        l_encoding_BB = encoding_criterion(BB_z, tmp_real_B_z)
        l_encoding_BA = encoding_criterion(BA_z, tmp_real_B_z)
        l_encoding_AB = encoding_criterion(AB_z, tmp_real_A_z)

        encoding_loss = (l_encoding_AA + l_encoding_BB + l_encoding_BA + l_encoding_AB) * config[
            'lambda_encoding']

        E_paras = cat([x.view(-1) for x in E.parameters()])
        G_A_paras = cat([x.view(-1) for x in G_A.parameters()])
        G_B_paras = cat([x.view(-1) for x in G_B.parameters()])
        E_regularization = torch_norm(E_paras, 1)
        G_A_regularization = torch_norm(G_A_paras, 1)
        G_B_regularization = torch_norm(G_B_paras, 1)
        l1_regularization = (E_regularization + G_A_regularization + G_B_regularization) * config["lambda_l1_reg"]

        G_loss = adv_loss + recon_loss + encoding_loss + l1_regularization

        # backward
        G_loss.backward()

        # step
        optimizerG_A.step()
        optimizerG_B.step()
        optimizerE.step()

        tune.report(D_A_loss=(D_A_loss.item()),
                    D_B_loss=(D_B_loss.item()),
                    adv_loss=(adv_loss.item()),
                    recon_loss=(recon_loss.item()),
                    encoding_loss=(encoding_loss.item()),
                    G_loss=(G_loss.item())
                    )

        if iteration % 300 == 0:
            print(
                '[%d/%d] D_A_loss: %.4f  D_B_loss: %.4f adv_loss: %.4f  recon_loss: %.4f encoding_loss: %.4f G_loss: %.4f'
                % (iteration, config['niter'], D_A_loss.item(), D_B_loss.item(), adv_loss.item(), recon_loss.item(),
                   encoding_loss.item(), G_loss.item()))
            D_A_loss_val = 0.0
            D_B_loss_val = 0.0
            adv_loss_val = 0.0
            recon_loss_val = 0.0
            encoding_loss_val = 0.0
            G_loss_val = 0.0

            counter = 0
            sparcity_cellA_val = 0.0
            sparcity_cellB_val = 0.0
            sparcity_G_A_val = 0.0
            sparcity_G_B_val = 0.0

            A_valid_loader_it = iter(A_valid_loader)
            B_valid_loader_it = iter(B_valid_loader)

            max_length = max(len(A_valid_loader), len(B_valid_loader))
            with torch.no_grad():
                for iteration_val in range(1, max_length):
                    try:
                        cellA_val = A_valid_loader_it.next()[0]
                        cellB_val = B_valid_loader_it.next()[0]
                    except StopIteration:
                        A_valid_loader_it, B_valid_loader_it = iter(A_valid_loader), iter(B_valid_loader)
                        cellA_val = A_valid_loader_it.next()[0]
                        cellB_val = B_valid_loader_it.next()[0]

                    counter += 1

                    real_A_z = E(cellA_val)
                    real_B_z = E(cellB_val)
                    AB = G_B(real_A_z)
                    BA = G_A(real_B_z)
                    AA = G_A(real_A_z)
                    BB = G_B(real_B_z)
                    AA_z = E(AA)
                    BB_z = E(BB)
                    AB_z = E(AB)
                    BA_z = E(BA)

                    ABA = G_A(AB_z)
                    BAB = G_B(BA_z)

                    outA_val = D_A(cellA_val)
                    outB_val = D_B(cellB_val)
                    out_AA = D_A(AA)
                    out_BB = D_B(BB)
                    out_AB = D_B(AB)
                    out_BA = D_A(BA)
                    out_ABA = D_A(ABA)
                    out_BAB = D_B(BAB)

                    if opt['gan_loss'] == 'vanilla':
                        D_A_real_loss_val = F.binary_cross_entropy_with_logits(outA_val, ones)
                        D_B_real_loss_val = F.binary_cross_entropy_with_logits(outB_val, ones)
                        D_A_fake_loss_val = F.binary_cross_entropy_with_logits(out_BA, zeros)
                        D_B_fake_loss_val = F.binary_cross_entropy_with_logits(out_AB, zeros)
                    elif opt['gan_loss'] == 'lsgan':
                        D_A_real_loss_val = F.mse_loss(outA_val, ones)
                        D_B_real_loss_val = F.mse_loss(outB_val, ones)
                        D_A_fake_loss_val = F.mse_loss(out_BA, zeros)
                        D_B_fake_loss_val = F.mse_loss(out_AB, zeros)
                    elif opt['gan_loss'] == 'wgan':
                        D_A_real_loss_val = -torch.mean(outA_val)
                        D_B_real_loss_val = -torch.mean(outB_val)
                        D_A_fake_loss_val = torch.mean(out_BA)
                        D_B_fake_loss_val = torch.mean(out_AB)

                    else:
                        NotImplementedError("not implement loss")
                    D_A_loss_val += (D_A_real_loss_val + D_A_fake_loss_val).item()
                    D_B_loss_val += (D_B_real_loss_val + D_B_fake_loss_val).item()

                    if opt['gan_loss'] == 'vanilla':
                        G_AA_adv_loss_val = F.binary_cross_entropy_with_logits(out_AA, ones)
                        G_BA_adv_loss_val = F.binary_cross_entropy_with_logits(out_BA, ones)
                        G_ABA_adv_loss_val = F.binary_cross_entropy_with_logits(out_ABA, ones)

                        G_BB_adv_loss_val = F.binary_cross_entropy_with_logits(out_BB, ones)
                        G_AB_adv_loss_val = F.binary_cross_entropy_with_logits(out_AB, ones)
                        G_BAB_adv_loss_val = F.binary_cross_entropy_with_logits(out_BAB, ones)

                    elif opt['gan_loss'] == 'lsgan':
                        G_AA_adv_loss_val = F.mse_loss(out_AA, ones)
                        G_BA_adv_loss_val = F.mse_loss(out_BA, ones)
                        G_ABA_adv_loss_val = F.mse_loss(out_ABA, ones)

                        G_BB_adv_loss_val = F.mse_loss(out_BB, ones)
                        G_AB_adv_loss_val = F.mse_loss(out_AB, ones)
                        G_BAB_adv_loss_val = F.mse_loss(out_BAB, ones)

                    elif opt['gan_loss'] == 'wgan':
                        G_AA_adv_loss_val = -torch.mean(out_AA)
                        G_BA_adv_loss_val = -torch.mean(out_BA)
                        G_ABA_adv_loss_val = -torch.mean(out_ABA)

                        G_BB_adv_loss_val = -torch.mean(out_BB)
                        G_AB_adv_loss_val = -torch.mean(out_AB)
                        G_BAB_adv_loss_val = -torch.mean(out_BAB)
                    else:
                        NotImplementedError("not implement loss")
                    G_A_adv_loss_val = G_AA_adv_loss_val + G_BA_adv_loss_val + G_ABA_adv_loss_val
                    G_B_adv_loss_val = G_BB_adv_loss_val + G_AB_adv_loss_val + G_BAB_adv_loss_val
                    adv_loss_val += (G_A_adv_loss_val + G_B_adv_loss_val).item() * config['lambda_adv']

                    # reconstruction loss
                    l_rec_AA_val = recon_criterion(AA, cellA_val)
                    l_rec_BB_val = recon_criterion(BB, cellB_val)
                    recon_loss_val += (l_rec_AA_val + l_rec_BB_val).item() * config['lambda_recon']

                    # encoding loss
                    l_encoding_AA_val = encoding_criterion(AA_z, real_A_z)
                    l_encoding_BB_val = encoding_criterion(BB_z, real_B_z)
                    l_encoding_BA_val = encoding_criterion(BA_z, real_B_z)
                    l_encoding_AB_val = encoding_criterion(AB_z, real_A_z)
                    encoding_loss_val += (
                                                 l_encoding_AA_val + l_encoding_BB_val + l_encoding_BA_val + l_encoding_AB_val).item() * \
                                         config['lambda_encoding']
                    G_loss_val += adv_loss_val + recon_loss_val + encoding_loss_val

                    # real data sparsity
                    cellA_val_np = cellA_val.cpu().detach().numpy()
                    cellB_val_np = cellB_val.cpu().detach().numpy()

                    cellA_val_zero = cellA_val_np[cellA_val_np <= 0]
                    cellB_val_zero = cellB_val_np[cellB_val_np <= 0]

                    sparcity_cellA_val += cellA_val_zero.shape[0] / (cellA_val_np.shape[0] * cellA_val_np.shape[1])
                    sparcity_cellB_val += cellB_val_zero.shape[0] / (cellB_val_np.shape[0] * cellB_val_np.shape[1])
                    # fake data sparsity
                    AB_val_np = AB.cpu().detach().numpy()
                    BA_val_np = BA.cpu().detach().numpy()
                    AA_val_np = AA.cpu().detach().numpy()
                    BB_val_np = BB.cpu().detach().numpy()

                    AB_val_zero = AB_val_np[AB_val_np <= 0]
                    BA_val_zero = BA_val_np[BA_val_np <= 0]
                    AA_val_zero = AA_val_np[AA_val_np <= 0]
                    BB_val_zero = BB_val_np[BB_val_np <= 0]

                    sparcity_G_A_val += (BA_val_zero.shape[0] / (BA_val_np.shape[0] * BA_val_np.shape[1])) + \
                                        (AA_val_zero.shape[0] / (AA_val_np.shape[0] * AA_val_np.shape[1]))
                    sparcity_G_B_val += (AB_val_zero.shape[0] / (AB_val_np.shape[0] * AB_val_np.shape[1])) + \
                                        (BB_val_zero.shape[0] / (BB_val_np.shape[0] * BB_val_np.shape[1]))

            print(
                '[%d/%d] adv_loss_val: %.4f  recon_loss_val: %.4f encoding_loss_val: %.4f  G_loss: %.4f D_A_loss_val: %.4f D_B_loss_val: %.4f'
                % (iteration, config['niter'], adv_loss_val / counter, recon_loss_val / counter,
                   encoding_loss_val / counter, G_loss / counter, D_A_loss_val / counter, D_B_loss_val / counter))

            sparcity_G_A_diff = abs(1 - (sparcity_G_A_val / sparcity_cellA_val))
            sparcity_G_B_diff = abs(1 - (sparcity_G_B_val / sparcity_cellB_val))
            tune.report(adv_loss_val=(adv_loss_val / counter),
                        recon_loss_val=(recon_loss_val / counter),
                        encoding_loss_val=(encoding_loss_val / counter),
                        G_loss=(G_loss / counter),
                        D_A_loss_val=(D_A_loss_val / counter),
                        D_B_loss_val=(D_B_loss_val / counter),
                        sparcity_diff=(sparcity_G_A_diff + sparcity_G_B_diff) * 0.5)
    with tune.checkpoint_dir(iteration) as checkpoint_dir:
        path = os.path.join(checkpoint_dir, "checkpoint_E")
        torch.save((E.state_dict(), optimizerE.state_dict()), path)
    with tune.checkpoint_dir(iteration) as checkpoint_dir:
        path = os.path.join(checkpoint_dir, "checkpoint_G_A")
        torch.save((G_A.state_dict(), optimizerG_A.state_dict()), path)
    with tune.checkpoint_dir(iteration) as checkpoint_dir:
        path = os.path.join(checkpoint_dir, "checkpoint_G_B")
        torch.save((G_B.state_dict(), optimizerG_B.state_dict()), path)
    with tune.checkpoint_dir(iteration) as checkpoint_dir:
        path = os.path.join(checkpoint_dir, "checkpoint_D_A")
        torch.save((D_A.state_dict(), optimizerD_A.state_dict()), path)
    with tune.checkpoint_dir(iteration) as checkpoint_dir:
        path = os.path.join(checkpoint_dir, "checkpoint_D_B")
        torch.save((D_B.state_dict(), optimizerD_B.state_dict()), path)
    print("Finished Training")


def test(opt, E, G_A, G_B, D_A, D_B, test_dataPath, model_name):
    print(f"++++++++++++++++++++++{model_name}++++++++++++++++++++++++++++")
    test_adata = sc.read(test_dataPath)
    print("shape of test adata: ", test_adata.shape)
    A_test_adata = test_adata[test_adata.obs[opt['condition_key']] == opt['condition']['control']]
    B_test_adata = test_adata[test_adata.obs[opt['condition_key']] == opt['condition']['case']]
    if sparse.issparse(test_adata.X):
        expr_testA = A_test_adata.X.A
        expr_testB = B_test_adata.X.A
    else:
        expr_testA = A_test_adata.X
        expr_testB = B_test_adata.X
    expr_testA_tensor = Tensor(expr_testA)
    expr_testB_tensor = Tensor(expr_testB)

    if opt['cuda'] and cuda_is_available():
        expr_testA_tensor = expr_testA_tensor.cuda()
        expr_testB_tensor = expr_testB_tensor.cuda()
    A_z = E(expr_testA_tensor)
    B_z = E(expr_testB_tensor)
    AB = G_B(A_z)
    BA = G_A(B_z)
    AB_z = E(AB)
    BA_z = E(BA)
    AB_adata = anndata.AnnData(X=AB.cpu().detach().numpy(),
                               obs={opt['condition_key']: ["transfer_AtoB"] * len(AB),
                                    opt['cell_type_key']: A_test_adata.obs[opt['cell_type_key']].tolist()})
    AB_adata.var_names = A_test_adata.var_names

    BA_adata = anndata.AnnData(X=BA.cpu().detach().numpy(),
                               obs={opt['condition_key']: ["transfer_BtoA"] * len(BA),
                                    opt['cell_type_key']: B_test_adata.obs[opt['cell_type_key']].tolist()})
    BA_adata.var_names = B_test_adata.var_names

    AA = G_A(A_z)
    BB = G_B(B_z)

    AA_z = E(AA)
    BB_z = E(BB)

    test_A_z_adata = anndata.AnnData(X=A_z.cpu().detach().numpy(),
                                     obs={opt['condition_key']: ["test_A_z"] * len(A_z),
                                          opt['cell_type_key']: A_test_adata.obs[opt['cell_type_key']].tolist()})
    test_B_z_adata = anndata.AnnData(X=B_z.cpu().detach().numpy(),
                                     obs={opt['condition_key']: ["test_B_z"] * len(B_z),
                                          opt['cell_type_key']: B_test_adata.obs[opt['cell_type_key']].tolist()})
    AA_z_adata = anndata.AnnData(X=AA_z.cpu().detach().numpy(),
                                 obs={opt['condition_key']: ["AA_z"] * len(AA_z),
                                      opt['cell_type_key']: A_test_adata.obs[opt['cell_type_key']].tolist()}
                                 )
    BB_z_adata = anndata.AnnData(X=BB_z.cpu().detach().numpy(),
                                 obs={opt['condition_key']: ["BB_z"] * len(BB_z),
                                      opt['cell_type_key']: B_test_adata.obs[opt['cell_type_key']].tolist()}
                                 )

    AB_z_adata = anndata.AnnData(X=AB_z.cpu().detach().numpy(),
                                 obs={opt['condition_key']: ["AB_z"] * len(AB_z),
                                      opt['cell_type_key']: A_test_adata.obs[opt['cell_type_key']].tolist()}
                                 )
    BA_z_adata = anndata.AnnData(X=BA_z.cpu().detach().numpy(),
                                 obs={opt['condition_key']: ["BA_z"] * len(BA_z),
                                      opt['cell_type_key']: B_test_adata.obs[opt['cell_type_key']].tolist()}
                                 )

    AA_adata = anndata.AnnData(X=AA.cpu().detach().numpy(),
                               obs={opt['condition_key']: ["recon_AtoA"] * len(AA),
                                    opt['cell_type_key']: A_test_adata.obs[opt['cell_type_key']].tolist()})
    AA_adata.var_names = A_test_adata.var_names
    BB_adata = anndata.AnnData(X=BB.cpu().detach().numpy(),
                               obs={opt['condition_key']: ["recon_BtoB"] * len(BB),
                                    opt['cell_type_key']: B_test_adata.obs[opt['cell_type_key']].tolist()})
    BB_adata.var_names = B_test_adata.var_names

    # ============umap plot=========================
    test_z_adata = test_A_z_adata.concatenate(test_B_z_adata)
    sc.pp.neighbors(test_z_adata)
    sc.tl.umap(test_z_adata)
    sc.pl.umap(test_z_adata, color=[opt['cell_type_key'], opt['condition_key']],
               wspace=0.4,
               legend_fontsize=14,
               save=f"test_z_{opt['data_name']}_{model_name}.pdf",
               show=False,
               frameon=False)

    z_all_adata = test_A_z_adata.concatenate(test_B_z_adata, AA_z_adata, BB_z_adata, AB_z_adata, BA_z_adata)
    sc.pp.neighbors(z_all_adata)
    sc.tl.umap(z_all_adata)
    sc.pl.umap(z_all_adata,
               color=[opt['cell_type_key'], opt['condition_key']],
               wspace=0.4,
               legend_fontsize=14,
               save=f"_z_all_adata_{opt['data_name']}_{model_name}.pdf",
               show=False,
               frameon=False)

    all_adata = A_test_adata.concatenate(B_test_adata,
                                         AB_adata, BA_adata,
                                         AA_adata, BB_adata)
    sc.pp.neighbors(all_adata)
    sc.tl.umap(all_adata)
    sc.pl.umap(all_adata,
               color=[opt['condition_key'], opt['cell_type_key']],
               wspace=0.4,
               legend_fontsize=14,
               save=f"_all_adata_{opt['data_name']}_{model_name}.pdf",
               show=False,
               frameon=False)

    AB_stim_control_adata = A_test_adata.concatenate(B_test_adata, AB_adata)
    sc.pp.neighbors(AB_stim_control_adata)
    sc.tl.umap(AB_stim_control_adata)
    sc.pl.umap(AB_stim_control_adata,
               color=[opt['condition_key'], opt['cell_type_key']],
               wspace=0.4,
               legend_fontsize=14,
               save=f"_AB_stim_control_adata_{opt['data_name']}_{model_name}.pdf",
               show=False,
               frameon=False)

    BA_stim_control_adata = A_test_adata.concatenate(B_test_adata, BA_adata)
    sc.pp.neighbors(BA_stim_control_adata)
    sc.tl.umap(BA_stim_control_adata)
    sc.pl.umap(BA_stim_control_adata,
               color=[opt['condition_key'], opt['cell_type_key']],
               wspace=0.4,
               legend_fontsize=14,
               save=f"_BA_stim_control_adata_{opt['data_name']}_{model_name}.pdf",
               show=False,
               frameon=False)

    # ============DEGs====================
    cell_types = test_adata.obs[opt['cell_type_key']].unique().tolist()
    total = 0
    DEG_path = f'{opt["outf"]}/DEGs'
    if not os.path.exists(DEG_path):
        os.makedirs(DEG_path)
    for t in cell_types:
        t_A_test_adata = A_test_adata[A_test_adata.obs[opt['cell_type_key']] == t]
        t_B_test_adata = B_test_adata[B_test_adata.obs[opt['cell_type_key']] == t]
        t_AB_adata = AB_adata[AB_adata.obs[opt['cell_type_key']] == t]
        t_BA_adata = BA_adata[BA_adata.obs[opt['cell_type_key']] == t]
        minCells = min(t_A_test_adata.shape[0], t_B_test_adata.shape[0])
        t_A_test_adata = t_A_test_adata[0:minCells, ]
        t_B_test_adata = t_B_test_adata[0:minCells, ]
        t_AB_adata = t_AB_adata[0:minCells, ]
        t_BA_adata = t_BA_adata[0:minCells, ]
        t_test_adata = t_A_test_adata.concatenate(t_B_test_adata)
        sc.tl.rank_genes_groups(t_test_adata, groupby=opt['condition_key'],
                                reference=opt['condition']['control'], method='wilcoxon')
        sc.pl.rank_genes_groups(t_test_adata, n_genes=25, sharey=False)
        t_test_BtoA_DEGs = (t_test_adata.uns['rank_genes_groups']['names'][0:99]).tolist()
        t_AB_and_control_adata = t_A_test_adata.concatenate(t_AB_adata)
        sc.tl.rank_genes_groups(t_AB_and_control_adata, groupby=opt['condition_key'],
                                reference=opt['condition']['control'], method='wilcoxon')
        sc.pl.rank_genes_groups(t_AB_and_control_adata, n_genes=25, sharey=False)
        t_AB_and_control_DEGs = (t_AB_and_control_adata.uns['rank_genes_groups']['names'][0:99]).tolist()
        t_BtoA_intersec_DEGs = list(set(t_test_BtoA_DEGs).intersection(set(t_AB_and_control_DEGs)))
        t_BtoA = {
            f'{t}_BtoA_intersec_length': len(t_BtoA_intersec_DEGs),
            f'{t}_test_BtoA_DEGs': t_test_BtoA_DEGs,
            f'{t}_transfer_BtoA_DEGs': t_AB_and_control_DEGs,
            f'{t}_BtoA_intersec_DEGs': t_BtoA_intersec_DEGs
        }
        with open(Path(opt['outf']) / f'DEGs/{model_name}_{t}_BtoA.json', 'w') as file:
            json.dump(t_BtoA, file, indent=4)
        print("===============DEGs========================")
        print(
            f"the length of same DEGS between transfered(AB, control) and real(case, control) for cell type {t}: {len(t_BtoA_intersec_DEGs)} ")
        total += len(t_BtoA_intersec_DEGs)
    print(f"total same DEGs: {total}")


def main(num_samples=10, gpus_per_trial=0):
    opt = {
        'cuda': True,
        'dataPath': '/home/wxj/scPreGAN/datasets/train_pbmc.h5ad',
        'valid_dataPath': '/home/wxj/scPreGAN/datasets/valid_pbmc.h5ad',
        'checkpoint_dir': None,
        'condition_key': 'condition',
        'condition': {"case": "stimulated", "control": "control"},
        'cell_type_key': 'cell_type',
        'manual_seed': 3060,
        'data_name': 'pbmc',
        'model_name': 'sn_lsgan',
        'outf': '/home/wxj/scPreGAN/datasets/pbmc_opt',
        'test_path': '/home/wxj/scPreGAN/datasets/valid_pbmc.h5ad',
        'use_sn': True,
        'use_wgan_div': True,
        'gan_loss': 'wgan'

    }

    if cuda_is_available():
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass
        cudnn.benchmark = True

    config = {
        "lr_disc": tune.choice([0.01, 0.001, 0.0001]),
        "lr_e": tune.choice([0.01, 0.001, 0.0001, 0.00005]),
        "lr_g": tune.choice([0.01, 0.001, 0.0001, 0.00005]),
        "lambda_adv": tune.choice([1, 0.1, 0.01, 0.001]),
        "lambda_recon": tune.choice([1, 0.1, 0.01, 0.001]),
        "lambda_encoding": tune.choice([0.1, 0.01, 0.001, 0.0001]),
        "lambta_gp": tune.choice([1, 0.1, 0.01, 0.001, 0.0001]),
        "lambda_l1_reg": tune.choice([0]),
        "min_hidden_size": tune.choice([256, 128, 64]),
        "batch_size": tune.choice([64, 256, 128]),
        "z_dim": tune.choice([16, 32, 64, 128]),
        "niter": tune.choice([5000, 10000, 20000, 30000, 40000])

    }
    if not os.path.exists(opt['outf']):
        os.makedirs(opt['outf'])
    reporter = CLIReporter(metric_columns=["D_A_loss", "D_B_loss", "adv_loss", "recon_loss", "encoding_loss", "G_loss",
                                           "adv_loss_val", "recon_loss_val", "encoding_loss_val", "G_loss",
                                           "D_A_loss_val", "D_B_loss_val",
                                           "sparcity_diff", "training_iteration"])

    result = tune.run(
        partial(train_BranchGAN, opt=opt),
        name="hyper_scPreGAN",
        resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        progress_reporter=reporter)

    A_pd, A_celltype_ohe_pd, B_pd, B_celltype_ohe_pd = load_anndata(path=opt['dataPath'],
                                                                    condition_key=opt['condition_key'],
                                                                    condition=opt['condition'],
                                                                    cell_type_key=opt['cell_type_key'])
    n_features = A_pd.shape[1]

    for i in range(0, num_samples):
        print("============the ", i, "th trial:========================")
        print(result.trials[i].config)
        trial_i = result.trials[i]

        E, G_A, G_B, D_A, D_B = create_model(n_features=n_features,
                                             z_dim=trial_i.config['z_dim'],
                                             min_hidden_size=trial_i.config['min_hidden_size'],
                                             use_cuda=opt['cuda'], use_sn=opt['use_sn'])
        checkpoint_dir = trial_i.checkpoint.value
        E_state, optimizerE_state = torch.load(os.path.join(checkpoint_dir, "checkpoint_E"))
        E.load_state_dict(E_state)
        G_A_state, optimizerG_A_state = torch.load(os.path.join(checkpoint_dir, "checkpoint_G_A"))
        G_A.load_state_dict(G_A_state)
        G_B_state, optimizerG_B_state = torch.load(os.path.join(checkpoint_dir, "checkpoint_G_B"))
        G_B.load_state_dict(G_B_state)
        D_A_state, optimizerD_A_state = torch.load(os.path.join(checkpoint_dir, "checkpoint_D_A"))
        D_A.load_state_dict(D_A_state)
        D_B_state, optimizerD_B_state = torch.load(os.path.join(checkpoint_dir, "checkpoint_D_B"))
        D_B.load_state_dict(D_B_state)

        opt['batch_size'] = trial_i.config['batch_size']
        test(opt, E, G_A, G_B, D_A, D_B, opt['test_path'], f'{opt["model_name"]}_trial_{i}')


if __name__ == "__main__":

    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    main(num_samples=50, gpus_per_trial=1)
