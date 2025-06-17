from __future__ import print_function
import os
from pathlib import Path
from pyexpat import model
import random
import numpy as np
from scPreGAN.model.scPreGAN import is_model_trained
import torch
import torch.nn as nn
from torch import mean, exp, unique, cat, isnan
from torch import norm as torch_norm

import multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.cuda import is_available as cuda_is_available
from torch import Tensor, FloatTensor
from torch.utils.data import random_split
from torch import autograd
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import scanpy as sc
import anndata
from scipy import sparse
from scPreGAN.reproducibility.util import load_anndata
from scPreGAN.reproducibility.model.Discriminator import Discriminator
from scPreGAN.reproducibility.model.Generator import Generator
from scPreGAN.reproducibility.model.Encoder import Encoder
from anndata import AnnData


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform(m.weight, 1e-2)
        m.bias.data.fill_(0.01)


def create_model(n_features, z_dim, min_hidden_size, use_cuda, use_sn):
    if use_sn:
        D_A = Discriminator(
            n_features=n_features, min_hidden_size=min_hidden_size, out_dim=1
        )
        D_B = Discriminator(
            n_features=n_features, min_hidden_size=min_hidden_size, out_dim=1
        )
    else:
        D_A = Discriminator(
            n_features=n_features, min_hidden_size=min_hidden_size, out_dim=1
        )
        D_B = Discriminator(
            n_features=n_features, min_hidden_size=min_hidden_size, out_dim=1
        )

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


# 计算损失项
def calc_gradient_penalty(
    netD, real_data, fake_data, batch_size, use_cuda, lambta, use_wgan_div, k=2, p=6
):
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=(
            torch.ones(disc_interpolates.size()).cuda()
            if use_cuda
            else torch.ones(disc_interpolates.size())
        ),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    if use_wgan_div:
        gradient_penalty = torch.pow(gradients.norm(2, dim=1), p).mean() * k
    else:
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambta
    return gradient_penalty


def train_and_predict(config, opt, tensorboard_path: Path, load_model=False) -> AnnData:
    output_path = opt['outf']
    model_path = output_path / "model"
    
    if load_model:
        assert is_model_trained(output_path)
        print("Loading model from", model_path)
        return
        
    
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(tensorboard_path, exist_ok=True)
    
    
    if opt["manual_seed"] is None:
        opt["manual_seed"] = random.randint(1, 10000)
    print("Random Seed: ", opt["manual_seed"])
    random.seed(opt["manual_seed"])
    torch.manual_seed(opt["manual_seed"])
    if opt["cuda"]:
        torch.cuda.manual_seed_all(opt["manual_seed"])
    # load data===============================
    A_pd, A_celltype_ohe_pd, B_pd, B_celltype_ohe_pd = load_anndata(
        adata=opt["dataset"],
        condition_key=opt["condition_key"],
        condition=opt["condition"],
        cell_type_key=opt["cell_type_key"],
        prediction_type=opt["prediction_type"],
        out_sample_prediction=opt["out_sample_prediction"],
    )
    A_tensor = Tensor(np.array(A_pd))
    B_tensor = Tensor(np.array(B_pd))

    if opt["cuda"] and torch.cuda.is_available():
        A_tensor = A_tensor.cuda()
        B_tensor = B_tensor.cuda()

    A_Dataset = torch.utils.data.TensorDataset(A_tensor)
    B_Dataset = torch.utils.data.TensorDataset(B_tensor)
    if opt["validation"] and opt["valid_dataPath"] is None:
        print("splite dataset to train subset and validation subset")
        A_test_abs = int(len(A_Dataset) * 0.8)
        A_train_subset, A_val_subset = random_split(
            A_Dataset, [A_test_abs, len(A_Dataset) - A_test_abs]
        )

        B_test_abs = int(len(B_Dataset) * 0.8)
        B_train_subset, B_val_subset = random_split(
            B_Dataset, [B_test_abs, len(B_Dataset) - B_test_abs]
        )

        A_train_loader = torch.utils.data.DataLoader(
            dataset=A_train_subset,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            drop_last=True,
        )

        B_train_loader = torch.utils.data.DataLoader(
            dataset=B_train_subset,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            drop_last=True,
        )
        A_valid_loader = torch.utils.data.DataLoader(
            dataset=A_val_subset,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            drop_last=True,
        )
        B_valid_loader = torch.utils.data.DataLoader(
            dataset=B_val_subset,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            drop_last=True,
        )
    elif opt["validation"] and opt["valid_dataPath"] is not None:
        A_pd_val, A_celltype_ohe_pd_val, B_pd_val, B_celltype_ohe_pd_val = load_anndata(
            path=opt["valid_dataPath"],
            condition_key=opt["condition_key"],
            condition=opt["condition"],
            cell_type_key=opt["cell_type_key"],
        )

        print(
            f"use validation dataset, lenth of A: {A_pd_val.shape}, lenth of B: {B_pd_val.shape}"
        )

        A_tensor_val = Tensor(np.array(A_pd_val))
        B_tensor_val = Tensor(np.array(B_pd_val))

        if opt["cuda"] and torch.cuda.is_available():
            A_tensor_val = A_tensor_val.cuda()
            B_tensor_val = B_tensor_val.cuda()

        A_Dataset_val = torch.utils.data.TensorDataset(A_tensor_val)
        B_Dataset_val = torch.utils.data.TensorDataset(B_tensor_val)

        A_train_loader = torch.utils.data.DataLoader(
            dataset=A_Dataset,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            drop_last=True,
        )

        B_train_loader = torch.utils.data.DataLoader(
            dataset=B_Dataset,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            drop_last=True,
        )
        A_valid_loader = torch.utils.data.DataLoader(
            dataset=A_Dataset_val,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            drop_last=True,
        )
        B_valid_loader = torch.utils.data.DataLoader(
            dataset=B_Dataset_val,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            drop_last=True,
        )

    else:
        print("No validation.")
        A_train_loader = torch.utils.data.DataLoader(
            dataset=A_Dataset,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            drop_last=True,
        )

        B_train_loader = torch.utils.data.DataLoader(
            dataset=B_Dataset,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            drop_last=True,
        )

    opt["n_features"] = A_pd.shape[1]
    print("feature length: ", opt["n_features"])
    A_train_loader_it = iter(A_train_loader)
    B_train_loader_it = iter(B_train_loader)

    E, G_A, G_B, D_A, D_B = create_model(
        n_features=opt["n_features"],
        z_dim=config["z_dim"],
        min_hidden_size=config["min_hidden_size"],
        use_cuda=opt["cuda"],
        use_sn=opt["use_sn"],
    )

    recon_criterion = nn.MSELoss()
    encoding_criterion = nn.MSELoss()

    optimizerD_A = torch.optim.Adam(
        D_A.parameters(), lr=config["lr_disc"], betas=(0.5, 0.9)
    )
    optimizerD_B = torch.optim.Adam(
        D_B.parameters(), lr=config["lr_disc"], betas=(0.5, 0.9)
    )
    optimizerG_A = torch.optim.Adam(
        G_A.parameters(), lr=config["lr_g"], betas=(0.5, 0.9)
    )
    optimizerG_B = torch.optim.Adam(
        G_B.parameters(), lr=config["lr_g"], betas=(0.5, 0.9)
    )
    optimizerE = torch.optim.Adam(E.parameters(), lr=config["lr_e"])

    ones = torch.ones(config["batch_size"], 1)
    zeros = torch.zeros(config["batch_size"], 1)

    if opt["cuda"]:
        ones.cuda()
        zeros.cuda()

    D_A.train()
    D_B.train()
    G_A.train()
    G_B.train()
    E.train()

    D_A_loss = 0.0
    D_B_loss = 0.0

    writer = SummaryWriter(tensorboard_path)

    for iteration in range(1, config["niter"] + 1):
        if iteration % 10000 == 0:
            for param_group in optimizerD_A.param_groups:
                param_group["lr"] = param_group["lr"] * 0.9
            for param_group in optimizerD_B.param_groups:
                param_group["lr"] = param_group["lr"] * 0.9
            for param_group in optimizerG_A.param_groups:
                param_group["lr"] = param_group["lr"] * 0.9
            for param_group in optimizerG_B.param_groups:
                param_group["lr"] = param_group["lr"] * 0.9
            for param_group in optimizerE.param_groups:
                param_group["lr"] = param_group["lr"] * 0.9

        for count in range(0, 5):
            try:
                real_A = next(A_train_loader_it)[0]
                real_B = next(B_train_loader_it)[0]
            except StopIteration:
                A_train_loader_it, B_train_loader_it = iter(A_train_loader), iter(
                    B_train_loader
                )
                real_A = next(A_train_loader_it)[0]
                real_B = next(B_train_loader_it)[0]

            if (opt["cuda"]) and cuda_is_available():
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

            D_A_gradient_penalty = calc_gradient_penalty(
                D_A,
                real_A.detach(),
                BA.detach(),
                batch_size=config["batch_size"],
                use_cuda=opt["cuda"],
                lambta=config["lambta_gp"],
                use_wgan_div=opt["use_wgan_div"],
            )
            D_B_gradient_penalty = calc_gradient_penalty(
                D_B,
                real_B.detach(),
                AB.detach(),
                batch_size=config["batch_size"],
                use_cuda=opt["cuda"],
                lambta=config["lambta_gp"],
                use_wgan_div=opt["use_wgan_div"],
            )

            if opt["gan_loss"] == "vanilla":
                D_A_real_loss = F.binary_cross_entropy_with_logits(out_A, ones)
                D_B_real_loss = F.binary_cross_entropy_with_logits(out_B, ones)
                D_A_fake_loss = F.binary_cross_entropy_with_logits(out_BA, zeros)
                D_B_fake_loss = F.binary_cross_entropy_with_logits(out_AB, zeros)
                D_A_loss = D_A_real_loss + D_A_fake_loss
                D_B_loss = D_B_real_loss + D_B_fake_loss
            elif opt["gan_loss"] == "lsgan":
                D_A_real_loss = F.mse_loss(out_A, ones)
                D_B_real_loss = F.mse_loss(out_B, ones)
                D_A_fake_loss = F.mse_loss(out_BA, zeros)
                D_B_fake_loss = F.mse_loss(out_AB, zeros)
                D_A_loss = D_A_real_loss + D_A_fake_loss
                D_B_loss = D_B_real_loss + D_B_fake_loss
            elif opt["gan_loss"] == "wgan":
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
            real_A = next(A_train_loader_it)[0]
            real_B = next(B_train_loader_it)[0]
        except StopIteration:
            A_train_loader_it, B_train_loader_it = iter(A_train_loader), iter(
                B_train_loader
            )
            real_A = next(A_train_loader_it)[0]
            real_B = next(B_train_loader_it)[0]

        if (opt["cuda"]) and cuda_is_available():
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

        if opt["gan_loss"] == "vanilla":
            G_AA_adv_loss = F.binary_cross_entropy_with_logits(out_AA, ones)
            G_BA_adv_loss = F.binary_cross_entropy_with_logits(out_BA, ones)
            G_ABA_adv_loss = F.binary_cross_entropy_with_logits(out_ABA, ones)

            G_BB_adv_loss = F.binary_cross_entropy_with_logits(out_BB, ones)
            G_AB_adv_loss = F.binary_cross_entropy_with_logits(out_AB, ones)
            G_BAB_adv_loss = F.binary_cross_entropy_with_logits(out_BAB, ones)
        elif opt["gan_loss"] == "lsgan":
            G_AA_adv_loss = F.mse_loss(out_AA, ones)
            G_BA_adv_loss = F.mse_loss(out_BA, ones)
            G_ABA_adv_loss = F.mse_loss(out_ABA, ones)

            G_BB_adv_loss = F.mse_loss(out_BB, ones)
            G_AB_adv_loss = F.mse_loss(out_AB, ones)
            G_BAB_adv_loss = F.mse_loss(out_BAB, ones)
        elif opt["gan_loss"] == "wgan":
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
        adv_loss = (G_A_adv_loss + G_B_adv_loss) * config["lambda_adv"]

        # reconstruction loss
        l_rec_AA = recon_criterion(AA, real_A)
        l_rec_BB = recon_criterion(BB, real_B)

        recon_loss = (l_rec_AA + l_rec_BB) * config["lambda_recon"]

        # encoding loss
        tmp_real_A_z = real_A_z.detach()
        tmp_real_B_z = real_B_z.detach()
        l_encoding_AA = encoding_criterion(AA_z, tmp_real_A_z)
        l_encoding_BB = encoding_criterion(BB_z, tmp_real_B_z)
        l_encoding_BA = encoding_criterion(BA_z, tmp_real_B_z)
        l_encoding_AB = encoding_criterion(AB_z, tmp_real_A_z)

        encoding_loss = (
            l_encoding_AA + l_encoding_BB + l_encoding_BA + l_encoding_AB
        ) * config["lambda_encoding"]

        E_paras = cat([x.view(-1) for x in E.parameters()])
        G_A_paras = cat([x.view(-1) for x in G_A.parameters()])
        G_B_paras = cat([x.view(-1) for x in G_B.parameters()])
        E_regularization = torch_norm(E_paras, 1)
        G_A_regularization = torch_norm(G_A_paras, 1)
        G_B_regularization = torch_norm(G_B_paras, 1)
        l1_regularization = (
            E_regularization + G_A_regularization + G_B_regularization
        ) * config["lambda_l1_reg"]

        G_loss = adv_loss + recon_loss + encoding_loss + l1_regularization

        # backward
        G_loss.backward()

        # step
        optimizerG_A.step()
        optimizerG_B.step()
        optimizerE.step()

        writer.add_scalar("D_A_loss", D_A_loss, global_step=iteration)
        writer.add_scalar("D_B_loss", D_B_loss, global_step=iteration)
        writer.add_scalar("adv_loss", adv_loss, global_step=iteration)
        writer.add_scalar("recon_loss", recon_loss, global_step=iteration)
        writer.add_scalar("encoding_loss", encoding_loss, global_step=iteration)
        writer.add_scalar("G_loss", G_loss, global_step=iteration)

        if iteration % 100 == 0:
            print(
                "[%d/%d] D_A_loss: %.4f  D_B_loss: %.4f adv_loss: %.4f  recon_loss: %.4f encoding_loss: %.4f G_loss: %.4f"
                % (
                    iteration,
                    config["niter"],
                    D_A_loss.item(),
                    D_B_loss.item(),
                    adv_loss.item(),
                    recon_loss.item(),
                    encoding_loss.item(),
                    G_loss.item(),
                )
            )

            if opt["validation"]:
                D_A_loss_val = 0.0
                D_B_loss_val = 0.0

                adv_loss_val = 0.0
                recon_loss_val = 0.0
                encoding_loss_val = 0.0
                G_loss_val = 0.0

                counter = 0
                sparcity_cellA_val = 0.0
                sparcity_cellB_val = 0.0

                A_valid_loader_it = iter(A_valid_loader)
                B_valid_loader_it = iter(B_valid_loader)

                max_length = max(len(A_valid_loader), len(B_valid_loader))
                with torch.no_grad():
                    for iteration_val in range(1, max_length):
                        try:
                            cellA_val = next(A_valid_loader_it)[0]
                            cellB_val = next(B_valid_loader_it)[0]
                        except StopIteration:
                            A_valid_loader_it, B_valid_loader_it = iter(
                                A_valid_loader
                            ), iter(B_valid_loader)
                            cellA_val = next(A_valid_loader_it)[0]
                            cellB_val = next(B_valid_loader_it)[0]

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

                        if opt["gan_loss"] == "vanilla":
                            D_A_real_loss_val = F.binary_cross_entropy_with_logits(
                                outA_val, ones
                            )
                            D_B_real_loss_val = F.binary_cross_entropy_with_logits(
                                outB_val, ones
                            )
                            D_A_fake_loss_val = F.binary_cross_entropy_with_logits(
                                out_BA, zeros
                            )
                            D_B_fake_loss_val = F.binary_cross_entropy_with_logits(
                                out_AB, zeros
                            )
                        elif opt["gan_loss"] == "lsgan":
                            D_A_real_loss_val = F.mse_loss(outA_val, ones)
                            D_B_real_loss_val = F.mse_loss(outB_val, ones)
                            D_A_fake_loss_val = F.mse_loss(out_BA, zeros)
                            D_B_fake_loss_val = F.mse_loss(out_AB, zeros)
                        elif opt["gan_loss"] == "wgan":
                            D_A_real_loss_val = -torch.mean(outA_val)
                            D_B_real_loss_val = -torch.mean(outB_val)
                            D_A_fake_loss_val = torch.mean(out_BA)
                            D_B_fake_loss_val = torch.mean(out_AB)

                        else:
                            NotImplementedError("not implement loss")
                        D_A_loss_val += (D_A_real_loss_val + D_A_fake_loss_val).item()
                        D_B_loss_val += (D_B_real_loss_val + D_B_fake_loss_val).item()

                        if opt["gan_loss"] == "vanilla":
                            G_AA_adv_loss_val = F.binary_cross_entropy_with_logits(
                                out_AA, ones
                            )
                            G_BA_adv_loss_val = F.binary_cross_entropy_with_logits(
                                out_BA, ones
                            )
                            G_ABA_adv_loss_val = F.binary_cross_entropy_with_logits(
                                out_ABA, ones
                            )

                            G_BB_adv_loss_val = F.binary_cross_entropy_with_logits(
                                out_BB, ones
                            )
                            G_AB_adv_loss_val = F.binary_cross_entropy_with_logits(
                                out_AB, ones
                            )
                            G_BAB_adv_loss_val = F.binary_cross_entropy_with_logits(
                                out_BAB, ones
                            )

                        elif opt["gan_loss"] == "lsgan":
                            G_AA_adv_loss_val = F.mse_loss(out_AA, ones)
                            G_BA_adv_loss_val = F.mse_loss(out_BA, ones)
                            G_ABA_adv_loss_val = F.mse_loss(out_ABA, ones)

                            G_BB_adv_loss_val = F.mse_loss(out_BB, ones)
                            G_AB_adv_loss_val = F.mse_loss(out_AB, ones)
                            G_BAB_adv_loss_val = F.mse_loss(out_BAB, ones)

                        elif opt["gan_loss"] == "wgan":
                            G_AA_adv_loss_val = -torch.mean(out_AA)
                            G_BA_adv_loss_val = -torch.mean(out_BA)
                            G_ABA_adv_loss_val = -torch.mean(out_ABA)

                            G_BB_adv_loss_val = -torch.mean(out_BB)
                            G_AB_adv_loss_val = -torch.mean(out_AB)
                            G_BAB_adv_loss_val = -torch.mean(out_BAB)
                        else:
                            NotImplementedError("not implement loss")
                        G_A_adv_loss_val = (
                            G_AA_adv_loss_val + G_BA_adv_loss_val + G_ABA_adv_loss_val
                        )
                        G_B_adv_loss_val = (
                            G_BB_adv_loss_val + G_AB_adv_loss_val + G_BAB_adv_loss_val
                        )
                        adv_loss_val += (
                            G_A_adv_loss_val + G_B_adv_loss_val
                        ).item() * config["lambda_adv"]

                        # reconstruction loss
                        l_rec_AA_val = recon_criterion(AA, cellA_val)
                        l_rec_BB_val = recon_criterion(BB, cellB_val)
                        recon_loss_val += (l_rec_AA_val + l_rec_BB_val).item() * config[
                            "lambda_recon"
                        ]

                        # encoding loss
                        l_encoding_AA_val = encoding_criterion(AA_z, real_A_z)
                        l_encoding_BB_val = encoding_criterion(BB_z, real_B_z)
                        l_encoding_BA_val = encoding_criterion(BA_z, real_B_z)
                        l_encoding_AB_val = encoding_criterion(AB_z, real_A_z)
                        encoding_loss_val = (
                            l_encoding_AA_val
                            + l_encoding_BB_val
                            + l_encoding_BA_val
                            + l_encoding_AB_val
                        ).item() * config["lambda_encoding"]
                        G_loss_val += adv_loss_val + recon_loss_val + encoding_loss_val

                        cellA_val_np = cellA_val.cpu().detach().numpy()
                        cellB_val_np = cellB_val.cpu().detach().numpy()

                        cellA_val_zero = cellA_val_np[cellA_val_np <= 0]
                        cellB_val_zero = cellB_val_np[cellB_val_np <= 0]

                        sparcity_cellA_val += cellA_val_zero.shape[0] / (
                            cellA_val_np.shape[0] * cellA_val_np.shape[1]
                        )
                        sparcity_cellB_val += cellB_val_zero.shape[0] / (
                            cellB_val_np.shape[0] * cellB_val_np.shape[1]
                        )

                print(
                    "[%d/%d] adv_loss_val: %.4f  recon_loss_val: %.4f encoding_loss_val: %.4f  G_loss: %.4f D_A_loss_val: %.4f D_B_loss_val: %.4f"
                    % (
                        iteration,
                        config["niter"],
                        adv_loss_val / counter,
                        recon_loss_val / counter,
                        encoding_loss_val / counter,
                        G_loss / counter,
                        D_A_loss_val / counter,
                        D_B_loss_val / counter,
                    )
                )

                writer.add_scalar(
                    "adv_loss_val", adv_loss_val / counter, global_step=iteration
                )
                writer.add_scalar(
                    "recon_loss_val", recon_loss_val / counter, global_step=iteration
                )
                writer.add_scalar(
                    "encoding_loss_val",
                    encoding_loss_val / counter,
                    global_step=iteration,
                )
                writer.add_scalar("G_loss", G_loss / counter, global_step=iteration)
                writer.add_scalar(
                    "D_A_loss_val", D_A_loss_val / counter, global_step=iteration
                )
                writer.add_scalar(
                    "D_B_loss_val", D_B_loss_val / counter, global_step=iteration
                )

                print(
                    "[%d/%d] adv_loss_val: %.4f  recon_loss_val: %.4f encoding_loss_val: %.4f  G_loss: %.4f D_A_loss_val: %.4f D_B_loss_val: %.4f"
                    % (
                        iteration,
                        config["niter"],
                        adv_loss_val / counter,
                        recon_loss_val / counter,
                        encoding_loss_val / counter,
                        G_loss / counter,
                        D_A_loss_val / counter,
                        D_B_loss_val / counter,
                    )
                )
        if opt["checkpoint_dir"] is not None and iteration % 10000 == 0:
            path = os.path.join(opt["checkpoint_dir"], "checkpoint_E.pth")
            torch.save((E.state_dict(), optimizerE.state_dict()), path)
            path = os.path.join(opt["checkpoint_dir"], "checkpoint_G_A.pth")
            torch.save((G_A.state_dict(), optimizerG_A.state_dict()), path)
            path = os.path.join(opt["checkpoint_dir"], "checkpoint_G_B.pth")
            torch.save((G_B.state_dict(), optimizerG_B.state_dict()), path)
            path = os.path.join(opt["checkpoint_dir"], "checkpoint_D_A.pth")
            torch.save((D_A.state_dict(), optimizerD_A.state_dict()), path)
            path = os.path.join(opt["checkpoint_dir"], "checkpoint_D_B.pth")
            torch.save((D_B.state_dict(), optimizerD_B.state_dict()), path)

    torch.save(E.state_dict(), os.path.join(model_path, 'E.pth'))
    torch.save(G_A.state_dict(), os.path.join(model_path, 'G_A.pth'))
    torch.save(G_B.state_dict(), os.path.join(model_path, 'G_B.pth'))
    torch.save(D_A.state_dict(), os.path.join(model_path, 'D_A.pth'))
    torch.save(D_B.state_dict(), os.path.join(model_path, 'D_B.pth'))
    writer.close()
    print("Finished Training")
    
    adata = opt["dataset"]
    
    control_adata = adata[
        (adata.obs[opt["cell_type_key"]] == opt["prediction_type"])
        & (adata.obs[opt["condition_key"]] == opt["condition"]["control"])
    ]
    
    
    """
    Evaluate
    """
    D_A.eval()
    D_B.eval()
    G_A.eval()
    G_B.eval()
    E.eval()

    if sparse.issparse(control_adata.X):
        control_tensor = Tensor(control_adata.X.toarray())
    else:
        control_tensor = Tensor(control_adata.X)
    if opt["cuda"] and cuda_is_available():
        print("Using CUDA for evaluation")
        control_tensor = control_tensor.cuda()
        
    with torch.no_grad():
        control_z = E(control_tensor)
        case_pred = G_B(control_z)
        
    pred_perturbed_adata = sc.AnnData(
        X=case_pred.cpu().detach().numpy(),
        obs={
            opt["condition_key"]: ["pred_perturbed"] * len(case_pred),
            opt["cell_type_key"]: control_adata.obs[opt["cell_type_key"]].tolist(),
        },
    )
    
    pred_perturbed_adata.var_names = control_adata.var_names
    return pred_perturbed_adata


def main(data_name):
    if data_name == "pbmc":
        opt = {
            "cuda": True,
            "dataPath": "/home/wxj/scPreGAN/datasets/pbmc/pbmc.h5ad",
            "checkpoint_dir": None,
            "condition_key": "condition",
            "condition": {"case": "stimulated", "control": "control"},
            "cell_type_key": "cell_type",
            "prediction_type": None,
            "out_sample_prediction": True,
            "manual_seed": 3060,
            "data_name": "pbmc",
            "model_name": "pbmc_OOD",
            "outf": "/home/wxj/scPreGAN/datasets/pbmc/pbmc_OOD",
            "validation": False,
            "valid_dataPath": None,
            "use_sn": True,
            "use_wgan_div": True,
            "gan_loss": "wgan",
        }
    elif data_name == "hpoly":
        opt = {
            "cuda": True,
            "dataPath": "/home/wxj/scPreGAN/datasets/Hpoly/hpoly.h5ad",
            "checkpoint_dir": None,
            "condition_key": "condition",
            "condition": {"case": "Hpoly.Day10", "control": "Control"},
            "cell_type_key": "cell_label",
            "prediction_type": None,
            "out_sample_prediction": True,
            "manual_seed": 3060,
            "data_name": "hpoly",
            "model_name": "hpoly_OOD",
            "outf": "/home/wxj/scPreGAN/datasets/Hpoly/hpoly_OOD",
            "validation": False,
            "valid_dataPath": None,
            "test_dataPath": "/home/wxj/scPreGAN/datasets/Hpoly/valid_hpoly.h5ad",
            "use_sn": True,
            "use_wgan_div": True,
            "gan_loss": "wgan",
        }
    elif data_name == "species":
        opt = {
            "cuda": True,
            "dataPath": "/home/wxj/scPreGAN/datasets/species/species.h5ad",
            "checkpoint_dir": None,
            "condition_key": "condition",
            "condition": {"case": "LPS6", "control": "unst"},
            "cell_type_key": "species",
            "prediction_type": None,
            "out_sample_prediction": True,
            "manual_seed": 3060,
            "data_name": "species",
            "model_name": "species_OOD",
            "outf": "/home/wxj/scPreGAN/datasets/species/species_OOD",
            "validation": False,
            "valid_dataPath": None,
            "use_sn": True,
            "use_wgan_div": True,
            "gan_loss": "wgan",
        }
    else:
        NotImplementedError()

    if cuda_is_available():
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass
        cudnn.benchmark = True
    config = {
        "batch_size": 64,
        "lambda_adv": 0.001,
        "lambda_encoding": 0.1,
        "lambda_l1_reg": 0,
        "lambda_recon": 1,
        "lambta_gp": 1,
        "lr_disc": 0.001,
        "lr_e": 0.0001,
        "lr_g": 0.001,
        "min_hidden_size": 256,
        "niter": 20000,
        "z_dim": 16,
    }

    opt["out_sample_prediction"] = True
    adata = sc.read(opt["dataPath"])
    cell_type_list = adata.obs[opt["cell_type_key"]].unique().tolist()
    print("cell type list: " + str(cell_type_list))
    for cell_type in cell_type_list:
        print("=================" + cell_type + "=========================")
        opt["prediction_type"] = cell_type

        if not os.path.exists(opt["outf"]):
            os.makedirs(opt["outf"])
        train_and_predict(config, opt)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main(data_name="pbmc")
