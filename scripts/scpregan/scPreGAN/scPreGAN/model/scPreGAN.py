from __future__ import print_function
import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.cuda import is_available as cuda_is_available
from torch import Tensor
from torch.utils.data import random_split
from torch import autograd
from torch.utils.tensorboard import SummaryWriter
import scanpy as sc
from scipy import sparse
import pandas as pd
import random
from pathlib import Path

from scPreGAN.model.Discriminator import Discriminator_AC
from scPreGAN.model.Generator import Generator_AC_layer
from scPreGAN.model.Encoder import Encoder_AC_layer


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform(m.weight, 1e-2)
        m.bias.data.fill_(0.01)


def is_model_trained(output_path: Path) -> bool:
    model_path = output_path / "model"
    return (
        (model_path / "E.pth").exists()
        and (model_path / "G_A.pth").exists()
        and (model_path / "G_B.pth").exists()
        and (model_path / "D_A.pth").exists()
        and (model_path / "D_B.pth").exists()
    )


class Model:
    def __init__(
        self,
        n_features,
        n_classes,
        z_dim=16,
        min_hidden_size=256,
        use_cuda=True,
        manual_seed=3060,
    ):
        if manual_seed is None:
            manual_seed = random.randint(1, 10000)
        print("Random Seed: ", manual_seed)
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        if use_cuda:
            torch.cuda.manual_seed_all(manual_seed)

        def init_weights(m):
            if type(m) == torch.nn.Linear:
                torch.nn.init.xavier_uniform(m.weight, 1e-2)
                m.bias.data.fill_(0.01)

        D_A = Discriminator_AC(
            n_features=n_features,
            min_hidden_size=min_hidden_size,
            out_dim=1,
            n_classes=n_classes,
        )
        D_B = Discriminator_AC(
            n_features=n_features,
            min_hidden_size=min_hidden_size,
            out_dim=1,
            n_classes=n_classes,
        )

        G_A = Generator_AC_layer(
            z_dim=z_dim, min_hidden_size=min_hidden_size, n_features=n_features
        )
        G_B = Generator_AC_layer(
            z_dim=z_dim, min_hidden_size=min_hidden_size, n_features=n_features
        )
        E = Encoder_AC_layer(
            n_features=n_features, min_hidden_size=min_hidden_size, z_dim=z_dim
        )

        init_weights(E)
        init_weights(G_A)
        init_weights(G_B)
        init_weights(D_A)
        init_weights(D_B)

        if use_cuda and torch.cuda.is_available():
            D_A = D_A.cuda()
            D_B = D_B.cuda()
            G_A = G_A.cuda()
            G_B = G_B.cuda()
            E = E.cuda()

        self.E = E
        self.G_A = G_A
        self.G_B = G_B
        self.D_A = D_A
        self.D_B = D_B
        self.use_cuda = use_cuda
        print("Successfully created the model")

    def train(
        self,
        train_data,
        output_path: Path,
        load_model: bool = False,
        valid_data=None,
        niter=20_000,
        lr_e=0.0001,
        lr_g=0.001,
        lr_d=0.001,
        batch_size=64,
        lambda_adv=0.001,
        lambda_recon=1,
        lambda_encoding=0.1,
        betas=(0.5, 0.9),
        k=2,
        p=6,
    ):
        output_path = Path(output_path)
        model_path = output_path / "model"

        if load_model:
            print("Loading model from", model_path)
            assert is_model_trained(output_path)
            self.E.load_state_dict(torch.load(model_path / "E.pth"))
            self.G_A.load_state_dict(torch.load(model_path / "G_A.pth"))
            self.G_B.load_state_dict(torch.load(model_path / "G_B.pth"))
            self.D_A.load_state_dict(torch.load(model_path / "D_A.pth"))
            self.D_B.load_state_dict(torch.load(model_path / "D_B.pth"))
            return
        
        os.makedirs(model_path, exist_ok=True)

        # load data===============================
        A_pd, A_celltype_ohe_pd, B_pd, B_celltype_ohe_pd = train_data
        trainA = [np.array(A_pd), np.array(A_celltype_ohe_pd)]
        trainB = [np.array(B_pd), np.array(B_celltype_ohe_pd)]

        expr_trainA, cell_type_trainA = trainA
        expr_trainB, cell_type_trainB = trainB
        expr_trainA_tensor = Tensor(expr_trainA)
        expr_trainB_tensor = Tensor(expr_trainB)
        cell_type_trainA_tensor = Tensor(cell_type_trainA)
        cell_type_trainB_tensor = Tensor(cell_type_trainB)

        if self.use_cuda and torch.cuda.is_available():
            expr_trainA_tensor = expr_trainA_tensor.cuda()
            expr_trainB_tensor = expr_trainB_tensor.cuda()
            cell_type_trainA_tensor = cell_type_trainA_tensor.cuda()
            cell_type_trainB_tensor = cell_type_trainB_tensor.cuda()

        A_Dataset = torch.utils.data.TensorDataset(
            expr_trainA_tensor, cell_type_trainA_tensor
        )
        B_Dataset = torch.utils.data.TensorDataset(
            expr_trainB_tensor, cell_type_trainB_tensor
        )

        A_train_loader = torch.utils.data.DataLoader(
            dataset=A_Dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

        B_train_loader = torch.utils.data.DataLoader(
            dataset=B_Dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

        A_train_loader_it = iter(A_train_loader)
        B_train_loader_it = iter(B_train_loader)

        if valid_data is not None:
            A_pd_val, A_celltype_ohe_pd_val, B_pd_val, B_celltype_ohe_pd_val = (
                valid_data
            )

            valA = [np.array(A_pd_val), np.array(A_celltype_ohe_pd_val)]
            valB = [np.array(B_pd_val), np.array(B_celltype_ohe_pd_val)]

            expr_valA, cell_type_valA = valA
            expr_valB, cell_type_valB = valB

            expr_valA_tensor = Tensor(expr_valA)
            expr_valB_tensor = Tensor(expr_valB)
            cell_type_valA_tensor = Tensor(cell_type_valA)
            cell_type_valB_tensor = Tensor(cell_type_valB)

            if self.use_cuda and torch.cuda.is_available():
                expr_valA_tensor = expr_valA_tensor.cuda()
                expr_valB_tensor = expr_valB_tensor.cuda()
                cell_type_valA_tensor = cell_type_valA_tensor.cuda()
                cell_type_valB_tensor = cell_type_valB_tensor.cuda()

            A_Dataset_val = torch.utils.data.TensorDataset(
                expr_valA_tensor, cell_type_valA_tensor
            )
            B_Dataset_val = torch.utils.data.TensorDataset(
                expr_valB_tensor, cell_type_valB_tensor
            )

            A_valid_loader = torch.utils.data.DataLoader(
                dataset=A_Dataset_val,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
            )
            B_valid_loader = torch.utils.data.DataLoader(
                dataset=B_Dataset_val,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
            )

        # loss function
        recon_criterion = nn.MSELoss()
        encoding_criterion = nn.MSELoss()
        dis_criterion = nn.BCELoss()
        aux_criterion = nn.NLLLoss()
        # Optimizer
        optimizerD_A = torch.optim.Adam(self.D_A.parameters(), lr=lr_d, betas=betas)
        optimizerD_B = torch.optim.Adam(self.D_B.parameters(), lr=lr_d, betas=betas)
        optimizerG_A = torch.optim.Adam(self.G_A.parameters(), lr=lr_g, betas=betas)
        optimizerG_B = torch.optim.Adam(self.G_B.parameters(), lr=lr_g, betas=betas)
        optimizerE = torch.optim.Adam(self.E.parameters(), lr=lr_e)

        ones = torch.ones(batch_size, 1)
        zeros = torch.zeros(batch_size, 1)

        if self.use_cuda and torch.cuda.is_available():
            ones = ones.cuda()
            zeros = zeros.cuda()

        self.D_A.train()
        self.D_B.train()
        self.G_A.train()
        self.G_B.train()
        self.E.train()


        # Training
        for iteration in range(1, niter + 1):
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

            # Train discriminator
            D_A_loss_item = 0.0
            D_B_loss_item = 0.0

            try:
                real_A, cell_type_A = next(A_train_loader_it)
                real_B, cell_type_B = next(B_train_loader_it)
            except StopIteration:
                A_train_loader_it, B_train_loader_it = iter(A_train_loader), iter(
                    B_train_loader
                )
                real_A, cell_type_A = next(A_train_loader_it)
                real_B, cell_type_B = next(B_train_loader_it)

            if self.use_cuda and cuda_is_available():
                real_A = real_A.cuda()
                real_B = real_B.cuda()
                cell_type_A = cell_type_A.cuda()
                cell_type_B = cell_type_B.cuda()

            self.D_A.zero_grad()
            self.D_B.zero_grad()

            out_A, out_A_cls = self.D_A(real_A, cell_type_A)  # real A
            out_B, out_B_cls = self.D_B(real_B, cell_type_B)  # real B

            real_A_z = self.E(real_A)
            AB = self.G_B(real_A_z)

            real_B_z = self.E(real_B)
            BA = self.G_A(real_B_z)

            out_BA, out_BA_cls = self.D_A(BA.detach(), cell_type_B)  # false A
            out_AB, out_AB_cls = self.D_B(AB.detach(), cell_type_A)  # false B

            _cell_type_A = torch.argmax(cell_type_A, dim=-1)
            _cell_type_B = torch.argmax(cell_type_B, dim=-1)

            dis_D_A_real = dis_criterion(out_A, ones)
            aux_D_A_real = aux_criterion(out_A_cls, _cell_type_A)
            D_A_real = dis_D_A_real + aux_D_A_real
            dis_D_A_fake = dis_criterion(out_BA, zeros)
            aux_D_A_fake = aux_criterion(out_BA_cls, _cell_type_B)
            D_A_fake = dis_D_A_fake + aux_D_A_fake

            dis_D_B_real = dis_criterion(out_B, ones)
            aux_D_B_real = aux_criterion(out_B_cls, _cell_type_B)
            D_B_real = dis_D_B_real + aux_D_B_real
            dis_D_B_fake = dis_criterion(out_AB, zeros)
            aux_D_B_fake = aux_criterion(out_AB_cls, _cell_type_A)
            D_B_fake = dis_D_B_fake + aux_D_B_fake

            D_A_loss = D_A_real + D_A_fake
            D_B_loss = D_B_real + D_B_fake

            D_A_loss.backward()
            D_B_loss.backward()
            optimizerD_A.step()
            optimizerD_B.step()

            D_A_loss_item += D_A_loss.item()
            D_B_loss_item += D_B_loss.item()


            # Train encoder and decoder
            try:
                real_A, cell_type_A = next(A_train_loader_it)
                real_B, cell_type_B = next(B_train_loader_it)
            except StopIteration:
                A_train_loader_it, B_train_loader_it = iter(A_train_loader), iter(
                    B_train_loader
                )
                real_A, cell_type_A = next(A_train_loader_it)
                real_B, cell_type_B = next(B_train_loader_it)

            if self.use_cuda and cuda_is_available():
                real_A = real_A.cuda()
                real_B = real_B.cuda()
                cell_type_A = cell_type_A.cuda()
                cell_type_B = cell_type_B.cuda()

            self.G_A.zero_grad()
            self.G_B.zero_grad()
            self.E.zero_grad()

            real_A_z = self.E(real_A)
            AA = self.G_A(real_A_z)
            AB = self.G_B(real_A_z)

            AA_z = self.E(AA)
            AB_z = self.E(AB)
            ABA = self.G_A(AB_z)

            real_B_z = self.E(real_B)
            BA = self.G_A(real_B_z)
            BB = self.G_B(real_B_z)
            BA_z = self.E(BA)
            BB_z = self.E(BB)
            BAB = self.G_B(BA_z)

            out_AA, out_AA_cls = self.D_A(AA, cell_type_A)
            out_AB, out_AB_cls = self.D_B(AB, cell_type_A)
            out_BA, out_BA_cls = self.D_A(BA, cell_type_B)
            out_BB, out_BB_cls = self.D_B(BB, cell_type_B)
            out_ABA, out_ABA_cls = self.D_A(ABA, cell_type_A)
            out_BAB, out_BAB_cls = self.D_B(BAB, cell_type_B)

            # adversarial loss
            G_AA_adv_loss = dis_criterion(out_AA, ones) + aux_criterion(
                out_AA_cls, _cell_type_A
            )
            G_BA_adv_loss = dis_criterion(out_BA, ones) + aux_criterion(
                out_BA_cls, _cell_type_B
            )
            G_ABA_adv_loss = dis_criterion(out_ABA, ones) + aux_criterion(
                out_ABA_cls, _cell_type_A
            )

            G_BB_adv_loss = dis_criterion(out_BB, ones) + aux_criterion(
                out_BB_cls, _cell_type_B
            )
            G_AB_adv_loss = dis_criterion(out_AB, ones) + aux_criterion(
                out_AB_cls, _cell_type_A
            )
            G_BAB_adv_loss = dis_criterion(out_BAB, ones) + aux_criterion(
                out_BAB_cls, _cell_type_B
            )

            G_A_adv_loss = G_AA_adv_loss + G_BA_adv_loss + G_ABA_adv_loss
            G_B_adv_loss = G_BB_adv_loss + G_AB_adv_loss + G_BAB_adv_loss
            adv_loss = (G_A_adv_loss + G_B_adv_loss) * lambda_adv

            # reconstruction loss
            l_rec_AA = recon_criterion(AA, real_A)
            l_rec_BB = recon_criterion(BB, real_B)
            recon_loss = (l_rec_AA + l_rec_BB) * lambda_recon

            # encoding loss
            tmp_real_A_z = real_A_z.detach()
            tmp_real_B_z = real_B_z.detach()
            l_encoding_AA = encoding_criterion(AA_z, tmp_real_A_z)
            l_encoding_BB = encoding_criterion(BB_z, tmp_real_B_z)
            l_encoding_BA = encoding_criterion(BA_z, tmp_real_B_z)
            l_encoding_AB = encoding_criterion(AB_z, tmp_real_A_z)
            encoding_loss = (
                l_encoding_AA + l_encoding_BB + l_encoding_BA + l_encoding_AB
            ) * lambda_encoding

            G_loss = adv_loss + recon_loss + encoding_loss

            G_loss.backward()

            optimizerG_A.step()
            optimizerG_B.step()
            optimizerE.step()


            if iteration % 100 == 0:
                print(
                    "[%d/%d] adv_loss: %.4f  recon_loss: %.4f encoding_loss: %.4f G_loss: %.4f D_A_loss: %.4f  D_B_loss: %.4f"
                    % (
                        iteration,
                        niter,
                        adv_loss.item(),
                        recon_loss.item(),
                        encoding_loss.item(),
                        G_loss.item(),
                        D_A_loss_item,
                        D_B_loss_item,
                    )
                )

                # Validation
                if valid_data is not None:
                    D_A_loss_val = 0.0
                    D_B_loss_val = 0.0
                    adv_loss_val = 0.0
                    recon_loss_val = 0.0
                    encoding_loss_val = 0.0
                    G_loss_val = 0.0
                    counter = 0

                    A_valid_loader_it = iter(A_valid_loader)
                    B_valid_loader_it = iter(B_valid_loader)

                    max_length = max(len(A_valid_loader), len(B_valid_loader))
                    with torch.no_grad():
                        for iteration_val in range(1, max_length):
                            try:
                                cellA_val, cell_type_A_val = A_valid_loader_it.next()[0]
                                cellB_val, cell_type_B_val = B_valid_loader_it.next()[0]
                            except StopIteration:
                                A_valid_loader_it, B_valid_loader_it = iter(
                                    A_valid_loader
                                ), iter(B_valid_loader)
                                cellA_val, cell_type_A_val = A_valid_loader_it.next()[0]
                                cellB_val, cell_type_B_val = B_valid_loader_it.next()[0]

                            counter += 1

                            real_A_z = self.E(cellA_val)
                            real_B_z = self.E(cellB_val)
                            AB = self.G_B(real_A_z)
                            BA = self.G_A(real_B_z)
                            AA = self.G_A(real_A_z)
                            BB = self.G_B(real_B_z)
                            AA_z = self.E(AA)
                            BB_z = self.E(BB)
                            AB_z = self.E(AB)
                            BA_z = self.E(BA)
                            ABA = self.G_A(AB_z)
                            BAB = self.G_B(BA_z)

                            outA_val, out_A_cls = self.D_A(cellA_val, cell_type_A_val)
                            outB_val, out_B_cls = self.D_B(cellB_val, cell_type_B_val)
                            out_AA, out_AA_cls = self.D_A(AA, cell_type_A_val)
                            out_BB, out_BB_cls = self.D_B(BB, cell_type_B_val)
                            out_AB, out_AB_cls = self.D_B(AB, cell_type_A_val)
                            out_BA, out_BA_cls = self.D_A(BA, cell_type_B_val)
                            out_ABA, out_ABA_cls = self.D_A(ABA, cell_type_A_val)
                            out_BAB, out_BAB_cls = self.D_B(BAB, cell_type_B_val)

                            _cell_type_A = torch.argmax(cell_type_A_val, dim=-1)
                            _cell_type_B = torch.argmax(cell_type_B_val, dim=-1)

                            D_A_real_loss_val = dis_criterion(
                                outA_val, ones
                            ) + aux_criterion(out_A_cls, _cell_type_A)
                            D_B_real_loss_val = dis_criterion(
                                outB_val, ones
                            ) + aux_criterion(out_B_cls, _cell_type_B)
                            D_A_fake_loss_val = dis_criterion(
                                out_BA, zeros
                            ) + aux_criterion(out_BA_cls, _cell_type_B)
                            D_B_fake_loss_val = dis_criterion(
                                out_AB, zeros
                            ) + aux_criterion(out_AB_cls, _cell_type_A)

                            D_A_loss_val += (
                                D_A_real_loss_val + D_A_fake_loss_val
                            ).item()
                            D_B_loss_val += (
                                D_B_real_loss_val + D_B_fake_loss_val
                            ).item()

                            G_AA_adv_loss_val = dis_criterion(
                                out_AA, ones
                            ) + aux_criterion(out_AA_cls, _cell_type_A)
                            G_BA_adv_loss_val = dis_criterion(
                                out_BA, ones
                            ) + aux_criterion(out_BA_cls, _cell_type_B)
                            G_ABA_adv_loss_val = dis_criterion(
                                out_ABA, ones
                            ) + aux_criterion(out_ABA_cls, _cell_type_A)

                            G_BB_adv_loss_val = dis_criterion(
                                out_BB, ones
                            ) + aux_criterion(out_BB_cls, _cell_type_B)
                            G_AB_adv_loss_val = dis_criterion(
                                out_AB, ones
                            ) + aux_criterion(out_AB_cls, _cell_type_A)
                            G_BAB_adv_loss_val = dis_criterion(
                                out_BAB, ones
                            ) + aux_criterion(out_BAB_cls, _cell_type_B)

                            G_A_adv_loss_val = (
                                G_AA_adv_loss_val
                                + G_BA_adv_loss_val
                                + G_ABA_adv_loss_val
                            )
                            G_B_adv_loss_val = (
                                G_BB_adv_loss_val
                                + G_AB_adv_loss_val
                                + G_BAB_adv_loss_val
                            )
                            adv_loss_val += (
                                G_A_adv_loss_val + G_B_adv_loss_val
                            ).item() * lambda_adv

                            # reconstruction loss
                            l_rec_AA_val = recon_criterion(AA, cellA_val)
                            l_rec_BB_val = recon_criterion(BB, cellB_val)
                            recon_loss_val += (
                                l_rec_AA_val + l_rec_BB_val
                            ).item() * lambda_recon

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
                            ).item() * lambda_encoding

                            G_loss_val += (
                                adv_loss_val + recon_loss_val + encoding_loss_val
                            )

                    print(
                        "[%d/%d] adv_loss_val: %.4f  recon_loss_val: %.4f encoding_loss_val: %.4f  G_loss: %.4f D_A_loss_val: %.4f D_B_loss_val: %.4f"
                        % (
                            iteration,
                            niter,
                            adv_loss_val / counter,
                            recon_loss_val / counter,
                            encoding_loss_val / counter,
                            G_loss / counter,
                            D_A_loss_val / counter,
                            D_B_loss_val / counter,
                        )
                    )

                    
                    print(
                        "[%d/%d] adv_loss_val: %.4f  recon_loss_val: %.4f encoding_loss_val: %.4f  G_loss: %.4f D_A_loss_val: %.4f D_B_loss_val: %.4f"
                        % (
                            iteration,
                            niter,
                            adv_loss_val / counter,
                            recon_loss_val / counter,
                            encoding_loss_val / counter,
                            G_loss / counter,
                            D_A_loss_val / counter,
                            D_B_loss_val / counter,
                        )
                    )

        print("Saving model to", model_path)
        torch.save(self.E.state_dict(), os.path.join(model_path, "E.pth"))
        torch.save(self.G_A.state_dict(), os.path.join(model_path, "G_A.pth"))
        torch.save(self.G_B.state_dict(), os.path.join(model_path, "G_B.pth"))
        torch.save(self.D_A.state_dict(), os.path.join(model_path, "D_A.pth"))
        torch.save(self.D_B.state_dict(), os.path.join(model_path, "D_B.pth"))
        print("Training finished.")

    def predict(self, control_adata, cell_type_key, condition_key):
        self.D_A.eval()
        self.D_B.eval()
        self.G_A.eval()
        self.G_B.eval()
        self.E.eval()
        
        if sparse.issparse(control_adata.X):
            control_tensor = Tensor(control_adata.X.toarray())
        else:
            control_tensor = Tensor(control_adata.X)
        if self.use_cuda and cuda_is_available():
            print("Using CUDA for evaluation")
            control_tensor = control_tensor.cuda()
            
        with torch.no_grad():
            control_z = self.E(control_tensor)
            case_pred = self.G_B(control_z)
            
        pred_perturbed_adata = sc.AnnData(
            X=case_pred.cpu().detach().numpy(),
            obs={
                condition_key: ["Predicted data"] * len(case_pred),
                cell_type_key: control_adata.obs[cell_type_key].tolist(),
            },
        )
        pred_perturbed_adata.var_names = control_adata.var_names
        
        print("Predicting data finished")
        return pred_perturbed_adata