import torch
from torch import nn
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi import _CONSTANTS
import numpy as np
from vidr.modules import *
from collections import Counter

# at beginning of the script
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class VIDRModel(BaseModuleClass):
    """
    Variational AutoEncoder for latent space arithmetic for perutrbation prediction.
    
    Parameters
    ----------
    input_dim
        Number of input genes
    hidden_dim
        Number of nodes per hidden layer
    latent_dim
        Dimensionality of the latent space
    n_hidden_layers
        Number of hidden layers used for encoder and decoder NNs
    dropout_rate
        Dropout rate for neural networks
    kl_weight
        Weight for kl divergence
    linear_decoder
        Boolean for whether or not to use linear decoder for a more interpretable model.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 800,
        latent_dim: int = 10,
        n_hidden_layers: int = 2,
        dropout_rate: float = 0.1,
        latent_distribution: str = "normal",
        kl_weight: float = 0.00005,
        linear_decoder: bool = True,
        nca_loss: bool = False,
        dose_loss = None,
    ):
        super().__init__()
        self.n_hidden_layers = n_hidden_layers
        self.latent_dim = latent_dim
        self.latent_distribution = "normal"
        self.kl_weight = kl_weight
        self.nca_loss = nca_loss
        self.dose_loss = dose_loss
        
        #         self.encoder = Encoder(
#             input_dim,
#             latent_dim,
#             n_layers=n_layers,
#             hidden_dim=hidden_dim,
#             dropout_rate=dropout_rate,
#             distribution=latent_distribution,
#             use_batch_norm= True,
#             use_layer_norm=False,
#             activation_fn=torch.nn.LeakyReLU,
#         )

        self.encoder = VIDREncoder(
            input_dim,
            latent_dim,
            hidden_dim,
            n_hidden_layers
        )
        
        
        #Decoder
        #Include Input Module
        
        self.nonlin_decoder = VIDRDecoder(
            input_dim,
            latent_dim,
            hidden_dim,
            n_hidden_layers
        )
        
        self.lin_decoder =  torch.nn.Sequential(
                        nn.Linear(latent_dim, input_dim),
                        nn.BatchNorm1d(input_dim, momentum=0.01, eps=0.001)
                        )
        
        self.decoder = self.lin_decoder if linear_decoder else self.nonlin_decoder
    
    def _get_inference_input(self, tensors):
        x = tensors[_CONSTANTS.X_KEY]
        input_dict = dict(
            x=x,
        )
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        input_dict = {
            "z": z,
        }
        return input_dict

    @auto_move_data
    def inference(self, x):
        """
        High level inference method.
        Runs the inference (encoder) model.
        """
        mean, var, latent_rep = self.encoder(x)

        outputs = dict(z=latent_rep, qz_m=mean, qz_v=var)
        return outputs

    @auto_move_data
    def generative(self, z):
        """Runs the generative model."""
        px = self.decoder(z)

        return dict(px=px)

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
    ):
        x = tensors[_CONSTANTS.X_KEY]
        mean = inference_outputs["qz_m"]
        var = inference_outputs["qz_v"]
        x_hat = generative_outputs["px"]
        
        std = var.sqrt()
        
        kld = kl(
            Normal(mean, std),
            Normal(0, 1),
        ).sum(dim=1)
        
        rl = self.get_reconstruction_loss(x, x_hat)
        if self.nca_loss:
            if np.any(self.dose_loss != None):
                disc_labels = [tensors['labels']]
                cont_inds = tensors['batch_indices'].cpu().detach().numpy()
                cont_labels = []
                cont_labels.append(
                        torch.tensor([[self.dose_loss[int(i[0])]] for i in cont_inds]).to(device))
                ncal, disc_loss, cont_loss = self.get_nca_loss(mean, disc_labels, cont_labels)
            else:
                disc_labels = [tensors['batch_indices'], tensors['labels']]
                cont_labels = None
                ncal, disc_loss, cont_loss = self.get_nca_loss(mean, disc_labels, cont_labels)
        else:
            ncal = torch.tensor(0, device = device)

        loss = (0.5 * rl + 0.5 * (kld * self.kl_weight)).mean() - 10 * ncal
        return LossRecorder(loss, rl, kld, kl_global=0.0)

    @torch.no_grad()
    def sample(
        self,
        tensors,
        n_samples=1,
    ) -> np.ndarray:
        r"""
        Generate observation samples from the posterior predictive distribution.
        The posterior predictive distribution is written as :math:`p(\hat{x} \mid x)`.
        Parameters
        ----------
        tensors
            Tensors dict
        n_samples
            Number of required samples for each cell
        library_size
            Library size to scale scamples to
        Returns
        -------
        x_new : :py:class:`torch.Tensor`
            tensor with shape (n_cells, n_genes, n_samples)
        """
        inference_kwargs = dict(n_samples=n_samples)
        inference_outputs, generative_outputs, = self.forward(
            tensors,
            inference_kwargs=inference_kwargs,
            compute_loss=False,
        )
        px = Normal(generative_outputs["px"], 1).sample()
        return px.cpu().numpy()

    def get_reconstruction_loss(self, x, x_hat) -> torch.Tensor:
        loss = ((x - x_hat) ** 2).sum(dim=1)
        return loss
    
    def get_nca_loss(self, z, disc, cont) -> torch.Tensor: 
        #losses list
        losses = []

        #Distance Matrix
        d1 = z.clone()
        d2 = z.clone()
        dist = torch.cdist(d1, d2, p = 2)
        p = dist.clone()

        #Set diagonal to inf
        p.diagonal().copy_(np.inf*torch.ones(len(p)))
        #Softmax 
        p = torch.softmax(-p, dim =1)

        #Calculating Latent Loss for Discrete Labels
        if disc != None:
            cells = len(disc[0])
            masks = np.zeros((cells,cells))
            maxVal = 0
            for Y in disc:
                Y = Y.cpu().detach().numpy()
                Y = np.asarray([y[0] for y in Y])
                counts = Counter(Y)
                area_counts = {k:  v for k, v in list(counts.items())} 
                m = Y[:, np.newaxis] == Y[np.newaxis, :]
                #Count number of labels (for each unique label)
                for k, v in list(area_counts.items()):
                    if 'nan' != str(k): 
                        maxVal += 1
                        masks = masks + m*(Y == k)*(1/v)
            if maxVal != 0:
                    masks = masks*(1/maxVal)
            
            masks = torch.from_numpy(masks).float().to(device)
            
            masked_p = p * masks
            losses += [torch.sum(masked_p)]
       
        else:
            losses += [torch.tensor(0, device = device)]

        if cont != None:
            cells = len(cont[0])
            n_cont_weights = len(cont)
            
            weights = torch.empty((cells, cells*len(cont)), dtype=torch.float, device = device)
            for n in range(n_cont_weights):
                Y = cont[n]
                Y1 = Y.clone()
                Y2 = Y.clone()
                dist = torch.cdist(Y1, Y2) #Get distances between continuous labels
                cont_dists = dist.clone()
                cont_dists.diagonal().copy_(np.inf*torch.ones(len(cont_dists))) #Set diagonal to inf
                cont_dists = torch.nan_to_num(cont_dists, nan=np.inf) #Use nans for data missing labels
                cont_dists = torch.softmax(-cont_dists, dim = 1) #soft max
                s = cells*n
                e = cells*(n+1)
                weights[:, s:e] = cont_dists

            for n in range(0,n_cont_weights):
                s = cells*n
                e = cells*(n+1)
                weight_calc = p * weights[:,s:e]
                m, m_indexes = torch.max(weights[:,s:e],dim=1)
                masked_p = weight_calc / torch.sum(m)
                cont_max, inds = torch.max(masked_p,dim=1)
                losses += [torch.sum(masked_p)]

        lossVals = torch.stack(losses,dim=0)

        scaled_losses = lossVals

        disc_loss = torch.sum(scaled_losses[0])
        
        if cont != None:
            cont_loss = torch.sum(*scaled_losses[1:])
        else:
            cont_loss = torch.tensor(0, device = device)

        loss = disc_loss + cont_loss
        return loss, disc_loss, cont_loss

