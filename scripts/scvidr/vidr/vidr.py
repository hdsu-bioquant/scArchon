from typing import Optional, Sequence
import sys
sys.path.append('../')
import numpy
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from adjustText import adjust_text
from anndata import AnnData
from matplotlib import pyplot
from scipy import sparse, stats
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin, VAEMixin

#Create Access to my code

from vidr.utils import *
from vidr.vidr_model import VIDRModel

class VIDR(VAEMixin, UnsupervisedTrainingMixin, BaseModelClass):
    """
    Implementation of scGen model for batch removal and perturbation prediction.
    Parameters
    ----------
    adata
        AnnData object that has been registered via :func:`~scgen.setup_anndata`.
    hidden_dim
        Number of nodes per hidden layer.
    latent_dim
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    dropout_rate
        Dropout rate for neural networks.
    **model_kwargs
        Keyword args for :class:`~scgen.SCGENVAE`
    Examples
    --------
    >>> vae = scgen.SCGEN(adata)
    >>> vae.train()
    >>> adata.obsm["X_scgen"] = vae.get_latent_representation()
    """

    def __init__(
        self,
        adata: AnnData,
        hidden_dim: int = 800,
        latent_dim: int = 100,
        n_hidden_layers: int = 2,
        dropout_rate: float = 0.2,
        kl_weight = 5e-5,
        linear_decoder: bool = False,
        continuous: bool = False,
        nca_loss: bool = False,
        dose_loss: bool = False,
        **model_kwargs,
    ):
        super(VIDR, self).__init__(adata)
        self.adata = adata
        self._is_linear_decoder = linear_decoder
        dose_loss = np.log1p(self.adata.uns['_scvi']['categorical_mappings']['_scvi_batch']['mapping'].astype(float)) if dose_loss else None
        print(dose_loss)
        self.module = VIDRModel(
            input_dim=adata.shape[1],
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_hidden_layers=n_hidden_layers,
            dropout_rate=dropout_rate,
            linear_decoder = linear_decoder,
	    kl_weight = kl_weight,
	    nca_loss = nca_loss,
	    dose_loss = dose_loss,
        )
        self._model_summary_string = (
            "VIDR Model with the following params: \nhidden_dim: {}, latent_dim: {}, n_layers: {}, dropout_rate: "
            "{}"
        ).format(
            hidden_dim,
            latent_dim,
            n_hidden_layers,
            dropout_rate,
        )
        self.init_params_ = self._get_init_params(locals())

    def predict(
        self,
        ctrl_key=None,
        treat_key=None,
        cell_type_to_predict=None,
        regression = False,
        continuous = False,
        low_dose = False,
        doses = None,
    ) -> AnnData:
        """
        Predicts the cell type provided by the user in treatulated condition.
        Parameters
        ----------
        ctrl_key: basestring
            key for `control` part of the `data` found in `condition_key`.
        treat_key: basestring
            key for `treated` part of the `data` found in `condition_key`.
        adata_to_predict: `~anndata.AnnData`
            Adata for unperturbed cells you want to be predicted.
        celltype_to_predict: basestring
            The cell type you want to be predicted.
        restrict_arithmetic_to: basestring or dict
            Dictionary of celltypes you want to be observed for prediction.
        Returns
        -------
        predicted_cells: numpy nd-array
            `numpy nd-array` of predicted cells in primary space.
        delta: float
            Difference between treatulated and control cells in latent space
        """
        if not self.is_trained_:
            raise Exception(
                            "It looks like your model hasn't been trained, please train it with model.train"
                            )
        # use keys registered from `setup_anndata()`
        cell_type_key = self.scvi_setup_dict_["categorical_mappings"]["_scvi_labels"][
            "original_key"
        ]
        treatment_key = self.scvi_setup_dict_["categorical_mappings"]["_scvi_batch"][
            "original_key"
        ]

        ctrl_x = self.adata[self.adata.obs[treatment_key] == ctrl_key]
        treat_x = self.adata[self.adata.obs[treatment_key] == treat_key]
        ctrl_x = random_sample(ctrl_x, cell_type_key)
        treat_x = random_sample(treat_x, cell_type_key)
        
        #Balancing across treatments 
        new_adata = ctrl_x.concatenate(treat_x)
        new_adata = random_sample(new_adata, treatment_key, max_or_min = "min", replacement = False)
        
        #Densify Sparse Matrix
        if sparse.issparse(new_adata.X):
                new_adata.X = new_adata.X.A
        
        #Control Data 
        if not low_dose:
            ctrl_data = new_adata[(new_adata.obs[cell_type_key] == cell_type_to_predict) & (new_adata.obs[treatment_key] == ctrl_key)]
            latent_cd = self.get_latent_representation(ctrl_data)
        else:
            ctrl_data = new_adata[(new_adata.obs[cell_type_key] == cell_type_to_predict) & (new_adata.obs[treatment_key] == treat_key)]
            latent_cd = self.get_latent_representation(ctrl_data)

        if sparse.issparse(new_adata.X):
                new_adata.X = new_adata.X.A
        
        #Are we regressing the delta on the latent space or not
        if not regression:
            #No regression on latent space
            ctrl_x = new_adata[new_adata.obs[treatment_key] == ctrl_key].copy()
            treat_x = new_adata[new_adata.obs[treatment_key] == treat_key].copy()
            #Compress data to latent space and then calculate the delta
            latent_ctrl = np.average(self.get_latent_representation(ctrl_x), axis = 0)
            latent_treat = np.average(self.get_latent_representation(treat_x), axis = 0)
            delta = latent_treat - latent_ctrl
        else:
            #Regression on latent space
            latent_X =  self.get_latent_representation(new_adata)
            latent_adata = sc.AnnData(X=latent_X, obs = new_adata.obs.copy())
            #Get deltas and control centroids for each cell tpye in the training dataset
            deltas = []
            latent_centroids = []
            cell_types = np.unique(latent_adata.obs[cell_type_key])
            for cell in cell_types:
                if cell != cell_type_to_predict:
                    latent_ctrl = latent_adata[(latent_adata.obs[cell_type_key] == cell) & (latent_adata.obs[treatment_key] == ctrl_key)]
                    latent_treat = latent_adata[(latent_adata.obs[cell_type_key] == cell) & (latent_adata.obs[treatment_key] == treat_key)]
                    ctrl_centroid = np.average(latent_ctrl.X, axis = 0)
                    treat_centroid = np.average(latent_treat.X, axis = 0)
                    delta = np.average(latent_treat.X, axis = 0) - np.average(latent_ctrl.X, axis = 0)
                    deltas.append(delta)
                    latent_centroids.append(ctrl_centroid)
            lr = LinearRegression()
            reg = lr.fit(latent_centroids, deltas)
            delta = reg.predict([np.average(latent_cd, axis = 0)])[0]
        
        #Continuous or Binary Perturbation
        if not continuous:
            treat_pred = delta + latent_cd
            predicted_cells = self.module.generative(torch.Tensor(treat_pred))["px"].cpu().detach().numpy()
            predicted_adata = sc.AnnData(X=predicted_cells , obs=ctrl_data.obs.copy(), var=ctrl_data.var.copy(),obsm=ctrl_data.obsm.copy(),)
            if not regression:
                return predicted_adata, delta
            else:
                return predicted_adata, delta, reg
        else:
            if not low_dose:
                treat_pred_dict = {d:delta*(np.log1p(d)/np.log1p(max(doses))) + latent_cd  for d in doses if d > min(doses)}
                predicted_cells_dict = {d:self.module.generative(torch.Tensor(treat_pred_dict[d]))["px"].cpu().detach().numpy() for d in doses if d > min(doses)}
                predicted_adata_dict = {d:sc.AnnData(X=predicted_cells_dict[d], obs=ctrl_data.obs.copy(), var=ctrl_data.var.copy(),obsm=ctrl_data.obsm.copy(),) for d in doses if d > min(doses)}
            else:
                treat_pred_dict = {d:latent_cd - delta*((np.log1p(max(doses)) - np.log1p(d))/np.log1p(max(doses)))  for d in doses if d < max(doses)}
                predicted_cells_dict = {d:self.module.generative(torch.Tensor(treat_pred_dict[d]))["px"].cpu().detach().numpy() for d in doses if d < max(doses)}
                predicted_adata_dict = {d:sc.AnnData(X=predicted_cells_dict[d], obs=ctrl_data.obs.copy(), var=ctrl_data.var.copy(),obsm=ctrl_data.obsm.copy(),) for d in doses if d < max(doses)}
            if not regression:
                return predicted_adata_dict, delta
            else:
                return predicted_adata_dict, delta, reg
   #Code taken from Lotfollahi et al's scGen from its pytorch implementation.
    def reg_mean_plot(
        self,
        adata,
        axis_keys,
        labels,
        path_to_save="./reg_mean.pdf",
        save=True,
        gene_list=None,
        show=False,
        top_100_genes=None,
        verbose=False,
        legend=True,
        title=None,
        x_coeff=0.30,
        y_coeff=0.8,
        fontsize=14,
        **kwargs,
    ):
        """
        Plots mean matching figure for a set of specific genes.
        Parameters
        ----------
        adata: `~anndata.AnnData`
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model. Must have been setup with `batch_key` and `labels_key`,
            corresponding to batch and cell type metadata, respectively.
        axis_keys: dict
            Dictionary of `adata.obs` keys that are used by the axes of the plot. Has to be in the following form:
             `{"x": "Key for x-axis", "y": "Key for y-axis"}`.
        labels: dict
            Dictionary of axes labels of the form `{"x": "x-axis-name", "y": "y-axis name"}`.
        path_to_save: basestring
            path to save the plot.
        save: boolean
            Specify if the plot should be saved or not.
        gene_list: list
            list of gene names to be plotted.
        show: bool
            if `True`: will show to the plot after saving it.
        Examples
        --------
        >>> import anndata
        >>> import scgen
        >>> import scanpy as sc
        >>> train = sc.read("./tests/data/train.h5ad", backup_url="https://goo.gl/33HtVh")
        >>> scgen.setup_anndata(train)
        >>> network = scgen.SCGEN(train)
        >>> network.train()
        >>> unperturbed_data = train[((train.obs["cell_type"] == "CD4T") & (train.obs["condition"] == "control"))]
        >>> pred, delta = network.predict(
        >>>     adata=train,
        >>>     adata_to_predict=unperturbed_data,
        >>>     ctrl_key="control",
        >>>     treat_key="treatulated"
        >>>)
        >>> pred_adata = anndata.AnnData(
        >>>     pred,
        >>>     obs={"condition": ["pred"] * len(pred)},
        >>>     var={"var_names": train.var_names},
        >>>)
        >>> CD4T = train[train.obs["cell_type"] == "CD4T"]
        >>> all_adata = CD4T.concatenate(pred_adata)
        >>> network.reg_mean_plot(
        >>>     all_adata,
        >>>     axis_keys={"x": "control", "y": "pred", "y1": "treatulated"},
        >>>     gene_list=["ISG15", "CD3D"],
        >>>     path_to_save="tests/reg_mean.pdf",
        >>>     show=False
        >>> )
        """
        import seaborn as sns

        sns.set()
        sns.set(color_codes=True)

        if sparse.issparse(adata.X):
            adata.X = adata.X.A
        condition_key = self.scvi_setup_dict_["categorical_mappings"]["_scvi_batch"][
            "original_key"
        ]

        diff_genes = top_100_genes
        treat = adata[adata.obs[condition_key] == axis_keys["y"]]
        ctrl = adata[adata.obs[condition_key] == axis_keys["x"]]
        if diff_genes is not None:
            if hasattr(diff_genes, "tolist"):
                diff_genes = diff_genes.tolist()
            adata_diff = adata[:, diff_genes]
            treat_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["y"]]
            ctrl_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["x"]]
            x_diff = numpy.average(ctrl_diff.X, axis=0)
            y_diff = numpy.average(treat_diff.X, axis=0)
            m, b, r_value_diff, p_value_diff, std_err_diff = stats.linregress(
                x_diff, y_diff
            )
            if verbose:
                print("top_100 DEGs mean: ", r_value_diff ** 2)
        x = numpy.average(ctrl.X, axis=0)
        y = numpy.average(treat.X, axis=0)
        m, b, r_value, p_value, std_err = stats.linregress(x, y)
        if verbose:
            print("All genes mean: ", r_value ** 2)
        df = pd.DataFrame({axis_keys["x"]: x, axis_keys["y"]: y})
        ax = sns.regplot(x=axis_keys["x"], y=axis_keys["y"], data=df)
        ax.tick_params(labelsize=fontsize)
        if "range" in kwargs:
            start, stop, step = kwargs.get("range")
            ax.set_xticks(numpy.arange(start, stop, step))
            ax.set_yticks(numpy.arange(start, stop, step))
        ax.set_xlabel(labels["x"], fontsize=fontsize)
        ax.set_ylabel(labels["y"], fontsize=fontsize)
        if gene_list is not None:
            texts = []
            for i in gene_list:
                j = adata.var_names.tolist().index(i)
                x_bar = x[j]
                y_bar = y[j]
                texts.append(pyplot.text(x_bar, y_bar, i, fontsize=11, color="black"))
                pyplot.plot(x_bar, y_bar, "o", color="red", markersize=5)
                # if "y1" in axis_keys.keys():
                # y1_bar = y1[j]
                # pyplot.text(x_bar, y1_bar, i, fontsize=11, color="black")
        if gene_list is not None:
            adjust_text(
                texts,
                x=x,
                y=y,
                arrowprops=dict(arrowstyle="->", color="grey", lw=0.5),
                force_points=(0.0, 0.0),
            )
        if legend:
            pyplot.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        if title is None:
            pyplot.title("", fontsize=fontsize)
        else:
            pyplot.title(title, fontsize=fontsize)
        ax.text(
            max(x) - max(x) * x_coeff,
            max(y) - y_coeff * max(y),
            r"$\mathrm{R^2_{\mathrm{\mathsf{all\ genes}}}}$= " + f"{r_value ** 2:.2f}",
            fontsize=kwargs.get("textsize", fontsize),
        )
        if diff_genes is not None:
            ax.text(
                max(x) - max(x) * x_coeff,
                max(y) - (y_coeff + 0.15) * max(y),
                r"$\mathrm{R^2_{\mathrm{\mathsf{top\ 100\ DEGs}}}}$= "
                + f"{r_value_diff ** 2:.2f}",
                fontsize=kwargs.get("textsize", fontsize),
            )
        if save:
            pyplot.savefig(f"{path_to_save}", bbox_inches="tight", dpi=100)
        if show:
            pyplot.show()
        pyplot.close()
        if diff_genes is not None:
            return r_value ** 2, r_value_diff ** 2
        else:
            return r_value ** 2
        
