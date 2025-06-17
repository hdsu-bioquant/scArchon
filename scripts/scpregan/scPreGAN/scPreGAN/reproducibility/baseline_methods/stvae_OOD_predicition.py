import scanpy as sc
import stvae
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from scipy import sparse
from torch import zeros, Tensor, LongTensor, log, cuda, save, load
import torch
from torch.cuda import is_available as cuda_is_available
from scipy import sparse
import os

data_name = 'pbmc'
if data_name == 'pbmc':
    adata = sc.read_h5ad("/home/wxj/scBranchGAN/datasets/pbmc/pbmc.h5ad")
    cell_type_key = 'cell_type'
    condition_key = 'condition'
    condition = {"case": "stimulated", "control": "control"}
elif data_name == 'hpoly':
    adata = sc.read_h5ad("/home/wxj/scBranchGAN/datasets/Hpoly/hpoly.h5ad")
    cell_type_key = 'cell_label'
    condition_key = 'condition'
    condition = {"case": "Hpoly.Day10", "control": "Control"}
elif data_name == 'species':
    adata = sc.read_h5ad("/home/wxj/scBranchGAN/datasets/species/species.h5ad")
    cell_type_key = 'species'
    condition_key = 'condition'
    condition = {"case": "LPS6", "control": "unst"}
else:
    raise Exception("InValid data name")

cell_type_list = adata.obs[cell_type_key].unique().tolist()
print(cell_type_list)
for cell_type in cell_type_list:
    print("=================processing " + cell_type + "=================")
    train_set = adata[~((adata.obs[condition_key] == condition["case"]) & (adata.obs[cell_type_key] == cell_type))]
    if sparse.issparse(train_set.X):
        train_expr = train_set.X.A
    else:
        train_expr = train_set.X
    train_expr = train_expr if isinstance(train_expr, np.ndarray) else np.array(train_expr)
    train_condition = np.array(train_set.obs[condition_key].tolist())
    train_condition = OneHotEncoder(sparse=False).fit_transform(train_condition.reshape(-1, 1))
    train_labels = np.array(train_set.obs[cell_type_key].tolist())
    train_labels = OneHotEncoder(sparse=False).fit_transform(train_labels.reshape(-1, 1))
    cfg = stvae.Config()
    cfg.epochs = 100
    cfg.input_dim = train_expr.shape[1]
    cfg.n_genes = train_expr.shape[1]
    cfg.count_labels = train_labels.shape[1]
    cfg.count_classes = train_condition.shape[1]
    model = stvae.stVAE(cfg)
    indices = np.array(train_expr.shape[0] * [0])
    test_size = 2
    train_expression, val_expression, train_condition_ohe, val_condition_ohe, train_label_ohe, test_label_ohe, train_ind, test_ind = train_test_split(
        train_expr, train_condition, train_labels,
        indices,
        random_state=cfg.random_state,
        stratify=train_condition.argmax(1), test_size=test_size
    )
    model.train((train_expression, train_condition_ohe, train_label_ohe), None)
    ge_transfer_raw = adata[
        (adata.obs[condition_key] == condition['control']) & (adata.obs[cell_type_key] == cell_type)]
    if sparse.issparse(ge_transfer_raw.X):
        ge_transfer_raw = Tensor(ge_transfer_raw.X.A)
    else:
        ge_transfer_raw = Tensor(ge_transfer_raw.X)

    if (data_name == 'pbmc') or (data_name == 'hpoly'):
        source_classes = np.zeros(shape=(ge_transfer_raw.shape[0], 2))
        source_classes[:, 0] = 1
        target_classes = np.zeros(shape=(ge_transfer_raw.shape[0], 2))
        target_classes[:, 1] = 1
    elif (data_name == 'species') or (data_name == 'covid-19-pbmc'):
        source_classes = np.zeros(shape=(ge_transfer_raw.shape[0], 2))
        source_classes[:, 1] = 1
        target_classes = np.zeros(shape=(ge_transfer_raw.shape[0], 2))
        target_classes[:, 0] = 1
    else:
        raise Exception("InValid data name")

    source_classes = Tensor(source_classes)
    target_classes = Tensor(target_classes)

    if cfg.use_cuda and cuda_is_available():
        source_classes = source_classes.cuda()
        target_classes = target_classes.cuda()
        ge_transfer_raw = ge_transfer_raw.cuda()
    pred_expression_tensor = model.model(ge_transfer_raw, target_classes)[0]
    pred_expression_np = pred_expression_tensor[0].detach().cpu().numpy()
    pred_expression_np[pred_expression_np < 0] = 0
    pred_adata = sc.AnnData(X=pred_expression_np,
                            obs={condition_key: ["pred_perturbed"] * len(pred_expression_np),
                                 cell_type_key: [cell_type] * len(pred_expression_np)
                                 })
    pred_adata.var_names = adata.var_names
    if not os.path.exists(f"./{data_name}_pred_data"):
        os.mkdir(f"./{data_name}_pred_data")
    pred_adata.write_h5ad(f"./{data_name}_pred_data/pred_adata_{cell_type}.h5ad")
    del cfg
    del model
    del ge_transfer_raw
    del source_classes
    del target_classes
    torch.cuda.empty_cache()
print("training all finished")
