import sys
import numpy as np
import scanpy as sc
import os
import trvae

data_name = 'pbmc'
if data_name == "hpoly":
    train_path = "/home/wxj/scBranchGAN/datasets/Hpoly/hpoly.h5ad"
    conditions = ["Control", "Hpoly.Day10"]
    source_condition = "Control"
    target_condition = "Hpoly.Day10"
    labelencoder = {"Control": 0, "Hpoly.Day10": 1}
    cell_type_key = "cell_label"
    condition_key = "condition"
elif data_name == "pbmc":
    train_path = "/home/wxj/scBranchGAN/datasets/pbmc/pbmc.h5ad"
    conditions = ["control", "stimulated"]
    source_condition = "control"
    target_condition = "stimulated"
    labelencoder = {"control": 0, "stimulated": 1}
    cell_type_key = "cell_type"
    condition_key = "condition"
elif data_name == "species":
    train_path = "/home/wxj/scBranchGAN/datasets/species/species.h5ad"
    conditions = ["unst", "LPS6"]
    source_condition = "unst"
    target_condition = "LPS6"
    labelencoder = {"unst": 0, "LPS6": 1}
    cell_type_key = "species"
    condition_key = "condition"
else:
    raise Exception("InValid data name")

train_adata = sc.read(train_path)
train_adata = train_adata[train_adata.obs[condition_key].isin(conditions)]

if train_adata.shape[1] > 2000:
    sc.pp.highly_variable_genes(train_adata, n_top_genes=2000)
    train_adata = train_adata[:, train_adata.var['highly_variable']]
print(train_adata)

for specific_celltype in train_adata.obs[cell_type_key].unique().tolist():
    print("process for" + specific_celltype)
    net_train_adata = train_adata[
        ~((train_adata.obs[cell_type_key] == specific_celltype) & (
            train_adata.obs[condition_key].isin([target_condition])))]
    print(net_train_adata)
    network = trvae.models.trVAE(x_dimension=net_train_adata.shape[1],
                                 z_dimension=40,
                                 conditions=conditions,
                                 model_path=f"./models/trVAE/{data_name}_{specific_celltype}/",
                                 output_activation='relu',
                                 verbose=5
                                 )

    network.train(net_train_adata,
                  condition_key,
                  n_epochs=1000,
                  batch_size=512,
                  verbose=2,
                  early_stop_limit=20,
                  lr_reducer=10,
                  )

    cell_type_adata = train_adata[train_adata.obs[cell_type_key] == specific_celltype]
    source_adata = cell_type_adata[cell_type_adata.obs[condition_key] == source_condition]
    target_adata = cell_type_adata[cell_type_adata.obs[condition_key] == target_condition]
    pred_adata = network.predict(source_adata,
                                 condition_key,
                                 target_condition
                                 )

    pred_adata.obs[condition_key] = [f"pred_perturbed"] * pred_adata.shape[0]
    pred_adata.obs[cell_type_key] = specific_celltype
    if not os.path.exists(f"./{data_name}_pred_adata"):
        os.mkdir(f"./{data_name}_pred_adata")
    pred_adata.write_h5ad(f"./{data_name}_pred_adata/{specific_celltype}_pred.h5ad")
