import sys
import logging
import scanpy as sc
import scgen
import os

logger = logging.getLogger("scvi.inference.autotune")
logger.setLevel(logging.WARNING)

data_name = "pbmc"
if data_name == "pbmc":
    train = sc.read("/home/wxj/scBranchGAN/datasets/pbmc/pbmc.h5ad")
    cell_type_key = "cell_type"
    condition_key = "condition"
    condition = {"case": "stimulated", "control": "control"}
elif data_name == 'hpoly':
    train = sc.read("/home/wxj/scBranchGAN/datasets/Hpoly/hpoly.h5ad")
    cell_type_key = 'cell_label'
    condition_key = 'condition'
    condition = {"case": "Hpoly.Day10", "control": "Control"}
elif data_name == 'species':
    train = sc.read("/home/wxj/scBranchGAN/datasets/species/species.h5ad")
    cell_type_key = 'species'
    condition_key = 'condition'
    condition = {"case": "LPS6", "control": "unst"}
else:
    raise Exception("InValid data name")

cell_type_list = train.obs[cell_type_key].unique().tolist()
for cell_type in cell_type_list[6:]:
    print("=================processing " + cell_type + "=================")
    train_new = train[~((train.obs[cell_type_key] == cell_type) &
                        (train.obs[condition_key] == condition["case"]))]
    train_new = scgen.setup_anndata(train_new, copy=True, batch_key=condition_key, labels_key=cell_type_key)
    model = scgen.SCGEN(train_new)
    model.train(max_epochs=100, batch_size=32, early_stopping=True, early_stopping_patience=25)
    pred_adata, delta = model.predict(ctrl_key=condition["control"], stim_key=condition["case"],
                                      celltype_to_predict=cell_type)
    pred_adata.obs['condition'] = 'pred_perturbed'
    if not os.path.exists(f"./{data_name}_pred_data"):
        os.mkdir(f"./{data_name}_pred_data")
    pred_adata.write_h5ad(f"./{data_name}_pred_data/pred_adata_{cell_type}.h5ad")
