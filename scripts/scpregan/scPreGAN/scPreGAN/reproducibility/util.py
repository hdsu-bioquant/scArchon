import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn import preprocessing


def label_encoder(adata, cell_type_key, encode_attr):
    le = preprocessing.LabelEncoder()
    encode_list = encode_attr + adata.obs[cell_type_key].tolist()
    labels = le.fit_transform(encode_list)
    labels = labels.reshape(-1, 1)

    le_OneHot = preprocessing.OneHotEncoder(sparse_output=False)
    labels_onehot = le_OneHot.fit_transform(labels)

    return labels_onehot[len(encode_attr):, ]


def load_anndata(adata, condition_key, condition, cell_type_key, prediction_type=None, out_sample_prediction=False):

    encode_attr = adata.obs[cell_type_key].unique().tolist()
    adata_celltype_ohe = label_encoder(adata, cell_type_key, encode_attr)

    adata_celltype_ohe_pd = pd.DataFrame(data=adata_celltype_ohe, index=adata.obs_names)

    if out_sample_prediction:
        case_adata = adata[
            ~(adata.obs[cell_type_key] == prediction_type) & (adata.obs[condition_key] == condition['case'])]
    else:
        case_adata = adata[adata.obs[condition_key] == condition['case']]

    control_adata = adata[adata.obs[condition_key] == condition['control']]

    if sparse.issparse(adata.X):
        control_pd = pd.DataFrame(data=control_adata.X.toarray(), index=control_adata.obs_names,
                                  columns=control_adata.var_names)
        case_pd = pd.DataFrame(data=case_adata.X.toarray(), index=case_adata.obs_names, columns=case_adata.var_names)
    else:
        control_pd = pd.DataFrame(data=control_adata.X, index=control_adata.obs_names,
                                  columns=control_adata.var_names)
        case_pd = pd.DataFrame(data=case_adata.X, index=case_adata.obs_names, columns=case_adata.var_names)

    control_celltype_ohe_pd = adata_celltype_ohe_pd.loc[control_pd.index, :]
    case_celltype_ohe_pd = adata_celltype_ohe_pd.loc[case_pd.index, :]

    return control_pd, control_celltype_ohe_pd, case_pd, case_celltype_ohe_pd
