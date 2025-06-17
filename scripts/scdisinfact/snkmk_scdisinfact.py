import sys, os
import torch
import numpy as np 
import pandas as pd
from scDisInFact_ import scdisinfact, create_scdisinfact_dataset
from scDisInFact_ import utils
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.sparse as sp
from umap import UMAP
from pathlib import Path
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import logging
import scanpy as sc
import argparse
import yaml
import warnings
import os
warnings.filterwarnings("ignore")

# Function to load the config file
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Argument parser setup
def get_arguments():
    parser = argparse.ArgumentParser(description="Run experiment with configuration.")
    parser.add_argument("--config", type=str, required=True, help="Path to the config.yml file")
    parser.add_argument("--file_path", type=str, help="Override input_data.file_path")
    parser.add_argument("--output_dir", type=str, help="Override input_data.file_path")
    parser.add_argument("--condition", type=str, help="Override experiment.condition")
    parser.add_argument("--condition_control", type=str, help="Override experiment.condition_control")
    parser.add_argument("--condition_stimulated", type=str, help="Override experiment.condition_stimulated")
    parser.add_argument("--batch", type=str, help="Override experiment.batch")
    parser.add_argument("--target", type=str, help="Override experiment.target")
    parser.add_argument("--experiment_name", type=str, help="Override experiment.experiment_name")
    return parser.parse_args()

# Parse command-line arguments
args = get_arguments()

# Load the config file
config = load_config(args.config)

# Override config values with command-line arguments, if provided
input_file_path = args.file_path or config['input_data']['file_path']
output_dir_path = args.output_dir or config['experiment']['output_dir']
condition = args.condition or config['experiment']['condition']
condition_control = args.condition_control or config['experiment']['condition_control']
condition_stimulated = args.condition_stimulated or config['experiment']['condition_stimulated']
batch = args.batch or config['experiment']['batch']
target = args.target or config['experiment']['target']
experiment_name = args.experiment_name or config['experiment']['experiment_name']

# Print the final values (or use them in your script)
print(f"Input file path: {input_file_path}")
print(f"Condition: {condition}")
print(f"Condition (Control): {condition_control}")
print(f"Condition (Stimulated): {condition_stimulated}")
print(f"Batch: {batch}")
print(f"Target: {target}")
print(f"Experiment Name: {experiment_name}")
print(f"Results will be saved at: {output_dir_path}")

from scipy.sparse import issparse, csr_matrix

# Load the data
adata = sc.read_h5ad(input_file_path)

# Ensure that adata.X is sparse
if not issparse(adata.X):
    adata.X = csr_matrix(adata.X)

# Check the data
if adata is None:
    logging.error("Data not loaded correctly")
    exit(1)

if condition not in adata.obs.columns:
    logging.error(f"Condition column not found in data. Please check the config file. Possibilities based on input data are: {adata.obs.columns.to_list()}.")
    exit(1)

if batch not in adata.obs.columns:
    logging.error("Batch column not found in data. Please check the config file.")
    exit(1)

if condition_control not in adata.obs[condition].values:
    logging.error(f"Condition (Control) '{condition_control}' not found in '{condition}' column in data.")
    exit(1)

if condition_stimulated not in adata.obs[condition].values:
    logging.error(f"Condition (Stimulated) '{condition_stimulated}' not found in '{condition}' column in data.")
    exit(1)

if target not in adata.obs[batch].values:
    logging.error(f"Target '{target}' not found in data.")
    exit(1)

if not os.path.exists(output_dir_path):
    os.makedirs(output_dir_path)

# Run scDisInFact

print('---------------------------------------------')
print('--------------- Running scDisInFact ---------------')
print('---------------------------------------------')

print(f'--------------- Dealing with {target} ---------------')
adata_train = adata[~((adata.obs[batch] == target) &
                (adata.obs[condition] == condition_stimulated))].copy()
data_dict = create_scdisinfact_dataset(adata_train.X, adata_train.obs, condition_key = [condition], batch_key = batch)


print(f'--------------- Training scDisInFact ---------------')

# default setting of hyper-parameters
reg_mmd_comm = 1e-4
reg_mmd_diff = 1e-4
reg_kl_comm = 1e-5
reg_kl_diff = 1e-2
reg_class = 1
reg_gl = 1
Ks = [8, 2]

batch_size = 64
nepochs = 200
interval = 10
lr = 5e-4
lambs = [reg_mmd_comm, reg_mmd_diff, reg_kl_comm, reg_kl_diff, reg_class, reg_gl]
model = scdisinfact(data_dict = data_dict, batch_size = batch_size, interval = interval, lr = lr, 
                    reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl, reg_class = reg_class, 
                    reg_kl_comm = reg_kl_comm, reg_kl_diff = reg_kl_diff, seed = 0, device = device)
model.train()
losses = model.train_model(nepochs = nepochs, recon_loss = "NB")


print(f'--------------- Predicting ---------------')

_ = model.eval()
train_cells = adata_train.obs_names
adata_val = adata[((adata.obs[batch] == target) &
                (adata.obs[condition] == condition_control))].copy()
pred = model.predict_counts(input_counts = adata_val.X, meta_cells = adata_val.obs, condition_keys = [condition], 
                                            batch_key = batch, predict_conds = [condition_stimulated], predict_batch = target)

pred = sc.AnnData(pred, obs=adata_val.obs, var=adata_val.var)
pred.obs[condition] = 'predicted'

ctrl_adata = adata[((adata.obs[batch] == target) & (adata.obs[condition] == condition_control))]
ctrl_denoised = model.predict_counts(input_counts = ctrl_adata.X, meta_cells = ctrl_adata.obs, condition_keys = [condition], 
                                          batch_key = batch, predict_conds = None, predict_batch = None)
ctrl_adata.obs[condition] = 'control'

stim_adata = adata[((adata.obs[batch] == target) & (adata.obs[condition] == condition_stimulated))]
stim_denoised = model.predict_counts(input_counts = stim_adata.X, meta_cells = stim_adata.obs, condition_keys = [condition], 
                                          batch_key = batch, predict_conds = None, predict_batch = None)
stim_adata.obs[condition] = 'stimulated'

stim_adata.obs_names_make_unique()
ctrl_adata.obs_names_make_unique()
pred.obs_names_make_unique()

stim_adata.var_names_make_unique()
ctrl_adata.var_names_make_unique()
pred.var_names_make_unique()

eval_adata = ctrl_adata.concatenate(stim_adata, pred)


print(f'scDisInFact sussesfully predicted stimulated {target}')

print(f'--------------- Saving results ---------------')

output_dir_path = output_dir_path.rstrip(os.sep)
h5ad_dir = os.path.join(output_dir_path, 'h5ad')
if not os.path.exists(h5ad_dir):
    os.makedirs(h5ad_dir)
eval_adata.write(f'{h5ad_dir}/{experiment_name}_scdisinfact_{target}.h5ad')

# Output flag
file_path = Path(f'flags/h5ad/output_run_flag_{experiment_name}_scdisinfact_{target}_h5ad.txt')
file_path.parent.mkdir(parents=True, exist_ok=True)
file_path.touch()

print('scDisInFact successfully saved the results')