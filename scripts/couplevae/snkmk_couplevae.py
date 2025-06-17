from couplevae_ import *

import anndata
import numpy as np
import scanpy as sc
import seaborn as sns
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
from scipy import sparse
from scipy.sparse import issparse
from adjustText import adjust_text
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import logging
import scanpy as sc
import argparse
import yaml
import warnings
import os
from pathlib import Path
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

# Load the data
adata = sc.read_h5ad(input_file_path)
if sparse.issparse(adata.X):
    adata.X = adata.X.toarray()

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

# Run couplevae

print('---------------------------------------------')
print('--------------- Running couplevae ---------------')
print('---------------------------------------------')

print(f'--------------- Dealing with {target} ---------------')
#adata_train = adata[~((adata.obs[batch] == target) &
#                (adata.obs[condition] == condition_stimulated))].copy()

print(f'--------------- Training couplevae ---------------')

trainloader = load_h5ad_to_dataloader(adata, condition, batch, 
                                    target, condition_control, condition_stimulated, 'cuda')

network = VAE(x_dim=adata.X.shape[1],
              z_dim=200,
              alpha=0.00005,
              beta=0.05,
              dropout_rate=0.1,
              learning_rate=0.0001)

trainer = Trainer(model=network, learning_rate=0.0001, n_epochs=200, patience=20, batch_size=32)
trainer.train(train_loader=trainloader)

print(f'--------------- Predicting ---------------')
adata_val = adata[((adata.obs[batch] == target) & (adata.obs[condition] == condition_control))].copy()
pred = network.predict(adata_val, adata_val)

pred = sc.AnnData(pred, obs=adata_val.obs, var=adata_val.var)
ctrl_adata = adata[((adata.obs[batch] == target) & (adata.obs[condition] == condition_control))]
stim_adata = adata[((adata.obs[batch] == target) & (adata.obs[condition] == condition_stimulated))]

pred.obs[condition] = 'predicted'
ctrl_adata.obs[condition] = 'control'
stim_adata.obs[condition] = 'stimulated'

eval_adata = ctrl_adata.concatenate(stim_adata, pred)


print(f'couplevae sussesfully predicted stimulated {target}')

print(f'--------------- Saving results ---------------')

output_dir_path = output_dir_path.rstrip(os.sep)
h5ad_dir = os.path.join(output_dir_path, 'h5ad')
if not os.path.exists(h5ad_dir):
    os.makedirs(h5ad_dir)
eval_adata.write(f'{h5ad_dir}/{experiment_name}_couplevae_{target}.h5ad')

# Output flag
file_path = Path(f'flags/h5ad/output_run_flag_{experiment_name}_couplevae_{target}_h5ad.txt')
file_path.parent.mkdir(parents=True, exist_ok=True)
file_path.touch()
print('couplevae successfully saved the results')