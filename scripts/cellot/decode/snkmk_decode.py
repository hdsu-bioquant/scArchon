import logging
import scanpy as sc
import scgen
import argparse
import yaml
import warnings
from scipy.sparse import csr_matrix
import os
import anndata
import torch
import numpy as np
from pathlib import Path
import pandas as pd
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
adata.obs_names_make_unique()

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




# Run scGen

print('---------------------------------------------')
print('--------------- Running scGen decoder ---------------')
print('---------------------------------------------')

print(f'--------------- Decoding ---------------')
adata_train = adata[~((adata.obs[batch] == target) &
                (adata.obs[condition] == condition_stimulated))].copy()

scgen.setup_anndata(adata, batch_key=condition, labels_key=batch)

model = scgen.SCGEN(adata, n_latent=50)

parent_folder = os.path.dirname(os.path.dirname(input_file_path))
model_path = os.path.join(parent_folder, "results/cellot_intermediate", f"intermediate_results_{experiment_name}_cellot_{target}", "model.pt")

model = model.load(model_path, adata)

print(f'--------------- Decoding predicted ---------------')

path_pred = os.path.join(parent_folder, "results/cellot_intermediate", f"intermediate_results_{experiment_name}_cellot_{target}", "evals_ood_data_space", "imputed.h5ad")
predicted = sc.read_h5ad(path_pred)
latent_tensor = torch.tensor(predicted.X, dtype=torch.float32).to(model.device) 
decoded_expression = model.module.decoder(latent_tensor).cpu().detach().numpy()

pred = anndata.AnnData(decoded_expression)

ctrl_adata = adata[((adata.obs[batch] == target) & (adata.obs[condition] == condition_control))]
ctrl_adata.obs[condition] = condition_control
stim_adata = adata[((adata.obs[batch] == target) & (adata.obs[condition] == condition_stimulated))]
stim_adata.obs[condition] = condition_stimulated

ctrl_adata.obs_names_make_unique()
stim_adata.obs_names_make_unique()
pred.obs_names_make_unique()

ctrl_adata.var_names_make_unique()
stim_adata.var_names_make_unique()
pred.var_names_make_unique()

print(f'--------------- Merging predicted, control and stimulated ---------------')
# Check if any of the .X matrices are sparse
is_sparse = isinstance(ctrl_adata.X, csr_matrix) or isinstance(stim_adata.X, csr_matrix) or isinstance(pred.X, csr_matrix)

# Handle sparse and dense concatenation separately
if is_sparse:
    # Convert sparse matrices to dense arrays and then concatenate (to avoid issues with sparse concatenation)
    X_merged = csr_matrix(np.concatenate([ctrl_adata.X.toarray() if isinstance(ctrl_adata.X, csr_matrix) else ctrl_adata.X,
                                          stim_adata.X.toarray() if isinstance(stim_adata.X, csr_matrix) else stim_adata.X,
                                          pred.X.toarray() if isinstance(pred.X, csr_matrix) else pred.X], axis=0))
else:
    # Concatenate dense matrices directly
    X_merged = np.concatenate([ctrl_adata.X, stim_adata.X, pred.X], axis=0)

# Now create a new AnnData object with the concatenated .X matrix
adata_merged = sc.AnnData(X=X_merged, 
                          obs=pd.concat([ctrl_adata.obs, stim_adata.obs, pred.obs]), 
                          var=ctrl_adata.var)

# Add the condition labels for each dataset
adata_merged.obs[condition] = ['control'] * ctrl_adata.shape[0] + ['stimulated'] * stim_adata.shape[0] + ['predicted'] * pred.shape[0]

print(f'--------------- Saving results ---------------')

# Ensure the base directory exists
output_dir_path = output_dir_path.rstrip(os.sep)
h5ad_dir = os.path.join(output_dir_path, 'h5ad')
if not os.path.exists(h5ad_dir):
    os.makedirs(h5ad_dir)
adata_merged.write(f'{h5ad_dir}/{experiment_name}_cellot_{target}.h5ad')

# Output flag 
file_path = Path(f'flags/h5ad/output_run_flag_{experiment_name}_cellot_{target}_h5ad.txt')
file_path.parent.mkdir(parents=True, exist_ok=True)
file_path.touch()

print('cellOT successfully saved the results')