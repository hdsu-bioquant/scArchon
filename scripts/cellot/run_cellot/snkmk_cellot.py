import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
cellot_path = os.path.join(script_dir, "cellot-main")
sys.path.insert(0, cellot_path)
import scanpy as sc
import torch
import argparse
import yaml
import anndata
import warnings
import logging
import numpy as np
import subprocess

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



# Run cellot

print('---------------------------------------------')
print('-------------- Running cellot -------------')
print('---------------------------------------------')
parent_folder = os.path.dirname(os.path.dirname(input_file_path))
latent_path = os.path.join(parent_folder, "results/cellot_intermediate", f"intermediate_results_{experiment_name}_cellot_{target}", "latent.npy")
latent = np.load(latent_path)

adata = sc.read_h5ad(input_file_path)

adata_latent = anndata.AnnData(latent)
adata_latent.obs = adata.obs.copy()

save_latent_path = os.path.join(parent_folder, "results/cellot_intermediate", f"intermediate_results_{experiment_name}_cellot_{target}", "latent.h5ad")
adata_latent.write(save_latent_path)


print('-------------- Prepare yaml file -------------')
import yaml

parent_folder = os.path.dirname(os.path.dirname(input_file_path))
intermediate_path = os.path.join(parent_folder, "results/cellot_intermediate", f"intermediate_results_{experiment_name}_cellot_{target}")

# Define the configuration data that you want to save into the YAML file
config_data = {
    'data': {
        'type': 'cell',
        'source': condition_control,
        'target': condition_stimulated,
        'condition': condition,
        'path': save_latent_path
    },
    'dataloader': {
        'batch_size': 64,
        'shuffle': True,
    },
    'datasplit': {
        'holdout': target,
        'key': batch,
        'groupby': condition,
        'name': 'toggle_ood',
        'mode': 'ood',
        'test_size': 0.2,
        'random_state': 0
    }
}

# Step 3: Write the combined configuration to the new YAML file
output_config_path = os.path.join(parent_folder, "results/cellot_intermediate", f"intermediate_results_{experiment_name}_cellot_{target}", "config_cellot.yaml")

with open(output_config_path, 'w') as file:
    yaml.dump(config_data, file, default_flow_style=False)

print('-------------- Yaml file prepared -------------')

print('-------------- Training cellot -------------')

training_script = os.path.join(cellot_path, "scripts", "train.py")
model_config = os.path.join(cellot_path, "configs", "models", "cellot.yaml")
task_config = output_config_path

command = [
    "python", training_script,
    "--outdir", intermediate_path,
    "--config", model_config,
    "--config", task_config
]

try:
    subprocess.run(command, check=True)
except subprocess.CalledProcessError as e:
    print(f"Error occurred: {e}")
    exit(1)
print('-------------- Training finished -------------')
print('-------------- Predicting -------------')
evaluation_script = os.path.join(cellot_path, "scripts", "evaluate.py")
command = [
    "python", evaluation_script,
    "--outdir", intermediate_path,
    "--setting", 'ood',
    "--where", 'data_space'
]

try:
    subprocess.run(command, check=True)
except subprocess.CalledProcessError as e:
    print(f"Error occurred: {e}")
    exit(1)