import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
scpregan_path = os.path.join(script_dir, "scPreGAN")
sys.path.insert(0, scpregan_path)
from scPreGAN.model.util import load_anndata
from scPreGAN import Model
import scanpy as sc
import torch
import argparse
import yaml
import warnings
import logging
from pathlib import Path
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

# Load the data
adata = sc.read_h5ad(input_file_path)

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



# Run scpregan

print('---------------------------------------------')
print('-------------- Running scPreGan -------------')
print('---------------------------------------------')



target_cell_type= target
adata_split, train_data = load_anndata(adata=adata,
                condition_key=condition,
                condition={'case': condition_stimulated, 'control': condition_control},
                cell_type_key=batch,
                target_cell_type=target_cell_type
                )
control_adata, perturb_adata, case_adata = adata_split
control_pd, control_celltype_ohe_pd, perturb_pd, perturb_celltype_ohe_pd = train_data

cell_types = control_adata.obs[batch].unique().tolist()
n_features = control_pd.shape[1]
n_classes = len(adata.obs[batch].unique())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f'--------------- Training scPreGan ---------------')


model = Model(n_features=n_features, n_classes=n_classes, use_cuda=True)
model.train(train_data=train_data, output_path = output_dir_path)

print(f'--------------- Predicting ---------------')
control_test_adata = control_adata[control_adata.obs[batch] == target_cell_type]
perturb_test_adata = perturb_adata[perturb_adata.obs[batch] == target_cell_type]  
pred_perturbed_adata = model.predict(control_adata=control_test_adata,
                cell_type_key=batch,
                condition_key=condition,
                )

control_test_adata.obs_names_make_unique()
perturb_test_adata.obs_names_make_unique()
pred_perturbed_adata.obs_names_make_unique()

control_test_adata.var_names_make_unique()
perturb_test_adata.var_names_make_unique()
pred_perturbed_adata.var_names_make_unique()

control_test_adata.obs[condition] = 'control'
perturb_test_adata.obs[condition] = 'stimulated'
pred_perturbed_adata.obs[condition] = 'predicted'


combined_adata = control_test_adata.concatenate(
    perturb_test_adata,
    pred_perturbed_adata
)

combined_adata.obs.index = combined_adata.obs_names
print(f'scpregan sussesfully predicted stimulated {target}')

print(f'--------------- Saving results ---------------')

output_dir_path = output_dir_path.rstrip(os.sep)
h5ad_dir = os.path.join(output_dir_path, 'h5ad')
if not os.path.exists(h5ad_dir):
    os.makedirs(h5ad_dir)
combined_adata.write(f'{h5ad_dir}/{experiment_name}_scpregan_{target}.h5ad')

# Output flag
file_path = Path(f'flags/h5ad/output_run_flag_{experiment_name}_scpregan_{target}_h5ad.txt')
file_path.parent.mkdir(parents=True, exist_ok=True)
file_path.touch()
print('scpregan successfully saved the results')