import scanpy as sc
import argparse
import yaml
import warnings
import logging
import os
import cpa
import pandas as pd
from pathlib import Path
import anndata
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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



# Run cpa

print('---------------------------------------------')
print('-------------- Running cpa -------------')
print('---------------------------------------------')

adata.obs['dose'] = 1
cpa.CPA.setup_anndata(adata,
                      perturbation_key=condition,
                      control_group=condition_control,
                      dosage_key='dose',
                      categorical_covariate_keys=[batch],
                      is_count_data=False,
                      max_comb_len=1,
                     )

adata.obs['split_datasets'] = 'train' # Default start

# Test set is composed of the target at stimulated condition
adata.obs.loc[(adata.obs[condition] == condition_stimulated) & (adata.obs[batch] == target), 'split_datasets'] = 'test'

# Select 10% of the train set to be in the validation set
train_cells = adata.obs[adata.obs['split_datasets'] == 'train']
num_valid_cells = int(len(train_cells) * 0.10)
valid_cells = train_cells.sample(n=num_valid_cells, random_state=42)
adata.obs.loc[valid_cells.index, 'split_datasets'] = 'valid'


print(f'--------------- Training cpa ---------------')

model_params = {
    "n_latent": 64,
    "recon_loss": "nb",
    "doser_type": "linear",
    "n_hidden_encoder": 128,
    "n_layers_encoder": 2,
    "n_hidden_decoder": 512,
    "n_layers_decoder": 2,
    "use_batch_norm_encoder": True,
    "use_layer_norm_encoder": False,
    "use_batch_norm_decoder": False,
    "use_layer_norm_decoder": True,
    "dropout_rate_encoder": 0.0,
    "dropout_rate_decoder": 0.1,
    "variational": False,
    "seed": 6977,
}

trainer_params = {
    "n_epochs_kl_warmup": None,
    "n_epochs_pretrain_ae": 30,
    "n_epochs_adv_warmup": 50,
    "n_epochs_mixup_warmup": 0,
    "mixup_alpha": 0.0,
    "adv_steps": None,
    "n_hidden_adv": 64,
    "n_layers_adv": 3,
    "use_batch_norm_adv": True,
    "use_layer_norm_adv": False,
    "dropout_rate_adv": 0.3,
    "reg_adv": 20.0,
    "pen_adv": 5.0,
    "lr": 0.0003,
    "wd": 4e-07,
    "adv_lr": 0.0003,
    "adv_wd": 4e-07,
    "adv_loss": "cce",
    "doser_lr": 0.0003,
    "doser_wd": 4e-07,
    "do_clip_grad": True,
    "gradient_clip_value": 1.0,
    "step_size_lr": 10,
}

model = cpa.CPA(adata=adata,
                split_key='split_datasets',
                train_split='train',
                valid_split='valid',
                test_split='test',
                **model_params,
               )

model.train(max_epochs=100,
            use_gpu=True,
            batch_size=64,
            plan_kwargs=trainer_params,
            early_stopping_patience=5,
            check_val_every_n_epoch=5,
            save_path=output_dir_path
           )

print(f'--------------- Predicting ---------------')

model = cpa.CPA.load(dir_path=output_dir_path,
                      adata=adata,
                      use_gpu=False)

control = adata[(adata.obs[condition] == condition_control) & (adata.obs[batch] == target)]
stimulated = adata[adata.obs['split_datasets'] == 'test']

model.predict(control)

x_control = control.X
x_predicted = control.obsm['CPA_pred']
x_stimulated = stimulated.X

control_obs = pd.DataFrame({condition: ['control'] * x_control.shape[0]})
predicted_obs = pd.DataFrame({condition: ['predicted'] * x_predicted.shape[0]})
stimulated_obs = pd.DataFrame({condition: ['stimulated'] * x_stimulated.shape[0]})

# Create AnnData objects for each
control_adata = anndata.AnnData(X=x_control, obs=control_obs)
predicted_adata = anndata.AnnData(X=x_predicted, obs=predicted_obs)
stimulated_adata = anndata.AnnData(X=x_stimulated, obs=stimulated_obs)

# Step 3: Concatenate the AnnData objects
combined_adata = control_adata.concatenate(predicted_adata, stimulated_adata)
combined_adata.var.index = adata.var.index

print(f'--------------- Saving results ---------------')

output_dir_path = output_dir_path.rstrip(os.sep)
h5ad_dir = os.path.join(output_dir_path, 'h5ad')
if not os.path.exists(h5ad_dir):
    os.makedirs(h5ad_dir)
combined_adata.write(f'{h5ad_dir}/{experiment_name}_cpa_{target}.h5ad')

# Output flag
file_path = Path(f'flags/h5ad/output_run_flag_{experiment_name}_cpa_{target}_h5ad.txt')
file_path.parent.mkdir(parents=True, exist_ok=True)
file_path.touch()
print('cpa successfully saved the results')