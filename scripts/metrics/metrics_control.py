import os
import subprocess
import yaml
import argparse
import scanpy as sc
from pathlib import Path
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
from sklearn.metrics import mean_squared_error
import pandas as pd
import seaborn as sns
import gseapy as gp
import argparse
import os
import pertpy

# Function to load the config file
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Argument parser setup
def get_arguments():
    parser = argparse.ArgumentParser(description="Run experiment with configuration.")
    parser.add_argument("--config", type=str, required=True, help="Path to the config.yml file")
    parser.add_argument("--file_path", type=str, help="Override input_data.file_path")
    parser.add_argument("--output_dir", type=str, help="Override experiment.output_dir")
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
tool = 'control'

adata = sc.read_h5ad(f'data/{experiment_name}.h5ad')
adata = adata[adata.obs[batch] == target]

control = adata[adata.obs[condition] == condition_control]
stimulated = adata[adata.obs[condition] == condition_stimulated]    

distance_scores = pd.DataFrame()

def add_control(control, stimulated):
        metric_names = [
            "mse",
            "wasserstein",
            "pearson_distance",
            "mmd",
            "t_test",
            "cosine_distance",
            ]
    

        for distance_metric in metric_names:
            print(f"Computing distance {distance_metric}")
            distance = pertpy.tl.Distance(distance_metric)
            distance_scores[distance_metric] = [
                tuple(float(val) for val in distance.bootstrap(
                control.X.toarray(),
                stimulated.X.toarray(),
                n_bootstrap=1
            ))
            ]

def r2_control_vs_stimulated(
    adata,
    condition_key=condition,
    ctrl_key=condition_control,
    stim_key=condition_stimulated,
    n_degs=100,
    n_bootstrap=10,
    sample_ratio=0.8,
    random_seed=42
):
    import numpy as np
    import scanpy as sc
    import pandas as pd

    np.random.seed(random_seed)

    # Filter out duplicated indices
    adata = adata[~adata.obs.index.duplicated(keep='first')].copy()
    adata = adata[:, ~adata.var.index.duplicated(keep='first')].copy()

    # Ensure control and stimulated labels exist
    if not all(k in adata.obs[condition_key].unique() for k in [ctrl_key, stim_key]):
        raise ValueError(f"Control or stimulated condition not found in '{condition_key}'")

    # Run differential expression
    sc.tl.rank_genes_groups(adata, groupby=condition_key, reference=ctrl_key, method="wilcoxon")
    degs = adata.uns['rank_genes_groups']['names'][stim_key][:n_degs]

    # Get expression dataframe
    df = adata.to_df()
    df_ctrl = df[adata.obs[condition_key] == ctrl_key][degs]
    df_stim = df[adata.obs[condition_key] == stim_key][degs]

    # Bootstrap R²
    r2_scores = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        ctrl_sample = df_ctrl.sample(frac=sample_ratio, random_state=i)
        stim_sample = df_stim.sample(frac=sample_ratio, random_state=i)

        ctrl_mean = ctrl_sample.mean().values
        stim_mean = stim_sample.mean().values

        r = np.corrcoef(ctrl_mean, stim_mean)[0, 1]
        r2_scores[i] = r ** 2 if not np.isnan(r) else np.nan

    mean_r2 = float(np.nanmean(r2_scores))
    var_r2 = float(np.nanvar(r2_scores))

    return (mean_r2, var_r2)




add_control(control, stimulated)
for n_degs in [20, 100, adata.shape[1]]:
    print(f"Computing R2 for {n_degs} DEGs")
    r2 = r2_control_vs_stimulated(adata, n_degs=n_degs)
    if n_degs == adata.shape[1]:
        distance_scores[f"r2_all_degs"] = [r2]
    else:
        distance_scores[f"r2_{n_degs}_degs"] = [r2]


output_folder = f'results/{experiment_name}/metrics'
os.makedirs(output_folder, exist_ok=True)
distance_scores.to_csv(f'results/{experiment_name}/metrics/{experiment_name}_{tool}_{target}_distance_scores.csv', index=False)

# Output flag
file_path = Path(f'flags/metrics/output_run_flag_{experiment_name}_control_{target}_metrics.txt')
file_path.parent.mkdir(parents=True, exist_ok=True)
file_path.touch()