import os
import re
import scanpy as sc
import pandas as pd
import warnings
import argparse
from metrics import Metrics
from pathlib import Path

warnings.filterwarnings("ignore")

# Function to parse command-line arguments
def get_arguments():
    parser = argparse.ArgumentParser(description="Run metrics computation.")
    parser.add_argument("--experiment_name", type=str, required=True, help="Experiment name")
    parser.add_argument("--tool", type=str, required=True, help="Tool name")
    parser.add_argument("--target", type=str, required=True, help="Target name")
    return parser.parse_args()

# Parse command-line arguments
args = get_arguments()

# Retrieve the input arguments
experiment_name = args.experiment_name
tool = args.tool
target = args.target

# Define paths
data_path = f'results/{experiment_name}/h5ad/{experiment_name}_{tool}_{target}.h5ad'  # Path to the folder containing .h5ad files
results_path = f'results/{experiment_name}/'  # Path to save results

# Create the results directory if it doesn't exist
os.makedirs(results_path, exist_ok=True)

datasets_file = 'config/datasets.tsv'  
datasets_df = pd.read_csv(datasets_file, sep='\t')


# Fetch the condition name for the matching experiment_name
condition = None
if experiment_name in datasets_df['experiment_name'].values:
    condition = datasets_df.loc[datasets_df['experiment_name'] == experiment_name, 'condition'].values[0]
else:
    raise ValueError(f"Experiment name '{experiment_name}' not found in datasets.tsv")



# Define keys for metrics (using placeholders for the condition, control, predicted, and stimulated)
# These should ideally be passed or modified as needed, since they aren't part of the Snakemake input
keys = {
    'condition': condition,  # Modify as needed
    'control': 'control',  # Modify as needed
    'predicted': 'predicted',  # Modify as needed
    'stimulated': 'stimulated'  # Modify as needed
}

# Bootstrapping to do
bootstrap_count = 10

print(f"Processing: Experiment={experiment_name}, Tool={tool}, Target={target}, File={results_path}")

# Read the .h5ad file
adata = sc.read_h5ad(data_path)


# Initialize Metrics
metrics = Metrics(adata, keys, bootstrap=bootstrap_count, experiment_name=experiment_name)

# Save results in the appropriate directory structure
os.makedirs(results_path, exist_ok=True)

exp_index = f'{experiment_name}_{tool}_{target}'

# Define output subdirectories
output_path = Path(results_path)
biology_dir = Path(results_path) / 'biology'
metrics_dir = Path(results_path) / 'metrics'

# Create directories if they do not exist
output_path.mkdir(parents=True, exist_ok=True)
biology_dir.mkdir(parents=True, exist_ok=True)
metrics_dir.mkdir(parents=True, exist_ok=True)
try:
    metrics.get_scores(output_path=str(output_path), exp_index=exp_index)
    # Create a flag file indicating completion
    Path(f'flags/metrics/output_run_flag_{exp_index}_metrics.txt').touch()
    print(f"Results saved for Experiment={experiment_name}, Tool={tool}, Target={target}, Exp_Index={exp_index}")
except Exception as e:
    print(f"Error processing metrics for {results_path}: {e}")