import os
import subprocess
import yaml
import argparse
import scanpy as sc
from pathlib import Path

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

# Print the final values (optional)
print(f"Input file path: {input_file_path}")
print(f"Condition: {condition}")
print(f"Condition (Control): {condition_control}")
print(f"Condition (Stimulated): {condition_stimulated}")
print(f"Batch: {batch}")
print(f"Target: {target}")
print(f"Experiment Name: {experiment_name}")
print(f"Results will be saved at: {output_dir_path}")

# Run the SCREEN command
def run_screen(input_file, output_dir, label, condition_key, cell_type_key, ctrl_key, stim_key,
               latent_dim=100, batch_size=64, epochs=40, full_quadratic=False, activation="leaky_relu", optimizer="Adam"):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Path to the script inside the screen folder
    screen_script = os.path.join("scripts/screen/screen", "screen.py")

    # Build the command
    command = [
        "python", screen_script,
        "-in", input_file,
        "-ou", output_dir,
        "--label", label,
        "--condition_key", condition_key,
        "--cell_type_key", cell_type_key,
        "--ctrl_key", ctrl_key,
        "--stim_key", stim_key,
        "--latent_dim", str(latent_dim),
        "--batch_size", str(batch_size),
        "--epochs", str(epochs),
        "--full_quadratic", str(full_quadratic),
        "--activation", activation,
        "--optimizer", optimizer
    ]

    # Execute the command
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        exit(1)

# Call the function with parsed values
run_screen(
    input_file=input_file_path,
    output_dir=output_dir_path,
    label=target,
    condition_key=condition,
    cell_type_key=batch,
    ctrl_key=condition_control,
    stim_key=condition_stimulated
)

# Renaming the SCREEN output to match the other namings
os.rename(f"{output_dir_path}/SCREEN_{target}.h5ad", f"{output_dir_path}/{experiment_name}_screen_{target}.h5ad")

# Change condition name to control, stimulated and predicted
adata = sc.read_h5ad(f"{output_dir_path}/{experiment_name}_screen_{target}.h5ad")
adata.obs[condition] = adata.obs[condition].replace(
    {f'{target}_Ctrl': 'control', f'{target}_Real': 'stimulated', f'{target}_SCREEN': 'predicted'}
)
adata.write(f"{output_dir_path}/h5ad/{experiment_name}_screen_{target}.h5ad")

os.remove(f"{output_dir_path}/{experiment_name}_screen_{target}.h5ad")

# Output flag
file_path = Path(f'flags/h5ad/output_run_flag_{experiment_name}_screen_{target}_h5ad.txt')
file_path.parent.mkdir(parents=True, exist_ok=True)
file_path.touch()