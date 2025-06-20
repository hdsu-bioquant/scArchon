import yaml
from snakemake.utils import min_version
import os
os.makedirs("logs", exist_ok=True)


report: "report/workflow.rst"

##### open datasets and create specific config #####
dataset_file = "config/datasets.tsv"

# create the input and output names list expected for the rule all
input_all = []
output_metrics = []
output_tools = []
output_benchmark = []

# I open the general datasets file and create specific config files for each run
with open(dataset_file) as config:
    first = True
    for line in config:
        if first:
            first = False
        
        

        else:
            line_split_length = len(line.splitlines())
            print(f"Length of splitlines: {line_split_length}")

            # splitting the line to get the different arguments
            parameters = line.splitlines()[0].split("\t")

            # getting list of tools to use
            tools = parameters[8].split(",")
            
            # getting name of h5ad
            experiment_name = parameters[6]
            output_benchmark.append(f"flags/benchmark/output_run_flag_{experiment_name}_benchmark.txt")

            # getting targets name
            targets = parameters[5].split(",")

            # starting creation of config files for every target and every tool

            # everey target
            for target in targets:

                config_file_path = f"config/config_{experiment_name}_control_{target}.yaml"
                input_all.append(config_file_path)

                with open(config_file_path, "w") as out_conf:
                    # writting the config file
                    out_conf.write(f"input_data:\n  file_path: \"{parameters[0]}\"\n\n")
                    out_conf.write(f"experiment:\n  condition: \"{parameters[1]}\"\n  condition_control: \"{parameters[2]}\"\n  ")
                    out_conf.write(f"condition_stimulated: \"{parameters[3]}\"\n  batch: \"{parameters[4]}\"\n  ")
                    out_conf.write(f"target: \"{target}\"\n  experiment_name: \"{experiment_name}\"\n  ")
                    out_conf.write(f"output_dir: \"{parameters[7]}\"")

                # every tool
                for tool in tools:

                    # increment the input names list
                    output_tools.append(f"flags/h5ad/output_run_flag_{experiment_name}_{tool}_{target}_h5ad.txt")
                    output_metrics.append(f"flags/metrics/output_run_flag_{experiment_name}_{tool}_{target}_metrics.txt")
                    output_metrics.append(f"flags/metrics/output_run_flag_{experiment_name}_control_{target}_metrics.txt")
                    
                    # path of the created config files
                    config_file_path = f"config/config_{experiment_name}_{tool}_{target}.yaml"
                    input_all.append(config_file_path)


                    with open(config_file_path, "w") as out_conf:
                        # writting the config file
                        out_conf.write(f"input_data:\n  file_path: \"{parameters[0]}\"\n\n")
                        out_conf.write(f"experiment:\n  condition: \"{parameters[1]}\"\n  condition_control: \"{parameters[2]}\"\n  ")
                        out_conf.write(f"condition_stimulated: \"{parameters[3]}\"\n  batch: \"{parameters[4]}\"\n  ")
                        out_conf.write(f"target: \"{target}\"\n  experiment_name: \"{experiment_name}\"\n  ")
                        out_conf.write(f"output_dir: \"{parameters[7]}\"")

                    


# Define the final output file using wildcards
rule all:
    input:
        output_tools,
        output_metrics,
        output_benchmark



rule run_scgen:
    input:
        config="config/config_{experiment_name}_scgen_{target}.yaml"

    output:
        "flags/h5ad/output_run_flag_{experiment_name}_scgen_{target}_h5ad.txt"

    singularity:
        "docker://hdsu/scgen_env:latest"  

    shell:
        """
        start_time=$(date +%s)
        
        # Activate conda environment and run the script
        source /opt/conda/etc/profile.d/conda.sh 
        conda activate scgen 
        python scripts/scgen/snkmk_scgen.py --config {input.config}
        
        end_time=$(date +%s)
        runtime=$((end_time - start_time))
        
        # Log the runtime
        log_file="logs/{wildcards.experiment_name}_scgen_{wildcards.target}.txt"
        echo "Runtime (seconds): $runtime" > "$log_file"
        """




rule run_disinfact:
    input:
        config="config/config_{experiment_name}_scdisinfact_{target}.yaml"
    output:
        "flags/h5ad/output_run_flag_{experiment_name}_scdisinfact_{target}_h5ad.txt"
    singularity:
        "docker://hdsu/scdisinfact_env:latest"  
    shell:
        """
        start_time=$(date +%s) 

        source /opt/conda/etc/profile.d/conda.sh 
        conda activate scdisinfact
        python scripts/scdisinfact/snkmk_scdisinfact.py --config {input.config}

        end_time=$(date +%s)
        runtime=$((end_time - start_time))
        
        # Log the runtime
        log_file="logs/{wildcards.experiment_name}_scdisinfact_{wildcards.target}.txt"
        echo "Runtime (seconds): $runtime" > "$log_file"
        """

rule run_scpregan:
    input:
        config="config/config_{experiment_name}_scpregan_{target}.yaml"
    output:
        "flags/h5ad/output_run_flag_{experiment_name}_scpregan_{target}_h5ad.txt"
    singularity:
        "docker://hdsu/scpregan_env:latest"  
    shell:
        """
        start_time=$(date +%s) 

        export MKL_INTERFACE_LAYER=LP64
        source /opt/conda/etc/profile.d/conda.sh 
        conda activate scpregan
        python scripts/scpregan/snkmk_scpregan.py --config {input.config}

        end_time=$(date +%s)
        runtime=$((end_time - start_time))
        
        # Log the runtime
        log_file="logs/{wildcards.experiment_name}_scpregan_{wildcards.target}.txt"
        echo "Runtime (seconds): $runtime" > "$log_file"
        """

rule run_scpram:
    input:
        config="config/config_{experiment_name}_scpram_{target}.yaml"
    output:
        "flags/h5ad/output_run_flag_{experiment_name}_scpram_{target}_h5ad.txt"
    singularity:
        "docker://hdsu/scpram_env:latest"  
    shell:
        """
        start_time=$(date +%s) 
        source /opt/conda/etc/profile.d/conda.sh 
        conda activate scpram
        python scripts/scpram/snkmk_scpram.py --config {input.config}

        end_time=$(date +%s)
        runtime=$((end_time - start_time))
        
        # Log the runtime
        log_file="logs/{wildcards.experiment_name}_scpram_{wildcards.target}.txt"
        echo "Runtime (seconds): $runtime" > "$log_file"
        """

rule run_screen:
    input:
        config="config/config_{experiment_name}_screen_{target}.yaml"
    output:
        "flags/h5ad/output_run_flag_{experiment_name}_screen_{target}_h5ad.txt"
    singularity:
        "docker://hdsu/screen_env:latest"  
    shell:
        """
        start_time=$(date +%s) 
        source /opt/conda/etc/profile.d/conda.sh 
        conda activate screen
        python scripts/screen/snkmk_screen.py --config {input.config}
        end_time=$(date +%s)
        runtime=$((end_time - start_time))
        
        # Log the runtime
        log_file="logs/{wildcards.experiment_name}_screen_{wildcards.target}.txt"
        echo "Runtime (seconds): $runtime" > "$log_file"
        """

rule run_scvidr:
    input:
        config="config/config_{experiment_name}_scvidr_{target}.yaml"
    output:
        "flags/h5ad/output_run_flag_{experiment_name}_scvidr_{target}_h5ad.txt"
    singularity:
        "docker://hdsu/scvidr_env:latest"  
    shell:
        """
        start_time=$(date +%s) 
        source /opt/conda/etc/profile.d/conda.sh 
        conda activate scvidr
        python scripts/scvidr/snkmk_scvidr.py --config {input.config}

        end_time=$(date +%s)
        runtime=$((end_time - start_time))
        
        # Log the runtime
        log_file="logs/{wildcards.experiment_name}_scvidr_{wildcards.target}.txt"
        echo "Runtime (seconds): $runtime" > "$log_file"
        """

rule run_cpa:
    input:
        config="config/config_{experiment_name}_cpa_{target}.yaml"
    output:
        "flags/h5ad/output_run_flag_{experiment_name}_cpa_{target}_h5ad.txt"
    singularity:
        "docker://hdsu/cpa_env:latest"  
    shell:
        """
        start_time=$(date +%s) 
        source /opt/conda/etc/profile.d/conda.sh 
        conda activate cpa
        python scripts/cpa/snkmk_cpa.py --config {input.config}

        end_time=$(date +%s)
        runtime=$((end_time - start_time))
        
        # Log the runtime
        log_file="logs/{wildcards.experiment_name}_cpa_{wildcards.target}.txt"
        echo "Runtime (seconds): $runtime" > "$log_file"
        """

rule run_trvae:
    input:
        config="config/config_{experiment_name}_trvae_{target}.yaml"
    output:
        "flags/h5ad/output_run_flag_{experiment_name}_trvae_{target}_h5ad.txt"
    singularity:
        "docker://hdsu/trvae_env:latest"  
    shell:
        """
        start_time=$(date +%s) 
        source /opt/conda/etc/profile.d/conda.sh 
        conda activate trvae
        python scripts/trvae/snkmk_trvae.py --config {input.config}

        end_time=$(date +%s)
        runtime=$((end_time - start_time))
        
        # Log the runtime
        log_file="logs/{wildcards.experiment_name}_trvae_{wildcards.target}.txt"
        echo "Runtime (seconds): $runtime" > "$log_file"
        """

rule encode:
    input:
        config="config/config_{experiment_name}_cellot_{target}.yaml"
    output:
        "results/cellot_intermediate/intermediate_results_{experiment_name}_cellot_{target}/latent.npy"
    singularity:
        "docker://hdsu/scgen_env:latest"  
    shell:
        """
        source /opt/conda/etc/profile.d/conda.sh 
        conda activate scgen
        python scripts/cellot/encode/snkmk_encode.py --config {input.config}
        """

rule run_cellot:
    input:
        "results/cellot_intermediate/intermediate_results_{experiment_name}_cellot_{target}/latent.npy",
        config="config/config_{experiment_name}_cellot_{target}.yaml"    
    output:
        "results/cellot_intermediate/intermediate_results_{experiment_name}_cellot_{target}/evals_ood_data_space/imputed.h5ad"
    singularity:
        "docker://hdsu/cellot_env:latest"
    shell:
        """
        start_time=$(date +%s) 
        
        source /opt/conda/etc/profile.d/conda.sh 
        conda activate cellot
        pip install -e scripts/cellot/run_cellot/cellot-main
        python scripts/cellot/run_cellot/snkmk_cellot.py --config {input.config}

        end_time=$(date +%s)
        runtime=$((end_time - start_time))
        log_file="logs/{wildcards.experiment_name}_cellot_{wildcards.target}.txt"
        echo "Runtime (seconds): $runtime" > "$log_file"
        """


rule decode:
    input:
        "results/cellot_intermediate/intermediate_results_{experiment_name}_cellot_{target}/evals_ood_data_space/imputed.h5ad",
        config="config/config_{experiment_name}_cellot_{target}.yaml" 
    output:
        "flags/h5ad/output_run_flag_{experiment_name}_cellot_{target}_h5ad.txt"
    singularity:
        "docker://hdsu/scgen_env:latest"  
    shell:
        """
        source /opt/conda/etc/profile.d/conda.sh 
        conda activate scgen
        python scripts/cellot/decode/snkmk_decode.py --config {input.config}
        """
rule run_linear:
    input:
        config="config/config_{experiment_name}_linear_{target}.yaml"
    output:
        "flags/h5ad/output_run_flag_{experiment_name}_linear_{target}_h5ad.txt"
    singularity:
        "docker://hdsu/metrics_env:latest"  
    shell:
        """
        start_time=$(date +%s) 
        source /opt/conda/etc/profile.d/conda.sh 
        conda activate metrics
        python scripts/linear/snkmk_linear.py --config {input.config}

        end_time=$(date +%s)
        runtime=$((end_time - start_time))
        
        # Log the runtime
        log_file="logs/{wildcards.experiment_name}_linear_{wildcards.target}.txt"
        echo "Runtime (seconds): $runtime" > "$log_file"
        """


rule compute_metrics:
    input:
        "flags/h5ad/output_run_flag_{experiment_name}_{tool}_{target}_h5ad.txt"
    output: 
        "flags/metrics/output_run_flag_{experiment_name}_{tool}_{target}_metrics.txt"
    singularity: 
        "docker://hdsu/metrics_env:latest"
    shell:
        """
        source /opt/conda/etc/profile.d/conda.sh 
        conda activate metrics
        python scripts/metrics/compute_metrics.py --experiment_name {wildcards.experiment_name} --tool {wildcards.tool} --target {wildcards.target}
        """

rule metrics_control:
    input:
        config="config/config_{experiment_name}_control_{target}.yaml"
    output:
        "flags/metrics/output_run_flag_{experiment_name}_control_{target}_metrics.txt"
    singularity:
        "docker://hdsu/metrics_env:latest"
    shell:
        """
        source /opt/conda/etc/profile.d/conda.sh 
        conda activate metrics
        python scripts/metrics/metrics_control.py --config {input.config}
        """

rule run_benchmark:
    input:
        lambda wildcards: output_metrics
    output: 
        "flags/benchmark/output_run_flag_{experiment_name}_benchmark.txt"
    singularity:
        "docker://hdsu/metrics_env:latest"
    shell:
        """
        source /opt/conda/etc/profile.d/conda.sh 
        conda activate metrics
        python scripts/metrics/benchmark.py --experiment_name {wildcards.experiment_name}
        """


