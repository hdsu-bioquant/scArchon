# User guide
## Requirements: Singularity, environment with snakemake installed
## Example on own cluster
- Connect to login1, go to your workspace
- Download the code there
- Request resource: `srun --partition=gpu --gres=gpu:K80:4 --cpus-per-task=12 --time=96:00:00 --pty /bin/bash`
- Create/activate conda environment with snakemake installed `conda activate snakemake_env`
- Load singularity: `module load system/singularity/3.10.4`
- If you run the tools, in the Snakefile, set rule all to: 
```python
rule all:
    input:
        output_tools
        # output_metrics to compute the metrics
```
- Run the pipeline using: `snakemake --use-singularity --singularity-args '--nv -B .:/dum' --cores all --jobs 1 --keep-going`



# Installation
- Connect to login1 at `/net/data.isilon/ag-cherrmann/jradig`
- Request resource: `srun --mem=250G --time=96:00:00 --pty /bin/bash` 
- Create conda env:
```python 
conda create -c conda-forge -c bioconda -n snakemake_env snakemake
conda activate snakemake_env
```
- Load singularity:
```python
module avail
module load system/singularity/3.10.4
```
- If requested to initiate conda:
```python
echo 'export PATH="/var/localfs/working2/jradig/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
conda init
```

# Running the pipeline
- activate snakemake_env

```python
# When running the tools
snakemake --use-singularity --singularity-args '--nv -B .:/dum' --cores all --jobs 1 --keep-going

# When running metrics
snakemake --use-singularity --singularity-args '--nv -B .:/dum' --cores all
```



```python 
srun --partition=gpu-legacy --gres=gpu:K40GT730:2 --cpus-per-task=12 --mem=8G --time=02:00:00 --pty /bin/bash
conda activate snakemak_env
module load system/singularity/3.10.4
snakemake --use-singularity --singularity-args '--nv -B .:/dum'
```

# Utils login1
- Check available GPUs: `scontrol show nodes | grep -i gpu`
- cancel a job: `squeue | scancel #jobid`



# Snakemake pipeline
We want to patch together the single cell perturbation methods. Given an input, we would like to run as many tools as possible, given they are compatible with the task. 

# Using singularity
To use the code:
```python
rule run_scgen:
    input:
        config="config.yaml"
    output:
        "results/{experiment_name}_scgen_{target}.h5ad"
    singularity:
        "scgen_env-latest.simg"
    shell:
        """
        conda init
        exit
        bash
        conda activate scgen 
        echo "Starting Python script execution"
        python scripts/scgen/snkmk_scgen.py --config {input.config} > run_scgen.log 2>&1
        echo "Python script finished"
        """
```

Works if we use:
```python
FROM snakemake/snakemake
```
If there is an error `ERROR  : Unknown image format/type: /workspace/.snakemake/singularity/698bd43a399a581b827f6fdca1a0ebb2.simg` you can do: 
```python 
singularity pull docker://hdsu/scgen_env:latest
```
Which saves the singulariy as `scgen_env-latest.simg`. 
We can then run the pipeline using:
```python 
snakemake --use-singularity
```

# Using conda environment
Trying to use code:
```python
rule run_scgen:
    input:
        config="config.yaml"
    output:
        "results/{experiment_name}_scgen_{target}.h5ad"
    conda:
        "scgen-env.yml"  # The generated Conda environment YAML file
    shell:
        """
        echo "Starting Python script execution" > run_scgen.log 2>&1
        python scripts/scgen/snkmk_scgen.py --config {input.config} >> run_scgen.log 2>&1
        echo "Python script finished" >> run_scgen.log 2>&1
        """
```

I have a local installation of the conda scgen. I created a yml thereof, saved as `scgen-env.yml`. 

Run using `snakemake --use-conda`.

# Creating conda yml files
It does not work when we try to create the conda env yaml files from the docker. Therefore, either we manage to create the conda env in the current folder and we can then use it, or we need to pursue in the direction of the similarity container. 

# Loading files relative to snakamake

```python
script_dir = os.path.dirname(os.path.abspath(__file__))
scpregan_path = os.path.join(script_dir, "scPreGAN")
sys.path.insert(0, scpregan_path)
```

# datasets.tsv
- All datasets but gb
```python
file_path	condition	condition_control	condition_stimulated	batch	target	experiment_name	output_dir	Tools
data/hpoly.h5ad	condition	Control	HpolyDay10	cell_label	Stem,EnterocyteProgenitor,TA,TAEarly,Enterocyte	hpoly	results/hpoly/	cellot
data/species.h5ad	condition	unst	LPS6	species	mouse,pig,rabbit,rat	species	results/species/	scgen,scdisinfact
data/kang.h5ad	condition	control	stimulated	cell_type	CD4T,CD14+Mono,B,CD8T,NK,FCGR3A+Mono,Dendritic	kang	results/kang/	scgen,scdisinfact
``` 
- 3x, 9x experience
```python
file_path	condition	condition_control	condition_stimulated	batch	target	experiment_name	output_dir	Tools
data/species.h5ad	condition	unst	LPS6	species	rat	speciesx9	results/speciesx9/	cellot,cpa,scdisinfact,scpram,scvidr,scgen,scpregan,screen,linear
data/kang.h5ad	condition	control	stimulated	cell_type	CD4T	kangx9	results/kangx9/	cellot,cpa,scdisinfact,scpram,scvidr,scgen,scpregan,screen,linear
```



# Flagged problems:
- CellOT dataset sensitivity. CellOT runs on very particular versions of scanpy and anndata. Loading data that was not written in these vesions can yield some reading problems: `AnnDataReadError: Above error raised while reading key '/layers' of type <class 'h5py._hl.group.Group'> from /.`. Before running the pipeline on cellOT, one should therefore try to load the data in the environment. If it does not work, try re-creating the adata in cellOT's environment. 

# Running time: made a mistake and saved cellot's running time under cpa: re-run cpa and re-name existing to cellot 
Cellot: sometimes started training and stopped. It started again at a later stage. This also inserts mistakes.

# Experiment x3
We increase the number of epochs for which the tools are trained by a factor 3, when available (not available for scPreGan, for cellOT we increase from 100'000 to 250'000, which is what they recommend in their paper).  

# Colors for tools:
colors = ['#9e0142', '#d53e4f', '#f46d43', '#fdae61', 
          '#fee08b', '#e6f598', '#abdda4', '#66c2a5', 
          '#3288bd', '#5e4fa2']

tools: cellot,cpa,scdisinfact,scpram,scvidr,scgen,scpregan,screen,linear

# Usage of benchmark
- Modify tools you want to run by adding `--without tool1 tool2` etc.. e.g. `--without cellot scgen scpregan` in `Snakefile` at run run_benchmark.