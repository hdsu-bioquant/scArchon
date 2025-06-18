<p align="center">
  <img src="images/002.png" alt="Description of Image" width="80%">
</p>

# scArchon: benchmark or run single-cell prediciton tools on your own dataset

scArchon is a modular, reproducible benchmarking platform for evaluating single-cell perturbation response prediction tools. Built on Snakemake, it provides an extensible framework to compare deep learning methods across diverse datasets using both statistical and biological metrics. Why scArchon?
While many tools exist to predict single-cell responses to perturbations (e.g., drug treatments), their systematic comparison has been limited. scArchon helps standardize benchmarking and highlights important nuances—such as when models with high quantitative scores fail to retain key biological signals.

We invite the community to adopt and contribute to scArchon, helping accelerate progress in single-cell perturbation modeling.

# Requirements
Running the deep learning models require GPU with CUDA 12.4+. To pull the environments from Dockerhub, Singularity 3.6+ needs to be installed on your machine. To store the environments, a disk space of about 50 Go is required. 

- CUDA 12.4+ (tested on 12.4)
- Singularity 3.6+ (tested on 3.6 and 4.1)
- About 50 Go disk space to download all environments 

In details, the different tools require following CUDA versions:

||CUDA version|
|--|----------|
|cellot| 10.2 |
|cpa| 11.7|
|scdisinfact|12.4|
|scpram|11.6|
|scvidr|12.1|
|scpregan| 12.1 |
|screen| 11.7|
|scgen| 11.7|

Below CUDA 11.6, no tool can be ran. After CUDA 12.4 all tools can be ran. 

# Installation
- Create a conda environment with snakemake:
    ```python
    conda create -c conda-forge -c bioconda -n snakemake_env snakemake
    conda activate snakemake_env
    ```
- Activate the environment: `conda activate snakemake_env`
- Ensure that you have a GPU with CUDA 12.4+ and Singularity 3.6+ available

# Running your experiments
- You can set up your experiments in `config/datasets.tsv`.

<div align="center">
    <img src="images/001.png" alt="Description of Image" style="width: 100%; margin: 0 auto;">
</div>

- If you are running the tools on a single GPU, it is suggested to run the tools one by one, otherwise the tasks will swap and will take overall longer. We suggest to run the pipeline with following command:

    ```python
    snakemake --use-singularity --singularity-args '--nv -B .:/dum' --cores all --jobs 1 --keep-going
    ```

    - `--use-singularity` will pull the docker images from the web
    - `--singularity-args '--nv -B .:/dum'` ensures GPU usage
    -  `--cores all` requests all CPUs available
    - `--jobs 1` runs one job after the other
    - `--keep-going` ensures the pipeline continues running even if there is a bug in one of the tool to not lose time

# User-useful information
- Pulling environment via singularity may take some time depending on your downloading speed. The environment only needs to be pulled once. It will be stored under `.snakemake/singularity`.
- The running time of some tools can be long. Given the performance of cellOT, CPA and scPreGAN, we suggest you to leave them out of your run.

<div align="center">
    <img src="images/003.png" alt="Description of Image" style="width: 100%; margin: 0 auto;">
</div>



