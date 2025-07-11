<p align="center">
  <img src="images/002.png" alt="Description of Image" width="80%">
</p>

<!-- badges: start -->
[![bioRxiv Preprint](https://img.shields.io/badge/bioRxiv-Preprint-orange)](https://www.biorxiv.org/content/10.1101/2025.06.23.661046v1)
[![license: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![GitHub repo views](https://komarev.com/ghpvc/?username=hdsu-bioquant&repo=scArchon&label=Views&color=blue)](https://github.com/hdsu-bioquant/scArchon)
[![GitHub stars](https://img.shields.io/github/stars/hdsu-bioquant/scArchon.svg?style=social&label=Star)](https://github.com/hdsu-bioquant/scArchon/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/hdsu-bioquant/scArchon.svg?style=social&label=Fork)](https://github.com/hdsu-bioquant/scArchon/network)
<!-- badges: end -->



# scArchon: benchmark or run single-cell prediction tools on your own dataset

scArchon is a modular, reproducible benchmarking platform for evaluating single-cell perturbation response prediction tools. Built on Snakemake, it provides an extensible framework to compare deep learning methods across diverse datasets using both statistical and biological metrics. Why scArchon?
While many tools exist to predict single-cell responses to perturbations (e.g., drug treatments), their systematic comparison has been limited. Importantly, scArchon provides environments for each of the tools to aleviate problems related to their installation. scArchon helps standardize benchmarking and highlights important nuances—such as when models with high quantitative scores fail to retain key biological signals.

We invite the community to adopt and contribute to scArchon, helping accelerate progress in single-cell perturbation modeling.

## Citation

"Tracking biological hallucinations in single-cell perturbation predictions using scArchon, a comprehensive benchmarking platform" 
Jean Radig, Robin Droit, Daria Ivona Doncevic, Albert Li, Duc Thien Bui, Thaddeus Kuehn, Luis Herfurth, Carl Herrmann

[bioRxiv 2025.06.23.661046](https://doi.org/10.1101/2025.06.23.661046) 

## Youtube tutorials

- [scArchon: how to run on your own data](https://youtu.be/OEM51lyS5as) 
- [scArchon: how to add a new tool](https://youtu.be/sOf-wCVfiUA)

# Requirements
Running the deep learning models require GPU with CUDA 12.4+. To pull the environments from Dockerhub, Singularity 3.6+ needs to be installed on your machine. To store the environments, a disk space of about 60 GB is required. 

- CUDA 12.4+ (tested on 12.4)
- Singularity 3.6+ (tested on 3.6 and 4.1)
- About 60 GB disk space to download all environments (but we recommend selecting a subset of tools) 

# Installation
- Create a conda environment with snakemake:
    ```python
    conda create -c conda-forge -c bioconda -n snakemake_env snakemake
    ```
- Activate the environment: `conda activate snakemake_env`
- Ensure that you have a GPU with CUDA 12.4+ and Singularity 3.6+ available

# Running your experiments

- Clone or download scArchon and cd into the directory.
- You can set up your experiments in `config/datasets.tsv`.

<div align="center">
    <img src="images/001.png" alt="Description of Image" style="width: 100%; margin: 0 auto;">
</div>

- Prepare your adata: if your batch values have spaces, e.g. "T Cell", remove the space (also in your .h5ad), i.e. change it to "TCell", because of the tsv format it won't work otherwise. To ensure correct data format, use `adata.write("adata.h5ad", compression='gzip')` when saving your adata before running the pipeline.

- Do not put spaces between the comas separating the different targets or tools. Write the tools in lower caps. 

- If you are running the tools on a single GPU, it is suggested to run the tools one by one, otherwise the tasks will swap and will take overall longer. We suggest to run the pipeline with following command:

    ```python
    snakemake --use-singularity --singularity-args '--nv -B .:/dum' --cores all --jobs 1 --keep-going
    ```

    - `--use-singularity` will pull the docker images from the web
    - `--singularity-args '--nv -B .:/dum'` ensures GPU usage
    -  `--cores all` requests all CPUs available
    - `--jobs 1` runs one job after the other
    - `--keep-going` ensures the pipeline continues running even if a job fails to not lose time

# Input / Outputs
- Input: annotated dataset (adata) in .h5ad format. The dataset should ideally be count normalised (typically to 10,000) and log-normalised. The dataset should contain the couples control-perturbed necessary for the training along the control you want to get the prediction from. Ensure unique variables and observations. See the Kang dataset and the section *Running your experiments* for an example. Care, if your batch values have spaces, e.g. "T Cell", remove the space, i.e. change it to "TCell", because of the tsv format it won't work otherwise. 
- Outputs:
    - .h5ad with prediction, alongside the control and perturbed data. Stored in `results/{experiment_name}/h5ad/{experiment_name}_{tool}_{target}.h5ad`
    - Metrics results. Stored in `results/{experiment_name}/metrics/{experiment_name}_{tool}_{target}_distance_scores.csv`
    - Dimension reduction visualisation. Stored in `results/{experiment_name}/biology/{experiment_name}_{tool}_{target}_dim_red_vis.pdf`
    - Gene set enrichment analysis. Stored in `results/{experiment_name}/biology`
        - the file `{experiment_name}_{tool}_{target}_predicted_singificantly_enriched_terms.csv` contains the the enriched terms from the top 1,000 DEGs between control and predicted
        - the file `{experiment_name}_{tool}_{target}_stimulated_singificantly_enriched_terms.csv` contains the the enriched terms from the top 1,000 DEGs between control and stimulated
        - the file `{experiment_name}_{tool}_{target}_common_singificantly_enriched_terms.csv` contains the the enriched terms from the top 1,000 DEGs between stimulated and predicted (and not the intersection of the two previous files!)
        - the image `{experiment_name}_{tool}_{target}_shared_enriched_terms.pdf` shows the terms from predicted and stimulated files that are common to both
        - the image `{experiment_name}_{tool}_{target}_score_genes_enriched_terms_only_in_predicted.pdf`shows the gene score for top 6 most statistically significant GO terms from the predicted file (compated to control)
        - the image `{experiment_name}_{tool}_{target}_score_genes_enriched_terms_only_in_stimulated.pdf`shows the gene score for top 6 most statistically significant GO terms from the perturbed file (compated to control)
        -  the image `{experiment_name}_{tool}_{target}_score_genes_enriched_terms_common.pdf` shows the gene score for top 6 most statistically significant GO terms that are shared between the predicted and perturbed files. 
    - results/{experiment_name}/benchmark: comparison of the different scores obtained on the different targets by the different tools.

# User-useful information
- The running time of some tools can be long. Given the performance of cellOT, CPA and scPreGAN, we suggest you to leave them out of your run.

<div align="center">
    <img src="images/003.png" alt="Description of Image" style="width: 100%; margin: 0 auto;">
</div>

- Pulling environments via singularity may take some time depending on your downloading speed. The environments only need to be pulled once. They will be stored under `.snakemake/singularity`. The environments will take up following disk space.

| | Singularity image disk space|
|-----|:-----------------------:|
|cellot (+scgen)| 2.22 GB (+6.08 GB)|
|cpa| 6.48 GB|
|scgen| 6.08 GB|
|scvidr| 5.97 GB|
|scpram| 4.67 GB|
|scpregan| 7.19 GB|
|scdisinfact| 6.71 GB|
|trvae| 6.48 GB|
|screen| 6.37 GB|
|metrics/linear/control| 8.04 GB|
|**Total**|60.21 GB|

- In details, the different tools require following CUDA versions.

||CUDA version|
|--|:----------:|
|cellot| 10.2|
|cpa| 11.7|
|scdisinfact|12.4|
|scpram|11.6|
|scvidr|12.1|
|scpregan| 12.1 |
|screen| 11.7|
|scgen| 11.7|
|trvae| 12.4|

Below CUDA 11.6, no tool can be run. After CUDA 12.4 all tools can be run.

# Tools to be added
More tools are coming out and need to be benchmarked. Adding the tools also require to be able to run and reproduce the results in the papers when available. These two steps might require more information and changes from the authors of the given paper, which may lead to some delay in their integration in the pipeline. Hereafter is a list of tool we are planning to add to the pipeline. 

- [scCADE](https://ieeexplore.ieee.org/document/10822339). Added to IEEE Xplore on the 10th of January 2025. Compared against scGen, scPreGAN, CPA and scPRAM.
- [scVAEder](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-025-03519-4) Published on the 21st of March 2025. They compare against scGen and scPreGAN.
- [coupleVAE](https://academic.oup.com/bib/article/26/2/bbaf126/8104857) Published on the 3rd of April 2025. They compare against scPreGAN, trVAE, scGen, CVAE, scPRAM and scVIDR.

If you would like to add your own tool or need any help, please do not hesitate to contact us.