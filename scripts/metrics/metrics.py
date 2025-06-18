# R2 function as implemented in scButterfly, defined in scGen
# MMD as defined in cellOT
# Other metrics from pertpy, using default values
import warnings
warnings.filterwarnings("ignore")
def fxn():
    warnings.warn("deprecated", DeprecationWarning)
with warnings.catch_warnings(action="ignore"):
    fxn()
import logging
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)


import numpy as np
import os
import scanpy as sc
from anndata import AnnData
import pandas as pd
import matplotlib.pyplot as plt
import anndata as ad
import pertpy
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse import issparse
from sklearn.metrics import mean_squared_error

import gseapy as gp
from gseapy import Biomart
from gseapy.plot import dotplot
import requests
import json
import scanpy as sc
import numpy as np
import pandas as pd
import requests
import json
import textwrap
import os


class Metrics:
    def __init__(self, adata, keys = None, bootstrap = 1, experiment_name = 'exp'):
        self.adata = adata
        self.keys = keys
        self.predicted = self.adata[adata.obs[self.keys["condition"]] == self.keys["predicted"]]
        self.control = self.adata[adata.obs[self.keys["condition"]] == self.keys["control"]]
        self.ground_truth = self.adata[adata.obs[self.keys["condition"]] == self.keys["stimulated"]]
        self.bootstrap = bootstrap
        self.distance_scores = pd.DataFrame()
        self.experiment_name = experiment_name

#### log number of cells in each condition ####
    def cell_numbers(self):
        print('Logging number of cells in each condition')
        self.distance_scores.loc[0, "#cells_ctrl"] = int(self.control.shape[0])
        self.distance_scores.loc[0, "#cells_pred"] = int(self.predicted.shape[0])
        self.distance_scores.loc[0, "#cells_sti"] = int(self.ground_truth.shape[0])

#### compute metrics using pertpy ####
    def pertpy_metrics(self):
        metric_names = [
            "mse",
            "wasserstein",
            "pearson_distance",
            "mmd",
            "t_test",
            "cosine_distance"
            ]
    

        for distance_metric in metric_names:
            print(f"Computing distance {distance_metric}")
            distance = pertpy.tl.Distance(distance_metric)
            self.distance_scores[distance_metric] = [
                tuple(float(val) for val in distance.bootstrap(
                self.ground_truth.X.toarray(),
                self.predicted.X.toarray(),
                n_bootstrap=self.bootstrap
            ))
            ]

#### fetch common DEGs ####
    def common_degs(self):
        print('Computing common DEGs')
        sc.tl.rank_genes_groups(
            self.adata,
            groupby=self.keys['condition'],
            reference=self.keys['control'],
            method="t-test",
            show=False,
        )
        degs_sti = self.adata.uns["rank_genes_groups"]["names"][self.keys['stimulated']]
        degs_pred = self.adata.uns["rank_genes_groups"]["names"][self.keys['predicted']]

        common_degs = list(set(degs_sti[0:100]) & set(degs_pred[0:100]))
        common_nums = len(common_degs)

        self.distance_scores["common_DEGs_top_100"] = common_nums

        common_degs_20 = list(set(degs_sti[0:20]) & set(degs_pred[0:20]))
        common_nums_20 = len(common_degs_20)

        self.distance_scores["common_DEGs_top_20"] = common_nums_20


#### compute r2 scores ####
    def r2_scores(self):
        print('Computing r2 scores')

        def get_pearson2(eval_adata, key_dic, n_degs=100, sample_ratio=0.8, times=100):

            stim_key = self.keys['stimulated']
            pred_key = self.keys['predicted']
            ctrl_key = self.keys['control']
            condition_key = self.keys['condition']

            # Ensure unique cell indices (obs) and gene names (var)
            eval_adata = eval_adata[~eval_adata.obs.index.duplicated(keep='first')].copy()
            eval_adata = eval_adata[:, ~eval_adata.var.index.duplicated(keep='first')].copy()

            # Ensure that the necessary groups exist in the AnnData object
            unique_conditions = eval_adata.obs[condition_key].unique()
            if stim_key not in unique_conditions:
                raise ValueError(f"Stimulated condition '{stim_key}' not found in condition key '{condition_key}'")
            if pred_key not in unique_conditions:
                raise ValueError(f"Predicted condition '{pred_key}' not found in condition key '{condition_key}'")
            if ctrl_key not in unique_conditions:
                raise ValueError(f"Control condition '{ctrl_key}' not found in condition key '{condition_key}'")

            # Perform rank genes groups to get DEGs
            sc.tl.rank_genes_groups(eval_adata, groupby=condition_key, reference=ctrl_key, method="wilcoxon")

            # Select the top n DEGs
            degs = eval_adata.uns["rank_genes_groups"]["names"][stim_key][:n_degs]

            # Convert AnnData object to DataFrame
            df_adata = eval_adata.to_df()

            # Ensure unique column names (genes)
            df_adata = df_adata.loc[:, ~df_adata.columns.duplicated(keep='first')]

            # Filter the data for stimulated and predicted cells
            df_stim = df_adata.loc[eval_adata.obs[condition_key] == stim_key, :]
            df_pred = df_adata.loc[eval_adata.obs[condition_key] == pred_key, :]

            # Ensure unique row indices (obs) in filtered DataFrames
            df_stim = df_stim[~df_stim.index.duplicated(keep='first')]
            df_pred = df_pred[~df_pred.index.duplicated(keep='first')]

            data = np.zeros((times, 1))
            for i in range(times):
                # Bootstrap sampling
                stim = df_stim.sample(frac=sample_ratio, random_state=i)
                pred = df_pred.sample(frac=sample_ratio, random_state=i)

                # Compute mean of DEGs
                stim_degs_mean = stim.loc[:, degs].mean().values.reshape(1, -1)
                pred_degs_mean = pred.loc[:, degs].mean().values.reshape(1, -1)

                # Calculate R² score
                r2_degs_mean = (np.corrcoef(stim_degs_mean, pred_degs_mean)[0, 1]) ** 2

                data[i, :] = [r2_degs_mean]
            
            df = pd.DataFrame(data, columns=['r2_degs_mean'])
            return df

        # Define the keys for different conditions
        key_dict = {
            "condition_key": self.keys['condition'],
            "ctrl_key": self.keys['control'],
            "stim_key": self.keys['stimulated'],
            "pred_key": self.keys['predicted'],
        }

        # Perform calculations for various DEG sets
        df_deg_all = get_pearson2(
            self.adata, key_dic=key_dict, n_degs=self.adata.shape[1], sample_ratio=0.8, times=self.bootstrap
        )
        df_deg_20 = get_pearson2(
            self.adata, key_dic=key_dict, n_degs=20, sample_ratio=0.8, times=self.bootstrap
        )
        df_deg_100 = get_pearson2(
            self.adata, key_dic=key_dict, n_degs=100, sample_ratio=0.8, times=self.bootstrap
        )

        # Store results in distance_scores with normal float values
        self.distance_scores['r2_20_degs'] = [(float(df_deg_20.mean()['r2_degs_mean']), float(df_deg_20.var()['r2_degs_mean']))]
        self.distance_scores['r2_100_degs'] = [(float(df_deg_100.mean()['r2_degs_mean']), float(df_deg_100.var()['r2_degs_mean']))]
        self.distance_scores['r2_all_degs'] = [(float(df_deg_all.mean()['r2_degs_mean']), float(df_deg_all.var()['r2_degs_mean']))]

#### plot pca, umap and tsne ####
    def get_umap_pca_tsne_plots(self, conditions_to_plot = ['Predicted','Stimulated', 'Control'], points_to_keep = None, output_path = "../results", exp_index = 'exp1'):
        # Rename the condition values in adata.obs
        def rename_conditions(adata, condition_key='condition'):
            """
            Renames the condition labels in the specified column of adata.obs.

            Parameters:
                adata: AnnData
                    The annotated data matrix.
                condition_key: str
                    The key in `adata.obs` where the condition labels are stored.
            """
            if condition_key not in adata.obs:
                raise KeyError(f"'{condition_key}' not found in adata.obs.")

            # Define the mapping from old names to new names
            condition_mapping = {
                self.keys['control']: 'Control',
                self.keys['stimulated']: 'Stimulated',
                self.keys['predicted']: 'Predicted',
            }

            # Apply the mapping to rename conditions
            adata.obs[condition_key] = adata.obs[condition_key].replace(condition_mapping)
        
        def restore_conditions(adata, condition_key='condition'):

            # Define the reverse mapping from new names back to original keys
            reverse_mapping = {
                'Control': self.keys['control'],
                'Stimulated': self.keys['stimulated'],
                'Predicted': self.keys['predicted'],
            }

            # Apply the reverse mapping to restore original condition labels
            adata.obs[condition_key] = adata.obs[condition_key].replace(reverse_mapping)


        import os
        import numpy as np
        import matplotlib.pyplot as plt

        def plot_pca_umap_tsne(adata, condition_key, conditions_to_plot=["Control", "Stimulated", "Predicted"], output_path=output_path):
            """
            Plots PCA, UMAP, and t-SNE if they are present in the AnnData object, 
            ensuring equal point count across conditions and correct ordering.

            Parameters:
                adata: AnnData
                    The annotated data matrix with PCA, UMAP, and t-SNE representations.
                condition_key: str
                    The key in `adata.obs` to use for coloring the plots.
                conditions_to_plot: list or None
                    A list of conditions to plot. If None, all conditions are plotted.
                output_path: str
                    Path where the plots will be saved.
                exp_index: str or int
                    Experiment index used in the output filename.
            """

            # Define colors for conditions
            condition_colors = {'Control': '#fdae61', 'Stimulated': '#66c2a5', 'Predicted': '#5e4fa2'}

            # Check if condition_key exists
            if condition_key not in adata.obs:
                raise KeyError(f"'{condition_key}' not found in adata.obs.")

            # Filter based on conditions_to_plot
            if conditions_to_plot is not None:
                if not isinstance(conditions_to_plot, list):
                    raise TypeError("conditions_to_plot must be a list of conditions.")
                adata = adata[adata.obs[condition_key].isin(conditions_to_plot)].copy()

            # Ensure PCA, UMAP, or t-SNE exist
            has_pca = 'X_pca' in adata.obsm
            has_umap = 'X_umap' in adata.obsm
            has_tsne = 'X_tsne' in adata.obsm

            if not any([has_pca, has_umap, has_tsne]):
                raise ValueError("No PCA, UMAP, or t-SNE embeddings found in adata.")

            # Ensure equal number of points across conditions
            min_cells = min(adata.obs[condition_key].value_counts())  # Find smallest condition
            np.random.seed(42)  # For reproducibility

            def downsample_condition(adata, condition, size):
                return adata[adata.obs[condition_key] == condition].obs.sample(size, random_state=42).index

            selected_indices = np.concatenate([
                downsample_condition(adata, cond, min_cells) for cond in adata.obs[condition_key].unique()
            ])
            adata = adata[selected_indices, :]

            # Count active plots
            n_plots = sum([has_pca, has_umap, has_tsne])

            # Create square subplots
            fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6), constrained_layout=True)
            if n_plots == 1:
                axes = [axes]  # Ensure axes is iterable

            plot_index = 0  # Track subplot index

            def plot_embedding(ax, embedding, title, xlabel, ylabel):
                """Helper function to plot embeddings, ensuring predicted points are on top."""
                conditions = ["Control", "Stimulated", "Predicted"]
                
                # Plot normal conditions first
                for condition in conditions:
                    if condition in adata.obs[condition_key].unique():
                        mask = adata.obs[condition_key] == condition
                        ax.scatter(
                            embedding[mask, 0], 
                            embedding[mask, 1], 
                            c=condition_colors[condition],  
                            alpha=0.5, 
                            s=10, 
                            label=condition
                        )

                ax.set_title(title, fontsize=14)
                ax.set_xlabel(xlabel, fontsize=12)
                ax.set_ylabel(ylabel, fontsize=12)
                ax.set_box_aspect(1)  # Ensure square shape

            # Plot PCA
            if has_pca:
                plot_embedding(axes[plot_index], adata.obsm['X_pca'], "PCA Projection", "PC1", "PC2")
                plot_index += 1

            # Plot UMAP
            if has_umap:
                plot_embedding(axes[plot_index], adata.obsm['X_umap'], "UMAP Projection", "UMAP1", "UMAP2")
                plot_index += 1

            # Plot t-SNE
            if has_tsne:
                plot_embedding(axes[plot_index], adata.obsm['X_tsne'], "t-SNE Projection", "t-SNE1", "t-SNE2")

            # Add legend to the right
            handles = [plt.Line2D([], [], marker="o", linestyle="", color=color, label=cond) 
                    for cond, color in condition_colors.items()]
            fig.legend(
                handles=handles, 
                loc='center left', 
                bbox_to_anchor=(1.02, 0.5),  # Position legend just outside the right of plots
                fontsize=12
            )

            # Ensure the output directory exists
            os.makedirs(output_path, exist_ok=True)

            # Save the figure using os.path.join() to handle different OS path formats
            save_path = os.path.join(output_path, f"{exp_index}_dim_red_vis.pdf")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

        def recompute_pca_umap(adata, n_pcs=50, umap_min_dist=0.5, umap_spread=1.0, random_state=42):
            """
            Recomputes PCA and UMAP for the given AnnData object.

            Parameters:
                adata: AnnData
                    The AnnData object to recompute PCA and UMAP on.
                n_pcs: int
                    Number of principal components to compute.
                umap_min_dist: float
                    The minimum distance parameter for UMAP.
                umap_spread: float
                    The spread parameter for UMAP.
                random_state: int
                    Random state for reproducibility.

            Returns:
                AnnData
                    The AnnData object with updated PCA and UMAP embeddings.
            """
            
            # Recompute PCA
            sc.pp.pca(adata, n_comps=n_pcs, random_state=random_state)
            print("PCA recomputed.")
            
            # Recompute neighbors and UMAP
            sc.pp.neighbors(adata, n_pcs=n_pcs, random_state=random_state)
            sc.tl.umap(adata, min_dist=umap_min_dist, spread=umap_spread, random_state=random_state)
            print("UMAP recomputed.")

            # Recompute t-SNE
            sc.tl.tsne(adata, random_state=random_state)
            print("t-SNE recomputed.")
            
            return adata



        adata = self.adata
        rename_conditions(adata, condition_key=self.keys['condition'])
        adata = recompute_pca_umap(adata, n_pcs=50, umap_min_dist=0.5, umap_spread=1.0, random_state=42)
        plot_pca_umap_tsne(adata, condition_key=self.keys['condition'], conditions_to_plot=['Predicted','Stimulated', 'Control'])
        restore_conditions(adata, condition_key=self.keys['condition'])

    def checks(self, method = None):
        # Ensure that the AnnData object is valid
        self.adata.obs_names_make_unique()
        self.adata.var_names_make_unique()
        
        if method == 'add_min':
            # Add min to all values to avoid negative values
            self.adata.X = self.adata.X.toarray() if issparse(self.adata.X) else self.adata.X  
            self.adata.X += abs(self.adata.X.min())

        if method == 'neg_to_0':
            # Set negative values to 0
            self.adata.X = self.adata.X.toarray() if issparse(self.adata.X) else self.adata.X  
            self.adata.X = np.maximum(self.adata.X, 0)

    def get_scores(self, output_path = "../results", exp_index = 'exp1'):
        # Compute selected scores
        self.cell_numbers()
        self.pertpy_metrics()
        self.common_degs()
        self.r2_scores()

        biology_path = os.path.join(output_path, 'biology')
        self.get_umap_pca_tsne_plots(output_path=biology_path, exp_index=exp_index)
        self.enrichment_experiment(output_path=biology_path, exp_index=exp_index)

        # Save the results to a CSV file
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        self.distance_scores.to_csv(f"{output_path}/metrics/{exp_index}_distance_scores.csv")

    
    def enrichment_experiment(self, output_path = '../results/enrichment_experiment', exp_index='exp1'):
        
        # Compute differential expression
        def compute_differential_expression(adata):
            sc.tl.rank_genes_groups(adata, groupby=self.keys["condition"], reference=self.keys["control"], method="t-test")
            degs_sti = list(adata.uns["rank_genes_groups"]["names"][self.keys["stimulated"]][0:1000])
            degs_pred = list(adata.uns["rank_genes_groups"]["names"][self.keys["predicted"]][0:1000])
            return degs_sti, degs_pred

        # Perform gene set enrichment analysis
        def perform_enrichment_analysis(degs, adata):
            msigdb_results =  gp.enrichr(
                gene_list=degs,
                gene_sets=['MSigDB_Hallmark_2020'],
                organism='Human',
                outdir=None,
                background=list(adata.var.index)
            )

            go_results = gp.enrichr(
                gene_list=degs,
                gene_sets=['GO_Biological_Process_2021',
                        'GO_Molecular_Function_2021',
                        'GO_Cellular_Component_2021'],
                organism='Human',
                outdir=None)


            kegg_results = gp.enrichr(
                gene_list=degs,
                gene_sets='KEGG_2021_Human',
                organism='Human',
                outdir=None)
            
            return msigdb_results, go_results, kegg_results

        # Save enrichment results
        def save_enrichment_results(enr, output_dir, index, marker):
            significant_results = enr.results.loc[enr.results["Adjusted P-value"] <= 0.05]
            suffix = f'{index}_{marker}_singificantly_enriched_terms.csv'
            significant_results.to_csv(os.path.join(output_dir, suffix))
            return significant_results

        def yield_significant_enrichment_terms(adata, output_dir, index, experiment):
            # Compute DEGs 
            degs_sti, degs_pred = compute_differential_expression(adata)
            common_degs = list(set(degs_sti) & set(degs_pred))

            # If gb, use cancer hallmarks
            if experiment == 'gb' or experiment == 'gb_harmonised':
                enr1, _, _ = perform_enrichment_analysis(degs_sti, adata)
                enr2, _, _ = perform_enrichment_analysis(degs_pred, adata)
                enr3, _, _ = perform_enrichment_analysis(common_degs, adata)

            # For the others, use GO
            else:
                _, enr1, _ = perform_enrichment_analysis(degs_sti, adata)
                _, enr2, _ = perform_enrichment_analysis(degs_pred, adata)
                _, enr3, _ = perform_enrichment_analysis(common_degs, adata)
            
            # Fetch the enrichment terms for which p < 0.05
            sign_enr1 = save_enrichment_results(enr1, output_dir, index, 'stimulated')
            sign_enr2 = save_enrichment_results(enr2, output_dir, index, 'predicted')
            sign_enr3 = save_enrichment_results(enr3, output_dir, index, 'shared')

            return sign_enr1, sign_enr2, sign_enr3


        def shared_enrichment_terms(adata, output_dir, index, experiment):
            sign_enr1, sign_enr2, _ = yield_significant_enrichment_terms(adata, output_dir, index, experiment)

            # Plot the number of enrichment terms
            number_stim =len(sign_enr1["Term"])
            number_pred =len(sign_enr2["Term"])
            number_common = len(list(set(sign_enr1["Term"]).intersection(set(sign_enr2["Term"]))))
            self.distance_scores['common_enrichment_terms'] = number_common
            plt.bar(['stimulated', 'predicted', 'common'], [number_stim, number_pred, number_common])
            plt.title(f'{index} number of enriched terms')
            plt.ylabel('Number of enriched terms')
            suffix = f'{index}_shared_enriched_terms.pdf'
            plt.savefig(os.path.join(output_dir, suffix))
            plt.close()

        def gene_score_top_enriched_terms(adata, output_dir, index, experiment):
            print("Plotting score for top enriched terms...")

            # Get enriched terms
            sign_enr1, sign_enr2, sign_enr3 = yield_significant_enrichment_terms(adata, output_dir, index, experiment)

            # Find terms that are unique to sign_enr1 and sign_enr2
            unique_terms_enr1 = set(sign_enr1['Term']) - set(sign_enr2['Term'])
            unique_terms_enr2 = set(sign_enr2['Term']) - set(sign_enr1['Term'])
            common_terms = set(sign_enr1['Term']) & set(sign_enr2['Term'])  

            colors = ["orange", "green", "blue"]
            # Function to plot UMAPs + boxplots for given terms
            def plot_terms(terms, sign_enr, filename_suffix):
                if len(terms) == 0:
                    print(f"No terms found for {filename_suffix}. Skipping...")
                    return

                min_score, max_score = np.inf, -np.inf

                # First loop: Determine global min/max scores
                for term in list(terms)[:6]:  # Limit to 6 terms for consistent layout
                    gene_list = sign_enr.loc[sign_enr['Term'] == term, 'Genes'].values[0].split(';')
                    adata.var_names = adata.var_names.str.upper()
                    sc.tl.score_genes(adata, gene_list=gene_list)

                    # Update min and max scores across all gene sets
                    min_score = min(min_score, adata.obs['score'].min())
                    max_score = max(max_score, adata.obs['score'].max())

                # Set up a 2-row layout (UMAPs in row 1, Box plots in row 2)
                fig, axes = plt.subplots(2, 6, figsize=(30, 10))  # 2 rows, 6 columns
                axes = axes.reshape(2, 6)  # Ensure correct shape

                # Second loop: Plot UMAP + Box Plot for each term
                for i, term in enumerate(list(terms)[:6]):  # Ensure max 6 terms
                    wrapped_term = "\n".join(textwrap.wrap(term, width=25))
                    gene_list = sign_enr.loc[sign_enr['Term'] == term, 'Genes'].values[0].split(';')
                    adata.var_names = adata.var_names.str.upper()
                    sc.tl.score_genes(adata, gene_list=gene_list)

                    # Plot UMAP in row 1
                    sc.pl.umap(adata, color='score', ax=axes[0, i], show=False, vmin=min_score, vmax=max_score)
                    axes[0, i].set_title(f"{wrapped_term}")

                    # Create box plot for the same term in row 2
                    table_boxplot = [
                        adata.obs.loc[adata.obs[self.keys["condition"]] == self.keys["predicted"]]["score"],
                        adata.obs.loc[adata.obs[self.keys["condition"]] == self.keys["stimulated"]]["score"],
                        adata.obs.loc[adata.obs[self.keys["condition"]] == self.keys["control"]]["score"],
                    ]
                    labels = ["predicted", "stimulated", "control"]

                    # Create a box plot with custom colors
                    box = axes[1, i].boxplot(table_boxplot, labels=labels, patch_artist=True)
                    for patch, color in zip(box["boxes"], colors):
                        patch.set(facecolor=color, edgecolor=color)

                    axes[1, i].set_title(f"{wrapped_term}")
                    axes[1, i].set_ylabel("Gene score")

                # Adjust layout
                plt.tight_layout()

                # Save the figure
                filename = os.path.join(output_dir, f"{index}_{filename_suffix}.pdf")
                plt.savefig(filename, bbox_inches="tight")
                plt.close()

                print(f"Saved: {filename}")

            # Plot for unique terms in sign_enr1
            plot_terms(unique_terms_enr1, sign_enr1, "score_genes_enriched_terms_only_in_stimulated")

            # Plot for unique terms in sign_enr2
            plot_terms(unique_terms_enr2, sign_enr2, "score_genes_enriched_terms_only_in_predicted")

            # Plot for common terms in sign_enr1 and sign_enr2
            plot_terms(common_terms, sign_enr1[sign_enr1["Term"].isin(common_terms)], "score_genes_enriched_terms_common")

        print('Starting biology analysis...')
        shared_enrichment_terms(self.adata, output_path, exp_index, self.experiment_name)
        gene_score_top_enriched_terms(self.adata, output_path, exp_index, self.experiment_name)
        