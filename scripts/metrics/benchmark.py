import pandas as pd
import numpy as np
import ast
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from itertools import chain
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import math
import os
import numpy as np


metrics = [
    'mse', 'wasserstein', 'pearson_distance', 'mmd', 't_test', 'cosine_distance', 'common_DEGs_top_100', 'common_DEGs_top_20', 
    'r2_20_degs', 'r2_100_degs', 'r2_all_degs', 'common_diff_ge', 'precision_diff_ge', 'recall_diff_ge', 'f1_diff_ge', 
    'percent_common_over_true_diff_ge', 'common_enrichment_terms'
]

def get_arguments():
    parser = argparse.ArgumentParser(description="Run metrics computation.")
    parser.add_argument("--experiment_name", type=str, required=True, help="Experiment name to process.")
    parser.add_argument("--without", nargs='*', default=[], help="List of tools to ignore. E.g. --without cellot scgen")
    return parser.parse_args()

def generate_color_palette(experiment_name, cmap_name='Spectral'):
    cmap = cm.get_cmap(cmap_name)
    datasets_path = Path('./config/datasets.tsv')
    datasets = pd.read_csv(datasets_path, sep='\t')
    experiment_data = datasets[datasets["experiment_name"] == experiment_name]
    tools = experiment_data.iloc[0]["Tools"].split(',')
    tools.append('control') 
    num_tools = len(tools)
    colors = [mcolors.to_hex(cmap(i / (num_tools - 1))) for i in range(num_tools)]
    return dict(zip(tools, colors))

def create_summary_csv(experiment_name, without_tools):
    """
    Fetch all files in results/{experiment_name}/metrics/{experiment_name}_{tool}_{target}_distance_scores.csv with format:
    |  metric 1    |     metric 2  | metric 3 | ... |
    | '(mean, var)'| '(mean, var)' | value    | ... |

    Create a summary CSV with format:
    | Experiment_name | Tool | Target | metric 1 mean | metric 1 var| metric 2 mean | metric 2 var| metric 3 value | ... |
    """
    results_path = Path('./results')
    datasets_path = Path('./config/datasets.tsv')

    # Read dataset metadata
    datasets = pd.read_csv(datasets_path, sep='\t')

    # Get experiment-specific metadata
    experiment_data = datasets[datasets["experiment_name"] == experiment_name]
    if experiment_data.empty:
        print(f"Experiment '{experiment_name}' not found in datasets.tsv.")
        return

    targets = experiment_data.iloc[0]["target"].split(',')
    tools = experiment_data.iloc[0]["Tools"].split(',')
    tools.append('control')

    # Filter out tools that should be excluded
    filtered_tools = [tool for tool in tools if tool not in without_tools]

    output_dir = results_path / experiment_name / 'benchmark'
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_data = []  # List to store summary rows

    for target in targets:
        for tool in filtered_tools:
            metric_file = results_path / experiment_name / "metrics" / f"{experiment_name}_{tool}_{target}_distance_scores.csv"
            
            if not metric_file.exists():
                continue  # Skip missing files
            
            results = pd.read_csv(metric_file)

            summary_entry = {
                "Experiment_name": experiment_name,
                "Tool": tool,
                "Target": target
            }

            for metric in metrics:
                if metric not in results.columns:
                    continue  # Skip missing metrics

                value = results.loc[0, metric]

                if metric not in ['common_DEGs_top_100', 'common_DEGs_top_20', 'common_diff_ge', 'recall_diff_ge', 'precision_diff_ge', 'f1_diff_ge', 'percent_common_over_true_diff_ge', 'common_enrichment_terms']:
                    try:
                        mean, var = ast.literal_eval(value)  # Convert string tuple to numbers
                        summary_entry[f"{metric}_mean"] = mean
                        summary_entry[f"{metric}_var"] = var
                    except (ValueError, SyntaxError):
                        continue
                else:
                    mean = float(value)
                    summary_entry[f"{metric}_mean"] = mean  # Store only the mean

            summary_data.append(summary_entry)

    # Convert summary data to a DataFrame and save as CSV
    summary_df = pd.DataFrame(summary_data)
    summary_file = output_dir / f"{experiment_name}_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    
    print(f"Summary CSV saved: {summary_file}")



def plot_combined_tool_metrics_single_dataset(summary_csv, full_tool_color_mapping, tools_to_plot=None, metrics=None):
    sns.set_style('whitegrid')
    plt.rcParams.update({'axes.facecolor': 'white'})

    df = pd.read_csv(summary_csv)

    if metrics is None:
        metrics = [
            'mse_mean', 'wasserstein_mean', 't_test_mean', 'common_DEGs_top_20_mean',
            'common_enrichment_terms_mean', 'cosine_distance_mean', 'pearson_distance_mean',
            'common_DEGs_top_100_mean', 'mmd_mean', 'r2_20_degs_mean', 'r2_100_degs_mean',
            'r2_all_degs_mean'
        ]

    available_tools = df['Tool'].unique().tolist()
    experiment_name = df['Experiment_name'].unique()[0]

    # If user provides tools_to_plot, filter it to what's available in the dataframe
    if tools_to_plot is None:
        tools_to_plot = available_tools
    else:
        tools_to_plot = [tool for tool in tools_to_plot if tool in available_tools]

    # Final list of tools for plotting
    tools_in_plot = tools_to_plot.copy()

    # Build the color mapping only for tools present
    tool_color_mapping = {tool: full_tool_color_mapping[tool] for tool in tools_in_plot}

    integer_y_metrics = [
        'common_DEGs_top_20_mean', 'common_DEGs_top_100_mean',
        'common_DEGs_all_mean', 'common_enrichment_terms_mean',
        'r2_20_degs_mean', 'r2_100_degs_mean', 'r2_all_degs_mean'
    ]

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(6, 5))

        df_metric = df[df['Tool'].isin(tools_in_plot)]

        if 'cellot' in tools_in_plot:
            cellot_vals = df_metric[df_metric['Tool'] == 'cellot'][metric].dropna()
            tools_vals = df_metric[df_metric['Tool'] != 'cellot'][metric].dropna()

            if len(cellot_vals) > 0 and len(tools_vals) > 0:
                cellot_min = cellot_vals.min()
                tools_max = tools_vals.max()

                if cellot_min > tools_max:
                    # Split axis case
                    bbox = ax.get_position()
                    fig.delaxes(ax)
                    fig_width = bbox.width
                    fig_height = bbox.height
                    x0 = bbox.x0
                    y0 = bbox.y0
                    gap = 0.02

                    ax_bottom = fig.add_axes([x0, y0, fig_width, fig_height * 0.48])
                    ax_top = fig.add_axes([x0, y0 + fig_height * 0.48 + gap, fig_width, fig_height * 0.48], sharex=ax_bottom)

                    sns.boxplot(
                        x='Tool', y=metric,
                        data=df_metric[df_metric['Tool'] == 'cellot'],
                        palette={'cellot': tool_color_mapping['cellot']},
                        ax=ax_top,
                        order=['cellot'],
                        width=0.6,
                        dodge=False
                    )

                    sns.boxplot(
                        x='Tool', y=metric,
                        data=df_metric[df_metric['Tool'] != 'cellot'],
                        palette={tool: tool_color_mapping[tool] for tool in tools_in_plot if tool != 'cellot'},
                        ax=ax_bottom,
                        order=[tool for tool in tools_in_plot if tool != 'cellot'],
                        width=0.6,
                        dodge=False
                    )

                    ax_bottom.set_ylim(tools_vals.min() * 0.95, tools_max * 1.05)
                    ax_top.set_ylim(cellot_min * 0.95, cellot_vals.max() * 1.05)

                    ax_bottom.tick_params(axis='x', rotation=45, labelsize=8)
                    ax_bottom.tick_params(axis='y', labelsize=8)
                    ax_top.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
                    ax_top.tick_params(axis='y', labelsize=8)

                    ax_bottom.set_xlabel('')
                    ax_bottom.set_ylabel('')
                    ax_top.set_xlabel('')
                    ax_top.set_ylabel('')

                    if metric in integer_y_metrics:
                        ax_bottom.yaxis.get_major_locator().set_params(integer=True)
                        ax_top.yaxis.get_major_locator().set_params(integer=True)

                    ax_top.grid(False)
                    ax_bottom.grid(False)

                    ax_top.spines['bottom'].set_visible(False)
                    ax_bottom.spines['top'].set_visible(False)

                    d = 0.03
                    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
                    ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs)
                    ax_bottom.plot([0, 1], [1, 1], transform=ax_bottom.transAxes, **kwargs)

                    ax_top.set_title(metric, fontsize=12)

                    plt.tight_layout()
                    fig.savefig(f"results/benchmark/{metric}.pdf", dpi=300)
                    plt.close(fig)
                    continue

        # Simple boxplot (no split axis needed, or no cellot present)
        sns.boxplot(
            x='Tool', y=metric, data=df_metric,
            palette=tool_color_mapping, ax=ax,
            order=tools_in_plot, width=0.6,
            dodge=False
        )
        ax.set_xticklabels(tools_in_plot, rotation=45, fontsize=8)
        ax.set_title(metric, fontsize=12)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(axis='y', labelsize=8)

        if metric in integer_y_metrics:
            ax.yaxis.get_major_locator().set_params(integer=True)

        ax.grid(False)

        plt.tight_layout()
        fig.savefig(f"results/{experiment_name}/benchmark/{metric}.pdf", dpi=300)
        plt.close(fig)




def plot_tool_rankings_barplot(summary_csv, full_tool_color_mapping, tools_to_plot=None, metrics=None, figsize=(8,6)):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    sns.set_style('whitegrid')
    plt.rcParams.update({'axes.facecolor': 'white'})

    df = pd.read_csv(summary_csv)

    if tools_to_plot is None:
        # Keep tools in the order they appear in the csv
        tools_to_plot = df['Tool'].unique().tolist()

    if metrics is None:
        metrics = [
            'mse_mean', 'wasserstein_mean', 't_test_mean',
            'common_DEGs_top_20_mean', 'common_enrichment_terms_mean'
        ]

    default_metric_directions = {
        'mse_mean': False,
        'wasserstein_mean': False,
        't_test_mean': False,
        'common_DEGs_top_20_mean': True,
        'common_enrichment_terms_mean': True
    }

    # Compute rankings per metric
    all_ranks = []
    for metric in metrics:
        higher_is_better = default_metric_directions.get(metric, True)
        metric_means = df[df['Tool'].isin(tools_to_plot)].groupby('Tool')[metric].mean().reset_index()
        metric_means['Rank'] = metric_means[metric].rank(ascending=not higher_is_better, method='min')
        metric_means['Metric'] = metric
        all_ranks.append(metric_means[['Tool', 'Rank']])

    rankings_df = pd.concat(all_ranks)

    # Average rank per tool (in original tool order)
    avg_rank = rankings_df.groupby('Tool')['Rank'].mean().reset_index()
    avg_rank['Score'] = 1 - (avg_rank['Rank'] - 1) / (len(tools_to_plot) - 1)

    # Preserve original tool order
    avg_rank['Tool'] = pd.Categorical(avg_rank['Tool'], categories=tools_to_plot, ordered=True)
    avg_rank = avg_rank.sort_values('Tool')

    # Get colors from mapping for tools present
    palette = [full_tool_color_mapping.get(tool, '#333333') for tool in avg_rank['Tool']]

    plt.figure(figsize=figsize)
    sns.barplot(data=avg_rank, x='Score', y='Tool', palette=palette)
    plt.title('Tool Scores (Average Ranking)')
    plt.xlabel('Score (Higher is Better)')
    plt.ylabel('Tool')
    plt.grid(False)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()


    avg_rank.to_csv(f"results/{args.experiment_name}/benchmark/ranking.csv", index=False)
    plt.savefig(f"results/{args.experiment_name}/benchmark/ranking.pdf", dpi=300)
    plt.show()
    plt.close()





if __name__ == '__main__':
    args = get_arguments()
    tool_color_mapping = generate_color_palette(args.experiment_name)
    create_summary_csv(args.experiment_name, args.without)
    #plot_all_targets_per_cell_from_summary(f'results/{args.experiment_name}/benchmark/{args.experiment_name}_summary.csv', args.without)
    #plot_boxplots_per_tool(f'results/{args.experiment_name}/benchmark/{args.experiment_name}_summary.csv', tool_color_mapping = tool_color_mapping, exclude_tools = args.without)
    #plot_boxplots_per_tool_var(f'results/{args.experiment_name}/benchmark/{args.experiment_name}_summary.csv', tool_color_mapping = tool_color_mapping, exclude_tools = args.without)
    plot_combined_tool_metrics_single_dataset(f'results/{args.experiment_name}/benchmark/{args.experiment_name}_summary.csv', full_tool_color_mapping = tool_color_mapping)
    plot_tool_rankings_barplot(f'results/{args.experiment_name}/benchmark/{args.experiment_name}_summary.csv', full_tool_color_mapping = tool_color_mapping)
    Path(f'flags/benchmark/output_run_flag_{args.experiment_name}_benchmark.txt').touch()
