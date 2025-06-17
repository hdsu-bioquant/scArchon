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


def plot_all_targets_per_cell_from_summary(summary_csv, exclude_tools=[]):
    """
    Take the summary CSV and plot the mean and var for all the targets.
    On the x-axis plot the tools. On the y-axis plot the mean and var of the metric.
    If the metric does not have var, just plot the mean as a dot.
    """

    # Create suffix for filename if tools are excluded
    exclude_suffix = ""
    if exclude_tools:
        exclude_suffix = "_without_" + "_".join(exclude_tools)
    
    # Read the summary CSV which contains the computed metrics
    summary_df = pd.read_csv(summary_csv)
    
    # Get unique tools and targets from the summary data
    tools = summary_df['Tool'].unique()
    targets = summary_df['Target'].unique()

    # Define fixed colors for each target
    target_colors = {
        target: color for target, color in zip(targets, [
            '#9e0142', '#d53e4f', '#f46d43', '#fdae61', '#fee08b', 
            '#e6f598', '#abdda4', '#66c2a5', '#3288bd', '#5e4fa2'
        ])
    }

    # Dictionary to store handles for legend
    legend_handles = {}

    # Plotting for each metric
    for metric in metrics:
        plt.figure(figsize=(7, 4))
        plt.tight_layout(pad=0.5)
        plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.3)

        # Loop through all targets
        for target in targets:
            means, variances, labels = [], [], []
            
            # Loop through all tools to get mean and variance for each metric
            for tool in tools:
                if tool in exclude_tools:
                    continue

                # Filter for the specific tool and target
                filtered_df = summary_df[(summary_df['Tool'] == tool) & (summary_df['Target'] == target)]
                
                if filtered_df.empty:
                    continue

                # Extract mean and variance for this metric
                mean_col = f"{metric}_mean"
                var_col = f"{metric}_var"

                # Ensure the metric exists in the DataFrame
                if mean_col in filtered_df.columns:
                    mean = filtered_df[mean_col].values[0]
                else:
                    continue  # Skip if metric is missing
                
                if var_col in filtered_df.columns:
                    var = filtered_df[var_col].values[0]
                else:
                    var = 0  # Default to 0 if variance is missing

                # Append the results
                means.append(mean)
                variances.append(var)
                labels.append(tool)

            # Skip plotting if no valid data for this target
            if not means:
                continue

            # Use the defined color mapping
            color = target_colors[target]

            # Plot mean and variance for each tool
            for i, tool in enumerate(labels):
                if variances[i] > 0:
                    line = plt.errorbar(
                        x=[tool],
                        y=[means[i]],
                        yerr=[variances[i]],
                        fmt='o',
                        capsize=5,
                        color=color
                    )
                else:
                    line, = plt.plot(
                        tool,
                        means[i],
                        'o',  
                        color=color,
                        markersize=8  
                    )

            # Store the legend entry only once per target
            if target not in legend_handles:
                legend_handles[target] = line

        # Only integer on axis if only integers to plot
        # Check if all mean values are integers
        all_means = means  # Use the current `means` list for this specific plot
        if all(float(value).is_integer() for value in all_means):  
            plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

        # Customize the plot
        plt.title(f'{metric}', fontsize=10)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=10)
        plt.legend(handles=legend_handles.values(), labels=legend_handles.keys(), title='Targets', 
                   bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
        plt.tight_layout()

        # Save the plot with the appropriate suffix
        plot_filename = f"results/{args.experiment_name}/benchmark/{metric}_individual_targets{exclude_suffix}.pdf"
        plt.savefig(plot_filename, dpi=300)
        plt.close()

        print(f"Plot saved for {metric}_mean_and_variance_plot.pdf")


def plot_boxplots_per_tool(summary_csv, tool_color_mapping, exclude_tools=[]):
    """
    Create box plots of mean metric values for each tool, aggregated across targets.
    Each box represents the distribution of mean values across targets for a tool.
    Metrics not present for some tools (e.g. 'control') are skipped.
    """

    # Read the summary CSV
    summary_df = pd.read_csv(summary_csv)

    # Get unique tools
    tools = summary_df['Tool'].unique()

    # Determine which metrics are available
    metric_suffixes = [col for col in summary_df.columns if col.endswith('_mean')]
    available_metrics = [col.replace('_mean', '') for col in metric_suffixes]

    exclude_suffix = ""
    if exclude_tools:
        exclude_suffix = "_without_" + "_".join(exclude_tools)

    for metric in available_metrics:
        plt.figure(figsize=(6, 5))
        plt.tight_layout(pad=0.5)
        plt.subplots_adjust(left=0.1, right=0.75, top=0.9, bottom=0.3)

        plot_data = []
        tool_colors = []

        for tool in tools:
            if tool in exclude_tools:
                continue

            mean_col = f"{metric}_mean"
            if mean_col in summary_df.columns:
                tool_means = summary_df.loc[summary_df['Tool'] == tool, mean_col].dropna().tolist()
                if tool_means:
                    plot_data.append((tool, tool_means))
                    tool_colors.append(tool_color_mapping.get(tool, '#000000'))  # fallback to black

        if not plot_data:
            continue

        tool_labels, tool_means = zip(*plot_data)

        # Create the boxplot
        box = sns.boxplot(data=tool_means, palette=tool_colors)

        all_means = list(chain.from_iterable(tool_means))
        if all(float(value).is_integer() for value in all_means):
            plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

        plt.xticks(ticks=range(len(tool_labels)), labels=tool_labels, rotation=45, fontsize=12)
        plt.title(f'{metric}', fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.6)

        legend_labels = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=tool)
                         for tool, color in zip(tool_labels, tool_colors)]
        plt.legend(handles=legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

        plot_filename = f"results/{args.experiment_name}/benchmark/{metric}_boxplot{exclude_suffix}.pdf"
        plt.savefig(plot_filename, dpi=300)
        plt.close()

        print(f"Box plot saved for {metric}_boxplot.pdf")



def plot_boxplots_per_tool_var(summary_csv, tool_color_mapping, exclude_tools=[]):
    """
    Create box plots of variance metric values for each tool, aggregated across targets.
    Each box represents the distribution of variance values across targets for a tool.
    """

    # Read the summary CSV
    summary_df = pd.read_csv(summary_csv)

    # Get unique tools
    tools = summary_df['Tool'].unique()

    exclude_suffix = ""
    if exclude_tools:
        exclude_suffix = "_without_" + "_".join(exclude_tools)

    # Dynamically determine which metrics exist with variance values
    available_var_metrics = [col.replace('_var', '') for col in summary_df.columns if col.endswith('_var')]

    for metric in available_var_metrics:
        plt.figure(figsize=(6, 5))
        plt.tight_layout(pad=0.5)
        plt.subplots_adjust(left=0.1, right=0.75, top=0.9, bottom=0.3)

        # Prepare data for box plot
        plot_data = []
        tool_colors = []

        for tool in tools:
            if tool in exclude_tools:
                continue

            var_col = f"{metric}_var"
            if var_col in summary_df.columns:
                tool_vars = summary_df.loc[summary_df['Tool'] == tool, var_col].dropna().tolist()
                if tool_vars:
                    plot_data.append((tool, tool_vars))
                    tool_colors.append(tool_color_mapping.get(tool, '#000000'))

        if not plot_data:
            continue

        tool_labels, tool_vars = zip(*plot_data)
        box = sns.boxplot(data=tool_vars, palette=tool_colors)

        all_vars = list(chain.from_iterable(tool_vars))
        if all(float(value).is_integer() for value in all_vars):
            plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

        plt.xticks(ticks=range(len(tool_labels)), labels=tool_labels, rotation=45, fontsize=12)
        plt.title(f'{metric} Variance', fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.6)

        legend_labels = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=tool)
            for tool, color in zip(tool_labels, tool_colors)
        ]
        plt.legend(handles=legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

        plot_filename = f"results/{args.experiment_name}/benchmark/{metric}_var_boxplot{exclude_suffix}.pdf"
        plt.savefig(plot_filename, dpi=300)
        plt.close()

        print(f"Box plot saved: {plot_filename}")

def plot_combined_tool_metrics_single_dataset(summary_csv, full_tool_color_mapping, tools_to_plot=None, metrics=None):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Patch
    import math
    import os
    import numpy as np

    sns.set_style('whitegrid')
    plt.rcParams.update({'axes.facecolor': 'white'})

    df = pd.read_csv(summary_csv)

    if tools_to_plot is None:
        tools_to_plot = [tool for tool in df['Tool'].unique() if tool != 'cellot']

    if metrics is None:
        metrics = ['mse_mean', 'wasserstein_mean', 't_test_mean', 'common_DEGs_top_20_mean', 'common_enrichment_terms_mean']

    # Put cellot at far left
    tools_with_cellot = ['cellot'] + tools_to_plot

    # Tool color mapping including cellot
    tool_color_mapping = {tool: full_tool_color_mapping[tool] for tool in tools_with_cellot}

    # === Ranking ===
    default_metric_directions = {
        'mse_mean': False,
        'wasserstein_mean': False,
        't_test_mean': False,
        'common_DEGs_top_20_mean': True,
        'common_enrichment_terms_mean': True
    }

    all_ranks = []
    for metric in metrics:
        higher_is_better = default_metric_directions.get(metric, True)
        metric_data = (
            df[df['Tool'].isin(tools_with_cellot)]
            .groupby('Tool')[metric].mean()
            .reset_index()
        )
        metric_data['Rank'] = metric_data[metric].rank(ascending=not higher_is_better, method='min')
        metric_data['Metric'] = metric
        all_ranks.append(metric_data[['Tool', 'Metric', 'Rank']])

    rankings_df = pd.concat(all_ranks)
    avg_rank = rankings_df.groupby('Tool')['Rank'].mean().reset_index()
    avg_rank['Score'] = 1 - (avg_rank['Rank'] - 1) / (len(tools_with_cellot) - 1)
    avg_rank = avg_rank.sort_values('Score', ascending=False)

    sorted_tools = avg_rank['Tool'].tolist()
    palette_sorted = [full_tool_color_mapping[tool] for tool in sorted_tools]

    integer_y_metrics = ['common_DEGs_top_20_mean', 'common_enrichment_terms_mean']
    n_metrics = len(metrics)
    n_cols = 4
    n_rows = math.ceil(n_metrics / n_cols)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5), constrained_layout=True)
    axs = axs.flatten()

    for idx, metric in enumerate(metrics):
        ax = axs[idx]

        df_metric = df[df['Tool'].isin(tools_with_cellot)]
        cellot_vals = df_metric[df_metric['Tool'] == 'cellot'][metric].dropna()
        tools_vals = df_metric[df_metric['Tool'] != 'cellot'][metric].dropna()

        if len(cellot_vals) == 0 or len(tools_vals) == 0:
            # fallback: plot normally if data missing
            sns.boxplot(
                x='Tool', y=metric, data=df_metric,
                palette=tool_color_mapping, ax=ax,
                order=tools_with_cellot, width=0.6,
                dodge=False
            )
            ax.set_xticklabels(tools_with_cellot, rotation=45, fontsize=8)
            ax.set_title(metric, fontsize=10)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelsize=8)
            ax.grid(False)
            if metric in integer_y_metrics:
                ax.yaxis.get_major_locator().set_params(integer=True)
            continue

        cellot_min = cellot_vals.min()
        tools_max = tools_vals.max()

        if cellot_min > tools_max:
            # Create broken y-axis with two subplots inside this subplot's bounding box
            bbox = ax.get_position()
            fig.delaxes(ax)

            # Axes positions
            fig_height = bbox.height
            fig_width = bbox.width
            x0 = bbox.x0
            y0 = bbox.y0

            gap = 0.02

            ax_bottom = fig.add_axes([x0, y0, fig_width, fig_height * 0.6])
            ax_top = fig.add_axes([x0, y0 + fig_height * 0.6 + gap, fig_width, fig_height * 0.4], sharex=ax_bottom)

            # Bottom plot (tools)
            sns.boxplot(
                x='Tool', y=metric,
                data=df_metric[df_metric['Tool'] != 'cellot'],
                palette={tool: tool_color_mapping[tool] for tool in tools_to_plot},
                ax=ax_bottom,
                order=tools_to_plot,
                width=0.6,
                dodge=False
            )
            ax_bottom.set_ylim(tools_vals.min() * 0.95, tools_max * 1.05)
            ax_bottom.tick_params(axis='x', rotation=45, labelsize=8)
            ax_bottom.tick_params(axis='y', labelsize=8)
            if metric in integer_y_metrics:
                ax_bottom.yaxis.get_major_locator().set_params(integer=True)
            ax_bottom.grid(False)

            # Top plot (cellot)
            sns.boxplot(
                x='Tool', y=metric,
                data=df_metric[df_metric['Tool'] == 'cellot'],
                palette={'cellot': tool_color_mapping['cellot']},
                ax=ax_top,
                order=['cellot'],
                width=0.6,
                dodge=False
            )
            ax_top.set_ylim(cellot_min * 0.95, cellot_vals.max() * 1.05)
            ax_top.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
            ax_top.tick_params(axis='y', labelsize=8)
            if metric in integer_y_metrics:
                ax_top.yaxis.get_major_locator().set_params(integer=True)
            ax_top.grid(False)

            # Hide spines between ax_top and ax_bottom
            ax_top.spines['bottom'].set_visible(False)
            ax_bottom.spines['top'].set_visible(False)

            # Draw diagonal break marks
            d = 0.015
            kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                          linestyle="none", color='k', mec='k', mew=1, clip_on=False)
            ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs)
            ax_bottom.plot([0, 1], [1, 1], transform=ax_bottom.transAxes, **kwargs)

            # Set title on top axis
            ax_top.set_title(metric, fontsize=10)

        else:
            # Normal boxplot (no break)
            sns.boxplot(
                x='Tool', y=metric, data=df_metric,
                palette=tool_color_mapping, ax=ax,
                order=tools_with_cellot, width=0.6,
                dodge=False
            )
            ax.set_xticklabels(tools_with_cellot, rotation=45, fontsize=8)
            ax.set_title(metric, fontsize=10)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelsize=8)
            ax.grid(False)
            if metric in integer_y_metrics:
                ax.yaxis.get_major_locator().set_params(integer=True)

    # Remove unused subplots
    for i in range(n_metrics, len(axs)):
        fig.delaxes(axs[i])

    # === Score Plot ===
    fig_score, ax_score = plt.subplots(figsize=(5, n_rows * 1.5))
    sns.barplot(data=avg_rank, y='Tool', x='Score', palette=palette_sorted, ax=ax_score)
    ax_score.set_title('Tool Scores', fontsize=12)
    ax_score.set_xlabel('')
    ax_score.set_ylabel('')
    ax_score.grid(False)
    sns.despine(ax=ax_score, left=True, bottom=True)

    # === Legend ===
    handles = [Patch(color=tool_color_mapping['cellot'], label='cellot')] + [
        Patch(color=tool_color_mapping[tool], label=tool) for tool in tools_to_plot
    ]
    labels = ['cellot'] + tools_to_plot

    fig.legend(
        handles, labels,
        title='Tool', loc='center left',
        bbox_to_anchor=(1.02, 0.5), borderaxespad=0., frameon=False
    )

    plt.tight_layout(rect=[0, 0.03, 0.98, 0.95])

    os.makedirs("results/benchmark", exist_ok=True)
    fig.savefig("results/benchmark/main_summary.pdf", dpi=300)
    fig_score.savefig("results/benchmark/score_summary.pdf", dpi=300)
    plt.close(fig)
    plt.close(fig_score)














    



if __name__ == '__main__':
    args = get_arguments()
    tool_color_mapping = generate_color_palette(args.experiment_name)
    create_summary_csv(args.experiment_name, args.without)
    plot_all_targets_per_cell_from_summary(f'results/{args.experiment_name}/benchmark/{args.experiment_name}_summary.csv', args.without)
    plot_boxplots_per_tool(f'results/{args.experiment_name}/benchmark/{args.experiment_name}_summary.csv', tool_color_mapping = tool_color_mapping, exclude_tools = args.without)
    plot_boxplots_per_tool_var(f'results/{args.experiment_name}/benchmark/{args.experiment_name}_summary.csv', tool_color_mapping = tool_color_mapping, exclude_tools = args.without)
    plot_combined_tool_metrics_single_dataset(f'results/{args.experiment_name}/benchmark/{args.experiment_name}_summary.csv', full_tool_color_mapping = tool_color_mapping)
    Path(f'flags/benchmark/output_run_flag_{args.experiment_name}_benchmark.txt').touch()
