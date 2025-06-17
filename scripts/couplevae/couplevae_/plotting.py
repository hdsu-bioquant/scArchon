import os
import numpy as np
import scanpy as sc
import pandas as pd
from scipy import stats, sparse
from adjustText import adjust_text
import matplotlib
import matplotlib.pyplot as plt
font = {'family' : 'Arial',
        # 'weight' : 'bold',
        'size'   : 14}

matplotlib.rc('font', **font)
matplotlib.rc('ytick', labelsize=14)
matplotlib.rc('xtick', labelsize=14)
path_to_save = "./result"



def calc_R2(adata, cell_type, n_genes=6000, conditions=None, fraction=0.8, data_name=None):
    if n_genes != adata.shape[1]:
        celldata = adata.copy()[adata.obs["celltype"] == cell_type]
        print(celldata.obs["condition"].unique().tolist())
        if data_name is not None:
            sc.tl.rank_genes_groups(celldata, groupby="condition", n_genes=n_genes, method="logreg")
        else:
            sc.tl.rank_genes_groups(celldata, groupby="condition", n_genes=n_genes, method="wilcoxon")
        diff_genes = celldata.uns["rank_genes_groups"]["names"][conditions["real_stim"]]
        adata = adata[:, diff_genes.tolist()]
    r_values = np.zeros((1, 100))
    real_stim = adata[adata.obs["condition"] == conditions["real_stim"]]
    pred_stim = adata[adata.obs["condition"] == conditions["pred_stim"]]
    for i in range(100):
        pred_stim_idx = np.random.choice(range(0, pred_stim.shape[0]), int(fraction * pred_stim.shape[0]))
        real_stim_idx = np.random.choice(range(0, real_stim.shape[0]), int(fraction * real_stim.shape[0]))
        if sparse.issparse(pred_stim.X):
            pred_stim.X = pred_stim.X.A
            real_stim.X = real_stim.X.A
        x = np.average(pred_stim.X[pred_stim_idx], axis=0)
        y = np.average(real_stim.X[real_stim_idx], axis=0)

        m, b, r_value, p_value, std_err = stats.linregress(x, y)
        r_values[0, i] = r_value ** 2
    return r_values.mean(), r_values.std()




def calc_R2_mean_var(adata, cell_types, n_genes, fraction=0.8, data_name=None):
    r2_means, r2_vars = [], []
    for cell_type in cell_types:
        conditions = {"real_stim": cell_type+"_real_pert", "pred_stim": cell_type+"_pred_pert"}
        r2_mean, r2_var = calc_R2(adata, cell_type, n_genes=n_genes, conditions=conditions, fraction=fraction, data_name=data_name)
        r2_means.append(r2_mean)
        r2_vars.append(r2_var)
    return r2_means, r2_vars




def calc_R2_specific_model(adata, n_genes, conditions, cell_type, rank_gene_method="wilcixon"):
    if n_genes != adata.shape[1]:
        
        sc.tl.rank_genes_groups(adata, groupby="condition", n_genes=n_genes, method=rank_gene_method)
        diff_genes = adata.uns["rank_genes_groups"]["names"][conditions[f"{cell_type}_real_pert"]]
        adata = adata[:, diff_genes.tolist()]
    r2_means, r2_vars = [], []
    r_values = np.zeros((1, 100))
    real_stim = adata[adata.obs["condition"] == f"{cell_type}_real_pert"]
    pred_stim = adata[adata.obs["condition"] == f"{cell_type}_pred_pert"]
    for i in range(100):
        pred_stim_idx = np.random.choice(range(0, pred_stim.shape[0]), int(0.8 * pred_stim.shape[0]))
        real_stim_idx = np.random.choice(range(0, real_stim.shape[0]), int(0.8 * real_stim.shape[0]))
        if sparse.issparse(pred_stim.X):
            pred_stim.X = pred_stim.X.A
            real_stim.X = real_stim.X.A
        x = np.average(pred_stim.X[pred_stim_idx], axis=0)
        y = np.average(real_stim.X[real_stim_idx], axis=0)
        m, b, r_value, p_value, std_err = stats.linregress(x, y)
        r_values[0, i] = r_value ** 2
    print(r_values.mean(), r_values.std())
    return r_values.mean(), r_values.std()




def label(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2., 1.01 * height,
                '%.2f' % float(height),
                ha='center', va='bottom', fontsize=18)
        
        
        

def grouped_barplot(df, cat, subcat, val, err, filename, title=None, fontsize=14, put_label=False, legend=False, offset=0.375):
    plt.close("all")
#     import matplotlib
    matplotlib.rc('ytick', labelsize=25)
    matplotlib.rc('xtick', labelsize=30)
    u = df[cat].unique()
    x_pos = np.arange(0, 6*len(u), 6)
    subx = df[subcat].unique()
    plt.figure(figsize=(22, 10))
    colors=["#2b7eb8","#ff851b","#37a537","#d83233","#996fc0"]
#     g = sns.catplot(x=cat, y=val, hue=subcat, data=df, kind='bar', palette="muted", height=6, legend=False)
#     g.despine(left=True)
#     plt.yticks(np.arange(0, 1.2, 0.2))
#     g.set_xticklabels(rotation=90)
#     g.set_xlabels("")
    for i, gr in enumerate(subx):
        dfg = df[df[subcat] == gr]
        b = plt.bar(x_pos + i/1.25, dfg[val].values, capsize=5, alpha=0.95, label=f"{gr}", yerr=dfg[err].values, color=colors[i])
        a=np.random.normal(dfg[val].values, dfg[err].values, (10, len(u)))
#         print(a.shape)
#         dfc=pd.DataFrame({'x': x_pos + i/1.25, 'y': a[0]})
        plt.plot(x_pos + i/1.25, a.T, '.', color='black', alpha=0.5)
        if put_label:
            label(b)
    
    plt.ylabel(r"$\mathrm{R^2}$", fontsize=25)
    plt.xticks(x_pos+offset, u, rotation=20)
    plt.ylim(0,1)
    if title is None:
        plt.title(f"", fontsize=fontsize)
    else:
        plt.title(title, fontsize=fontsize)
    if legend:
        plt.legend(bbox_to_anchor=(1.05,0.5), loc="center left", borderaxespad=0, prop={'size': 18})
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_save, filename), dpi=300)
    plt.show()
    



def gene_number_barplot(df, cat, subcat, val, filename, title=None, fontsize=14, put_label=False, legend=False, offset=0.375):
    plt.close("all")
#     import matplotlib
    matplotlib.rc('ytick', labelsize=25)
    matplotlib.rc('xtick', labelsize=30)
    u = df[cat].unique()
    x_pos = np.arange(0, 6*len(u), 6)
    subx = df[subcat].unique()
    plt.figure(figsize=(22, 10))
    colors=["#2b7eb8","#ff851b","#37a537","#d83233","#996fc0"]
#     g = sns.catplot(x=cat, y=val, hue=subcat, data=df, kind='bar', palette="muted", height=6, legend=False)
#     g.despine(left=True)
#     plt.yticks(np.arange(0, 1.2, 0.2))
#     g.set_xticklabels(rotation=90)
#     g.set_xlabels("")
    for i, gr in enumerate(subx):
        dfg = df[df[subcat] == gr]
        #b = plt.bar(x_pos + i/1.25, dfg[val].values, capsize=5, alpha=0.95, label=f"{gr}", color=colors[i])
        b = plt.bar(x_pos + i/1.25, dfg[val].values, label=f"{gr}",color=colors[i])
       # a=np.random.normal(dfg[val].values,  (10, len(u)))
#         print(a.shape)
#         dfc=pd.DataFrame({'x': x_pos + i/1.25, 'y': a[0]})
#        plt.plot(x_pos + i/1.25, a.T, '.', color='black', alpha=0.5)
        if put_label:
            label(b)
    
    plt.ylabel(r"The Number of Genes", fontsize=25)
    plt.xticks(x_pos+offset, u, rotation=20)

    if title is None:
        plt.title(f"", fontsize=fontsize)
    else:
        plt.title(title, fontsize=fontsize)
    if legend:
        plt.legend(bbox_to_anchor=(1.05,0.5), loc="center left", borderaxespad=0, prop={'size': 18})
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_save, filename), dpi=300)
    plt.show()
    
    
    
    
def plot_gene_correlation(data, cond_col, axes_map, label_map, file_path='./output_plot.pdf', 
                          genes_of_interest=None, top_genes=None, show_plot=False, include_legend=True, 
                          plot_title='', x_offset=0.3, y_offset=0.8, text_size=14):
    from scipy.sparse import issparse                              
    # Check if optional parameters were passed and set default values
    if genes_of_interest is None:
        genes_of_interest = []
    if top_genes is None:
        top_genes = []

    # Convert sparse matrix to dense matrix
    if issparse(data.X):
        data.X = data.X.toarray()

    # Retrieve groups of cells based on the condition
    group_y = data[data.obs[cond_col] == axes_map['y'], :]
    group_x = data[data.obs[cond_col] == axes_map['x'], :]

    # If top_genes are provided, handle the differential gene part
    if len(top_genes) > 0:
        subset_data_x = group_x[:, top_genes].X
        subset_data_y = group_y[:, top_genes].X

        avg_x_genes = np.mean(subset_data_x, axis=0)
        avg_y_genes = np.mean(subset_data_y, axis=0)

        # Calculate R² value
        r_squared_top = np.corrcoef(avg_x_genes, avg_y_genes)[0, 1] ** 2
        print(f'R-squared for top genes: {r_squared_top:.2f}')

    # Compute the average expression for all genes
    avg_x = np.mean(group_x.X, axis=0)
    avg_y = np.mean(group_y.X, axis=0)

    # Calculate R² value
    r_squared_all = np.corrcoef(avg_x, avg_y)[0, 1] ** 2
    print(f'R-squared for all genes: {r_squared_all:.2f}')

    # Create a DataFrame with x and y data
    df = pd.DataFrame({label_map['x']: avg_x, label_map['y']: avg_y})

    # Create a scatter plot and use sns.regplot to draw the regression line
    plt.figure()
    ax = sns.regplot(x=label_map['x'], y=label_map['y'], data=df, scatter_kws={'s': 10}, line_kws={'color': 'green'})
    
    plt.xlabel(label_map['x'], fontsize=text_size)
    plt.ylabel(label_map['y'], fontsize=text_size)

    # If a title is specified
    if plot_title:
        plt.title(plot_title, fontsize=text_size)

    # If specific genes are provided, label them
    if len(genes_of_interest) > 0:
        texts = []
        for gene in genes_of_interest:
            gene_idx = np.where(data.var_names == gene)[0][0]  # Find the gene index in var_names
            x_pos = avg_x[gene_idx]
            y_pos = avg_y[gene_idx]
            texts.append(plt.text(x_pos, y_pos, gene, fontsize=10, color='black'))
            plt.scatter(x_pos, y_pos, color='red', s=40)
        adjust_text(texts, arrowprops=dict(arrowstyle="->", color='grey', lw=0.5))

    # Display R² value
    plt.text(max(avg_x) - max(avg_x) * x_offset, max(avg_y) - y_offset * max(avg_y),
             r'$\mathrm{R^2_{\mathrm{\mathsf{all\ genes}}}}$='+ f"{r_squared_all:.2f}", fontsize=text_size)

    if len(top_genes) > 0:
        plt.text(max(avg_x) - max(avg_x) * x_offset, max(avg_y) - (y_offset + 0.15) * max(avg_y),
                r'$\mathrm{R^2_{\mathrm{\mathsf{top\ 100\ DEGs}}}}$='+ f"{r_squared_top:.2f}", fontsize=text_size)

    # Display legend if needed
    if include_legend:
        plt.legend(loc='best')

    # Save the plot
    plt.savefig(file_path, bbox_inches='tight', dpi=300)

    # Show the plot if required
    if show_plot:
        plt.show()

    plt.close()
