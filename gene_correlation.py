#!/usr/bin/env python3
# gene_analysis.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.manifold import TSNE
from scipy.stats import ttest_ind, pearsonr
from matplotlib.backends.backend_pdf import PdfPages

sns.set(style="whitegrid", context="notebook", font_scale=1.0)

# -------------------------------
# Config
# -------------------------------
datasets = {
    "T2D": "T2D PATIENTS",
    "Liver": "LIVER",
    "Ovarian": "OVARIAN",
    "Pancreatic": "PANCREATIC",
}

file_path = "/home/abi/tsetlin_project/PyTsetlinMachine/pyTsetlinMachine/DATASET FOR RISK PRED.xlsx"
output_root = "/home/abi/tsetlin_project/PyTsetlinMachine/pyTsetlinMachine/newplotsresults"
os.makedirs(output_root, exist_ok=True)

# Parameters
TOP_N_GENES = 5           # for boxplots
CLUSTER_TOP_K = 40        # for clustermap / network (choose reasonable)
NETWORK_CORR_THRESH = 0.7 # absolute correlation threshold for edges

# -------------------------------
# Helpers
# -------------------------------
def normalize_labels(disease, df_or_series):
    # Accepts a Series (LABEL row) or a DataFrame with DIABETIC & Pancreatic Cancer columns
    if disease == "T2D":
        labels = df_or_series.astype(str).str.strip().str.upper()
        return labels.replace({"T2D": "T2D", "CONTROL": "NO_T2D"})
    elif disease == "Liver":
        labels = df_or_series.astype(str).str.strip().str.upper()
        return labels.replace({"LIVER CANCER": "LIVER CANCER", "CONTROL": "NOT_CANCER"})
    elif disease == "Ovarian":
        labels = df_or_series.astype(str).str.strip().str.upper()
        return labels.replace({"OVARIAN CANCER": "OVARIAN CANCER", "CONTROL": "NOT_CANCER"})
    elif disease == "Pancreatic":
        combined_labels = []
        for idx, row in df_or_series.iterrows():
            diabetic = str(row['DIABETIC']).strip().lower()
            pancreatic = str(row['Pancreatic Cancer']).strip().lower()
            if diabetic == "yes" and "pancreatic cancer" in pancreatic:
                combined_labels.append("PANCREATIC CANCER_T2D")
            elif diabetic == "yes" and "pancreatic cancer" not in pancreatic:
                combined_labels.append("NO PANCREATIC CANCER_T2D")
            elif diabetic == "no" and "pancreatic cancer" in pancreatic:
                combined_labels.append("PANCREATIC CANCER_NO_T2D")
            elif diabetic == "no" and "pancreatic cancer" not in pancreatic:
                combined_labels.append("NO PANCREATIC CANCER_NO_T2D")
            else:
                combined_labels.append("UNKNOWN")
        return pd.Series(combined_labels)
    else:
        return df_or_series

def disease_binary_from_labels(disease, labels_series):
    """
    Return binary array (1=disease, 0=control) based on label strings for each dataset.
    For pancreatic (multi labels) we treat any label containing 'PANCREATIC CANCER' as disease.
    """
    lab = labels_series.astype(str).str.strip().str.upper()
    if disease == "T2D":
        return (~lab.str.contains("NO_T2D") & lab.str.contains("T2D")).astype(int)
    if disease == "Liver":
        return (lab.str.contains("LIVER CANCER")).astype(int)
    if disease == "Ovarian":
        return (lab.str.contains("OVARIAN CANCER")).astype(int)
    if disease == "Pancreatic":
        return (lab.str.contains("PANCREATIC CANCER")).astype(int)
    return (lab != "").astype(int)

def safe_pearsonr(x, y):
    # Return correlation and p-value; handle constant x or y
    try:
        if np.nanstd(x) == 0 or np.nanstd(y) == 0:
            return np.nan, np.nan
        r, p = pearsonr(x, y)
        return r, p
    except Exception:
        return np.nan, np.nan

# -------------------------------
# Main loop
# -------------------------------
for disease, sheet_name in datasets.items():
    print(f"\n=== Processing {disease} ({sheet_name}) ===")
    ds_out = os.path.join(output_root, disease)
    os.makedirs(ds_out, exist_ok=True)

    # Read sheet and drop empty rows/cols
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df = df.dropna(how="all").dropna(axis=1, how="all")

    # Extract labels & expression matrix
    if disease == "Pancreatic":
        diabetic_row = df[df.iloc[:,0].astype(str).str.upper().str.contains("DIABETIC")]
        pancreatic_row = df[df.iloc[:,0].astype(str).str.upper().str.contains("PANCREATIC CANCER")]
        if diabetic_row.empty or pancreatic_row.empty:
            raise ValueError("DIABETIC or Pancreatic Cancer row not found in sheet")
        label_df = pd.DataFrame({
            'DIABETIC': diabetic_row.iloc[0, 1:].values,
            'Pancreatic Cancer': pancreatic_row.iloc[0, 1:].values
        })
        labels = normalize_labels(disease, label_df)
        expr_df = df[~df.iloc[:,0].astype(str).str.upper().str.contains("DIABETIC|PANCREATIC CANCER")]
        X = expr_df.iloc[:, 1:].T.reset_index(drop=True)
    else:
        label_row = df[df.iloc[:,0].astype(str).str.upper().str.contains("LABEL")]
        if label_row.empty:
            raise ValueError(f"LABEL row not found in sheet {sheet_name}")
        labels = normalize_labels(disease, label_row.iloc[0, 1:])
        expr_df = df[~df.iloc[:,0].astype(str).str.upper().str.contains("LABEL")]
        X = expr_df.iloc[:, 1:].T

    # Ensure gene names as columns (they come from first column of original; here X columns are sample columns?):
    # We expect X columns to be gene names, rows = samples. The earlier code transposes so columns are gene names.
    X.columns = [str(col).strip() for col in X.columns]

    # Numeric conversion and clean
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0).replace([np.inf, -np.inf], 0)

    # Ensure labels index aligns with X rows
    labels = labels.reset_index(drop=True)
    if len(labels) != X.shape[0]:
        # If labels are a Series from a single row, transpose may have given different shape.
        labels = labels.iloc[:X.shape[0]].reset_index(drop=True)

    # Binary disease vector for correlation / t-tests
    y_bin = disease_binary_from_labels(disease, labels)

    print(f"Samples: {X.shape[0]}, Genes: {X.shape[1]}")
    # Drop zero-variance genes
    nonconst = X.std(axis=0) != 0
    X = X.loc[:, nonconst]
    print(f"Non-constant genes: {X.shape[1]}")

    # -------------------------------
    # Compute per-gene stats: Pearson corr with binary label, p-value via t-test, fold-change
    # -------------------------------
    genes = X.columns
    corr_list = []
    pval_list = []
    fc_list = []

    for g in genes:
        vals = X[g].values
        # Pearson correlation with binary disease label
        r, rp = safe_pearsonr(vals, y_bin.values)
        corr_list.append(r)
        # t-test between disease and control groups
        group1 = vals[y_bin == 1]
        group0 = vals[y_bin == 0]
        try:
            if len(group1) > 1 and len(group0) > 1:
                tstat, p = ttest_ind(group1, group0, equal_var=False, nan_policy='omit')
            else:
                p = np.nan
        except Exception:
            p = np.nan
        pval_list.append(p)
        # log2 fold change (mean disease / mean control) with pseudocount
        mean1 = np.nanmean(group1) if len(group1)>0 else np.nan
        mean0 = np.nanmean(group0) if len(group0)>0 else np.nan
        # use +1 pseudocount to avoid zero issues
        if np.isnan(mean1) or np.isnan(mean0):
            fc = np.nan
        else:
            fc = np.log2((mean1 + 1.0) / (mean0 + 1.0))
        fc_list.append(fc)

    stats_df = pd.DataFrame({
        'gene': genes,
        'pearson_r': corr_list,
        'pval': pval_list,
        'log2FC': fc_list
    }).set_index('gene')

    # sort by absolute correlation
    stats_df['abs_r'] = stats_df['pearson_r'].abs()
    stats_df = stats_df.sort_values('abs_r', ascending=False)

    # Save stats table
    stats_df.to_csv(os.path.join(ds_out, f"{disease}_gene_stats.csv"))

    # -------------------------------
    # 1) Volcano plot: correlation vs -log10(p)
    #    and fold-change vs -log10(p) side by side
    # -------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14,6))
    # left: correlation volcano
    ax = axes[0]
    plot_df = stats_df.dropna(subset=['pearson_r', 'pval'])
    x = plot_df['pearson_r']
    yvals = -np.log10(plot_df['pval'].replace(0, np.nan))
    sc = ax.scatter(x, yvals, c=plot_df['pearson_r'], cmap='coolwarm', edgecolor='k', alpha=0.8)
    ax.set_xlabel('Pearson r (gene vs disease)')
    ax.set_ylabel('-log10(p-value)')
    ax.set_title(f"{disease} Volcano (correlation)")
    plt.colorbar(sc, ax=ax, label='Pearson r')

    # annotate top correlating genes
    top_corr = plot_df['abs_r'].nlargest(8).index
    for g in top_corr:
        ax.annotate(g, (plot_df.loc[g,'pearson_r'], -np.log10(plot_df.loc[g,'pval'] if plot_df.loc[g,'pval']>0 else 1e-300)),
                    fontsize=8)

    # right: log2FC volcano
    ax = axes[1]
    plot_df2 = stats_df.dropna(subset=['log2FC','pval'])
    x2 = plot_df2['log2FC']
    y2 = -np.log10(plot_df2['pval'].replace(0,np.nan))
    sc2 = ax.scatter(x2, y2, c=plot_df2['log2FC'], cmap='bwr', edgecolor='k', alpha=0.8)
    ax.set_xlabel('log2 Fold Change (disease / control)')
    ax.set_ylabel('-log10(p-value)')
    ax.set_title(f"{disease} Volcano (fold-change)")
    plt.colorbar(sc2, ax=ax, label='log2FC')

    top_fc = plot_df2['log2FC'].abs().nlargest(8).index
    for g in top_fc:
        ax.annotate(g, (plot_df2.loc[g,'log2FC'], -np.log10(plot_df2.loc[g,'pval'] if plot_df2.loc[g,'pval']>0 else 1e-300)),
                    fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(ds_out, f"{disease}_volcano_corr_fc.png"))
    plt.close(fig)

    # -------------------------------
    # 2) Boxplots for top N genes (by absolute correlation)
    # -------------------------------
    top_genes = stats_df.index[:TOP_N_GENES].tolist()
    if len(top_genes) == 0:
        print("No top genes found for boxplots.")
    else:
        fig, ax = plt.subplots(figsize=(6 + TOP_N_GENES, 6))
        plotdf = pd.melt(X[top_genes].assign(Disease=y_bin.values), id_vars='Disease', var_name='Gene', value_name='Expression')
        sns.boxplot(data=plotdf, x='Gene', y='Expression', hue='Disease', ax=ax)
        ax.set_title(f"{disease} Top {TOP_N_GENES} genes boxplots (disease vs control)")
        ax.legend(title='Disease', loc='upper right', labels=['Control' if x==0 else 'Disease' for x in ax.get_legend_handles_labels()[1]])
        plt.tight_layout()
        plt.savefig(os.path.join(ds_out, f"{disease}_top{TOP_N_GENES}_boxplots.png"))
        plt.close(fig)

    # -------------------------------
    # 3) Clustered heatmap (samples vs top K genes)
    # -------------------------------
    k = min(CLUSTER_TOP_K, X.shape[1])
    topk = stats_df.index[:k]
    if len(topk) > 1:
        try:
            cluster_data = X[topk]
            # add disease annotation colors
            row_colors = pd.Series(y_bin.values).map({0: 'lightblue', 1: 'salmon'})
            cg = sns.clustermap(cluster_data, standard_scale=1, row_cluster=True, col_cluster=True,
                                figsize=(10, max(8, k*0.15)),
                                row_colors=row_colors, cmap='vlag')
            plt.suptitle(f"{disease} Clustermap (top {k} genes by |r|)", y=1.02)
            cg.savefig(os.path.join(ds_out, f"{disease}_clustermap_top{k}.png"))
            plt.close()
        except Exception as e:
            print("Clustermap failed:", e)

    # -------------------------------
    # 4) Correlation network graph (top genes)
    # -------------------------------
    # Build correlation matrix for top genes and threshold
    net_k = min(30, X.shape[1])
    top_net_genes = stats_df.index[:net_k]
    corrmat = X[top_net_genes].corr()
    # Build graph
    G = nx.Graph()
    for g in top_net_genes:
        G.add_node(g, score=stats_df.loc[g,'pearson_r'])
    # Add edges for strong correlations
    for i, g1 in enumerate(top_net_genes):
        for j, g2 in enumerate(top_net_genes):
            if j <= i: continue
            val = corrmat.loc[g1, g2]
            if abs(val) >= NETWORK_CORR_THRESH:
                G.add_edge(g1, g2, weight=val)
    # Draw graph
    plt.figure(figsize=(10,8))
    pos = nx.spring_layout(G, seed=42, k=0.5)
    # node colors by pearson_r
    node_vals = [G.nodes[n]['score'] for n in G.nodes()]
    nodes = nx.draw_networkx_nodes(G, pos, node_size=400,
                                   node_color=node_vals, cmap='coolwarm', vmin=-1, vmax=1)
    if G.number_of_edges() > 0:
        edges = nx.draw_networkx_edges(G, pos, alpha=0.7,
                                       edge_color=[G[u][v]['weight'] for u,v in G.edges()],
                                       edge_cmap=plt.cm.PiYG, edge_vmin=-1, edge_vmax=1)
    labels_nx = nx.draw_networkx_labels(G, pos, font_size=8)
    plt.colorbar(nodes, label='pearson r')
    plt.title(f"{disease} Correlation Network (|r| >= {NETWORK_CORR_THRESH}) — top {net_k} genes")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(ds_out, f"{disease}_network_top{net_k}.png"))
    plt.close()

    print(f"Saved outputs for {disease} in {ds_out}")

print("\nAll done. Results in folder:", output_root)


























# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from matplotlib.backends.backend_pdf import PdfPages

# # -------------------------------
# # Dataset Definitions
# # -------------------------------
# datasets = {
#     "T2D": "T2D PATIENTS",
#     "Liver": "LIVER",
#     "Ovarian": "OVARIAN",
#     "Pancreatic": "PANCREATIC",
# }

# file_path = "/home/abi/tsetlin_project/PyTsetlinMachine/pyTsetlinMachine/DATASET FOR RISK PRED.xlsx"
# output_dir = "gene_tsne_combined_pdf"
# os.makedirs(output_dir, exist_ok=True)

# # -------------------------------
# # Label normalization rules
# # -------------------------------
# def normalize_labels(disease, df_or_series):
#     if disease == "T2D":
#         labels = df_or_series.astype(str).str.strip().str.upper()
#         return labels.replace({"T2D": "T2D", "CONTROL": "NO_T2D"})
#     elif disease == "Liver":
#         labels = df_or_series.astype(str).str.strip().str.upper()
#         return labels.replace({"LIVER CANCER": "LIVER CANCER", "CONTROL": "NOT_CANCER"})
#     elif disease == "Ovarian":
#         labels = df_or_series.astype(str).str.strip().str.upper()
#         return labels.replace({"OVARIAN CANCER": "OVARIAN CANCER", "CONTROL": "NOT_CANCER"})
#     elif disease == "Pancreatic":
#         combined_labels = []
#         for idx, row in df_or_series.iterrows():
#             diabetic = str(row['DIABETIC']).strip().lower()
#             pancreatic = str(row['Pancreatic Cancer']).strip().lower()
#             if diabetic == "yes" and pancreatic == "pancreatic cancer":
#                 combined_labels.append("PANCREATIC CANCER_T2D")
#             elif diabetic == "yes" and pancreatic != "pancreatic cancer":
#                 combined_labels.append("NO PANCREATIC CANCER_T2D")
#             elif diabetic == "no" and pancreatic == "pancreatic cancer":
#                 combined_labels.append("PANCREATIC CANCER_NO_T2D")
#             elif diabetic == "no" and pancreatic != "pancreatic cancer":
#                 combined_labels.append("NO PANCREATIC CANCER_NO_T2D")
#             else:
#                 combined_labels.append("UNKNOWN")
#         return pd.Series(combined_labels)
#     return df_or_series

# # -------------------------------
# # Process each dataset
# # -------------------------------
# for disease, sheet_name in datasets.items():
#     print(f"\n=== Processing {disease} ({sheet_name}) ===")
#     df = pd.read_excel(file_path, sheet_name=sheet_name)
#     df = df.dropna(how="all").dropna(axis=1, how="all")

#     # -------------------------------
#     # Extract labels and features
#     # -------------------------------
#     if disease == "Pancreatic":
#         diabetic_row = df[df.iloc[:,0].astype(str).str.upper().str.contains("DIABETIC")]
#         pancreatic_row = df[df.iloc[:,0].astype(str).str.upper().str.contains("PANCREATIC CANCER")]
#         label_df = pd.DataFrame({
#             'DIABETIC': diabetic_row.iloc[0, 1:].values,
#             'Pancreatic Cancer': pancreatic_row.iloc[0, 1:].values
#         })
#         y = normalize_labels(disease, label_df)

#         X_df = df[~df.iloc[:,0].astype(str).str.upper().str.contains("DIABETIC|PANCREATIC CANCER")]
#         X = X_df.iloc[:, 1:].T.reset_index(drop=True)
#     else:
#         label_row = df[df.iloc[:,0].astype(str).str.upper().str.contains("LABEL")]
#         y = normalize_labels(disease, label_row.iloc[0, 1:])
#         X_df = df[~df.iloc[:,0].astype(str).str.upper().str.contains("LABEL")]
#         X = X_df.iloc[:, 1:].T

#     # Convert to numeric, fill NaN/Inf
#     X = X.apply(pd.to_numeric, errors="coerce").fillna(0).replace([np.inf, -np.inf], 0)

#     # Ensure columns have proper gene names
#     X.columns = [str(col).strip() for col in X.columns]

#     # Drop zero-variance genes
#     X_nonconst = X.loc[:, X.std() != 0]

#     # Sort genes by variance (descending)
#     gene_variances = X_nonconst.var().sort_values(ascending=False)
#     X_sorted = X_nonconst[gene_variances.index]

#     print(f"Non-constant genes sorted by variance: {list(X_sorted.columns)}")

#     # t-SNE embedding
#     tsne = TSNE(n_components=2, random_state=42)
#     X_embedded = tsne.fit_transform(X_sorted.values)

#     # Save all plots in a single PDF
#     pdf_path = os.path.join(output_dir, f"{disease}_gene_tsne.pdf")
#     with PdfPages(pdf_path) as pdf:
#         # Plot colored by disease label
#         plt.figure(figsize=(6,5))
#         for lbl in np.unique(y):
#             idx = np.where(y == lbl)
#             plt.scatter(X_embedded[idx,0], X_embedded[idx,1], label=lbl, alpha=0.7, s=40)
#         plt.legend()
#         plt.title(f"{disease} t-SNE colored by disease")
#         plt.xlabel("t-SNE 1")
#         plt.ylabel("t-SNE 2")
#         plt.tight_layout()
#         pdf.savefig()
#         plt.close()

#         # Plot colored by each gene using its name (high variance genes first)
#         for gene in X_sorted.columns:
#             plt.figure(figsize=(6,5))
#             plt.scatter(X_embedded[:,0], X_embedded[:,1], c=X_sorted[gene], cmap="viridis", s=40)
#             plt.colorbar(label=f"{gene} Expression")
#             plt.title(f"{disease} t-SNE colored by {gene}")
#             plt.xlabel("t-SNE 1")
#             plt.ylabel("t-SNE 2")
#             plt.tight_layout()
#             pdf.savefig()
#             plt.close()
    
#     print(f"Saved combined t-SNE PDF for {disease} at {pdf_path}")
