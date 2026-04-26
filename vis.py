import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine

# -------------------------------
# 1. Dataset Definitions
# -------------------------------
datasets = {
    "T2D": "T2D PATIENTS",
    "Liver": "LIVER",
    "Ovarian": "OVARIAN",
    "Pancreatic": "PANCREATIC",
}

file_path = "/home/abi/tsetlin_project/PyTsetlinMachine/pyTsetlinMachine/DATASET FOR RISK PRED.xlsx"
output_dir = "/home/abi/tsetlin_project/PyTsetlinMachine/pyTsetlinMachine/vis_results"
os.makedirs(output_dir, exist_ok=True)

# -------------------------------
# 2. Label normalization
# -------------------------------
def normalize_labels(disease, df_or_series):
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
            if diabetic == "yes" and pancreatic == "pancreatic cancer":
                combined_labels.append("PANCREATIC CANCER_T2D")
            elif diabetic == "yes" and pancreatic != "pancreatic cancer":
                combined_labels.append("NO PANCREATIC CANCER_T2D")
            elif diabetic == "no" and pancreatic == "pancreatic cancer":
                combined_labels.append("PANCREATIC CANCER_NO_T2D")
            elif diabetic == "no" and pancreatic != "pancreatic cancer":
                combined_labels.append("NO PANCREATIC CANCER_NO_T2D")
            else:
                combined_labels.append("UNKNOWN")
        return pd.Series(combined_labels)
    return df_or_series

# -------------------------------
# 3. Process each dataset
# -------------------------------
for disease, sheet_name in datasets.items():
    print(f"\n=== Processing {disease} ({sheet_name}) ===")
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df = df.dropna(how="all").dropna(axis=1, how="all")

    # -------------------------------
    # Extract features and labels
    # -------------------------------
    if disease == "Pancreatic":
        diabetic_row = df[df.iloc[:,0].astype(str).str.upper().str.contains("DIABETIC")]
        pancreatic_row = df[df.iloc[:,0].astype(str).str.upper().str.contains("PANCREATIC CANCER")]
        label_df = pd.DataFrame({
            'DIABETIC': diabetic_row.iloc[0, 1:].values,
            'Pancreatic Cancer': pancreatic_row.iloc[0, 1:].values
        })
        y = normalize_labels(disease, label_df)
        X_df = df[~df.iloc[:,0].astype(str).str.upper().str.contains("DIABETIC|PANCREATIC CANCER")]
        X = X_df.iloc[:, 1:].T.reset_index(drop=True)
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0).replace([np.inf, -np.inf], 0).astype(np.float32)
    else:
        label_row = df[df.iloc[:,0].astype(str).str.upper().str.contains("LABEL")]
        y = normalize_labels(disease, label_row.iloc[0, 1:])
        X_df = df[~df.iloc[:,0].astype(str).str.upper().str.contains("LABEL")]
        X = X_df.iloc[:, 1:].T
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0).replace([np.inf, -np.inf], 0).astype(np.float32)

    # -------------------------------
    # Filter invalid labels
    # -------------------------------
    valid_labels = ["T2D", "NO_T2D", "LIVER CANCER", "NOT_CANCER",
                    "OVARIAN CANCER", "PANCREATIC CANCER_T2D",
                    "PANCREATIC CANCER_NO_T2D", "NO PANCREATIC CANCER_T2D",
                    "NO PANCREATIC CANCER_NO_T2D"]
    valid_idx = y.isin(valid_labels)
    X = X.loc[valid_idx.values].reset_index(drop=True)
    y = y.loc[valid_idx].reset_index(drop=True)

    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {len(np.unique(y))}")
    print(f"Class distribution:\n{y.value_counts()}")

    # -------------------------------
    # Encode labels
    # -------------------------------
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_array = X.values

    # -------------------------------
    # Compute mean vectors per class
    # -------------------------------
    class_means = np.array([X_array[y_enc==i].mean(axis=0) for i in range(len(le.classes_))])

    # -------------------------------
    # Compute distance difference for each sample (closest vs others)
    # -------------------------------
    dist_diff = np.zeros(X_array.shape[0])
    for i, x in enumerate(X_array):
        dists = np.array([np.linalg.norm(x - mv) for mv in class_means])
        closest = np.min(dists)
        second_closest = np.partition(dists, 1)[1] if len(dists) > 1 else closest
        dist_diff[i] = second_closest - closest  # positive difference → closer to a class

    # -------------------------------
    # Sigmoid probability mapping
    # -------------------------------
    k = 5.0  # steepness
    disease_probs = 1 / (1 + np.exp(-k * dist_diff))

    # -------------------------------
    # Scatter plot: distance difference → probability
    # -------------------------------
    plt.figure(figsize=(6,4))
    plt.scatter(dist_diff, disease_probs, c='blue', alpha=0.7)
    plt.xlabel("Distance Difference (2nd closest - closest)")
    plt.ylabel("Risk Probability")
    plt.title(f"{disease} Risk Probability vs Distance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{disease}_prob_vs_dist.png"))
    plt.close()

    # -------------------------------
    # t-SNE visualization
    # -------------------------------
    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(X_array)

    plt.figure(figsize=(6,5))
    for cls_idx, cls in enumerate(le.classes_):
        idx = np.where(y_enc==cls_idx)
        plt.scatter(X_embedded[idx,0], X_embedded[idx,1], alpha=0.7, label=cls)
    plt.legend()
    plt.title(f"{disease} t-SNE Clusters")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{disease}_tsne_clusters.png"))
    plt.close()

    # -------------------------------
    # Combined t-SNE + probability overlay
    # -------------------------------
    plt.figure(figsize=(6,5))
    for cls_idx, cls in enumerate(le.classes_):
        idx = np.where(y_enc==cls_idx)
        plt.scatter(X_embedded[idx,0], X_embedded[idx,1],
                    c=disease_probs[idx], cmap="coolwarm",
                    s=50, alpha=0.8, label=cls)
    plt.colorbar(label="Risk Probability")
    plt.legend()
    plt.title(f"{disease} t-SNE + Probability Gradient")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{disease}_tsne_prob_overlay.png"))
    plt.close()

    print(f"Plots saved for {disease} in {output_dir}")
