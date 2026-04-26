import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from pyTsetlinMachine.tm import MultiClassTsetlinMachine

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
output_dir = "/home/abi/tsetlin_project/PyTsetlinMachine/pyTsetlinMachine/euc_results"
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
# 3. Euclidean distance-based risk probability
# -------------------------------
def risk_proportion(patient, centroid_disease, centroid_healthy):
    d_disease = np.linalg.norm(patient - centroid_disease)
    d_healthy = np.linalg.norm(patient - centroid_healthy)
    # Proportion along the line connecting centroids: 1 = disease, 0 = healthy
    return d_healthy / (d_healthy + d_disease + 1e-8)

# -------------------------------
# 4. Process each dataset
# -------------------------------
for disease, sheet_name in datasets.items():
    print(f"\n=== Processing {disease} ({sheet_name}) ===")
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df = df.dropna(how="all").dropna(axis=1, how="all")

    # -------------------------------
    # Extract labels and features
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
    # Filter valid labels
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
    # Encode labels and split
    # -------------------------------
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # -------------------------------
    # Train Tsetlin Machine
    # -------------------------------
    tm = MultiClassTsetlinMachine(number_of_clauses=100, T=15, s=3.9)
    print(f"Training TM for {disease}...")
    tm.fit(X_train, y_train, epochs=50)

    # -------------------------------
    # Compute class centroids in feature space
    # -------------------------------
    X_array = np.array(X)
    class_means = np.array([X_array[y_enc == idx].mean(axis=0) for idx in range(len(le.classes_))])
    
    # For simplicity, assign one disease and one healthy centroid
    # (assuming binary classification; adjust for multiclass if needed)
    if len(le.classes_) == 2:
        centroid_disease = class_means[1]  # disease class
        centroid_healthy = class_means[0]  # healthy class
    else:
        # If multiclass, pick the largest disease class vs healthy (adjust as needed)
        centroid_disease = class_means[1]
        centroid_healthy = class_means[0]

    # -------------------------------
    # Compute per-patient distances and risk probability
    # -------------------------------
    dist_disease = np.linalg.norm(X_array - centroid_disease, axis=1)
    dist_healthy = np.linalg.norm(X_array - centroid_healthy, axis=1)

    # Proportion along the line connecting centroids (0=healthy, 1=disease)
    risk_probs = 1 - (dist_healthy / (dist_healthy + dist_disease + 1e-8))

    df_probs = pd.DataFrame({
        'Patient': X.index,
        'Label': y,
        'Dist_Disease': dist_disease,
        'Dist_Healthy': dist_healthy,
        'Risk_Probability': risk_probs
    })

    csv_path = os.path.join(output_dir, f"{disease}_per_patient_probability.csv")
    df_probs.to_csv(csv_path, index=False)
    print(f"Per-patient probability saved to {csv_path}")


    # -------------------------------
    # t-SNE for visualization
    # -------------------------------
    if len(X) > 2:
        tsne = TSNE(n_components=2, random_state=42)
        X_embedded = tsne.fit_transform(X_array)

        # Compute t-SNE centroids
        tsne_centroid_disease = X_embedded[y_enc == 1].mean(axis=0)
        tsne_centroid_healthy = X_embedded[y_enc == 0].mean(axis=0)

        # Plot t-SNE clusters
        plt.figure(figsize=(7,6))
        for cls in np.unique(y_enc):
            idx = np.where(y_enc == cls)
            plt.scatter(X_embedded[idx,0], X_embedded[idx,1], alpha=0.7, label=le.classes_[cls])
        plt.scatter(tsne_centroid_disease[0], tsne_centroid_disease[1], c='red', marker='X', s=150, label='Disease Centroid')
        plt.scatter(tsne_centroid_healthy[0], tsne_centroid_healthy[1], c='green', marker='X', s=150, label='Healthy Centroid')
        plt.legend()
        plt.title(f"{disease} t-SNE Cluster with Centroids")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{disease}_tsne_cluster_centroids.png"))
        plt.close()

        # Draw line from one example patient to centroids
        example_idx = 0  # change to visualize other patients
        plt.figure(figsize=(7,6))
        plt.scatter(X_embedded[:,0], X_embedded[:,1], alpha=0.5)
        plt.scatter(tsne_centroid_disease[0], tsne_centroid_disease[1], c='red', marker='X', s=150)
        plt.scatter(tsne_centroid_healthy[0], tsne_centroid_healthy[1], c='green', marker='X', s=150)
        patient_point = X_embedded[example_idx]
        plt.scatter(patient_point[0], patient_point[1], c='blue', s=100, label='Patient')
        plt.plot([patient_point[0], tsne_centroid_disease[0]],
                 [patient_point[1], tsne_centroid_disease[1]], 'r--', label='Distance to Disease')
        plt.plot([patient_point[0], tsne_centroid_healthy[0]],
                 [patient_point[1], tsne_centroid_healthy[1]], 'g--', label='Distance to Healthy')
        plt.legend()
        plt.title(f"{disease} t-SNE Patient Distance to Centroids")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{disease}_tsne_patient_distance.png"))
        plt.close()
