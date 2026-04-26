import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
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
output_dir = "/home/abi/tsetlin_project/PyTsetlinMachine/pyTsetlinMachine/prob_results"
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
# 3. Helper: stable risk probability
# -------------------------------
def stable_sigmoid(x):
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))

def risk_probability(x, mean_disease, mean_nondisease, k=0.5):
    d_disease = np.linalg.norm(x - mean_disease)
    d_nondisease = np.linalg.norm(x - mean_nondisease)
    return stable_sigmoid(k * (d_nondisease - d_disease))

# -------------------------------
# 4. Process each dataset
# -------------------------------
all_metrics = []

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
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0).replace([np.inf, -np.inf], 0).astype(np.int32)
    else:
        label_row = df[df.iloc[:,0].astype(str).str.upper().str.contains("LABEL")]
        y = normalize_labels(disease, label_row.iloc[0, 1:])
        X_df = df[~df.iloc[:,0].astype(str).str.upper().str.contains("LABEL")]
        X = X_df.iloc[:, 1:].T
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0).replace([np.inf, -np.inf], 0).astype(np.int32)

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
    # Compute class means for risk
    # -------------------------------
    X_train_array = np.array(X_train)
    class_means = np.array([X_train_array[y_train == idx].mean(axis=0) for idx in range(len(le.classes_))])

    # -------------------------------
    # Compute risk probabilities for ALL samples
    # -------------------------------
    disease_probs_all = np.zeros(len(X))
    for i, x in enumerate(X.values):
        idx_pred = le.transform([y[i]])[0]
        disease_probs_all[i] = risk_probability(x, class_means[idx_pred], class_means[(idx_pred+1)%len(le.classes_)])

    # -------------------------------
    # Prediction & metrics
    # -------------------------------
    y_pred = tm.predict(X_test)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=np.arange(len(le.classes_)), zero_division=0
    )
    accuracy_val = accuracy_score(y_test, y_pred)
    mean_probs = [disease_probs_all[y_enc == i].mean() for i in range(len(le.classes_))]

    metrics_df = pd.DataFrame({
        'Dataset': disease,
        'Class': le.classes_,
        'precision': precision,
        'recall': recall,
        'f1-score': f1,
        'support': support,
        'Probability': mean_probs,
        'Accuracy': accuracy_val
    })
    all_metrics.append(metrics_df)

    # -------------------------------
    # Confusion Matrix
    # -------------------------------
    def wrap_labels(labels, max_words_per_line=2):
        wrapped = []
        for lbl in labels:
            parts = lbl.replace("_", " ").split()
            lines = [" ".join(parts[i:i+max_words_per_line]) for i in range(0, len(parts), max_words_per_line)]
            wrapped.append("\n".join(lines))
        return wrapped

    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(le.classes_)))
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=wrap_labels(le.classes_),
                yticklabels=wrap_labels(le.classes_))
    plt.title(f"{disease} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{disease}_confusion_matrix.png"))
    plt.close()

    # -------------------------------
    # t-SNE Embedding
    # -------------------------------
    if len(X) > 2:
        tsne = TSNE(n_components=2, random_state=42)
        X_embedded = tsne.fit_transform(X.values)

        # Plot cluster only
        plt.figure(figsize=(7,6))
        for cls in le.classes_:
            idx = np.where(y_enc == le.transform([cls])[0])
            plt.scatter(X_embedded[idx,0], X_embedded[idx,1], label=cls, alpha=0.7)
        plt.legend()
        plt.title(f"{disease} t-SNE Cluster")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{disease}_tsne_cluster.png"))
        plt.close()

        # Plot cluster + probability overlay
        plt.figure(figsize=(7,6))
        for cls in le.classes_:
            idx = np.where(y_enc == le.transform([cls])[0])
            plt.scatter(X_embedded[idx,0], X_embedded[idx,1],
                        c=disease_probs_all[idx], cmap="coolwarm", label=cls, alpha=0.7)
        plt.colorbar(label="Risk Probability")
        plt.legend()
        plt.title(f"{disease} t-SNE Cluster + Risk Probability")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{disease}_tsne_cluster_prob.png"))
        plt.close()

# -------------------------------
# 5. Save metrics
# -------------------------------
final_metrics_df = pd.concat(all_metrics).reset_index(drop=True)
csv_path = os.path.join(output_dir, "all_metrics_classwise.csv")
final_metrics_df.to_csv(csv_path, index=False)
print(f"\nAll class-wise results saved to {csv_path}")
