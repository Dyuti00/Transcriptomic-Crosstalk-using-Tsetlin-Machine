import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
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
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# -------------------------------
# 2. Label normalization rules
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
# 3. Process each dataset separately
# -------------------------------
all_metrics = []

for disease, sheet_name in datasets.items():
    print(f"\n=== Processing {disease} ({sheet_name}) ===")
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df = df.dropna(how="all").dropna(axis=1, how="all")

    # -------------------------------
    # Labels and Features
    # -------------------------------
    if disease == "Pancreatic":
        # Pancreatic dataset has DIABETIC + Pancreatic Cancer
        diabetic_row = df[df.iloc[:,0].astype(str).str.upper().str.contains("DIABETIC")]
        pancreatic_row = df[df.iloc[:,0].astype(str).str.upper().str.contains("PANCREATIC CANCER")]

        if diabetic_row.empty or pancreatic_row.empty:
            raise ValueError("DIABETIC or Pancreatic Cancer row not found in sheet")

        label_df = pd.DataFrame({
            'DIABETIC': diabetic_row.iloc[0, 1:].values,
            'Pancreatic Cancer': pancreatic_row.iloc[0, 1:].values
        })
        y = normalize_labels(disease, label_df)

        # Features: all other rows
        X_df = df[~df.iloc[:,0].astype(str).str.upper().str.contains("DIABETIC|PANCREATIC CANCER")]
        X = X_df.iloc[:, 1:].T.reset_index(drop=True)
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0).replace([np.inf, -np.inf], 0).astype(np.int32)

    else:
        # Other datasets have LABEL row
        label_row = df[df.iloc[:,0].astype(str).str.upper().str.contains("LABEL")]
        if label_row.empty:
            raise ValueError(f"LABEL row not found in sheet {sheet_name}")
        y = normalize_labels(disease, label_row.iloc[0, 1:])

        X_df = df[~df.iloc[:,0].astype(str).str.upper().str.contains("LABEL")]
        X = X_df.iloc[:, 1:].T
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0).replace([np.inf, -np.inf], 0).astype(np.int32)

    # -------------------------------
    # Drop samples without valid label
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

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # -------------------------------
    # Train Tsetlin Machine
    # -------------------------------
    tm = MultiClassTsetlinMachine(number_of_clauses=100, T=15, s=3.9)
    print(f"Training TM for {disease}...")
    tm.fit(X_train, y_train, epochs=50)

    # Predict
    y_pred = tm.predict(X_test)

    # -------------------------------
    # Metrics
    # -------------------------------
    # Predicted probabilities
    try:
        y_proba = tm.predict_proba(X_test)  # shape: (n_samples, n_classes)
    except AttributeError:
        # If your TM does not have predict_proba, use 1.0 for predicted class
        y_proba = np.zeros((len(y_test), len(le.classes_)))
        y_proba[np.arange(len(y_test)), y_pred] = 1.0

    # Get probability of the predicted class for each sample
    pred_probs = y_proba[np.arange(len(y_pred)), y_pred]

    # Build DataFrame for per-class metrics
    report = classification_report(
        y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0
    )

    # Convert to DataFrame and keep only class rows
    metrics_df = pd.DataFrame(report).transpose()
    metrics_df = metrics_df.loc[le.classes_, ['precision', 'recall', 'f1-score', 'support']]
    metrics_df['Dataset'] = disease

    # Add predicted probability column (mean probability per class)
    mean_probs = []
    for cls_idx, cls_name in enumerate(le.classes_):
        cls_mask = y_test == cls_idx
        if np.sum(cls_mask) > 0:
            mean_probs.append(pred_probs[cls_mask].mean())
        else:
            mean_probs.append(np.nan)
    metrics_df['Probability'] = mean_probs

    # Add Accuracy column (same for all rows)
    accuracy_val = (y_test == y_pred).mean()
    metrics_df['Accuracy'] = accuracy_val

    all_metrics.append(metrics_df)

    # -------------------------------
    # Confusion Matrix
    # -------------------------------
    cm = confusion_matrix(y_test, y_pred, labels=range(len(le.classes_)))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
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
        plt.figure(figsize=(7, 6))
        for cls in le.classes_:
            idx = np.where(y_enc == le.transform([cls])[0])
            plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=cls, alpha=0.7)
        plt.legend()
        plt.title(f"{disease} t-SNE Embedding")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{disease}_tsne.png"))
        plt.close()


# -------------------------------
# 4. Save metrics
# -------------------------------
# Concatenate all dataset metrics
final_metrics_df = pd.concat(all_metrics).reset_index().rename(columns={'index': 'Class'})

# Reorder columns for clarity, keeping Accuracy column
final_metrics_df = final_metrics_df[['Dataset', 'Class', 'precision', 'recall', 'f1-score', 'support', 'Probability', 'Accuracy']]

# Save to CSV
csv_path = os.path.join(output_dir, "all_metrics.csv")
final_metrics_df.to_csv(csv_path, index=False)
print(f"\nAll results saved to {csv_path}")

