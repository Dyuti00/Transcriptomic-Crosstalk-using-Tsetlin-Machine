# Bioinfo and PyTsetlinMachine_RiskPrediction

**Bioinformatics R pipeline with PyTsetlinMachine-based pipeline for risk prediction using Euclidean distance probabilities**

---

## Overview

This project implements a **risk prediction pipeline** for patient datasets using **Tsetlin Machines**. Risk probabilities are computed using **Euclidean distance** from class centroids (prototype-based approach).  

The project processes **gene expression datasets** for multiple diseases:  
- **T2D (Type 2 Diabetes)**  
- **Liver Cancer**  
- **Ovarian Cancer**  
- **Pancreatic Cancer**  

It outputs **per-patient risk probabilities**, **class-wise metrics**, and **t-SNE visualizations** for easy inspection.

---
## Integrated Bioinformatics + Tsetlin Machine Pipeline

This repository follows a **two-stage framework** combining bioinformatics analysis with interpretable machine learning:

### Stage 1: Bioinformatics (Transcriptomic Crosstalk)

We first perform differential gene expression analysis on:

* **T2D dataset (GSE164416)**
* **Pancreatic Cancer dataset (GSE79668)**

Using **DESeq2**, we identify:

* Disease-specific DEGs
* **Shared DEGs (co-DEGs)** between T2D and pancreatic cancer

Filtering criteria:

* FDR (adjusted p-value) < 0.05
* |log2 Fold Change| > 1

These shared genes represent **transcriptomic crosstalk** and form the biological basis for downstream modeling.

---

### Stage 2: Tsetlin Machine Risk Prediction

The processed gene expression data is then used as input to the Tsetlin Machine pipeline.

* Learns interpretable patterns from gene expression data
* Predicts disease risk probabilities for individual samples
* Uses **Euclidean distance from class centroids** to compute risk scores

---

### Key Idea

Instead of applying machine learning directly on raw data:

> We first extract biologically meaningful features (DEGs),
> then apply interpretable AI for prediction and validation.

This ensures:

* Biological relevance
* Model interpretability
* Cross-disease insight (T2D → Pancreatic Cancer link)

---

### Workflow Summary

Bioinformatics → Feature Selection → Tsetlin Machine → Risk Prediction

1. Raw RNA-seq data
2. DESeq2 analysis (T2D & Cancer separately)
3. Identify shared DEGs
4. Use filtered gene set for ML
5. Train Tsetlin Machine
6. Compute risk probabilities

---

### Repository Structure

```
📂 bioinformatics/
    bioinformatics_pipeline.R   # DGE + overlap analysis

📂 tsetlin_model/
    main.py                     # Risk prediction pipeline
```

---

### Important Note

Currently, the integrated pipeline focuses on:

* **Type 2 Diabetes (T2D)**
* **Pancreatic Cancer**

Other datasets (liver, ovarian) in the repository are part of the general TM framework but are not included in the bioinformatics crosstalk analysis.

## Features

- **Tsetlin Machine** classifier for multiclass patient risk prediction.  
- **Euclidean distance-based risk probability** calculation.  
- Handles multiple disease datasets in Excel format.  
- Outputs:
  - Per-patient CSVs with true/predicted class and risk probability  
  - Class-wise metrics CSV (precision, recall, F1-score, support, mean probability, accuracy)  
  - t-SNE plots colored by risk probability  

---

# Bioinfo and PyTsetlinMachine_RiskPrediction

**Bioinformatics R pipeline with PyTsetlinMachine-based pipeline for risk prediction using Euclidean distance probabilities**

---

## Overview

This project implements a **risk prediction pipeline** for patient datasets using **Tsetlin Machines**. Risk probabilities are computed using **Euclidean distance** from class centroids (prototype-based approach).  

The project processes **gene expression datasets** for multiple diseases:  
- **T2D (Type 2 Diabetes)**  
- **Liver Cancer**  
- **Ovarian Cancer**  
- **Pancreatic Cancer**  

It outputs **per-patient risk probabilities**, **class-wise metrics**, and **t-SNE visualizations** for easy inspection.

---
## Integrated Bioinformatics + Tsetlin Machine Pipeline

This repository follows a **two-stage framework** combining bioinformatics analysis with interpretable machine learning:

### Stage 1: Bioinformatics (Transcriptomic Crosstalk)

We first perform differential gene expression analysis on:

* **T2D dataset (GSE164416)**
* **Pancreatic Cancer dataset (GSE79668)**

Using **DESeq2**, we identify:

* Disease-specific DEGs
* **Shared DEGs (co-DEGs)** between T2D and pancreatic cancer

Filtering criteria:

* FDR (adjusted p-value) < 0.05
* |log2 Fold Change| > 1

These shared genes represent **transcriptomic crosstalk** and form the biological basis for downstream modeling.

---

### Stage 2: Tsetlin Machine Risk Prediction

The processed gene expression data is then used as input to the Tsetlin Machine pipeline.

* Learns interpretable patterns from gene expression data
* Predicts disease risk probabilities for individual samples
* Uses **Euclidean distance from class centroids** to compute risk scores

---

### Key Idea

Instead of applying machine learning directly on raw data:

> We first extract biologically meaningful features (DEGs),
> then apply interpretable AI for prediction and validation.

This ensures:

* Biological relevance
* Model interpretability
* Cross-disease insight (T2D → Pancreatic Cancer link)

---

### Workflow Summary

Bioinformatics → Feature Selection → Tsetlin Machine → Risk Prediction

1. Raw RNA-seq data
2. DESeq2 analysis (T2D & Cancer separately)
3. Identify shared DEGs
4. Use filtered gene set for ML
5. Train Tsetlin Machine
6. Compute risk probabilities

---
## How to Run the Pipeline

### Step 1: Run Bioinformatics Analysis (R)

```r
source("bioinformatics/bioinformatics_pipeline.R")
```

This will generate:

* `Cancer_DEGs.csv`
* `T2D_DEGs.csv`
* `Common_DEGs.csv`

---

### Step 2: Run Tsetlin Machine Model (Python)

```bash
python tsetlin_model/main.py
```

Input:

* Processed gene expression data (optionally filtered using shared DEGs)

Output:

* Risk probabilities
* Classification metrics
* t-SNE visualization plots
The input feature space can be optionally restricted to shared DEGs to ensure biologically informed learning.

### Repository Structure

```
📂 bioinformatics/
    bioinformatics_pipeline.R   # DGE + overlap analysis

📂 tsetlin_model/
    main.py                     # Risk prediction pipeline
```

---

### Important Note

Currently, the integrated pipeline focuses on:

* **Type 2 Diabetes (T2D)**
* **Pancreatic Cancer**

Other datasets (liver, ovarian) in the repository are part of the general TM framework but are not included in the bioinformatics crosstalk analysis.

## Features

- **Tsetlin Machine** classifier for multiclass patient risk prediction.  
- **Euclidean distance-based risk probability** calculation.  
- Handles multiple disease datasets in Excel format.  
- Outputs:
  - Per-patient CSVs with true/predicted class and risk probability  
  - Class-wise metrics CSV (precision, recall, F1-score, support, mean probability, accuracy)  
  - t-SNE plots colored by risk probability  

---

## Tech Stack

### Bioinformatics

* R (≥ 4.2)
* DESeq2
* dplyr, tidyverse
* clusterProfiler (optional for enrichment)

### Machine Learning

* Python 3.10+
* NumPy, Pandas
* scikit-learn
* Matplotlib, Seaborn
* pyTsetlinMachine

### Conceptual Framework

* Transcriptomic Crosstalk Analysis
* Interpretable AI (Rule-based learning via Tsetlin Machine)




