# PyTsetlinMachine_RiskPrediction

**PyTsetlinMachine-based pipeline for risk prediction using Euclidean distance probabilities**

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

- Python 3.10+  
- Libraries: NumPy, Pandas, Matplotlib, Seaborn, scikit-learn  
- `pyTsetlinMachine`  

