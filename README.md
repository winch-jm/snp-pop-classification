# snp-pop-classification

**Population Classification Using SNP Genotyping Data**

## Data

* [HGDP Dataset (Huang et al. 2011)](https://rosenberglab.stanford.edu/data/huangEtAl2011/)

---

## Setup Instructions
1. Install the conda environment `snp-pop`.
```
cd /path/to/snp-pop-classification
conda env create -f environment.yml
```

2. Data setup:

   * To download manually:

     * Visit the dataset link above.
     * Download:
       `unphased_HGDP+India+Africa_2810SNPs-regions1to36.stru`
   * Otherwise:

     * The file is expected at:
       `data/unphased_HGDP+India+Africa_2810SNPs-regions1to36.stru`

---

## Overview

Run the Python scripts and notebooks in the order below using `snp-pop` env. Each notebook includes a brief description of its cells.

---

## 1. Data Processing 

**File:** `data/DataProcessing.py`

* Loads and preprocesses SNP data.
* Outputs:

  * Normalized allele dosage matrix → `normAlleleDosagemat.npy`
  * Metadata per individual → `metaDataIndex.csv`

---

## 2. Principal Component Analysis (PCA)

**File:** `Analyses/PCA/PrincipalComponentAnalysis.py`

* Performs PCA using **stratified k-fold cross-validation** (preserves class proportions).
* Inputs:

  * `normAlleleDosagemat.npy`
  * `metaDataIndex.csv`
* Outputs:

  * Stores PCA results in an **AnnData object**, containing:

    * Dosage matrix
    * Metadata
    * SNP index
    * Per-fold PCA score matrices

---

## 3. PCA Visualization

**Notebook:** `notebooks/PCA_plots.ipynb`

Run all cells:

* **Cell 1:** Import dependencies
* **Cell 2:** Load AnnData outputs from PCA
* **Cell 3:** Scree plot (average variance explained per PC across folds/splits)
* **Cells 4–5:** Project data using fold 1 PC scores across splits
* **Cell 6:** 3D PCA projection (first 3 PCs, fold 1)
* **Cell 7 (key):** 2D PCA (first two PCs, 80/20 split, fold 1) with geographic labels
* **Cell 8:** PCA using last two PCs (same split/fold)
* **Cell 9:** Reconstruction error and test variance explained across folds/splits

---

## 4. K-Nearest Neighbors (KNN) Classification

**Notebook:** `notebooks/KNNClassifier.ipynb`

Run all cells:

* **Cell 1:** Import dependencies
* **Cell 2:** Perform KNN classification across folds
* **Cell 3:** Compute per-class false positive rate (FPR)
* **Cell 4:** Bar chart of individuals per geographic region
* **Cell 5:** Hyperparameter sweep over k:

  * Metrics recorded:

    * Macro F1
    * FPR
    * FNR
* **Cell 6:** F1 vs k with variability across folds
* **Cell 7:** Specificity and sensitivity vs k (two-panel plot)
* **Cells 8–10:** Run after validating helper functions (Cells 2–6)

---

## 5. Random Forest Classification

**Notebook:** `notebooks/RandomForest.ipynb`

Run all cells:

* **Cell 1:** Import dependencies
* **Cell 2:** From-scratch Random Forest implementation
* **Cell 3:** Compute average FPR and FNR across populations
* **Cell 4:** Evaluate across splits and number of trees (stores F1 in 3D array)
* **Cell 5:** Bar plot of mean F1 per split
* **Cell 6:** Grid search over hyperparameters:

  * Metrics:

    * F1
    * FPR
    * FNR
  * Outputs sorted results dataframe (by average F1)
* **Cell 7:** Convert:

  * FNR → Sensitivity
  * FPR → Specificity
* **Cell 8:** Heatmap of hyperparameter interactions
* **Cell 9:** Confusion matrix visualization
* **Cells 10–13:** Run after completing Cells 1–8

---

## Notes

* AnnData objects are used to store intermediate representations and results.
* Stratified cross-validation ensures balanced representation of populations across folds.
* Evaluation emphasizes:

  * Macro F1 score
  * Sensitivity / specificity
  * Per-class performance
