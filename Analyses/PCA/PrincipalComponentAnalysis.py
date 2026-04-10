# PCA - principal component analysis

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

##### Workflow
# Goal -- compress 2810 SNPs from 1107 people into top PCs that capture most meaningful variance
# allele dosage matrix --> normalized allele dosage matrix --> eigenvectors --> top PC scores (feature matrix for classifer)
# *train/test split is performed before PCA so test data does not influence the principal axes

### 1) Loading in allele dosage matrix and metaIndex csv file saved from data processing

X_norm = np.load("../../data/normAlleleDosagemat.npy")
metaIndex = pd.read_csv("../../data/metaDataIndex.csv")

labels = metaIndex['geo_region_of_origin'].values

### 2) Train / test split (stratified by population label, 80/20)

trainIdx, testIdx = train_test_split(np.arange(len(labels)), test_size=0.2, random_state=42,stratify=labels)

X_train = X_norm[trainIdx]
X_test  = X_norm[testIdx]
labels_train = labels[trainIdx]
labels_test  = labels[testIdx]

### 3) Building Genomic Relationship matrix G on training data only

# G = nTrain x nTrain matrix of pairwise genetic similarity among training individuals
nNormSNP = X_train.shape[1]
G_train = (X_train @ X_train.T) / nNormSNP

### 4) Eigendecomposition of G_train

# eigenvectors of G_train give directions along which training individuals vary most
# eigenvalues reflect total variance along each direction
eigenvalues, eigenvectors = np.linalg.eigh(G_train)

# sorting values in descending order so most variance PC comes first
order = np.argsort(eigenvalues)[::-1]
eigenvalues  = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

# sanity check -- eigenvalue sum should equal trace of G_train
assert np.isclose(eigenvalues.sum(), np.trace(G_train)), "Eigenvalue sum does not match trace of G_train"

# sanity check -- no large negative eigenvalues / G_train should be positive semi-definite
assert eigenvalues[-1] > -1e-10, f"Large negative eigenvalue detected: {eigenvalues[-1]}"

# getting explained variance for each eigenvalue
explainedRatio = eigenvalues / eigenvalues.sum()

# finding number of PCs to explain 80% of variance in training data
cumVar = np.cumsum(explainedRatio)
nPCs80 = np.searchsorted(cumVar, 0.80) + 1
print(f"PCs needed to explain 80% variance: {nPCs80}")

### 5) Deriving SNP-space loadings from training eigenvectors

# eigenvectors of G_train are in sample space (n_train, )
# to project train and test data, need SNP-space loadings V (nSNPs x nComponents).

# from the SVD relationship:  X_train = U S V^T
# then G_train = X_train X_train^T / # SNPs = U (S^2/ # SNPs) U^T
# so converting to SNP loading space --> V = X_train^T U / sqrt(# SNPs * eigvals)
# PC scores then = x @ V

nComponents = 208
# top 208 eigenvectors from G train (each col is PC direction in sample space)
Uk = eigenvectors[:, :nComponents]

# converting to SNP loading space -- V = X_train^T U / sqrt(# SNPs * lambda)
snpLoadings = X_train.T @ Uk / np.sqrt(nNormSNP * eigenvalues[:nComponents])

### 6) Projecting train and test data onto the training PCs

pcScores_train = X_train @ snpLoadings
pcScores_test  = X_test  @ snpLoadings

print(f"Train set: {pcScores_train.shape}, Test set: {pcScores_test.shape}")

### 7) Save PC scores, labels, and explained ratio

# npy files for plots
# np.save("pcScores_train.npy", pcScores_train)
# np.save("pcScores_test.npy", pcScores_test)
# np.save("pc_geoRegion_labels_train.npy", labels_train)
# np.save("pc_geoRegion_labels_test.npy", labels_test)
# np.save("PCAexplainedRatio.npy", explainedRatio)

# CSV files for downstream classification
# pcCols = [f"PC{i+1}" for i in range(nComponents)]

# dfTrain = pd.DataFrame(pcScores_train, columns=pcCols)
# dfTrain["label"] = labels_train
# dfTrain.to_csv("../../data/pcScores_train_labeled.csv", index=False)

# dfTest = pd.DataFrame(pcScores_test, columns=pcCols)
# dfTest["label"] = labels_test
# dfTest.to_csv("../../data/pcScores_test_labeled.csv", index=False)