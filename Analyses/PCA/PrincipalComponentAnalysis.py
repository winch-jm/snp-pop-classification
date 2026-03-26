# PCA - principal component analysis

import pandas as pd
import numpy as np

##### Workflow
# Goal -- compress 2810 SNPs from 1107 people into top PCs that capture most meaningful variance
# allele dosage matrix --> normalized allele dosage matrix --> eigenvectors --> top PC scores (feature matrix for classifer)

### 1) Loading in allele dosage matrix and metaIndex csv file saved from data processing

X_norm = np.load("../../data/normAlleleDosagemat.npy")
metaIndex = pd.read_csv("../../data/metaDataIndex.csv")

### 2) Building Genomic Relationship matrix G (covariance matrix)

# G = nIndividuals x nIndividual matrix of pairwise genetic similarity

# making genomic relationship / covariance matrix
nNormSNP = X_norm.shape[1]
G = (X_norm @ X_norm.T) / nNormSNP

### 3) Eigendecomposition of G

# eigenvectors of G give scores (direction along which people vary)
# eigenvalues of G = total variation along vector

# getting eigenvalues
eigenvalues, eigenvectors = np.linalg.eigh(G)

# sorting values in descending order to most variance first
order = np.argsort(eigenvalues)[::-1]
eigenvalues  = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

# sanity check -- eigenvalue sum should equal trace of G
assert np.isclose(eigenvalues.sum(), np.trace(G)), "Eigenvalue sum does not match trace of G"

# sanity check -- no large negative eigenvalues / G should be positive semi-definite
assert eigenvalues[-1] > -1e-10, f"Large negative eigenvalue detected: {eigenvalues[-1]}"

# getting explained variance for each eigenvalue
explainedRatio = eigenvalues / eigenvalues.sum()

# finding number of PCs to explain 80% of variance in data
cumVar = np.cumsum(explainedRatio)
nPCs80 = np.searchsorted(cumVar, 0.80) + 1
print(f"PCs needed to explain 80% variance: {nPCs80}")

### 4) extracting PC scores for downstream classification

# PC scores: projection of each individual onto the top-k principal axes (eigenvectors) --> feature matrix for the classifier
# shape: (n_individuals, n_components)

# taking top 208 eigenvectors required to explain 80% of variance (as calculated above) and scaling by sqrt(nNormSNP * eigenvalue) to get true PC scores
nComponents = 208
pcScores = eigenvectors[:, :nComponents] * np.sqrt(nNormSNP * eigenvalues[:nComponents])

# labels for classification — geographic region of origin
labels = metaIndex['geo_region_of_origin'].values

### 5)  saving PC scores, labels and explained ratio for plots

# saving as npy files to easily load into notebook for plots
# np.save("pcScores.npy", pcScores)
# np.save("pc_geoRegion_labels.npy", labels)
# np.save("PCAexplainedRatio.npy", explainedRatio)

# saving as labeled PCs as CSV file for classification
# pcCols = [f"PC{i+1}" for i in range(nComponents)]
# dfOutput = pd.DataFrame(pcScores, columns = pcCols)
# dfOutput["label"] = labels
# dfOutput.to_csv("../../data/pcScores_labeled.csv", index = False)