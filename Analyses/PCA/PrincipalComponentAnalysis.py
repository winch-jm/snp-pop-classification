# PCA - principal component analysis

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

##### Workflow
# Goal -- compress 2810 SNPs from 1107 people into top PCs that capture most meaningful variance
# allele dosage matrix --> normalized allele dosage matrix --> eigenvectors --> top PC scores (feature matrix for classifer)
# *train/test splits are done before PCA so test data does not influence the principal axes

### 1) Loading in allele dosage matrix and metaIndex csv file saved from data processing

X_norm = np.load("../../data/normAlleleDosagemat.npy").astype(np.float64)
metaIndex = pd.read_csv("../../data/metaDataIndex.csv")
labels = metaIndex['geo_region_of_origin'].values

### 2) Cross-validation over train/test split ratios

test_sizes = [0.1, 0.2, 0.3, 0.4]
results = {}

for test_size in test_sizes:
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):

        trainIdx, testIdx = train_test_split(np.arange(len(labels)), test_size=test_size, random_state=42, stratify=labels)

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

        # sorting values in descending order so most variant PC comes first
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
        nComponents = np.searchsorted(cumVar, 0.80) + 1

        ### 5) Deriving SNP-space loadings from training eigenvectors

        # eigenvectors of G_train are in sample space (n_train, )
        # to project train and test data, need SNP-space loadings V (nSNPs x nComponents).

        # from the SVD relationship:  X_train = U S V^T
        # then G_train = X_train X_train^T / # SNPs = U (S^2/ # SNPs) U^T
        # so converting to SNP loading space --> V = X_train^T U / sqrt(# SNPs * eigvals)
        # PC scores then = x @ V

        Uk = eigenvectors[:, :nComponents]

        # converting to SNP loading space -- V = X_train^T U / sqrt(# SNPs * lambda)
        snpLoadings = X_train.T @ Uk / np.sqrt(nNormSNP * eigenvalues[:nComponents])

        # fixing any sign ambiguity
        # makes element with the largest absolute value in each loading vector positive
        signs = np.sign(snpLoadings[np.argmax(np.abs(snpLoadings), axis=0), np.arange(nComponents)])
        snpLoadings *= signs

        ### 6) Projecting train and test data onto the training PCs

        pcScores_train = X_train @ snpLoadings
        pcScores_test  = X_test  @ snpLoadings

        ### 7) Reconstruction error on test sets
        # mean squared error between held out test data and reconstructed test data from retained PC scores

        X_reconstructed = pcScores_test @ snpLoadings.T
        reconError = np.mean((X_test - X_reconstructed) ** 2)

        results[test_size] = {"train_n":len(trainIdx), "test_n": len(testIdx),
                              "nComponents": nComponents,"reconError":  reconError}

        print(f"test_size={test_size:.0%} | train={len(trainIdx)}, test={len(testIdx)} | "
              f"nComponents={nComponents} | recon_error={reconError:.4f}")

        ### 8) saving PC scores of split as a labeled CSV
        # rows = samples
        # cols -- PC1...PCM, LABEL, SPLIT (0/1)
        col = f"split_{int(test_size*100)}"
        pcCols = [f"PC{i+1}" for i in range(nComponents)]
        # stacking so train scores on top and test scores on bottom
        df = pd.DataFrame(np.vstack([pcScores_train, pcScores_test]), columns=pcCols)
        df["label"] = np.concatenate([labels_train, labels_test])
        df[col] = np.concatenate([np.ones(len(trainIdx)), np.zeros(len(testIdx))]).astype(int)
        df.to_csv(f"../../data/pcScores_{col}.csv", index=False)

### 9) Save cross-validation results summary
pd.DataFrame(results).T.rename_axis("test_size").reset_index().to_csv("../../data/pcaCV_results.csv", index=False)