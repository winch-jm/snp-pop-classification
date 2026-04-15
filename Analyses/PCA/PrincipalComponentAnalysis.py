# PCA - principal component analysis

import pandas as pd
import numpy as np
import anndata as ad
from sklearn.model_selection import StratifiedKFold

##### Workflow
# Goal -- compress 2810 SNPs from 1107 people into top PCs that capture most meaningful variance
# allele dosage matrix --> normalized allele dosage matrix --> eigenvectors --> top PC scores (feature matrix for classifer)
# *train/test splits are done before PCA so test data does not influence the principal axes

### 1) Loading in allele dosage matrix and metaIndex csv file saved from data processing

X_norm = np.load("../../data/normAlleleDosagemat.npy").astype(np.float64)
metaIndex = pd.read_csv("../../data/metaDataIndex.csv")
labels = metaIndex['geo_region_of_origin'].values

### 2) K-fold stratified cross-validation over train/test split ratios
# each test size maps to nsplits = round(1 / test_size)

# test sizes for 10/5/4/3 fold splits
test_sizes = [0.1, 0.2, 0.25, 0.33]
results = {}
indices = np.arange(len(labels))

for test_size in test_sizes:
    nSplits = round(1 / test_size)
    skf = StratifiedKFold(n_splits=nSplits, shuffle=True, random_state=42)

    fold_reconErrors = []
    fold_testVarExplained = []
    fold_nComponents = []
    fold_pcScores = {}

    for fold, (trainIdx, testIdx) in enumerate(skf.split(indices, labels)):
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):

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

            ### 7) Storing full-individual PC scores in original row order for fold
            # placing train and test scores back at their original indices so row i = individual i
            # last column is train/test indicator: 0 = train, 1 = test

            # creating array for fold pc scores for training and testing
            foldScores = np.full((len(indices), nComponents), np.nan)
            foldScores[trainIdx] = pcScores_train
            foldScores[testIdx]  = pcScores_test
            # adding in split col
            splitCol = np.zeros((len(indices), 1))
            splitCol[testIdx] = 1
            fold_pcScores[fold] = np.hstack([foldScores, splitCol])

            ### 8) Reconstruction error on test fold
            # mean squared error between held out test data and reconstructed test data from retained PC scores

            X_reconstructed = pcScores_test @ snpLoadings.T
            reconError = np.mean((X_test - X_reconstructed) ** 2)
            testVarExplained = 1 - reconError / np.var(X_test)

            fold_reconErrors.append(reconError)
            fold_testVarExplained.append(testVarExplained)
            fold_nComponents.append(nComponents)

            print(f"test_size={test_size:.0%} | fold={fold+1}/{nSplits} | "
                  f"train={len(trainIdx)}, test={len(testIdx)} | "
                  f"nComponents={nComponents} | recon_error={reconError:.4f} | test_var_explained={testVarExplained:.4f}")


### 10) summary of reconstruction error
    avg_reconError = np.mean(fold_reconErrors)
    avg_testVarExplained = np.mean(fold_testVarExplained)
    avg_nComponents = np.mean(fold_nComponents)
    results[test_size] = {
        "n_splits": nSplits,
        "avg_train_n": len(indices) - len(indices) // nSplits,
        "avg_test_n":  len(indices) // nSplits,
        "avg_nComponents": avg_nComponents,
        "avg_reconError":  avg_reconError,
        "std_reconError":  np.std(fold_reconErrors),
        "avg_testVarExplained": avg_testVarExplained,
        "std_testVarExplained": np.std(fold_testVarExplained),
    }
    print(f"\ntest_size={test_size:.0%} | {nSplits}-fold CV avg recon_error={avg_reconError:.4f} "
          f"± {np.std(fold_reconErrors):.4f} | avg test_var_explained={avg_testVarExplained:.4f} "
          f"± {np.std(fold_testVarExplained):.4f}\n")

    ### 11) Saving AnnData for split
    # adata.X -- normalized SNP allele dosage matrix (nIndividuals x nSNPs)
    # adata.obs -- sample metadata from metaIndex
    # adata.obsm -- PC scores and split num (0/1) per fold, each (nIndividuals x nComponents_fold)
    # individuals placed back in original row order

    adata = ad.AnnData(
        X   = X_norm.astype(np.float32),
        obs = metaIndex.copy().reset_index(drop=True),
        var = pd.DataFrame(index=[f"SNP{i}" for i in range(X_norm.shape[1])]),
    )
    for f in range(nSplits):
        adata.obsm[f"X_pca_fold_{f+1}"] = fold_pcScores[f].astype(np.float32)

    #adata.write_h5ad(f"../../data/pcScores_split_{int(test_size*100)}.h5ad")

### 9) Save cross-validation results summary
# pd.DataFrame(results).T.rename_axis("test_size").reset_index().to_csv("../../data/pcaCV_results.csv", index=False)