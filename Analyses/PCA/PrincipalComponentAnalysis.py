# PCA - principal component analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##### Workflow
# Goal -- compress 2810 SNPs from 1107 people into top PCs that capture most meaningful variance
# raw SNP genotypes --> allele dosage matrix --> normalized allele dosage matrix --> eigenvectors --> top PC scores (feature matrix for classifer)

### 1) loading in dataset

dataset = "../../data/unphased_HGDP+India+Africa_2810SNPs-regions1to36.stru"

df = pd.read_csv(dataset, sep = " ", skiprows = 5, header = None)

# separating meta data and SNP data
df.columns = (["hgdp_id", "population_id", "population_name",
               "country_of_origin", "geo_region_of_origin", "genotyping_id", "sex"] +
              [str(i) for i in range(2810)])
metaCols = ["hgdp_id", "population_id", "population_name",
               "country_of_origin", "geo_region_of_origin", "genotyping_id", "sex"]
snpCols = [str(i) for i in range(2810)]

snpRaw = df[snpCols]
meta = df[metaCols]

### 2) combining alleles for each individual into one row
# in raw data, every 2 rows represents individual's 2 alleles for each SNP
allele1 = snpRaw[0::2]
allele2 = snpRaw[1::2]

# combining every 2 metadata rows corresponding to combined allele rows
metaIndex = meta.iloc[0::2].reset_index(drop = True)

### 3) Converting genotypes into allele dosages (0, 1, 2)

# for each SNP, picking one allele to be alt and counting how many copies of alt each individual has
# missing data '?' becomes NaN and imputed with the column mean

def encodeGenotypes(a1, a2):

    nInd, nSnps = a1.shape
    X = np.full((nInd, nSnps), np.nan)

    # looping through each SNP to get valid genotypes from all individuals (not "?")
    for i in range(nSnps):

        # getting alleles from all individuals for SNP i and removing ?s
        c1, c2 = a1[:, i], a2[:, i]
        valid = (c1 != '?') & (c2 != '?')
        alleles = np.concatenate([c1[valid], c2[valid]])

        # getting all unique alleles for SNP i
        unique = [a for a in np.unique(alleles) if a != '?']

        # skipping monomorphic SNPs (all individuals share same genotype / same alleles) - will be dropped later
        if len(unique) < 2:
            continue

        # choosing arbitrary allele to be alt
        alt = unique[0]
        # looping through each individual and assigning allele dosage numbers for SNP i
        for j in range(nInd):
            if c1[j] != '?' and c2[j] != '?':
                X[j, i] = (c1[j] == alt) + (c2[j] == alt)
    return X

# running allele 1 and allele 2 values into encoding function to get allele dosage matrix
allele1Vals = allele1.values
allele2Vals = allele2.values
X = encodeGenotypes(allele1Vals, allele2Vals)

# replacing any Nans with mean of corresponding SNP
colMeans = np.nanmean(X, axis=0)
nanIdx = np.where(np.isnan(X))
X[nanIdx] = np.take(colMeans, nanIdx[1])

# dropping monomorphic SNPs (cols where variance = 0)
varMask = X.var(axis=0) > 0
X = X[:, varMask]

# sanity check -- no NaNs and all dosage values are between 0 and 2
assert not np.isnan(X).any(), "NaNs remain in dosage matrix"
assert np.all((X >= 0) & (X <= 2)), "Dosage values outside expected range [0, 2]"

### 4) normalizing allele dosage matrix X

# centering each SNP and Patterson scaling (dividing by sqrt(p*(1-p)) so all variants contribute equally to the principal axes

# p = minor allele frequency per SNP
p = X.mean(axis=0) / 2
scale = np.sqrt(p * (1 - p))

# in case of any remaining monomorphic SNPs, setting scale = 1
scale[scale == 0] = 1

# normalizing X (centering then scaling) -> shape (nIndividuals, nSNPs)
X_norm = (X - X.mean(axis=0)) / scale

# sanity check -- normalized matrix should have around ~0 mean per SNP after centering
assert np.abs(X_norm.mean(axis=0)).max() < 1e-10, "Normalization failed: non-zero column means"

### 6) Building Genomic Relationship matrix G

# G = nIndividuals x nIndividual matrix of pairwise genetic similarity

# making genomic relationship / covariance matrix
nNormSNP = X_norm.shape[1]
G = (X_norm @ X_norm.T) / nNormSNP

### 7) Eigendecomposition of G

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

### 8) extracting PC scores for downstream classification

# PC scores: projection of each individual onto the top-k principal axes (eigenvectors) --> feature matrix for the classifier
# shape: (n_individuals, n_components)

# taking top 208 eigenvectors required to explain 80% of variance (as calculated above) and scaling by sqrt(nNormSNP * eigenvalue) to get true PC scores
nComponents = 208
pcScores = eigenvectors[:, :nComponents] * np.sqrt(nNormSNP * eigenvalues[:nComponents])

# labels for classification — geographic region of origin
labels = metaIndex['geo_region_of_origin'].values

### 9)  saving PC scores, labels and explained ratio for plots

# saving as npy files to easily load into notebook for plots
# np.save("pcScores.npy", pcScores)
# np.save("pc_geoRegion_labels.npy", labels)
# np.save("PCAexplainedRatio.npy", explainedRatio)

# saving as labeled PCs as CSV file for classification
pcCols = [f"PC{i+1}" for i in range(nComponents)]
dfOutput = pd.DataFrame(pcScores, columns = pcCols)
dfOutput["label"] = labels
dfOutput.to_csv("../../data/pcScores_labeled.csv", index = False)