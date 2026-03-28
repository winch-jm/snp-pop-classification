# Processing raw SNP data into allele dosage matrix for PCA analysis

import numpy as np
import pandas as pd

### 1) loading in dataset

dataset = "unphased_HGDP+India+Africa_2810SNPs-regions1to36.stru"

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

#number monomorphic SNPs in data
nMonomorphic = np.sum(X.var(axis=0) == 0)
print(f"Monomorphic SNPs: {nMonomorphic}")
print(f"Polymorphic SNPs remaining: {X.shape[1] - nMonomorphic}")

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

np.save("normAlleleDosagemat.npy", X_norm)
metaIndex.to_csv("metaDataIndex.csv", index = False)
