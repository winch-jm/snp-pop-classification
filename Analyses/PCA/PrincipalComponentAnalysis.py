# PCA - principal component analysis
#
# Workflow
# Goal -- compress 2810 SNPs from 1107 people into top PCs that capture most meaningful variance
# allele dosage matrix --> normalized allele dosage matrix --> eigenvectors --> top PC scores (feature matrix for classifier)
# *train/test splits are done before PCA so test data does not influence the principal axes

import numpy as np
import pandas as pd
import anndata as ad
from sklearn.model_selection import StratifiedKFold


def load_data(allele_path, meta_path, label_col='geo_region_of_origin'):
    """Load the normalized allele dosage matrix and metadata.

    Returns (X_norm, metaIndex, labels).
    """
    X_norm = np.load(allele_path).astype(np.float64)
    metaIndex = pd.read_csv(meta_path)
    labels = metaIndex[label_col].values
    return X_norm, metaIndex, labels


def compute_pca_loadings(X_train, var_threshold=0.80):
    """Compute SNP-space PC loadings from a training matrix.

    Builds the genomic relationship matrix G = X X^T / nSNPs, eigendecomposes it,
    selects the smallest k explaining `var_threshold` of total variance, then
    converts the training-sample eigenvectors back to SNP-space loadings using
    the SVD identity X = U S V^T  ->  V = X^T U / sqrt(nSNPs * lambda).

    Sign ambiguity is resolved by forcing the element of largest absolute value
    in each loading vector to be positive.

    Returns (snp_loadings (nSNPs, k), eigenvalues_k, n_components).
    """
    n_snps = X_train.shape[1]
    G_train = (X_train @ X_train.T) / n_snps

    eigenvalues, eigenvectors = np.linalg.eigh(G_train)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    assert np.isclose(eigenvalues.sum(), np.trace(G_train)), \
        "Eigenvalue sum does not match trace of G_train"
    assert eigenvalues[-1] > -1e-10, \
        f"Large negative eigenvalue detected: {eigenvalues[-1]}"

    explained_ratio = eigenvalues / eigenvalues.sum()
    cum_var = np.cumsum(explained_ratio)
    n_components = int(np.searchsorted(cum_var, var_threshold) + 1)

    Uk = eigenvectors[:, :n_components]
    lambdas_k = eigenvalues[:n_components]
    snp_loadings = X_train.T @ Uk / np.sqrt(n_snps * lambdas_k)

    signs = np.sign(snp_loadings[np.argmax(np.abs(snp_loadings), axis=0),
                                 np.arange(n_components)])
    snp_loadings *= signs

    return snp_loadings, lambdas_k, n_components


def run_fold(X_norm, train_idx, test_idx, var_threshold=0.80):
    """Run PCA on one train/test fold.

    Returns dict with:
      - pcScores: (nIndividuals, n_components + 1) — train+test scores in original
        row order; trailing column is a 0/1 train/test indicator
      - recon_error: MSE between X_test and X_test reconstructed from PC scores
      - test_var_explained: 1 - recon_error / var(X_test)
      - n_components: number of PCs retained for this fold
    """
    n_total = X_norm.shape[0]
    X_train = X_norm[train_idx]
    X_test = X_norm[test_idx]

    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        snp_loadings, _, n_components = compute_pca_loadings(X_train, var_threshold)

        pc_scores_train = X_train @ snp_loadings
        pc_scores_test = X_test @ snp_loadings

        # Place scores back at their original row indices; last column = train(0)/test(1).
        fold_scores = np.full((n_total, n_components), np.nan)
        fold_scores[train_idx] = pc_scores_train
        fold_scores[test_idx] = pc_scores_test
        split_col = np.zeros((n_total, 1))
        split_col[test_idx] = 1
        pcScores = np.hstack([fold_scores, split_col])

        X_reconstructed = pc_scores_test @ snp_loadings.T
        recon_error = float(np.mean((X_test - X_reconstructed) ** 2))
        test_var_explained = float(1 - recon_error / np.var(X_test))

    return {
        'pcScores': pcScores,
        'recon_error': recon_error,
        'test_var_explained': test_var_explained,
        'n_components': n_components,
    }


def run_cross_validation(X_norm, labels, test_size, var_threshold=0.80,
                         random_state=42, verbose=True):
    """Stratified k-fold PCA CV. n_splits = round(1 / test_size).

    Returns dict with per-fold lists (recon_errors, test_var_explained, n_components),
    fold_pcScores (dict fold_idx -> pcScores array), and n_splits.
    """
    n_splits = round(1 / test_size)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    indices = np.arange(len(labels))

    fold_recon_errors = []
    fold_test_var_explained = []
    fold_n_components = []
    fold_pc_scores = {}

    for fold, (train_idx, test_idx) in enumerate(skf.split(indices, labels)):
        result = run_fold(X_norm, train_idx, test_idx, var_threshold)

        fold_pc_scores[fold] = result['pcScores']
        fold_recon_errors.append(result['recon_error'])
        fold_test_var_explained.append(result['test_var_explained'])
        fold_n_components.append(result['n_components'])

        if verbose:
            print(
                f"test_size={test_size:.0%} | fold={fold + 1}/{n_splits} | "
                f"train={len(train_idx)}, test={len(test_idx)} | "
                f"nComponents={result['n_components']} | "
                f"recon_error={result['recon_error']:.4f} | "
                f"test_var_explained={result['test_var_explained']:.4f}"
            )

    return {
        'n_splits': n_splits,
        'fold_pcScores': fold_pc_scores,
        'fold_recon_errors': fold_recon_errors,
        'fold_test_var_explained': fold_test_var_explained,
        'fold_n_components': fold_n_components,
    }


def summarize_cv(cv_result, test_size, n_individuals):
    """Aggregate per-fold lists into a summary row."""
    n_splits = cv_result['n_splits']
    return {
        'n_splits': n_splits,
        'avg_train_n': n_individuals - n_individuals // n_splits,
        'avg_test_n': n_individuals // n_splits,
        'avg_nComponents': float(np.mean(cv_result['fold_n_components'])),
        'avg_reconError': float(np.mean(cv_result['fold_recon_errors'])),
        'std_reconError': float(np.std(cv_result['fold_recon_errors'])),
        'avg_testVarExplained': float(np.mean(cv_result['fold_test_var_explained'])),
        'std_testVarExplained': float(np.std(cv_result['fold_test_var_explained'])),
    }


def build_split_anndata(X_norm, metaIndex, fold_pc_scores):
    """Build an AnnData with normalized SNPs in .X, metadata in .obs, and per-fold PC scores in .obsm."""
    adata = ad.AnnData(
        X=X_norm.astype(np.float32),
        obs=metaIndex.copy().reset_index(drop=True),
        var=pd.DataFrame(index=[f"SNP{i}" for i in range(X_norm.shape[1])]),
    )
    for f, scores in fold_pc_scores.items():
        adata.obsm[f"X_pca_fold_{f + 1}"] = scores.astype(np.float32)
    return adata


def main(
    allele_path='../../data/normAlleleDosagemat.npy',
    meta_path='../../data/metaDataIndex.csv',
    test_sizes=(0.1, 0.2, 0.25, 0.33),
    var_threshold=0.80,
    random_state=42,
    save_outputs=False,
    output_dir='../../data',
):
    X_norm, metaIndex, labels = load_data(allele_path, meta_path)
    n_individuals = len(labels)

    results = {}
    for test_size in test_sizes:
        cv = run_cross_validation(X_norm, labels, test_size,
                                  var_threshold=var_threshold,
                                  random_state=random_state)
        summary = summarize_cv(cv, test_size, n_individuals)
        results[test_size] = summary

        print(
            f"\ntest_size={test_size:.0%} | {summary['n_splits']}-fold CV "
            f"avg recon_error={summary['avg_reconError']:.4f} ± {summary['std_reconError']:.4f} | "
            f"avg test_var_explained={summary['avg_testVarExplained']:.4f} "
            f"± {summary['std_testVarExplained']:.4f}\n"
        )

        adata = build_split_anndata(X_norm, metaIndex, cv['fold_pcScores'])
        if save_outputs:
            adata.write_h5ad(f"{output_dir}/pcScores_split_{int(test_size * 100)}.h5ad")

    if save_outputs:
        (pd.DataFrame(results).T
            .rename_axis('test_size')
            .reset_index()
            .to_csv(f"{output_dir}/pcaCV_results.csv", index=False))

    return results


if __name__ == '__main__':
    main()
