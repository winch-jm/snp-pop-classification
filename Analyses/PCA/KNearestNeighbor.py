# k-Nearest Neighbor classification of geographic region from SNP PCs
#
# Implemented from scratch: distance matrix via vectorized numpy, top-k
# selection via np.argpartition, majority vote (uniform + distance-weighted).
#
# Uses precomputed PC scores and train/test split from PCA analysis:
#   ../../data/pcScores_split_20.csv
# Columns: PC1..PC193, label (region), split_20 (1=train, 0=test)

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


### 1) Load PC scores and split into train/test

df = pd.read_csv("../../data/pcScores_split_20.csv")

pcCols = [c for c in df.columns if c.startswith("PC")]
X_train = df.loc[df["split_20"] == 1, pcCols].to_numpy(dtype=np.float64)
X_test  = df.loc[df["split_20"] == 0, pcCols].to_numpy(dtype=np.float64)
y_train = df.loc[df["split_20"] == 1, "label"].to_numpy()
y_test  = df.loc[df["split_20"] == 0, "label"].to_numpy()

print(f"train: {X_train.shape}, test: {X_test.shape}, PCs: {len(pcCols)}")
print("\nClass distribution (test):")
unique, counts = np.unique(y_test, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  {u:20s} {c}")



### 2) Build the k-NN graph from scratch

def pairwise_sq_euclidean(A, B):
    aa = np.sum(A * A, axis=1, keepdims=True)   # (nA, 1)
    bb = np.sum(B * B, axis=1, keepdims=True).T # (1, nB)
    D2 = aa + bb - 2.0 * (A @ B.T)
    # numerical floor: inner products can produce tiny negative values
    np.maximum(D2, 0, out=D2)
    return D2


def knn_graph(X_query, X_ref, k):
    D2 = pairwise_sq_euclidean(X_query, X_ref)          # (n_query, n_ref)
    part = np.argpartition(D2, kth=k, axis=1)[:, :k]    # (n_query, k)

    # Fetch the partitioned distances and sort just those k per row
    part_d2 = np.take_along_axis(D2, part, axis=1)
    order = np.argsort(part_d2, axis=1)
    indices = np.take_along_axis(part, order, axis=1)
    distances = np.sqrt(np.take_along_axis(part_d2, order, axis=1))
    return indices, distances


### 3) Majority vote (uniform and distance-weighted)

def majority_vote(neighbor_labels):
    preds = np.empty(neighbor_labels.shape[0], dtype=neighbor_labels.dtype)
    for i, row in enumerate(neighbor_labels):
        labs, cnts = np.unique(row, return_counts=True)
        preds[i] = labs[np.argmax(cnts)]
    return preds


def distance_weighted_vote(neighbor_labels, neighbor_distances, eps=1e-12):
    """Inverse-distance weighted vote. Each neighbor contributes 1/(d + eps)."""
    preds = np.empty(neighbor_labels.shape[0], dtype=neighbor_labels.dtype)
    weights = 1.0 / (neighbor_distances + eps)
    for i in range(neighbor_labels.shape[0]):
        labs = neighbor_labels[i]
        w = weights[i]
        unique_labs = np.unique(labs)
        scores = np.array([w[labs == u].sum() for u in unique_labs])
        preds[i] = unique_labs[np.argmax(scores)]
    return preds


### 4) Sweep k and evaluate

k_values = [1, 3, 5, 7, 9, 15, 21]
k_max = max(k_values)

# Compute the graph once at k_max, then slice for smaller k
indices_max, distances_max = knn_graph(X_test, X_train, k_max)
neighbor_labels_max = y_train[indices_max]

print("\nk-NN classification results:")
print(f"{'k':>3}  {'acc (uniform)':>14}  {'macroF1 (uniform)':>18}  "
      f"{'acc (dist)':>11}  {'macroF1 (dist)':>15}")

results = []
for k in k_values:
    nl = neighbor_labels_max[:, :k]
    nd = distances_max[:, :k]

    pred_u = majority_vote(nl)
    pred_w = distance_weighted_vote(nl, nd)

    acc_u = accuracy_score(y_test, pred_u)
    f1_u  = f1_score(y_test, pred_u, average="macro")
    acc_w = accuracy_score(y_test, pred_w)
    f1_w  = f1_score(y_test, pred_w, average="macro")

    print(f"{k:>3}  {acc_u:>14.3f}  {f1_u:>18.3f}  {acc_w:>11.3f}  {f1_w:>15.3f}")
    results.append((k, acc_u, f1_u, acc_w, f1_w, pred_u, pred_w))


### 5) Confusion matrix for the best (uniform) k

best = max(results, key=lambda r: r[1])  # highest uniform accuracy
best_k, best_acc, best_f1, _, _, best_pred, _ = best
print(f"\nBest k = {best_k}  |  accuracy = {best_acc:.3f}  |  macro-F1 = {best_f1:.3f}")

classes = np.unique(np.concatenate([y_train, y_test]))
cm = confusion_matrix(y_test, best_pred, labels=classes)

print("\nConfusion matrix (rows = true, cols = predicted):")
header = "              " + "  ".join(f"{c[:6]:>6s}" for c in classes)
print(header)
for i, c in enumerate(classes):
    row = f"  {c[:12]:12s}  " + "  ".join(f"{cm[i, j]:>6d}" for j in range(len(classes)))
    print(row)

# Per-class recall
print("\nPer-class recall:")
for i, c in enumerate(classes):
    n_true = cm[i].sum()
    recall = cm[i, i] / n_true if n_true else 0.0
    print(f"  {c:20s} {recall:.3f}  (n = {n_true})")
