# snp-pop-classification
Population Classification Using SNP Genotyping Data

[Data](https://rosenberglab.stanford.edu/data/huangEtAl2011/)

Pre-Instructions:
    1.) download conda env
    2.) If you want to redownload the data then go to the link https://rosenberglab.stanford.edu/data/huangEtAl2011/, and download the file          named "unphased_HGDP+India+Africa_2810SNPs-regions1to36.stru", 
        a.) otherwise, the data will be in the filepath "data/unphased_HGDP+India+Africa_2810SNPs-regions1to36.stru"

(Run the .py files/notebooks in this the provided, a quick summary of each cell is also provided)


1.)DataProcessing.py
    - FilePath: data/DataProcessing.py
    - Loads in the data and preprocesses it.
        a.) saves the normalized allele dosage matrix to "normAlleleDosagemat.npy"
        b.) saves the per individual metaData dataframe to ""metaDataIndex.csv"

2.) PrincipalComponentAnalysis.py
    - FilePath: Analyses/PCA/PrincipalComponentAnalysis.py
    - Runs PCA analysis using stratified k-fold validation, where the class proportions in each fold are conserved for each split. So PCA is       run
     - The normAlleleDosageMat and metaDataIndex.csv passed in to all of the helper function
    1.)Run the code
    2.) The output(PC scores) for each fold is saved within an object of the anndata data structure which stores the dosage matrix,                  metadata, SNP index (one per row of SNP), and the per-fold PC score matrices

3.) PCA_plots.ipynb : Run all cells
    - FilePath: notebooks/PCA_plots.ipynb 
    Cell 1: all dependencies
    Cell 2: Each split output from anndata data structure is passed in
    Cell 3: Plots the scree plot where the average percentage of variance is explained per PC axis across all folds and splits
    Cell 4-5: Projects the data for PCs across all splits using fold 1 PC scores
    Cell 6: Graphs 3D PCA projection using 3 PCs with fold 1 
    Cell 7: (Most Important) Graphs the PCA projection using the first two PCs from the first fold of the 80/20 split labeling the clusters              with geographical region of origin
    Cell 8: Graphs the PCA projection using the last two PCs from the first fold of the 80/20 split labeling the clusters with                       geographical region of origin
    Cell 9: Plots the overall reconstruction error and the average test variance explained by training PCs across all folds per each split

4.) KNNClassifier.ipynb: Run all cells
    - Filepath: notebooks/KNNClassifier.ipynb
    Cell 1: all dependencies
    Cell 2: Performs KNN classification on each fold stored in the anndata data structure.
    Cell 3: Computes false positive rate for each class separately
    Cell 4: Creates a sorted bar chart showing how many individuals fall into each geographic region within the dataset
    Cell 5: Runs a hyperparameter sweep to evaluate the range of k values across all of the splits and folds for KNN while recording macro               F1, false positive rate, and false negative rate for every combination.
    Cell 6: Plots the F1 score vs the k value for all train/test splits while also showing how much the F1 score varies across each fold                 each k value
    Cell 7: Creates a plot with two panels that shows the specificity and sensitivity vs k for each split   
    Cells 8-10: Run the last three cells after testing all of the helper functions (Cells 2-6)

5.) RandomForest.ipynb: Run all cells
    - Filepath: notebooks/RandomForest.ipynb
    Cell 1: All dependencies
    Cell 2: From Scatch Random Forest Classifier
    Cell 3: Computes FPR and FNR averaged across all populations
    Cell 4: Evaluates Random Forest across all the splits and all n decision trees and stores the F1 scores in a 3D array
    Cell 5: Creates a bar plot showing the mean F1 score for each train/test split
    Cell 6: Trains a Random Forest model using several combinations hyperparameters from a parameter grid across all pre-defined folds 
            using F1, FPR, and FNR and returns a dataframe of combinations sorted by best avg. F1. 
    Cell 7: Converts FNR and FPR to sensitivity and specificity and adds two new columns to the results dataFrame
    Cell 8: Plots a heatmap to show how two hyperparameters interact to affect a performance metric to visualize which combo of
            hyperparameters can give mthe best performance at a glance
    Cell 9: Plots a confusion matrix showing the how well the model performs at predicting each population
    Cells 10-13: Run the last 4 cells after running all the helper functions from cells 1-8
