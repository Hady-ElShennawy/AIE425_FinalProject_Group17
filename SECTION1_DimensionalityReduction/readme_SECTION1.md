# Section 1: Dimensionality Reduction for Collaborative Filtering

This directory contains the implementation of dimensionality reduction techniques (PCA and SVD) applied to collaborative filtering using the MovieLens 20M dataset.

## Dataset

*   **Source**: [MovieLens Latest Datasets](https://grouplens.org/datasets/movielens/)
*   **Download Link**: [ml-20m.zip](https://files.grouplens.org/datasets/movielens/ml-20m.zip)

## Directory Structure

*   **`code/`**: Contains Jupyter notebooks and utility scripts for analysis.
    *   `statistical_analysis.ipynb`: Initial data loading, cleaning, sampling, and basic statistical analysis.
    *   `pca_mean_filling.ipynb`: Implementation of PCA using mean imputation for missing values.
    *   `pca_mle.ipynb`: Implementation of PCA using Maximum Likelihood Estimation (MLE) for covariance estimation.
    *   `svd_analysis.ipynb`: Implementation of Singular Value Decomposition (SVD) for dimensionality reduction.
    *   `utils.py`: Shared utility functions for data loading, result management, and manual implementation of statistical methods.
*   **`data/`**: Stores the raw and processed datasets (e.g., `ratings.csv`, `ratings_cleaned_sampled.csv`).
*   **`results/`**: Generated outputs.
    *   `tables/`: CSV files containing calculated statistics and results.
    *   `plots/`: Generated plots and visualizations.

## Prerequisites

*   Python 3.x
*   Required libraries: `pandas`, `numpy`, `scipy`, `matplotlib`, `sklearn`.

## Usage Instructions

It is recommended to run the notebooks in the following order to ensure data dependencies are met:

1.  **`code/statistical_analysis.ipynb`**: Run this first to clean the raw data and generate the sampled dataset (`ratings_cleaned_sampled.csv`) used by other notebooks.
2.  **`code/pca_mean_filling.ipynb`** OR **`code/pca_mle.ipynb`**: Run these to perform PCA analysis. `pca_mean_filling` uses simple mean imputation, while `pca_mle` uses a more robust covariance estimation method.
3.  **`code/svd_analysis.ipynb`**: Run this to perform SVD-based analysis.

## Key Features

*   **Data Processing**: Robust data loading that handles raw MovieLens data, cleaning (1-5 scale), and deterministic sampling.
*   **Manual Implementations**: Core statistical functions (Mean, Covariance) are implemented manually to demonstrate understanding, alongside optimized library usage.
*   **Dimensionality Reduction**: comparison of PCA (Mean Filling vs MLE) and SVD for item-item similarity and latent feature extraction.
