# Section 2: Domain-Specific Recommender System

## Overview
This section focuses on building a domain-specific recommender system for Amazon "Health & Household" products. The system leverages both **Content-Based Filtering** (using item metadata and text descriptions) and **Collaborative Filtering** (using user-item interaction history) to provide personalized recommendations. It also explores **Hybrid** approaches to combine the strengths of both methods and address common challenges like the cold-start problem.

## Directory Structure

```text
SECTION2_DomainRecommender/
├── code/
│   ├── collaborative.ipynb     # Item-Based Collaborative Filtering implementation
│   ├── content_based.ipynb     # Content-Based Filtering implementation
│   ├── data_preprocessing.ipynb # Data cleaning, feature engineering, and label encoding
│   ├── hybrid.ipynb            # Hybrid models (Weighted, Switching, Cascade) and evaluation
│   ├── main.ipynb              # Main pipeline orchestrating the entire workflow
│   └── utils.py                # Shared utility functions (data loading, metrics, cold-start simulation)
├── data/                       # Directory for raw and processed datasets (git-ignored)
├── results/                    # Directory for intermediate and final outputs
│   └── tables/                 # Saved evaluation metrics and user profiles
└── README_SECTION2.md          # This file
```

## Key Components

### 1. Data Preprocessing (`code/data_preprocessing.ipynb`)
*   **Data Cleaning**: Handles missing values (imputation), removes duplicates, and filters out irrelevant categories (e.g., medical items based on keywords).
*   **Feature Engineering**:
    *   Extracts sustainability signals to create an `is_green` feature.
    *   Normalizes text data for downstream processing.
*   **Encoding**: Applies Label Encoding to User and Item IDs for efficient matrix operations.
*   **EDA**: Visualizes rating distributions and long-tail patterns.

### 2. Content-Based Filtering (`code/content_based.ipynb`)
*   **Item Profiles**: Constructs feature vectors using TF-IDF on item descriptions and titles, combined with price and `is_green` flags.
*   **User Profiles**: Builds user profiles by calculating the weighted average of the item vectors they have rated.
*   **Cold-Start**: Handles new users by utilizing a global average item vector.

### 3. Collaborative Filtering (`code/collaborative.ipynb`)
*   **Matrix Construction**: Creates a sparse User-Item interaction matrix.
*   **Similarity Metric**: Uses **Pearson Correlation** on mean-centered ratings to calculate item-item similarity.
*   **Prediction**: Predicts ratings based on the weighted sum of ratings for similar items.

### 4. Hybrid Recommender (`code/hybrid.ipynb`)
*   **Strategies**: Implements multiple hybrid techniques:
    *   **Weighted**: Linearly combines scores from Content-Based and Collaborative models.
    *   **Switching**: select the best model based on confidence or user history length.
    *   **Cascade**: Refines candidates from one model using the other.
*   **Evaluation**: Compares performance (Hit Rate @ 10, 50, 100) against baselines (Popularity, Random) and individual models using Leave-One-Out Cross-Validation.
*   **Cold-Start Simulation**: Specifically evaluates performance for users with limited history.

### 5. Utilities (`code/utils.py`)
*   Contains reusable functions for loading/saving data, calculating similarities, generating recommendations, and computing evaluation metrics.

## Prerequisites

Ensure you have the following Python libraries installed:

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn
```

## Usage Instructions

The notebooks are designed to be run sequentially or via the `main.ipynb` orchestrator.

**Recommended Order:**

1.  **Data Preprocessing**:
    Run `code/data_preprocessing.ipynb` to clean the raw data and generate `Amazon_health&household_preprocessed.csv` and encoded files.

2.  **Model Training & Exploration**:
    *   Run `code/content_based.ipynb` to generate user profiles and content features.
    *   Run `code/collaborative.ipynb` to compute similarity matrices.

3.  **Hybrid Evaluation**:
    Run `code/hybrid.ipynb` to train hybrid models and run the comprehensive evaluation benchmark.

**Orchestrator:**
Alternatively, you can run the entire pipeline by executing:
`code/main.ipynb`

## Key Features
*   **Sustainability Focus**: Incorporates an `is_green` feature to promote eco-friendly product discovery.
*   **Robust Cold-Start Handling**: Explicitly simulates and solves for cold-start scenarios using hybrid switching strategies.
*   **Scalable Design**: Uses sparse matrices and efficient vectorization to handle large datasets.
