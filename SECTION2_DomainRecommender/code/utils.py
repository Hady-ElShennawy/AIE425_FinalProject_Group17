# Group 17: 
# Eyad Medhat 221100279 / Hady Aly 221101190 / Mohamed Mahfouz 221101743 / Omar Mady 221100745

import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from scipy import sparse
import matplotlib.pyplot as plt
import random
import time
from sklearn.linear_model import LinearRegression
from scipy.linalg import eigh
import re
import nltk
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
<<<<<<< HEAD
import json
import csv
import seaborn as sns

=======
from scipy.sparse import csr_matrix, diags
>>>>>>> fd9fdf68e96e99ded029c6322a082ab61628bc20



# --- 1. LOCATE THE RESULTS FOLDER ---
import os


def load_data(table_name=None):
    """
    Loads data:
    1. If table_name is provided, loads that specific table.
    2. Checks for sample_filename ('ratings_cleaned_sampled.csv'). If found, loads it.
    3. If not found, loads raw_filename ('ratings.csv'), cleans it (1-5 scale), and returns full dataset.
       Note: This function IN NO LONGER creates the sampled dataset automatically. 
       Use create_and_save_cleaned_ratings() for that.
    """
    # Locate utils.py directory to find relative paths robustly
    current_dir = os.path.dirname(os.path.abspath(__file__))
    section_root = os.path.dirname(current_dir) # Parent of 'code' folder i.e. SECTION1_DimensionalityReduction

    # 0. Direct Table Loading (if requested)
    if table_name:
        # Define search paths for the requested table
        possible_paths_specific = [
            os.path.join('results', table_name),
            os.path.join('..', 'results', table_name),
            os.path.join('data', table_name),
            os.path.join('..', 'data', table_name),
            # Robust paths relative to utils.py
            os.path.join(section_root, 'results', table_name),
            os.path.join(section_root, 'data', table_name),
            table_name
        ]
        
        for path in possible_paths_specific:
            if os.path.exists(path):
                print(f" Found requested table at: {path}")
                try:
                    return pd.read_csv(path)
                except Exception as e:
                    print(f" Error loading {path}: {e}")
                    return None
        
        print(f" Error: Could not find requested table '{table_name}' in standard locations.")
        return None

    # 1. Search for Sampled File First
    possible_paths_sample = [
        os.path.join('data', table_name),
        os.path.join('..', 'data', table_name),
        os.path.join(section_root, 'data', table_name)
    ]
    for path in possible_paths_sample:
        if os.path.exists(path):
            print(f" Found cached sample at: {path}")
            return pd.read_csv(path)

    # 2. If not found, load Raw Data
    print(" Sample not found. Loading and processing raw data...")
    possible_paths_raw = [
        os.path.join('data', table_name),
        os.path.join('..', 'data', table_name),
        os.path.join(section_root, 'data', table_name)
    ]
    
    df = None
    for path in possible_paths_raw:
        if os.path.exists(path):
            print(f" Found raw dataset at: {path}")
            # Load only necessary columns to save memory if needed, but full load is safer for now
            df = pd.read_csv(path)
            break
            
    if df is None:
        print(f"\n ERROR: Could not find '{table_name}'.")
        return None

    return df


def load_result_csv(filename):
    """Loads a CSV from the results/tables folder if it exists."""
    if not filename.endswith('.csv'):
        filename += '.csv'
        
    file_path = os.path.join(results_root, filename)
    
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            print(f"    Loaded cached results: {filename}")
            return df
        except Exception as e:
            print(f"    Error loading cached CSV {filename}: {e}")
            return None
    return None

# --- 2. DEFINE SAVING FUNCTIONS ---
def get_section_root():
    """
    Returns the absolute path to the section root directory (SECTION2_DomainRecommender).
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)

def ensure_results_folders():
    """
    Ensures that the 'results', 'results/plots', and 'results/tables' folders exist 
    inside the SECTION2_DomainRecommender directory.
    
    Returns:
        str: Path to the results folder.
    """
    section_root = get_section_root()
    results_path = os.path.join(section_root, "results")
    
    # Check/Create results
    if not os.path.exists(results_path):
        os.makedirs(results_path)
        print(f"Created results folder at: {results_path}")
    else:
        print(f"Results folder exists at: {results_path}")
        
    # Check/Create subfolders
    for sub in ["tables"]:
        sub_path = os.path.join(results_path, sub)
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)
            print(f"Created subfolder: {sub_path}")
        else:
             print(f"Subfolder exists: {sub_path}")
             
    return results_path


# Initialize results root globally using the new robust function
results_root = ensure_results_folders()

def get_output_path(section_name, subfolder_type):
    """
    Creates the section subfolder structure (e.g., results/Section_1/plots) if it doesn't exist.
    subfolder_type should be 'plots' or 'tables'.
    """
    # Structure: results_root / section_name / subfolder_type
    target_dir = os.path.join(results_root, section_name, subfolder_type)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"    Created new subfolder: {target_dir}")
    return target_dir

def save_csv(dataframe, filename):
    """Saves a DataFrame to results/tables/filename.csv"""
    try:
        folder_path = os.path.join(results_root, "tables")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Ensure filename ends with .csv
        if not filename.endswith('.csv'):
            filename += '.csv'
        file_path = os.path.join(folder_path, filename)
        
        dataframe.to_csv(file_path, index=False)
        print(f"    Saved CSV: tables/{filename}")
    except Exception as e:
        print(f"    Error saving CSV {filename}: {e}")

def save_plot(figure, filename):
    """Saves a matplotlib figure to results/section_name/plots/filename.png"""
    try:
        folder_path = os.path.join(results_root, "plots")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)
            
        # Ensure filename ends with .png
        if not filename.endswith('.png'):
            filename += '.png'
        file_path = os.path.join(folder_path, filename)
        
        figure.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"    Saved Plot: plots/{filename}")
    except Exception as e:
        print(f"    Error saving Plot {filename}: {e}")



def save_output(data, filename, columns=None, index_label='id'):
    """
    Smart save function that handles various data types and saves them as ready-to-use CSVs.
    
    Args:
        data: The data to save (pd.DataFrame, dict, list, or scalar).
        filename (str): Name of the file (without .csv extension is accurate).
        columns (list, optional): Column names for list data.
        index_label (str, optional): Name for the index column when saving dicts.
    """
    try:
        folder_path = os.path.join(results_root, "tables")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        if not filename.endswith('.csv'):
            filename += '.csv'
            
        file_path = os.path.join(folder_path, filename)
        
        # 1. Handle DataFrame -> Save directly
        if isinstance(data, pd.DataFrame):
            # If data has no index name, we might want to keep index=False unless it's meaningful
            # But specific request said "reusability", so standardizing is good.
            # If user passed a DF, we assume it's already structured.
            data.to_csv(file_path, index=False)
            print(f"    Saved DataFrame: tables/{filename}")
            
        # 2. Handle Dictionary -> Convert to DataFrame (Rows=Keys, Cols=Values)
        elif isinstance(data, dict):
            # Useful for User Profiles: {uid: [vector]} -> DF with index uid
            df = pd.DataFrame.from_dict(data, orient='index')
            df.index.name = index_label
            # If the values are scalars (e.g. {'k_10': 2.5}), columns will be [0]
            # If values are lists (vectors), columns will be 0, 1, 2...
            if columns:
                df.columns = columns
            
            # We save with index=True because the key (e.g. UserID) is crucial
            df.to_csv(file_path, index=True)
            print(f"    Saved Dict as CSV: tables/{filename}")

        # 3. Handle List (or List of Lists/Tuples) -> Convert to DataFrame
        elif isinstance(data, list):
            # If it's a list of primitives [1, 2, 3] -> Single col
            # If it's a list of tuples [(item, score), ...] -> Multiple cols
            
            # Determine columns if not provided
            if columns is None:
                # Try to peek at first element
                if len(data) > 0 and isinstance(data[0], (tuple, list)):
                    # Generic names
                    columns = [f'col_{i}' for i in range(len(data[0]))]
                else:
                    columns = ['value']
            
            df = pd.DataFrame(data, columns=columns)
            df.to_csv(file_path, index=False)
            print(f"    Saved List as CSV: tables/{filename}")
            
        # 4. Handle Scalar (float, int, str) -> Single row DataFrame
        elif isinstance(data, (int, float, str, np.number)):
            # Create a simple DF with one row
            df = pd.DataFrame({'value': [data]})
            df.to_csv(file_path, index=False)
            print(f"    Saved Scalar as CSV: tables/{filename}")
            
        else:
            print(f"    Warning: Unsupported data type {type(data)} for {filename}. Not saved.")
            
    except Exception as e:
        print(f"    Error saving {filename}: {e}")

# =============================================================================
# HYBRID RECOMENDER UTILITIES
# =============================================================================

def build_content_features(df_items):
    """
    Constructs the item feature matrix for Content-Based Filtering.
    Combines TF-IDF on text descriptions, normalized price, and boolean is_green.
    """
    # 1. Text Features (TF-IDF)
    tfidf = TfidfVectorizer(stop_words='english', max_features=100)
    text_matrix = tfidf.fit_transform(df_items['text']).toarray()
    
    # 2. Metadata Features
    scaler = MinMaxScaler()
    price_vec = scaler.fit_transform(df_items[['price']])
    green_vec = df_items[['is_green']].astype(int).values
    
    # 3. Combine
    item_features = np.hstack([text_matrix, price_vec, green_vec])
    return item_features

def build_user_profiles(df_interactions, item_feats):
    """
    Constructs user profiles for Content-Based Filtering using weighted average
    of rated item vectors.
    """
    user_profiles = {}
    cold_start_vector = np.mean(item_feats, axis=0) # Fallback
    
    grouped = df_interactions.groupby('user_id')
    for uid, group in grouped:
        indices = group['item_id_encoded'].values.astype(int)
        ratings = group['rating'].values.reshape(-1, 1)
        
        # Safe Indexing
        valid_mask = indices < item_feats.shape[0]
        indices = indices[valid_mask]
        ratings = ratings[valid_mask]
        
        if len(indices) == 0:
            user_profiles[uid] = cold_start_vector
            continue
            
        vectors = item_feats[indices]
        # Weighted Average Formula
        weighted = np.sum(vectors * ratings, axis=0) / np.sum(ratings)
        user_profiles[uid] = weighted
        
    return user_profiles, cold_start_vector

def build_cf_matrix(df_subset, users, items, user_map, item_map):
    """
    Constructs a Sparse CSR Matrix for Collaborative Filtering.
    """
    row = df_subset['user_id'].map(user_map).values
    col = df_subset['item_id_encoded'].map(item_map).values
    data = df_subset['rating'].values
    
    # Filter out NaNs if any map failed (shouldn't happen if users/items are synced)
    valid = ~np.isnan(row) & ~np.isnan(col)
    R = csr_matrix((data[valid], (row[valid], col[valid])), shape=(len(users), len(items)))
    return R

def get_centered_sim_matrix(R):
    """
    Computes User-Mean Centered Ratings and Item-Item Pearson Correlation Matrix.
    """
    # Mean centering
    row_means = np.array(R.sum(axis=1)).flatten() / (np.diff(R.indptr) + 1e-9)
    R_coo = R.tocoo()
    R_centered = csr_matrix((R_coo.data - row_means[R_coo.row], (R_coo.row, R_coo.col)), shape=R.shape)
    
    # Cosine on centered data = Pearson
    M = R_centered.T
    # Normalize
    norms = np.sqrt(np.array(M.power(2).sum(axis=1)).flatten()) + 1e-9
    M_norm = diags(1/norms) @ M
    
    # Item-Item Similarity
    sim_matrix = M_norm @ M_norm.T
    return sim_matrix, row_means

def predict_cb(uid, i_idx, profiles, item_feats, cold_vec, dynamic_profile=None):
    """
    Content-Based Prediction Score.
    """
    if dynamic_profile is not None:
        profile = dynamic_profile
    else:
        profile = profiles.get(uid, cold_vec)

    item_vec = item_feats[i_idx]
    
    # Cosine Similarity
    score = np.dot(profile, item_vec) / (np.linalg.norm(profile) * np.linalg.norm(item_vec) + 1e-9)
    
    # Scaling to 1-5 Logic (Approximate)
    # Cosine is -1 to 1. User ratings are 1 to 5.
    # Simple mapping: 1 + 4 * similarity (clipped 0-1)
    return 1 + 4 * max(0, score)

def predict_cf(u_idx, i_idx, R, sim_matrix, user_means, dynamic_ratings=None, dynamic_mean=None):
    """
    Item-Based Collaborative Filtering Prediction Score.
    """
    sim_row = sim_matrix.getrow(i_idx)
    
    # Check if we are using dynamic user data (Cold Start Simulation)
    if dynamic_ratings is not None:
        # dynamic_ratings is a dict: {item_idx: rating}
        # dynamic_mean is the mean of this user
        u_mean = dynamic_mean
        rated_indices = list(dynamic_ratings.keys())
        # To fetch ratings, we just lookup in the dict
    else:
        u_mean = user_means[u_idx]
        u_row = R.getrow(u_idx)
        rated_indices = u_row.indices
    
    if len(rated_indices) == 0: return u_mean
    
    # Find neighbors that the user has rated
    neighbors = sim_row.indices
    scores = sim_row.data
    
    relevant_mask = np.isin(neighbors, rated_indices)
    rel_indices = neighbors[relevant_mask]
    rel_scores = scores[relevant_mask]
    
    if len(rel_indices) == 0: return u_mean
    
    # Retrieve user's ratings for these neighbors
    if dynamic_ratings is not None:
        current_ratings = np.array([dynamic_ratings[idx] for idx in rel_indices])
    else:
        # Optimizing sparse access
        curr_ratings_dict = {k: v for k, v in zip(u_row.indices, u_row.data)}
        current_ratings = np.array([curr_ratings_dict[idx] for idx in rel_indices])
    
    # Weighted Average of Deviations
    num = np.sum(rel_scores * (current_ratings - u_mean))
    den = np.sum(np.abs(rel_scores))
    
    if den == 0: return u_mean
    
    pred = u_mean + num/den
    return max(1, min(5, pred))

def hybrid_weighted(cb_score, cf_score, alpha):
    """Weighted Hybrid Strategy"""
    return alpha * cb_score + (1 - alpha) * cf_score

def hybrid_switching(user_rating_count, cb_score, cf_score, threshold=10):
    """Switching Hybrid Strategy"""
    if user_rating_count >= threshold:
        return cf_score
    return cb_score

def hybrid_cascade(cb_score, cf_score, cb_threshold=0.5):
    """
    Cascade Hybrid Strategy (Pointwise Simulation).
    Stage 1: Filter by Content-Based Score.
    Stage 2: If passed, Rank by Collaborative Filtering Score.
    """
    if cb_score < cb_threshold:
        return 0.0
    return cf_score

# =============================================================================
# EVALUATION & COLD START UTILITIES
# =============================================================================

def simulate_cold_start(df_interactions, min_ratings=20, n_users=20, n_ratings_list=[3, 5, 10]):
    """
    Selects users with > min_ratings. Creates masked profiles for them.
    Returns:
        scenarios: { user_id: { n: masked_df } }
        ground_truth: { user_id: set_of_all_actual_liked_items }
        sampled_users: list of user_ids
    """
    print("\n--- Simulating Cold-Start Scenarios ---")
    
    # 1. Select eligible users
    user_counts = df_interactions['user_id'].value_counts()
    eligible_users = user_counts[user_counts >= min_ratings].index.tolist()
    
    if len(eligible_users) > n_users:
        sampled_users = random.sample(eligible_users, n_users)
    else:
        sampled_users = eligible_users
        
    scenarios = {}
    ground_truth = {}
    
    for uid in sampled_users:
        user_data = df_interactions[df_interactions['user_id'] == uid]
        # Ground Truth: Items rated positively (rating >= 3.0)
        actual_items = set(user_data[user_data['rating'] >= 3.0]['item_id_encoded'].values)
        ground_truth[uid] = actual_items
        
        user_scenarios = {}
        for n in n_ratings_list:
            # Create a masked dataframe mimicking a user with only N ratings
            if len(user_data) >= n:
                masked = user_data.sample(n=n, random_state=42)
                user_scenarios[n] = masked
            else:
                user_scenarios[n] = user_data.copy()
        scenarios[uid] = user_scenarios
        
    print(f"Selected {len(sampled_users)} users for simulation.")
    return scenarios, ground_truth, sampled_users

def recommend_random(all_item_ids, k=10):
    """Returns k random item IDs."""
    return random.sample(list(all_item_ids), k)

def recommend_popularity(df_interactions, top_k=10):
    """Returns list of top K popular item_ids."""
    return df_interactions['item_id_encoded'].value_counts().head(top_k).index.tolist()

def evaluate_baselines_comparison(df_interactions, 
                                  profiles_train, item_features, cold_vec_train, # CB Model
                                  R_train, sim_train, means_train, # CF Model
                                  user_map, item_map,
                                  n_test_users=20):
    """
    Compares Best Hybrid (Weighted) vs Random vs Popularity vs Pure CB
    using Leave-One-Out protocols on sampled users.
    """
    print("\n--- Running Baseline Comparison (Leave-One-Out) ---")
    
    # 1. Select Users with > 10 ratings 
    user_counts = df_interactions['user_id'].value_counts()
    eligible = user_counts[user_counts >= 20].index.tolist()
    if len(eligible) > n_test_users:
        test_users = random.sample(eligible, n_test_users)
    else:
        test_users = eligible
        
    all_items = set(df_interactions['item_id_encoded'].unique())
    
    # Results accumulators - Expanded for Analysis
    metrics = {
        'Random': {'HR@10': 0, 'HR@50': 0, 'HR@100': 0},
        'Popularity': {'HR@10': 0, 'HR@50': 0, 'HR@100': 0},
        'Pure CB': {'HR@10': 0, 'HR@50': 0, 'HR@100': 0},
        'Weighted Hybrid': {'HR@10': 0, 'HR@50': 0, 'HR@100': 0}
    }
    
    # Pre-compute Popularity (Global)
    pop_items_global = recommend_popularity(df_interactions, top_k=200)
    
    for uid in test_users:
        # User Data & Hidden Item
        u_data = df_interactions[df_interactions['user_id'] == uid]
        positive = u_data[u_data['rating'] >= 3.0]
        if positive.empty: continue
        hidden_item = positive.sample(1, random_state=42)['item_id_encoded'].values[0]
        
        # Candidate Set (Hidden + 500 Pop + 100 Random)
        rated_items_set = set(u_data['item_id_encoded'].values)
        rated_items_set.discard(hidden_item)
        
        candidates = list(set(list(set(pop_items_global[:500])) + [hidden_item] + list(set(recommend_random(all_items, 100)))))
        
        # Double check no duplicates
        if len(candidates) != len(set(candidates)):
             candidates = list(set(candidates))
        
        # --- SCORES ---
        # 1. Random Scores (Simulated by shuffling)
        rnd_cands = list(candidates)
        random.shuffle(rnd_cands)
        
        # 2. Popularity Scores (Rank by freq)
        pop_counts = df_interactions['item_id_encoded'].value_counts()
        pop_scores = {cand: pop_counts.get(cand, 0) for cand in candidates}
        
        # 3. CB & Hybrid Scores
        cb_scores = {}
        cf_scores = {}
        for cand in candidates:
            c_idx = item_map.get(cand)
            if c_idx is None: 
                cb_scores[cand] = -1
                cf_scores[cand] = 3.0
                continue
                
            cb_scores[cand] = predict_cb(uid, c_idx, profiles_train, item_features, cold_vec_train)
            
            u_idx = user_map.get(uid)
            if u_idx is not None:
                cf_scores[cand] = predict_cf(u_idx, c_idx, R_train, sim_train, means_train)
            else:
                cf_scores[cand] = 3.0
                
        hyb_scores = {cand: hybrid_weighted(cb_scores[cand], cf_scores[cand], alpha=0.7) for cand in candidates}
        
        # --- RANKS ---
        rank_rnd = rnd_cands.index(hidden_item) + 1
        rank_pop = sorted(pop_scores, key=pop_scores.get, reverse=True).index(hidden_item) + 1
        rank_cb = sorted(cb_scores, key=cb_scores.get, reverse=True).index(hidden_item) + 1
        rank_hyb = sorted(hyb_scores, key=hyb_scores.get, reverse=True).index(hidden_item) + 1
        
        # Update Metrics
        for method, rank in [('Random', rank_rnd), ('Popularity', rank_pop), ('Pure CB', rank_cb), ('Weighted Hybrid', rank_hyb)]:
            if rank <= 10: metrics[method]['HR@10'] += 1
            if rank <= 50: metrics[method]['HR@50'] += 1
            if rank <= 100: metrics[method]['HR@100'] += 1
            
        print(f"UID: {uid} | Hidden: {hidden_item}")
        print(f"  -> Ranks: Random={rank_rnd}, Pop={rank_pop}, CB={rank_cb}, Hybrid={rank_hyb}")

    # Compile
    final_data = []
    n = len(test_users)
    for method, res in metrics.items():
        final_data.append({
            'Method': method,
            'Hit Rate @ 10': res['HR@10'] / n,
            'Hit Rate @ 50': res['HR@50'] / n,
            'Hit Rate @ 100': res['HR@100'] / n
        })
    
    df_final = pd.DataFrame(final_data).sort_values(by='Hit Rate @ 10', ascending=False)
    
    try:
        save_csv(df_final, "baseline_comparison.csv")
    except NameError:
        print("Warning: save_csv function not found. Printing only.")
        
    return df_final
