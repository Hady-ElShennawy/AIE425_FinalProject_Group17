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
from sklearn.metrics.pairwise import cosine_similarity
import os




def load_data(raw_filename='ratings.csv', sample_filename='ratings_cleaned_sampled.csv', table_name=None):
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
            os.path.join('results', 'tables', table_name),
            os.path.join('..', 'results', 'tables', table_name),
            os.path.join('data', 'ml-20m', table_name),
            os.path.join('..', 'data', 'ml-20m', table_name),
            # Robust paths relative to utils.py
            os.path.join(section_root, 'results', 'tables', table_name),
            os.path.join(section_root, 'data', 'ml-20m', table_name),
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
        os.path.join('data', 'ml-20m', sample_filename),
        os.path.join('..', 'data', 'ml-20m', sample_filename),
        os.path.join(section_root, 'data', 'ml-20m', sample_filename)
    ]
    for path in possible_paths_sample:
        if os.path.exists(path):
            print(f" Found cached sample at: {path}")
            return pd.read_csv(path)

    # 2. If not found, load Raw Data
    print(" Sample not found. Loading and processing raw data...")
    possible_paths_raw = [
        os.path.join('data', 'ml-20m', raw_filename),
        os.path.join('..', 'data', 'ml-20m', raw_filename),
        os.path.join(section_root, 'data', 'ml-20m', raw_filename)
    ]
    
    df = None
    for path in possible_paths_raw:
        if os.path.exists(path):
            print(f" Found raw dataset at: {path}")
            # Load only necessary columns to save memory if needed, but full load is safer for now
            df = pd.read_csv(path)
            break
            
    if df is None:
        print(f"\n ERROR: Could not find '{raw_filename}'.")
        return None

    # 3. Clean Data
    if 'timestamp' in df.columns:
        df.drop(columns=['timestamp'], inplace=True)
    df['rating'] = df['rating'].clip(lower=1, upper=5).round().astype(int)
    
    print(" Data cleaned (1-5 scale, timestamp removed). Returning full dataset.")
    return df

def clean_data(df):
    """Standardizes ratings (1-5) and removes timestamp"""
    if 'timestamp' in df.columns:
        df.drop(columns=['timestamp'], inplace=True)
    
    # Clip and round ratings
    df['rating'] = df['rating'].clip(lower=1, upper=5).round().astype(int)
    return df

def load_result_csv(filename):
    """Loads a CSV from the results/tables folder if it exists."""
    if not filename.endswith('.csv'):
        filename += '.csv'
        
    file_path = os.path.join(results_root, "tables", filename)
    
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            print(f"    Loaded cached result: tables/{filename}")
            return df
        except Exception as e:
            print(f"    Error loading cached CSV {filename}: {e}")
            return None
    return None

# --- 2. DEFINE SAVING FUNCTIONS ---
def get_section_root():
    """
    Returns the absolute path to the section root directory (SECTION1_DimensionalityReduction).
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)

def ensure_results_folders():
    """
    Ensures that the 'results', 'results/plots', and 'results/tables' folders exist 
    inside the SECTION1_DimensionalityReduction directory.
    
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
    for sub in ["plots", "tables"]:
        sub_path = os.path.join(results_path, sub)
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)
            print(f"Created subfolder: {sub_path}")
        else:
             print(f"Subfolder exists: {sub_path}")
             
    return results_path

def create_and_save_cleaned_ratings(raw_filename='ratings.csv',cleaned_filename='ratings_cleaned.csv',sample_filename='ratings_cleaned_sampled.csv',seed=42,n_users=100000,n_items=1000,n_ratings=1000000):
    """
    Loads raw ratings, cleans them, saves 'ratings_cleaned.csv'.
    Then creates a deterministic sample and saves 'ratings_cleaned_sampled.csv'.
    """
    section_root = get_section_root()
    
    # 1. Locate Raw Data (to determine save location)
    possible_paths_raw = [
        os.path.join(section_root, 'data', 'ml-20m', raw_filename),
        os.path.join('data', 'ml-20m', raw_filename),
        os.path.join('..', 'data', 'ml-20m', raw_filename)
    ]
    
    raw_path = None
    for path in possible_paths_raw:
        if os.path.exists(path):
            raw_path = path
            break
            
    # Determine save directory
    if raw_path:
        save_dir = os.path.dirname(raw_path)
    else:
        # Fallback if raw not found, but maybe sample exists?
        # We try standard locations
        save_dir = os.path.join(section_root, 'data', 'ml-20m')

    # 1.5 Check if files already exist
    cleaned_path = os.path.join(save_dir, cleaned_filename)
    sample_path = os.path.join(save_dir, sample_filename)
    
    if os.path.exists(sample_path) and os.path.exists(cleaned_path):
        print(f"Files already exist in {save_dir}. Skipping recreation.")
        try:
            return pd.read_csv(sample_path)
        except Exception as e:
            print(f"Error loading existing sample: {e}. Recreating...")

    if raw_path is None:
        print(f"Error: Could not find raw file '{raw_filename}'")
        return None
        
    print(f"Loading raw data from: {raw_path}")
    df = pd.read_csv(raw_path)
    
    # 2. Clean
    print("Cleaning data (1-5 scale, removing timestamp)...")
    if 'timestamp' in df.columns:
        df.drop(columns=['timestamp'], inplace=True)
    df['rating'] = df['rating'].clip(lower=1, upper=5).round().astype(int)
    
    # 3. Save Cleaned Data
    try:
        df.to_csv(cleaned_path, index=False)
        print(f"Saved cleaned ratings to: {cleaned_path}")
    except Exception as e:
        print(f"Error saving cleaned file: {e}")

    # 4. Deterministic Sampling
    print(f"Sampling Data (Seed {seed})...")
    np.random.seed(seed)
    
    # Filter for top n_items items
    top_items = df['movieId'].value_counts().nlargest(n_items).index
    df_filtered_items = df[df['movieId'].isin(top_items)]
    
    # Filter for random n_users
    available_users = df_filtered_items['userId'].unique()
    
    if len(available_users) > n_users:
        selected_users = np.random.choice(available_users, n_users, replace=False)
        df_filtered_users = df_filtered_items[df_filtered_items['userId'].isin(selected_users)]
    else:
        df_filtered_users = df_filtered_items
        
    # Sample n_ratings
    if len(df_filtered_users) > n_ratings:
        df_sampled = df_filtered_users.sample(n=n_ratings, random_state=seed)
    else:
        df_sampled = df_filtered_users
        
    print(f"Sampled shape: {df_sampled.shape} (Users: {df_sampled['userId'].nunique()}, Items: {df_sampled['movieId'].nunique()})")
    
    # 5. Save Sampled Data
    try:
        df_sampled.to_csv(sample_path, index=False)
        print(f"Saved sampled data to: {sample_path}")
    except Exception as e:
        print(f"Error saving sampled file: {e}")
        
    return df_sampled

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


#Statistical Analysis

def Mean(df, group_col, target_col):

    """
    Calculates the mean AND count of target_col grouped by group_col manually using loops.
    """
    target_feature = group_col.replace("Id", "")

    # 1. Initialize dictionaries for running totals and counts
    sums = {}
    counts = {}

    print(f"Starting manual loop for Group: '{group_col}' -> Target: '{target_col}'...")

    # 2. Iterate through the specific columns dynamically
    for key, value in zip(df[group_col], df[target_col]):
        # basic error handling: skip if the value is not a number (NaN)
        if pd.isna(value):
            continue

        # Logic: Initialize if new key
        if key not in sums:
            sums[key] = 0.0
            counts[key] = 0
        
        # Accumulate
        sums[key] += value
        counts[key] += 1

    # 3. Calculate Means (Total / Count)
    results_data = []

    for key in sums:
        total = sums[key]
        n = counts[key]
        
        if n > 0:
            mean_val = total / n
            
            # Create a dictionary for this row
            results_data.append({
                group_col: key, 
                f'mean_{target_col}_per_{target_feature}': mean_val,
                f'{target_col}_count_per_{target_feature}': n
            })

    # 4. Convert list of dicts to DataFrame
    results_df = pd.DataFrame(results_data)
    
    return results_df

def get_random_user_and_check(df, condition_mask, label, condition_text, storage_list, seed=42):
    """
    Selects a random user from df based on condition_mask, prints verification, 
    and appends details to storage_list.
    """
    subset = df[condition_mask]
    
    if subset.empty:
        print(f"[{label}] FAILURE: No users found matching {condition_text}")
        return None
    
    # Random selection
    selected_row = subset.sample(n=1, random_state=seed)
    
    # Extract details
    uid = selected_row['userId'].values[0]
    # Check for correct column name dynamically or use fixed one consistent with Mean function
    if 'rating_count_per_user' in selected_row:
        count = selected_row['rating_count_per_user'].values[0]
    elif 'user_num_ratings' in selected_row:
        count = selected_row['user_num_ratings'].values[0]
    else:
        # Fallback or error
        print(f"[{label}] Warning: Rating count column not found. Available: {selected_row.columns}")
        count = 0

    pct = selected_row['pct_rated'].values[0]
    
    # Print Verification
    print(f"[{label}] Selected User: {uid}")
    print(f"   Condition: {condition_text}")
    print(f"   Actual:    {pct:.4f}% ({count} ratings)")
    
    # Verify status using the index from the original mask
    is_match = condition_mask.loc[selected_row.index].values[0]
    status = 'MATCH' if is_match else 'MISMATCH'
    print(f"   Status:    {status}")
    print("-" * 48)
    
    # --- STORE DATA FOR SAVING ---
    storage_list.append({
        'Target_Label': label,
        'UserId': uid,
        'Rating_Count': count,
        'Percentage': pct,
        'Condition': condition_text,
        'Status': status
    })
    
    return uid

#Eyad

def create_user_item_matrix(ratings_df):
    """
    Creates a User-Item Matrix efficiently.
    Returns: DataFrame (User x Item)
    """
    print("Converting to User-Item Matrix...")
    # Use float32 to save memory
    pivot_df = ratings_df.pivot(index='userId', columns='movieId', values='rating').astype('float32')
    return pivot_df

def perform_mean_filling(matrix_df):
    """
    Fills NaN values with the column (item) mean.
    """
    print("Performing Mean Filling...")
    # Fill NaNs with column means inplace
    return matrix_df.fillna(matrix_df.mean())

def calculate_target_covariance(centered_df, target_item_ids):
    """
    Calculates the covariance of specific target items ONLY against ALL other items.
    Uses a MANUAL LOOP approach as requested, optimizing by only iterating relevant users.
    Formula: Sum((Rat_u,i * Rat_u,j)) / (N - 1)
    
    Args:
        centered_df: DataFrame with ['userId', 'movieId', 'rating_diff']
        target_item_ids: List of movieIds to calculate covariance for.
        
    Returns:
        DataFrame (Num_Targets x Num_Items) containing the covariances.
    """
    print("Starting MANUAL covariance calculation...")
    
    # Global N (Total users in the full dataset, not just the intersection)
    num_users = getattr(centered_df['userId'], 'nunique', lambda: len(centered_df['userId'].unique()))()
    print(f"Total N (Users): {num_users}")
    if (num_users <= 1):
        print("Error: N <= 1, cannot divide by N-1")
        return None

    all_movie_ids = sorted(centered_df['movieId'].unique())
    print(f"Total Items: {len(all_movie_ids)}")
    
    # Optimize: We only need to iterate over users who actually rated our target items
    # 1. Find users who rated ANY of the target items
    target_records = centered_df[centered_df['movieId'].isin(target_item_ids)]
    relevant_user_ids = target_records['userId'].unique()
    print(f"Users who rated targets: {len(relevant_user_ids)}")
    
    # 2. Filter dataset to only these users (Significantly reduces iteration space)
    relevant_df = centered_df[centered_df['userId'].isin(relevant_user_ids)]
    
    # 3. Build lookup structure: User -> {MovieId -> RatingDiff}
    # Grouping by user to make iteration easy
    print("Building efficient lookup dictionary...")
    user_ratings_map = {}
    
    # Iterate rows manually or use groupby (groupby is faster to build structure)
    # Using a loop over values is manual and clear
    # To be "manual" but essentially fast enough for Python:
    # Convert dataframe to records to iterate
    records = relevant_df[['userId', 'movieId', 'rating_diff']].to_dict('records')
    
    for row in records:
        u = row['userId']
        m = row['movieId']
        r = row['rating_diff']
        
        if u not in user_ratings_map:
            user_ratings_map[u] = {}
        user_ratings_map[u][m] = r
            
    print("Lookup built. Calculating sums...")
    
    # 4. Main Manual Loop
    results = {} # target_id -> {other_id -> covariance}
    
    for target_id in target_item_ids:
        print(f"Processing Target Item: {target_id}...")
        
        # Initialize sums for all items to 0
        # We assume 0 for any item not encountered (Since 0 * x = 0)
        item_sums = {} 
        
        # Iterate over users who actually rated this target
        # (We can scan all users in our map, or filter. Scanning map is fine)
        for user_id, user_items in user_ratings_map.items():
            
            # Check if this user rated the current target item
            if target_id in user_items:
                val_target = user_items[target_id]
                
                # Multiply with ALL other items this user rated
                for other_id, val_other in user_items.items():
                    # Sum(X * Y)
                    product = val_target * val_other
                    
                    if other_id in item_sums:
                        item_sums[other_id] += product
                    else:
                        item_sums[other_id] = product
        
        # Calculate Covariance: Sum / (N - 1)
        cov_vector = {}
        denom = num_users - 1
        
        # Ensure all movies are in the result, even if 0
        # (Optimization: We can just use the item_sums key set plus default 0)
        # But for full matrix shape consistency, we might want all_movie_ids columns
        
        for m_id in all_movie_ids:
            sum_val = item_sums.get(m_id, 0.0)
            cov_vector[m_id] = sum_val / denom
            
        results[target_id] = cov_vector

    # 5. Convert to DataFrame
    print("Formatting results...")
    result_df = pd.DataFrame.from_dict(results, orient='index')
    
    # Sort columns to match all_movie_ids order
    result_df = result_df.reindex(columns=all_movie_ids).fillna(0.0)
    
    return result_df

def calculate_full_covariance_sparse(centered_df):
    """
    Calculates the FULL Item-Item Covariance Matrix for ALL items.
    Uses Sparse Matrix Multiplication: Cov = (X.T @ X) / (N - 1)
    
    Returns:
        coo_matrix: The sparse covariance matrix (N_items x N_items)
        list: The list of movieIds corresponding to the rows/cols (sorted)
    """
    print("Preparing for Full Sparse Covariance Calculation...")
    
    # 1. Map IDs to indices
    # We need a fixed sorted order of movieIds to know what row/col is what
    all_movie_ids = sorted(centered_df['movieId'].unique())
    movie_id_to_idx = {mid: i for i, mid in enumerate(all_movie_ids)}
    
    num_users = getattr(centered_df['userId'], 'nunique', lambda: len(centered_df['userId'].unique()))()
    num_items = len(all_movie_ids)
    
    print(f"Dimensions: {num_users} Users x {num_items} Items")
    
    # 2. Create Sparse Matrix X (Users x Items)
    print("Constructing User-Item Sparse Matrix...")
    
    # Efficient mapping
    # Since userId doesn't need to be sorted for covariance (just distinct rows), we can category fast map
    centered_df['userId'] = centered_df['userId'].astype('category')
    user_indices = centered_df['userId'].cat.codes
    
    # Function to map movie_id to fixed index
    # (map is slow on large df, but category matching is fast if we align categories)
    # Be careful: cat.codes relies on alphabetical/sort order of the type.
    # Our all_movie_ids is sorted. So if we cast movieId to category with explicit ordered categories:
    centered_df['movieId'] = centered_df['movieId'].astype('category')
    
    # Set categories explicitly to ensure code 0 = item all_movie_ids[0]
    centered_df['movieId'] = centered_df['movieId'].cat.set_categories(all_movie_ids)
    movie_indices = centered_df['movieId'].cat.codes
    
    rating_values = centered_df['rating_diff'].values
    
    # Create COO Matrix
    X = coo_matrix((rating_values, (user_indices, movie_indices)), shape=(num_users, num_items))
    X = X.tocsr() # Convert to CSR for arithmetic
    
    # 3. Compute Covariance
    print("Computing X.T @ X ... (This may take a moment)")
    # Matrix mult adds user products automatically
    # Result is (Num_Items x Num_Items)
    # Using float32 for output to save memory if possible
    XtX = X.T.dot(X)
    
    print("Dividing by N-1...")
    cov_matrix = XtX / (num_users - 1)
    
    return cov_matrix, all_movie_ids


#Mady

def compute_latent_neighbors(n_comp, n_neighbors, all_items, target_list, evec_matrix):
    """Find top-N similar items using cosine similarity in latent space."""
    # Project to latent space
    latent_repr = evec_matrix[:, :n_comp]
    latent_frame = pd.DataFrame(latent_repr, index=all_items)

    neighbor_dict = {}
    for target in target_list:
        if target not in latent_frame.index:
            continue

        # Compute similarities
        target_embedding = latent_frame.loc[target].values.reshape(1, -1)
        similarities = cosine_similarity(target_embedding, latent_frame.values).flatten()
        sim_scores = pd.Series(similarities, index=latent_frame.index).drop(target)
        top_neighbors = sim_scores.sort_values(ascending=False)

        neighbor_dict[target] = {
            'top_ids': top_neighbors.head(n_neighbors).index.tolist(),
            'sim_series': sim_scores
        }
    return neighbor_dict