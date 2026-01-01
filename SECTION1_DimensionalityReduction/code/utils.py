import pandas as pd

# --- 1. LOCATE THE RESULTS FOLDER ---
import os

def load_data(filename='ratings_cleaned.csv'):
    """Location independent data loader"""
    possible_paths = [
        os.path.join('data', filename),
        os.path.join('..', 'data', filename)
    ]
    
    # Check current directory too
    if os.path.exists(filename):
        possible_paths.insert(0, filename)

    for path in possible_paths:
        if os.path.exists(path):
            print(f" Found dataset at: {path}")
            try:
                df = pd.read_csv(path)
                print(f" SUCCESS: Data loaded! ({len(df)} rows)")
                return df
            except Exception as e:
                print(f" Error loading CSV: {e}")
                return None
    print(f"\\n ERROR: Could not find '{filename}'.")
    return None

# Check if 'results' is in the current folder (Root) or one level up (Subfolder)
if os.path.exists("results"):
    results_root = "results"
elif os.path.exists("../results"):
    results_root = "../results"
else:
    # Fallback: Create it in current dir if it's missing
    results_root = "results"
    os.makedirs(results_root, exist_ok=True)

# --- 2. DEFINE SAVING FUNCTIONS ---

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

def get_random_user_and_check(df, condition_mask, label, condition_text, storage_list):
    """
    Selects a random user from df based on condition_mask, prints verification, 
    and appends details to storage_list.
    """
    subset = df[condition_mask]
    
    if subset.empty:
        print(f"[{label}] FAILURE: No users found matching {condition_text}")
        return None
    
    # Random selection
    selected_row = subset.sample(n=1)
    
    # Extract details
    uid = selected_row['userId'].values[0]
    count = selected_row['user_num_ratings'].values[0]
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

