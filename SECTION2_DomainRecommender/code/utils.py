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
    for sub in ["plots", "tables"]:
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


