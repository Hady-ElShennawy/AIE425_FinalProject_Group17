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
import json
import csv
import seaborn as sns




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
