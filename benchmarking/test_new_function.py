#!/usr/bin/env python3

import pandas as pd
import numpy as np
import sys
import os

# Add the current directory to the path so we can import the function
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dtw_calculator import calculate_pairwise_cdr_rmsd

def test_small_subset():
    """Test the modified function with a small subset of data"""
    print("Testing the modified calculate_pairwise_cdr_rmsd function...")
    
    # Load a small subset of the data
    df = pd.read_parquet('dms_embeddedby_ablang-heavy.parquet').set_index('NAME')
    
    # Get a small sample with representatives from each dataset
    train_sample = df[df['DATASET'] == 'TRAIN'].head(3)
    val_sample = df[df['DATASET'] == 'VAL'].head(2)
    test_sample = df[df['DATASET'] == 'TEST'].head(2)
    
    # Combine samples
    test_df = pd.concat([train_sample, val_sample, test_sample])
    
    print(f"Test dataframe shape: {test_df.shape}")
    print("Dataset distribution:")
    print(test_df['DATASET'].value_counts())
    print("\nSample indices:", test_df.index[:5].tolist())
    
    # Note: This will likely fail because we don't have the actual PDB files
    # but it will test the logic and show us any errors in the function structure
    try:
        train_vs_val, train_vs_test = calculate_pairwise_cdr_rmsd(
            df=test_df,
            pdb_directory='./test_pdbs',  # Dummy directory
            save_file='test_output'
        )
        print("Function executed successfully!")
        if train_vs_val is not None:
            print(f"Train vs Val matrix shape: {train_vs_val.shape}")
        if train_vs_test is not None:
            print(f"Train vs Test matrix shape: {train_vs_test.shape}")
            
    except Exception as e:
        print(f"Expected error (no PDB files): {e}")
        # This is expected since we don't have the PDB files
        
    print("Test completed!")

if __name__ == "__main__":
    test_small_subset()
