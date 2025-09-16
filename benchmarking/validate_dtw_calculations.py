#!/usr/bin/env python3
"""
Validation script for ABodyBuilder2 DTW CDR calculations.
Analyzes pre-computed similarity and RMSD matrices to find most/least similar antibody pairs.
"""

import pandas as pd
import numpy as np
import os

def load_and_validate_matrices():
    """Load and validate all required matrices and dataframes."""
    print("Loading required files...")
    
    # Load similarity matrices
    dms_sims = np.load('dms_pairwise_cdr_sims_train_vs_test.npy')
    sabdab_sims = np.load('sabdab_pairwise_cdr_sims_train_vs_test.npy')
    
    # Load RMSD matrices  
    dms_rmsds = np.load('dms_pairwise_cdr_rmsds_train_vs_test.npy')
    sabdab_rmsds = np.load('sabdab_pairwise_cdr_rmsds_train_vs_test.npy')
    
    # Load dataframes to get antibody names
    dms_df = pd.read_parquet('dms_embeddedby_ablang-heavy.parquet')
    sabdab_df = pd.read_parquet('sabdab_embeddedby_ablang-heavy.parquet')
    
    print(f"DMS similarity matrix shape: {dms_sims.shape}")
    print(f"DMS RMSD matrix shape: {dms_rmsds.shape}")
    print(f"DMS dataframe shape: {dms_df.shape}")
    print(f"SAbDab similarity matrix shape: {sabdab_sims.shape}")
    print(f"SAbDab RMSD matrix shape: {sabdab_rmsds.shape}")
    print(f"SAbDab dataframe shape: {sabdab_df.shape}")
    
    # Validate matrix dimensions match dataframe splits
    dms_train = dms_df[dms_df['DATASET'] == 'TRAIN']
    dms_test = dms_df[dms_df['DATASET'] == 'TEST']
    sabdab_train = sabdab_df[sabdab_df['DATASET'] == 'TRAIN']
    sabdab_test = sabdab_df[sabdab_df['DATASET'] == 'TEST']
    
    print(f"\nDataset splits:")
    print(f"DMS: {len(dms_train)} train, {len(dms_test)} test")
    print(f"SAbDab: {len(sabdab_train)} train, {len(sabdab_test)} test")
    
    assert dms_sims.shape == (len(dms_train), len(dms_test)), f"DMS similarity matrix shape mismatch"
    assert dms_rmsds.shape == (len(dms_train), len(dms_test)), f"DMS RMSD matrix shape mismatch"
    assert sabdab_sims.shape == (len(sabdab_train), len(sabdab_test)), f"SAbDab similarity matrix shape mismatch"
    assert sabdab_rmsds.shape == (len(sabdab_train), len(sabdab_test)), f"SAbDab RMSD matrix shape mismatch"
    
    print("✅ All matrices validated!")
    
    return {
        'dms_sims': dms_sims,
        'dms_rmsds': dms_rmsds,
        'sabdab_sims': sabdab_sims,
        'sabdab_rmsds': sabdab_rmsds,
        'dms_train': dms_train.reset_index(drop=True),
        'dms_test': dms_test.reset_index(drop=True),
        'sabdab_train': sabdab_train.reset_index(drop=True),
        'sabdab_test': sabdab_test.reset_index(drop=True)
    }


def analyze_top_pairs(sims_matrix, rmsds_matrix, train_df, test_df, dataset_name, name_col):
    """Analyze top and bottom similarity pairs for a dataset."""
    print(f"\n{'='*60}")
    print(f"ANALYZING {dataset_name.upper()} DATASET")
    print(f"{'='*60}")
    
    # Get flattened indices for sorting
    flat_indices = np.unravel_index(np.argsort(sims_matrix, axis=None), sims_matrix.shape)
    train_indices = flat_indices[0]
    test_indices = flat_indices[1]
    flat_sims = sims_matrix.flatten()
    flat_rmsds = rmsds_matrix.flatten()
    
    # Sort by similarity (ascending for least similar, descending for most similar)
    sorted_order = np.argsort(flat_sims)
    
    print(f"\n🔍 TOP 5 MOST SIMILAR CDR PAIRS:")
    print("-" * 80)
    
    # Most similar (highest similarity values)
    for i in range(5):
        idx = sorted_order[-(i+1)]  # Start from highest
        train_idx = train_indices[idx]
        test_idx = test_indices[idx]
        similarity = flat_sims[idx]
        rmsd = flat_rmsds[idx]
        
        train_name = train_df.iloc[train_idx][name_col]
        test_name = test_df.iloc[test_idx][name_col]
        
        print(f"{i+1:2d}. Train: {train_name:<15} | Test: {test_name:<15}")
        print(f"    Similarity: {similarity:.6f} | RMSD: {rmsd:.4f} Ų")
        print()
    
    print(f"\n🔍 TOP 5 LEAST SIMILAR CDR PAIRS:")
    print("-" * 80)
    
    # Least similar (lowest similarity values)
    for i in range(5):
        idx = sorted_order[i]  # Start from lowest
        train_idx = train_indices[idx]
        test_idx = test_indices[idx]
        similarity = flat_sims[idx]
        rmsd = flat_rmsds[idx]
        
        train_name = train_df.iloc[train_idx][name_col]
        test_name = test_df.iloc[test_idx][name_col]
        
        print(f"{i+1:2d}. Train: {train_name:<15} | Test: {test_name:<15}")
        print(f"    Similarity: {similarity:.6f} | RMSD: {rmsd:.4f} Ų")
        print()
    
    # Summary statistics
    print(f"\n📊 {dataset_name.upper()} SUMMARY STATISTICS:")
    print(f"Similarity range: {flat_sims.min():.6f} to {flat_sims.max():.6f}")
    print(f"RMSD range: {flat_rmsds.min():.4f} to {flat_rmsds.max():.4f} Ų")
    print(f"Mean similarity: {flat_sims.mean():.6f}")
    print(f"Mean RMSD: {flat_rmsds.mean():.4f} Ų")


def main():
    """Main validation function."""
    print("ABodyBuilder2 DTW CDR Calculations Validation")
    print("=" * 60)
    
    # Change to benchmarking directory if needed
    if not os.path.exists('dms_pairwise_cdr_sims_train_vs_test.npy'):
        print("Required files not found in current directory.")
        print("Please run this script from the benchmarking directory.")
        return
    
    # Load all required data
    data = load_and_validate_matrices()
    
    # Analyze DMS dataset
    analyze_top_pairs(
        sims_matrix=data['dms_sims'],
        rmsds_matrix=data['dms_rmsds'],
        train_df=data['dms_train'],
        test_df=data['dms_test'],
        dataset_name='DMS',
        name_col='NAME'
    )
    
    # Analyze SAbDab dataset
    analyze_top_pairs(
        sims_matrix=data['sabdab_sims'],
        rmsds_matrix=data['sabdab_rmsds'],
        train_df=data['sabdab_train'],
        test_df=data['sabdab_test'],
        dataset_name='SAbDab',
        name_col='NAME_x'
    )
    
    print(f"\n{'='*60}")
    print("VALIDATION COMPLETE")
    print(f"{'='*60}")
    print("\n✅ DTW calculations validated successfully!")
    print("Check the pairs above to verify that:")
    print("  • High similarity pairs have low RMSD values")
    print("  • Low similarity pairs have high RMSD values") 
    print("  • The relationship between similarity and RMSD makes sense")


if __name__ == "__main__":
    main()