import pandas as pd
import torch
import numpy as np
import os
import Levenshtein as Lev
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, f1_score
import matplotlib.pyplot as plt
import typing as T

# --- Helper Functions for Sequence Identity ---
# (Adapted from your clone_utils.py to make this script self-contained)

def get_seq_id_matrix(query: tuple, train_db: pd.DataFrame) -> np.ndarray:
    """Calculates pairwise sequence identity for full heavy and light chains."""
    hc, lc = query
    query_len = len(hc) + len(lc)
    # Use .copy() to avoid SettingWithCopyWarning
    train_db_copy = train_db.copy()
    train_db_copy["MAX_LEN"] = train_db_copy["SEQ_LEN"].apply(lambda l: max(l, query_len))

    hc_dists = train_db_copy["HC_AA"].apply(lambda train_hc: Lev.distance(hc, train_hc))
    lc_dists = train_db_copy["LC_AA"].apply(lambda train_lc: Lev.distance(lc, train_lc))

    fract_identities = 1 - ((hc_dists + lc_dists) / train_db_copy["MAX_LEN"])
    return fract_identities.values

def get_seq_id_matrix_cdrh3(query: str, train_db: pd.DataFrame) -> np.ndarray:
    """Calculates pairwise sequence identity for CDRH3."""
    hc = query
    query_len = len(hc)
    train_db_copy = train_db.copy()
    train_db_copy["MAX_LEN"] = train_db_copy["SEQ_LEN"].apply(lambda l: max(l, query_len))
    hc_dists = train_db_copy["CDRH3"].apply(lambda train_hc: Lev.distance(hc, train_hc))
    fract_identities = 1 - (hc_dists / train_db_copy["MAX_LEN"])
    return fract_identities.values

# --- Core Analysis Functions ---

def prep_data_from_precomputed_matrix(matrix_file: str, labels_file: str, use_square_matrices: bool = False) -> T.Tuple[np.ndarray, np.ndarray]:
    """
    Prepares data from pre-computed similarity matrices (like ABodyBuilder2 DTW CDR similarities).
    
    Args:
        matrix_file (str): Path to the numpy file containing pre-computed similarity matrix.
        labels_file (str): Path to the torch tensor file with pre-computed labels.
        use_square_matrices (bool): If True, applies lower triangle mask for square matrices (excludes diagonal).
        
    Returns:
        A tuple containing:
            - y_scores (np.ndarray): A 1D array of prediction scores for each pair.
            - y_true_continuous (np.ndarray): A 1D array of the corresponding continuous ground truth labels.
    """
    print(f"Loading pre-computed similarity matrix from: {matrix_file}")
    
    # Load the pre-computed similarity matrix
    score_matrix = torch.from_numpy(np.load(matrix_file)).float()
    
    # Load true labels and create a mask to filter out irrelevant pairs
    true_labels_matrix = torch.load(labels_file)
    nonexistent_pairs = torch.where((-.1 < true_labels_matrix) & (true_labels_matrix < 0.1))
    to_use_mask = torch.ones_like(score_matrix, dtype=torch.bool)
    to_use_mask[nonexistent_pairs[0], nonexistent_pairs[1]] = False
    
    # For square matrices, apply lower triangle mask (excludes diagonal)
    if use_square_matrices and score_matrix.shape[0] == score_matrix.shape[1]:
        # Create lower triangle mask (excluding diagonal) 
        lower_triangle_mask = torch.tril(torch.ones_like(score_matrix, dtype=torch.bool), diagonal=-1)
        to_use_mask = to_use_mask & lower_triangle_mask
        print(f"  🔄 Square matrix mode: Using lower triangle (excluding diagonal)")
        print(f"  📊 Matrix shape: {score_matrix.shape}, Using {to_use_mask.sum()} pairs (lower triangle)")
    
    # Flatten and apply the mask
    y_scores = score_matrix[to_use_mask].numpy()
    y_true_continuous = true_labels_matrix[to_use_mask].numpy()
    
    return y_scores, y_true_continuous

def prep_data(df: pd.DataFrame, score_type: str, dataset1: str, dataset2: str, labels_file: str, use_square_matrices: bool = False, embedding_column: str = "EMBEDDING") -> T.Tuple[np.ndarray, np.ndarray]:
    """
    Prepares data by calculating pairwise scores and aligning them with true labels.

    Args:
        df (pd.DataFrame): DataFrame with antibody sequences and/or embeddings.
        score_type (str): The method for scoring pairs. 
                          Options: 'cosine', 'seq_identity', 'cdrh3_identity', 'abodybuilder2_dtw_cdrs'.
        dataset1 (str): Name of the first dataset subset (e.g., "TRAIN").
        dataset2 (str): Name of the second dataset subset (e.g., "VAL").
        labels_file (str): Path to the torch tensor file with pre-computed labels.
        use_square_matrices (bool): If True, applies lower triangle mask for square matrices (excludes diagonal).

    Returns:
        A tuple containing:
            - y_scores (np.ndarray): A 1D array of prediction scores for each pair.
            - y_true_continuous (np.ndarray): A 1D array of the corresponding continuous ground truth labels.
    """
    print(f"Preparing data with score_type: '{score_type}' for {dataset1} vs {dataset2}...")
    if use_square_matrices and dataset1 == dataset2:
        print(f"  🔄 Square matrix mode: Using lower triangle (excluding diagonal)")
    
    set1_abs = df[df["DATASET"] == dataset1].copy()
    set2_abs = df[df["DATASET"] == dataset2].copy()

    if score_type == 'cosine':
        embeds1 = torch.Tensor(set1_abs[embedding_column].to_list())
        embeds2 = torch.Tensor(set2_abs[embedding_column].to_list())
        score_matrix = embeds1 @ embeds2.t()
    elif score_type == 'seq_identity':
        set1_abs["HCLC"] = set1_abs.apply(lambda row: (row["HC_AA"], row["LC_AA"]), axis=1)
        set2_abs["SEQ_LEN"] = set2_abs["HC_AA"].str.len() + set2_abs["LC_AA"].str.len()
        score_matrix = torch.Tensor(np.vstack(set1_abs["HCLC"].apply(lambda seq_tup: get_seq_id_matrix(seq_tup, set2_abs)).values))
    elif score_type == 'cdrh3_identity':
        set2_abs["SEQ_LEN"] = set2_abs["CDRH3"].astype(str).str.len()
        score_matrix = torch.Tensor(np.vstack(set1_abs["CDRH3"].apply(lambda cdrh3: get_seq_id_matrix_cdrh3(cdrh3, set2_abs)).values))
    else:
        raise ValueError("Invalid score_type. Choose from 'cosine', 'seq_identity', 'cdrh3_identity'.")

    # Load true labels and create a mask to filter out irrelevant pairs
    true_labels_matrix = torch.load(labels_file)
    nonexistent_pairs = torch.where((-.1 < true_labels_matrix) & (true_labels_matrix < 0.1))
    to_use_mask = torch.ones_like(score_matrix, dtype=torch.bool)
    to_use_mask[nonexistent_pairs[0], nonexistent_pairs[1]] = False
    
    # For square matrices, apply lower triangle mask (excludes diagonal)
    if use_square_matrices and dataset1 == dataset2:
        # Create lower triangle mask (excluding diagonal) 
        lower_triangle_mask = torch.tril(torch.ones_like(score_matrix, dtype=torch.bool), diagonal=-1)
        to_use_mask = to_use_mask & lower_triangle_mask
        print(f"  📊 Matrix shape: {score_matrix.shape}, Using {to_use_mask.sum()} pairs (lower triangle)")
    
    # Flatten and apply the mask
    y_scores = score_matrix[to_use_mask].numpy()
    y_true_continuous = true_labels_matrix[to_use_mask].numpy()
    
    return y_scores, y_true_continuous

def find_optimal_f1_threshold(y_scores: np.ndarray, y_true_binary: np.ndarray) -> float:
    """
    Finds the optimal threshold to maximize the F1 score by iterating through unique score values.
    """
    thresholds = np.unique(y_scores)
    # To handle cases with many unique values, we can sample thresholds
    if len(thresholds) > 1000:
        thresholds = np.linspace(thresholds.min(), thresholds.max(), 1000)

    f1_scores = [f1_score(y_true_binary, (y_scores >= t).astype(int)) for t in thresholds]
    
    best_threshold_idx = np.argmax(f1_scores)
    return thresholds[best_threshold_idx]

def calculate_roc_auc(y_scores: np.ndarray, y_true_continuous: np.ndarray, positive_threshold: float) -> T.Tuple[float, np.ndarray, np.ndarray]:
    """Calculates ROC-AUC score and curve points."""
    y_true_binary = (y_true_continuous >= positive_threshold).astype(int)
    
    auc_score = roc_auc_score(y_true_binary, y_scores)
    fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
    
    return auc_score, fpr, tpr

def calculate_pw_avg_prec(y_scores: np.ndarray, y_true_continuous: np.ndarray, positive_threshold: float) -> T.Tuple[float, np.ndarray, np.ndarray, float]:
    """Calculates pairwise average precision and precision-recall curve points."""
    y_true_binary = (y_true_continuous >= positive_threshold).astype(int)
    
    avg_prec = average_precision_score(y_true_binary, y_scores)
    precision, recall, _ = precision_recall_curve(y_true_binary, y_scores)
    no_skill_baseline = np.sum(y_true_binary) / len(y_true_binary)
    
    return avg_prec, precision, recall, no_skill_baseline

def get_metrics(df_path: str, labels_file_val: str, labels_file_test: str, score_type: str, model_name: str, output_folder: str, dataset1: str = "TRAIN", dataset2_val: str = "VAL", dataset2_test: str = "TEST", epitope_threshold: float = None, antigen_threshold: float = None, matrix_file_val: str = None, matrix_file_test: str = None, use_square_matrices: bool = False, embedding_column: str = "EMBEDDING"):
    """
    Main function to run the full analysis pipeline for a given scoring method.
    
    Args:
        df_path (str): Path to the dataset parquet file
        labels_file_val (str): Path to validation labels file
        labels_file_test (str): Path to test labels file
        score_type (str): Type of scoring ('cosine', 'seq_identity', 'cdrh3_identity', 'abodybuilder2_dtw_cdrs')
        model_name (str): Name of the model for output files
        output_folder (str): Directory to save results
        dataset1 (str): Name of first dataset (default: "TRAIN")
        dataset2_val (str): Name of validation dataset (default: "VAL")
        dataset2_test (str): Name of test dataset (default: "TEST")
        epitope_threshold (float, optional): Pre-computed optimal threshold for epitope classification. If None, will compute from validation data.
        antigen_threshold (float, optional): Pre-computed optimal threshold for antigen classification. If None, will compute from validation data.
        matrix_file_val (str, optional): Path to pre-computed similarity matrix for validation (required for 'abodybuilder2_dtw_cdrs')
        matrix_file_test (str, optional): Path to pre-computed similarity matrix for test (required for 'abodybuilder2_dtw_cdrs')
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    df = pd.read_parquet(df_path)

    # --- Find Optimal Thresholds using VAL data ---
    if epitope_threshold is None or antigen_threshold is None:
        print("\n--- Finding optimal F1 thresholds using VAL data ---")
        if score_type == 'abodybuilder2_dtw_cdrs':
            if matrix_file_val is None:
                raise ValueError("matrix_file_val is required for abodybuilder2_dtw_cdrs score_type")
            y_scores_val, y_true_continuous_val = prep_data_from_precomputed_matrix(matrix_file_val, labels_file_val, use_square_matrices)
        else:
            y_scores_val, y_true_continuous_val = prep_data(df, score_type, dataset1, dataset2_val, labels_file_val, use_square_matrices, embedding_column)
    
    # Threshold for Epitope
    if epitope_threshold is None:
        y_true_binary_val_ep = (y_true_continuous_val >= 0.5).astype(int)
        optimal_threshold_ep = find_optimal_f1_threshold(y_scores_val, y_true_binary_val_ep)
        print(f"Optimal F1 Threshold for Epitope (>=0.5): {optimal_threshold_ep:.4f}")
    else:
        optimal_threshold_ep = epitope_threshold
        print(f"Using provided Epitope threshold: {optimal_threshold_ep:.4f}")

    # Threshold for Antigen
    if antigen_threshold is None:
        y_true_binary_val_ag = (y_true_continuous_val >= 0.2).astype(int)
        optimal_threshold_ag = find_optimal_f1_threshold(y_scores_val, y_true_binary_val_ag)
        print(f"Optimal F1 Threshold for Antigen (>=0.2): {optimal_threshold_ag:.4f}")
    else:
        optimal_threshold_ag = antigen_threshold
        print(f"Using provided Antigen threshold: {optimal_threshold_ag:.4f}")

    # --- Epitope Analysis (>= 0.5) on TEST data ---
    # Prepare test data for final evaluation
    if score_type == 'abodybuilder2_dtw_cdrs':
        if matrix_file_test is None:
            raise ValueError("matrix_file_test is required for abodybuilder2_dtw_cdrs score_type")
        y_scores_test, y_true_continuous_test = prep_data_from_precomputed_matrix(matrix_file_test, labels_file_test, use_square_matrices)
    else:
        # For test vs test mode, use TEST vs TEST for final evaluation
        final_dataset1 = dataset2_test if use_square_matrices and dataset1 == dataset2_val else dataset1
        y_scores_test, y_true_continuous_test = prep_data(df, score_type, final_dataset1, dataset2_test, labels_file_test, use_square_matrices, embedding_column)

    print(f"\n--- Analyzing Epitope-level performance on {dataset2_test} data (Positive label >= 0.5) ---")
    auc_ep, fpr_ep, tpr_ep = calculate_roc_auc(y_scores_test, y_true_continuous_test, 0.5)
    ap_ep, prec_ep, rec_ep, ns_ep = calculate_pw_avg_prec(y_scores_test, y_true_continuous_test, 0.5)
    
    # Calculate F1 score using the optimal threshold found on the TEST set
    y_true_binary_val_ep = (y_true_continuous_test >= 0.5).astype(int)
    y_pred_binary_test_ep = (y_scores_test >= optimal_threshold_ep).astype(int)
    f1_ep = f1_score(y_true_binary_val_ep, y_pred_binary_test_ep)
    
    print(f"ROC-AUC: {auc_ep:.4f}, Average Precision: {ap_ep:.4f}, F1 Score: {f1_ep:.4f}")

    # --- Antigen Analysis (>= 0.2) on TEST data ---
    print(f"\n--- Analyzing Antigen-level performance on {dataset2_test} data (Positive label >= 0.2) ---")
    auc_ag, fpr_ag, tpr_ag = calculate_roc_auc(y_scores_test, y_true_continuous_test, 0.2)
    ap_ag, prec_ag, rec_ag, ns_ag = calculate_pw_avg_prec(y_scores_test, y_true_continuous_test, 0.2)

    # Calculate F1 score using the optimal threshold found on the TEST set
    y_true_binary_val_ag = (y_true_continuous_test >= 0.2).astype(int)
    y_pred_binary_test_ag = (y_scores_test >= optimal_threshold_ag).astype(int)
    f1_ag = f1_score(y_true_binary_val_ag, y_pred_binary_test_ag)

    print(f"ROC-AUC: {auc_ag:.4f}, Average Precision: {ap_ag:.4f}, F1 Score: {f1_ag:.4f}")

    # --- Save Results ---
    roc_ep_results = pd.DataFrame({'FPR': fpr_ep, 'TPR': tpr_ep, 'AUC': auc_ep, 'F1': f1_ep, 'Model': model_name})
    roc_ag_results = pd.DataFrame({'FPR': fpr_ag, 'TPR': tpr_ag, 'AUC': auc_ag, 'F1': f1_ag, 'Model': model_name})
    pr_ep_results = pd.DataFrame({'Recall': rec_ep, 'Precision': prec_ep, 'AP': ap_ep, 'NO_SKILL': ns_ep, 'Model': model_name})
    pr_ag_results = pd.DataFrame({'Recall': rec_ag, 'Precision': prec_ag, 'AP': ap_ag, 'NO_SKILL': ns_ag, 'Model': model_name})

    roc_ep_results.to_csv(f'{output_folder}/{model_name}_sabdab_roc_epitope.csv', index=False)
    roc_ag_results.to_csv(f'{output_folder}/{model_name}_sabdab_roc_antigen.csv', index=False)
    pr_ep_results.to_csv(f'{output_folder}/{model_name}_sabdab_pr_epitope.csv', index=False)
    pr_ag_results.to_csv(f'{output_folder}/{model_name}_sabdab_pr_antigen.csv', index=False)
    
    # --- Save Summary Metrics ---
    # Epitope-level summary
    epitope_summary_filename = f'{output_folder}/{model_name}_sabdab_ep_summarymetrics.txt'
    with open(epitope_summary_filename, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: sabdab_ep\n")
        f.write(f"Score_Type: {score_type}\n")
        f.write(f"ROC_AUC: {auc_ep:.6f}\n")
        f.write(f"Average_Precision: {ap_ep:.6f}\n")
        f.write(f"F1_Score: {f1_ep:.6f}\n")
        f.write(f"Threshold: {optimal_threshold_ep:.6f}\n")
    
    # Antigen-level summary
    antigen_summary_filename = f'{output_folder}/{model_name}_sabdab_ag_summarymetrics.txt'
    with open(antigen_summary_filename, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: sabdab_ag\n")
        f.write(f"Score_Type: {score_type}\n")
        f.write(f"ROC_AUC: {auc_ag:.6f}\n")
        f.write(f"Average_Precision: {ap_ag:.6f}\n")
        f.write(f"F1_Score: {f1_ag:.6f}\n")
        f.write(f"Threshold: {optimal_threshold_ag:.6f}\n")
    
    print(f"\nResults saved to {output_folder}")
    print(f"Summary metrics saved to {epitope_summary_filename} and {antigen_summary_filename}")


if __name__ == '__main__':
    # --- Configuration ---
    DF_PATH = "ablangpdb_renameddatasets.parquet"
    LABELS_PATH_VAL = 'ablangpdb_train_val_label_mat.pt'
    LABELS_PATH_TEST = 'ablangpdb_train_test_label_mat.pt'
    MODEL_NAME = "AbLangPDB"
    OUTPUTS = "output_csvs"
    
    # --- Run Analysis ---
    get_metrics(
        df_path=DF_PATH,
        labels_file_val=LABELS_PATH_VAL,
        labels_file_test=LABELS_PATH_TEST,
        score_type='cosine',  # Change this to 'seq_identity' or 'cdrh3_identity' to test others
        model_name=MODEL_NAME,
        output_folder=OUTPUTS
    )