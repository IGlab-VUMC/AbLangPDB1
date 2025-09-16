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

def prep_data_dms_from_precomputed_matrix(matrix_file: str, labels_file: str, use_square_matrices: bool = False) -> T.Tuple[np.ndarray, np.ndarray]:
    """
    Prepares DMS data from pre-computed similarity matrices (like ABodyBuilder2 DTW CDR similarities).
    
    Args:
        matrix_file (str): Path to the numpy file containing pre-computed similarity matrix.
        labels_file (str): Path to the torch tensor file with binary epitope equality matrix.
        
    Returns:
        A tuple containing:
            - y_scores (np.ndarray): A 1D array of prediction scores for each pair.
            - y_true_binary (np.ndarray): A 1D array of binary epitope equality labels.
    """
    print(f"Loading pre-computed similarity matrix from: {matrix_file}")
    
    # Load the pre-computed similarity matrix
    score_matrix = torch.from_numpy(np.load(matrix_file)).float()
    
    # Load binary labels (DMS uses binary epitope equality)
    binary_labels_matrix = torch.load(labels_file).bool()
    
    # For square matrices, only use lower triangle (excludes diagonal)
    if use_square_matrices:
        lower_triangle_mask = torch.tril(torch.ones_like(score_matrix, dtype=torch.bool), diagonal=-1)
        y_scores = score_matrix[lower_triangle_mask].numpy()
        y_true_binary = binary_labels_matrix[lower_triangle_mask].numpy().astype(int)
    else:
        # Flatten matrices for rectangular case
        y_scores = score_matrix.flatten().numpy()
        y_true_binary = binary_labels_matrix.flatten().numpy().astype(int)
    
    return y_scores, y_true_binary

def prep_data_dms(df: pd.DataFrame, score_type: str, dataset1: str, dataset2: str, labels_file: str, use_square_matrices: bool = False, embedding_column: str = "EMBEDDING") -> T.Tuple[np.ndarray, np.ndarray]:
    """
    Prepares data by calculating pairwise scores and aligning them with binary epitope labels for DMS dataset.

    Args:
        df (pd.DataFrame): DataFrame with antibody sequences and/or embeddings.
        score_type (str): The method for scoring pairs. 
                          Options: 'cosine', 'seq_identity', 'cdrh3_identity', 'abodybuilder2_dtw_cdrs'.
        dataset1 (str): Name of the first dataset subset (e.g., "TRAIN").
        dataset2 (str): Name of the second dataset subset (e.g., "VAL").
        labels_file (str): Path to the torch tensor file with binary epitope equality matrix.

    Returns:
        A tuple containing:
            - y_scores (np.ndarray): A 1D array of prediction scores for each pair.
            - y_true_binary (np.ndarray): A 1D array of binary epitope labels (1=same epitope, 0=different).
    """
    print(f"Preparing data with score_type: '{score_type}' for {dataset1} vs {dataset2}...")
    
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

    # Load binary epitope equality matrix
    true_labels_matrix = torch.load(labels_file)
    
    # For square matrices, only use lower triangle (excludes diagonal)
    if use_square_matrices and dataset1 == dataset2:
        lower_triangle_mask = torch.tril(torch.ones_like(score_matrix, dtype=torch.bool), diagonal=-1)
        y_scores = score_matrix[lower_triangle_mask].numpy()
        y_true_binary = true_labels_matrix[lower_triangle_mask].numpy().astype(int)
    else:
        # Convert to binary (True->1, False->0) and flatten
        y_scores = score_matrix.flatten().numpy()
        y_true_binary = true_labels_matrix.flatten().numpy().astype(int)
    
    return y_scores, y_true_binary

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

def calculate_roc_auc(y_scores: np.ndarray, y_true_binary: np.ndarray) -> T.Tuple[float, np.ndarray, np.ndarray]:
    """Calculates ROC-AUC score and curve points for binary classification."""
    auc_score = roc_auc_score(y_true_binary, y_scores)
    fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
    
    return auc_score, fpr, tpr

def calculate_pw_avg_prec(y_scores: np.ndarray, y_true_binary: np.ndarray) -> T.Tuple[float, np.ndarray, np.ndarray, float]:
    """Calculates pairwise average precision and precision-recall curve points for binary classification."""
    avg_prec = average_precision_score(y_true_binary, y_scores)
    precision, recall, _ = precision_recall_curve(y_true_binary, y_scores)
    no_skill_baseline = np.sum(y_true_binary) / len(y_true_binary)
    
    return avg_prec, precision, recall, no_skill_baseline

def get_metrics_dms(df_path: str, labels_file_val: str, labels_file_test: str, score_type: str, model_name: str, output_folder: str, dataset1: str = "TRAIN", dataset2_val: str = "VAL", dataset2_test: str = "TEST", epitope_threshold: float = None, matrix_file_val: str = None, matrix_file_test: str = None, use_square_matrices: bool = False, embedding_column: str = "EMBEDDING"):
    """
    Main function to run the full analysis pipeline for DMS dataset with binary epitope labels.
    
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
        epitope_threshold (float, optional): Pre-computed optimal threshold. If None, will compute from validation data.
        matrix_file_val (str, optional): Path to pre-computed similarity matrix for validation (required for 'abodybuilder2_dtw_cdrs')
        matrix_file_test (str, optional): Path to pre-computed similarity matrix for test (required for 'abodybuilder2_dtw_cdrs')
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    df = pd.read_parquet(df_path)

    # --- Find Optimal Threshold using VAL data ---
    if epitope_threshold is None:
        print("\n--- Finding optimal F1 threshold using VAL data ---")
        if score_type == 'abodybuilder2_dtw_cdrs':
            if matrix_file_val is None:
                raise ValueError("matrix_file_val is required for abodybuilder2_dtw_cdrs score_type")
            y_scores_val, y_true_binary_val = prep_data_dms_from_precomputed_matrix(matrix_file_val, labels_file_val, use_square_matrices)
        else:
            y_scores_val, y_true_binary_val = prep_data_dms(df, score_type, dataset1, dataset2_val, labels_file_val, use_square_matrices, embedding_column)
        optimal_threshold = find_optimal_f1_threshold(y_scores_val, y_true_binary_val)
        print(f"Optimal F1 Threshold for Epitope matching: {optimal_threshold:.4f}")
    else:
        optimal_threshold = epitope_threshold
        print(f"\n--- Using provided F1 threshold: {optimal_threshold:.4f} ---")

    # --- Epitope Analysis on TEST data ---
    # Prepare test data for final evaluation
    if score_type == 'abodybuilder2_dtw_cdrs':
        if matrix_file_test is None:
            raise ValueError("matrix_file_test is required for abodybuilder2_dtw_cdrs score_type")
        y_scores_test, y_true_binary_test = prep_data_dms_from_precomputed_matrix(matrix_file_test, labels_file_test, use_square_matrices)
    else:
        # For test vs test mode, use TEST vs TEST for final evaluation
        final_dataset1 = dataset2_test if use_square_matrices and dataset1 == dataset2_val else dataset1
        y_scores_test, y_true_binary_test = prep_data_dms(df, score_type, final_dataset1, dataset2_test, labels_file_test, use_square_matrices, embedding_column)

    print(f"\n--- Analyzing Epitope-level performance on {dataset2_test} data (Binary epitope matching) ---")
    auc, fpr, tpr = calculate_roc_auc(y_scores_test, y_true_binary_test)
    ap, precision, recall, no_skill_baseline = calculate_pw_avg_prec(y_scores_test, y_true_binary_test)
    
    # Calculate F1 score using the optimal threshold found on VAL data
    y_pred_binary_test = (y_scores_test >= optimal_threshold).astype(int)
    f1 = f1_score(y_true_binary_test, y_pred_binary_test)
    
    print(f"ROC-AUC: {auc:.4f}, Average Precision: {ap:.4f}, F1 Score: {f1:.4f}")

    # --- Save Results ---
    roc_results = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'AUC': auc, 'F1': f1, 'Model': model_name})
    pr_results = pd.DataFrame({'Recall': recall, 'Precision': precision, 'AP': ap, 'NO_SKILL': no_skill_baseline, 'Model': model_name})

    roc_results.to_csv(f'{output_folder}/{model_name}_dms_roc_epitope.csv', index=False)
    pr_results.to_csv(f'{output_folder}/{model_name}_dms_pr_epitope.csv', index=False)
    
    # --- Save Summary Metrics ---
    summary_filename = f'{output_folder}/{model_name}_dms_summarymetrics.txt'
    with open(summary_filename, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: dms\n")
        f.write(f"Score_Type: {score_type}\n")
        f.write(f"ROC_AUC: {auc:.6f}\n")
        f.write(f"Average_Precision: {ap:.6f}\n")
        f.write(f"F1_Score: {f1:.6f}\n")
        f.write(f"Threshold: {optimal_threshold:.6f}\n")
    
    print(f"\nResults saved to {output_folder}")
    print(f"Summary metrics saved to {summary_filename}")


if __name__ == '__main__':
    # --- Configuration ---
    DF_PATH = "dms_embeddedby_ablangpdb.parquet"  # or your DMS dataset path
    LABELS_PATH_VAL = 'dms_train_val_label_mat.pt'
    LABELS_PATH_TEST = 'dms_train_test_label_mat.pt'
    MODEL_NAME = "AbLangPDB_on_DMS"
    OUTPUTS = "output_csvs"
    
    # --- Run Analysis ---
    get_metrics_dms(
        df_path=DF_PATH,
        labels_file_val=LABELS_PATH_VAL,
        labels_file_test=LABELS_PATH_TEST,
        score_type='cosine',  # Change this to 'seq_identity' or 'cdrh3_identity' to test others
        model_name=MODEL_NAME,
        output_folder=OUTPUTS
    )