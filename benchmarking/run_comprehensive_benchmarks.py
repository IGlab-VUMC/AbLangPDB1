#!/usr/bin/env python3
"""
Comprehensive Benchmarking Pipeline Script

This script provides a complete workflow for:
1. Running comprehensive benchmarks across multiple models and datasets
2. Generating standardized summary files
3. Creating formatted Excel reports with performance rankings

Models included:
- AbLang Family: AbLangPDB, AbLangRBD, AbLangPre, AbLang2, AbLang-Heavy (cosine similarity)
- Other Protein LMs: AntiBERTy, BALM, ESM-2, IgBERT, Parapred (cosine similarity)
- Sequence-based: SEQID (sequence identity), CDRH3ID (CDRH3 identity)

Datasets:
- SAbDab (structural antibody database)
- DMS (deep mutational scanning)

Usage:
    python run_comprehensive_benchmarks.py [--recalculate-embeddings] [--output-folder OUTPUT_FOLDER] [--excel-filename FILENAME]

Options:
    --recalculate-embeddings    Re-calculate embeddings instead of using existing files
    --output-folder             Output folder for results (default: output_csvs)
    --excel-filename            Excel filename (default: comprehensive_benchmarking_results.xlsx)
    --batch-size                Batch size for embedding generation (default: 256)
"""

import pandas as pd
import torch
import numpy as np
import os
import sys
import argparse
from pathlib import Path
import warnings
import typing as T
from torch.utils.data import DataLoader, TensorDataset
import traceback
warnings.filterwarnings('ignore')

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run comprehensive benchmarking pipeline')
    parser.add_argument('--recalculate-embeddings', action='store_true', 
                       help='Re-calculate embeddings instead of using existing files')
    parser.add_argument('--output-folder', default='output_csvs', 
                       help='Output folder for results (default: output_csvs)')
    parser.add_argument('--excel-filename', default='comprehensive_benchmarking_results.xlsx',
                       help='Excel filename (default: comprehensive_benchmarking_results.xlsx)')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size for embedding generation (default: 256)')
    
    args = parser.parse_args()
    
    # =============================================================================
    # CONFIGURATION FLAGS
    # =============================================================================
    RECALCULATE_EMBEDDINGS = args.recalculate_embeddings
    OUTPUT_FOLDER = args.output_folder
    EXCEL_FILENAME = args.excel_filename
    BATCH_SIZE = args.batch_size
    
    # Model paths - update these paths as needed
    MODEL_PATHS = {
        "AbLangPDB": "../../../huggingface/AbLangPDB1/ablangpdb_model.safetensors",
        "AbLangRBD": "../../../huggingface/AbLangRBD1/model.safetensors"
    }
    
    print("="*70)
    print("COMPREHENSIVE BENCHMARKING PIPELINE")
    print("="*70)
    print(f"Configuration:")
    print(f"  • Recalculate embeddings: {RECALCULATE_EMBEDDINGS}")
    print(f"  • Output folder: {OUTPUT_FOLDER}")
    print(f"  • Excel filename: {EXCEL_FILENAME}")
    print(f"  • Batch size: {BATCH_SIZE}")
    print(f"\n📝 Note: This script supports ALL available parquet files:")
    print(f"  • Models: AbLangPDB, AbLangRBD, AbLangPre, AbLang2, AbLang-Heavy,")
    print(f"           AntiBERTy, BALM, ESM-2, IgBERT, Parapred, SEQID, CDRH3ID")
    print(f"  • Datasets: SAbDab and DMS")
    print(f"  • Only existing parquet files will be processed")
    
    # =============================================================================
    # SETUP AND IMPORTS
    # =============================================================================
    
    # Add parent directory to path for imports
    module_path = os.path.abspath(os.path.join('..'))
    if module_path not in sys.path:
        sys.path.append(module_path)
    
    try:
        # Import local modules
        import calculate_metrics
        import calculate_metrics_dms
        from excel_generator import generate_results_excel, print_summary_stats
        
        print("✅ Successfully imported all required modules")
    except ImportError as e:
        print(f"❌ Error importing required modules: {e}")
        print("Make sure you're running this script from the benchmarking directory")
        print("and that all required Python files are present.")
        sys.exit(1)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # =============================================================================
    # HELPER FUNCTIONS
    # =============================================================================
    
    def check_file_exists(filepath, description=""):
        """Check if a file exists and return status."""
        exists = os.path.exists(filepath)
        status = "✅" if exists else "❌"
        print(f"  {status} {description}: {filepath}")
        return exists
    
    def embed_with_ablangpaired(input_path: str, output_path: str, model_path: str, model_name: str):
        """Generate embeddings using AbLangPaired models.
        
            Args:
                model_name: str. If "ablangpre" then the model architecture will no longer have the mixer layer.
        """
        print(f"\n🔄 Generating {model_name} embeddings...")
        
        # Load data
        df = pd.read_parquet(input_path)
        if "EMBEDDING" in df.columns:
            df = df.drop(columns=["EMBEDDING"])
        
        # Setup model
        model_config = AbLangPairedConfig(checkpoint_filename=model_path)
        is_ablangpre = model_name == "ablangpre"
        model = AbLangPaired(model_config, device=device, use_pretrained=is_ablangpre)
        
        # Tokenize and embed using enhanced methods
        tokenized_dataloader = models.tokenize_data(df, model_config, batch_size=BATCH_SIZE)
        all_embeds = models.embed_dataloader(tokenized_dataloader, model, device)
        
        # Save
        df['EMBEDDING'] = list(all_embeds.cpu().numpy())
        df.to_parquet(output_path)
        
        print(f"✅ {model_name} embeddings saved to {output_path}")
        return df
    
    # =============================================================================
    # CHECK REQUIRED BASE FILES
    # =============================================================================
    
    print("\n📋 Checking base dataset files...")
    
    base_files = {
        "SAbDab base dataset": "ablangpdb_renameddatasets.parquet",
        "DMS base dataset": "ablangrbd_renameddatasets.parquet",
        "SAbDab validation labels": "ablangpdb_train_val_label_mat.pt",
        "SAbDab test labels": "ablangpdb_train_test_label_mat.pt",
        "DMS validation labels": "dms_train_val_label_mat.pt",
        "DMS test labels": "dms_train_test_label_mat.pt"
    }
    
    missing_base_files = []
    for desc, filepath in base_files.items():
        if not check_file_exists(filepath, desc):
            missing_base_files.append(filepath)
    
    if missing_base_files:
        print(f"\n❌ Missing required base files: {missing_base_files}")
        print("Please ensure all required data files are present before running.")
        sys.exit(1)
    
    print("\n✅ All base files found!")
    
    # =============================================================================
    # EMBEDDING FILE STATUS CHECK
    # =============================================================================
    
    # Define all embedding files that should exist
    embedding_files = {
        # SAbDab dataset embeddings
        "sabdab_embeddedby_ablangrbd.parquet": ("ablangpdb_renameddatasets.parquet", None, "AbLangRBD"),
        "sabdab_embeddedby_ablangpre.parquet": ("ablangpdb_renameddatasets.parquet", None, "AbLangPre"),
        "sabdab_embeddedby_ablang2.parquet": ("ablangpdb_renameddatasets.parquet", None, "AbLang2"),
        "sabdab_embeddedby_ablang-heavy.parquet": ("ablangpdb_renameddatasets.parquet", None, "AbLang-Heavy"),
        "sabdab_embeddedby_antiberty.parquet": ("ablangpdb_renameddatasets.parquet", None, "AntiBERTy"),
        "sabdab_embeddedby_balm.parquet": ("ablangpdb_renameddatasets.parquet", None, "BALM"),
        "sabdab_embeddedby_esm-2.parquet": ("ablangpdb_renameddatasets.parquet", None, "ESM-2"),
        "sabdab_embeddedby_igbert.parquet": ("ablangpdb_renameddatasets.parquet", None, "IgBERT"),
        "sabdab_embeddedby_parapred.parquet": ("ablangpdb_renameddatasets.parquet", None, "Parapred"),

        # DMS dataset embeddings
        "dms_embeddedby_ablangpdb.parquet": ("ablangrbd_renameddatasets.parquet", None, "AbLangPDB"),
        "dms_embeddedby_ablangpre.parquet": ("ablangrbd_renameddatasets.parquet", None, "AbLangPre"),
        "dms_embeddedby_ablang2.parquet": ("ablangrbd_renameddatasets.parquet", None, "AbLang2"),
        "dms_embeddedby_ablang-heavy.parquet": ("ablangrbd_renameddatasets.parquet", None, "AbLang-Heavy"),
        "dms_embeddedby_antiberty.parquet": ("ablangrbd_renameddatasets.parquet", None, "AntiBERTy"),
        "dms_embeddedby_balm.parquet": ("ablangrbd_renameddatasets.parquet", None, "BALM"),
        "dms_embeddedby_esm-2.parquet": ("ablangrbd_renameddatasets.parquet", None, "ESM-2"),
        "dms_embeddedby_igbert.parquet": ("ablangrbd_renameddatasets.parquet", None, "IgBERT"),
        "dms_embeddedby_parapred.parquet": ("ablangrbd_renameddatasets.parquet", None, "Parapred")
    }
    
    print(f"\n{'='*60}")
    print("EMBEDDING FILE STATUS CHECK")
    print(f"{'='*60}")
    
    if not RECALCULATE_EMBEDDINGS:
        print("📂 Checking for existing embedding files (no regeneration)...")
    else:
        print("🔄 Will regenerate embedding files...")
    
    existing_files = []
    missing_files = []
    
    for output_file, (input_file, model_path, model_name) in embedding_files.items():
        if os.path.exists(output_file):
            print(f"✅ Found: {output_file}")
            existing_files.append(output_file)
        else:
            print(f"❌ Missing: {output_file}")
            missing_files.append(output_file)
    
    print(f"\n📊 Embedding Files Summary:")
    print(f"  • Existing files: {len(existing_files)}/{len(embedding_files)}")
    print(f"  • Missing files: {len(missing_files)}")
    
    if missing_files and not RECALCULATE_EMBEDDINGS:
        print(f"\n⚠️ Missing embedding files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nNote: Configurations using missing files will be skipped automatically.")
    
    print(f"\n✅ Embedding file check complete!")
    
    # =============================================================================
    # COMPREHENSIVE MODEL CONFIGURATION
    # =============================================================================
    
    # Complete configuration for all model/dataset/metric combinations
    CONFIGS = {
        # SAbDab Dataset Configurations
        "ablangpdb_sabdab_cosine": {
            "df_path": "ablangpdb_renameddatasets.parquet",
            "labels_val": "ablangpdb_train_val_label_mat.pt",
            "labels_test": "ablangpdb_train_test_label_mat.pt",
            "model_name": "AbLangPDB",
            "score_type": "cosine",
            "function": calculate_metrics.get_metrics,
            "dataset_type": "sabdab"
        },
        "ablangrbd_sabdab_cosine": {
            "df_path": "sabdab_embeddedby_ablangrbd.parquet",
            "labels_val": "ablangpdb_train_val_label_mat.pt",
            "labels_test": "ablangpdb_train_test_label_mat.pt",
            "model_name": "AbLangRBD",
            "score_type": "cosine",
            "function": calculate_metrics.get_metrics,
            "dataset_type": "sabdab"
        },
        "ablangpre_sabdab_cosine": {
            "df_path": "sabdab_embeddedby_ablangpre.parquet",
            "labels_val": "ablangpdb_train_val_label_mat.pt",
            "labels_test": "ablangpdb_train_test_label_mat.pt",
            "model_name": "AbLangPre",
            "score_type": "cosine",
            "function": calculate_metrics.get_metrics,
            "dataset_type": "sabdab"
        },
        "ablang2_sabdab_cosine": {
            "df_path": "sabdab_embeddedby_ablang2.parquet",
            "labels_val": "ablangpdb_train_val_label_mat.pt",
            "labels_test": "ablangpdb_train_test_label_mat.pt",
            "model_name": "AbLang2",
            "score_type": "cosine",
            "function": calculate_metrics.get_metrics,
            "dataset_type": "sabdab"
        },
        "ablang_heavy_sabdab_cosine": {
            "df_path": "sabdab_embeddedby_ablang-heavy.parquet",
            "labels_val": "ablangpdb_train_val_label_mat.pt",
            "labels_test": "ablangpdb_train_test_label_mat.pt",
            "model_name": "AbLang-Heavy",
            "score_type": "cosine",
            "function": calculate_metrics.get_metrics,
            "dataset_type": "sabdab"
        },
        "antiberty_sabdab_cosine": {
            "df_path": "sabdab_embeddedby_antiberty.parquet",
            "labels_val": "ablangpdb_train_val_label_mat.pt",
            "labels_test": "ablangpdb_train_test_label_mat.pt",
            "model_name": "AntiBERTy",
            "score_type": "cosine",
            "function": calculate_metrics.get_metrics,
            "dataset_type": "sabdab"
        },
        "balm_sabdab_cosine": {
            "df_path": "sabdab_embeddedby_balm.parquet",
            "labels_val": "ablangpdb_train_val_label_mat.pt",
            "labels_test": "ablangpdb_train_test_label_mat.pt",
            "model_name": "BALM",
            "score_type": "cosine",
            "function": calculate_metrics.get_metrics,
            "dataset_type": "sabdab"
        },
        "esm2_sabdab_cosine": {
            "df_path": "sabdab_embeddedby_esm-2.parquet",
            "labels_val": "ablangpdb_train_val_label_mat.pt",
            "labels_test": "ablangpdb_train_test_label_mat.pt",
            "model_name": "ESM-2",
            "score_type": "cosine",
            "function": calculate_metrics.get_metrics,
            "dataset_type": "sabdab"
        },
        "igbert_sabdab_cosine": {
            "df_path": "sabdab_embeddedby_igbert.parquet",
            "labels_val": "ablangpdb_train_val_label_mat.pt",
            "labels_test": "ablangpdb_train_test_label_mat.pt",
            "model_name": "IgBERT",
            "score_type": "cosine",
            "function": calculate_metrics.get_metrics,
            "dataset_type": "sabdab"
        },
        "parapred_sabdab_cosine": {
            "df_path": "sabdab_embeddedby_parapred.parquet",
            "labels_val": "ablangpdb_train_val_label_mat.pt",
            "labels_test": "ablangpdb_train_test_label_mat.pt",
            "model_name": "Parapred",
            "score_type": "cosine",
            "function": calculate_metrics.get_metrics,
            "dataset_type": "sabdab"
        },
        "seqid_sabdab": {
            "df_path": "ablangpdb_renameddatasets.parquet",
            "labels_val": "ablangpdb_train_val_label_mat.pt",
            "labels_test": "ablangpdb_train_test_label_mat.pt",
            "model_name": "SEQID",
            "score_type": "seq_identity",
            "function": calculate_metrics.get_metrics,
            "dataset_type": "sabdab"
        },
        "cdrh3id_sabdab": {
            "df_path": "ablangpdb_renameddatasets.parquet",
            "labels_val": "ablangpdb_train_val_label_mat.pt",
            "labels_test": "ablangpdb_train_test_label_mat.pt",
            "model_name": "CDRH3ID",
            "score_type": "cdrh3_identity",
            "function": calculate_metrics.get_metrics,
            "dataset_type": "sabdab"
        },
        
        # DMS Dataset Configurations
        "ablangpdb_dms_cosine": {
            "df_path": "dms_embeddedby_ablangpdb.parquet",
            "labels_val": "dms_train_val_label_mat.pt",
            "labels_test": "dms_train_test_label_mat.pt",
            "model_name": "AbLangPDB",
            "score_type": "cosine",
            "function": calculate_metrics_dms.get_metrics_dms,
            "dataset_type": "dms"
        },
        "ablangrbd_dms_cosine": {
            "df_path": "ablangrbd_renameddatasets.parquet",
            "labels_val": "dms_train_val_label_mat.pt",
            "labels_test": "dms_train_test_label_mat.pt",
            "model_name": "AbLangRBD",
            "score_type": "cosine",
            "function": calculate_metrics_dms.get_metrics_dms,
            "dataset_type": "dms"
        },
        "ablangpre_dms_cosine": {
            "df_path": "dms_embeddedby_ablangpre.parquet",
            "labels_val": "dms_train_val_label_mat.pt",
            "labels_test": "dms_train_test_label_mat.pt",
            "model_name": "AbLangPre",
            "score_type": "cosine",
            "function": calculate_metrics_dms.get_metrics_dms,
            "dataset_type": "dms"
        },
        "ablang2_dms_cosine": {
            "df_path": "dms_embeddedby_ablang2.parquet",
            "labels_val": "dms_train_val_label_mat.pt",
            "labels_test": "dms_train_test_label_mat.pt",
            "model_name": "AbLang2",
            "score_type": "cosine",
            "function": calculate_metrics_dms.get_metrics_dms,
            "dataset_type": "dms"
        },
        "ablang_heavy_dms_cosine": {
            "df_path": "dms_embeddedby_ablang-heavy.parquet",
            "labels_val": "dms_train_val_label_mat.pt",
            "labels_test": "dms_train_test_label_mat.pt",
            "model_name": "AbLang-Heavy",
            "score_type": "cosine",
            "function": calculate_metrics_dms.get_metrics_dms,
            "dataset_type": "dms"
        },
        "antiberty_dms_cosine": {
            "df_path": "dms_embeddedby_antiberty.parquet",
            "labels_val": "dms_train_val_label_mat.pt",
            "labels_test": "dms_train_test_label_mat.pt",
            "model_name": "AntiBERTy",
            "score_type": "cosine",
            "function": calculate_metrics_dms.get_metrics_dms,
            "dataset_type": "dms"
        },
        "balm_dms_cosine": {
            "df_path": "dms_embeddedby_balm.parquet",
            "labels_val": "dms_train_val_label_mat.pt",
            "labels_test": "dms_train_test_label_mat.pt",
            "model_name": "BALM",
            "score_type": "cosine",
            "function": calculate_metrics_dms.get_metrics_dms,
            "dataset_type": "dms"
        },
        "esm2_dms_cosine": {
            "df_path": "dms_embeddedby_esm-2.parquet",
            "labels_val": "dms_train_val_label_mat.pt",
            "labels_test": "dms_train_test_label_mat.pt",
            "model_name": "ESM-2",
            "score_type": "cosine",
            "function": calculate_metrics_dms.get_metrics_dms,
            "dataset_type": "dms"
        },
        "igbert_dms_cosine": {
            "df_path": "dms_embeddedby_igbert.parquet",
            "labels_val": "dms_train_val_label_mat.pt",
            "labels_test": "dms_train_test_label_mat.pt",
            "model_name": "IgBERT",
            "score_type": "cosine",
            "function": calculate_metrics_dms.get_metrics_dms,
            "dataset_type": "dms"
        },
        "parapred_dms_cosine": {
            "df_path": "dms_embeddedby_parapred.parquet",
            "labels_val": "dms_train_val_label_mat.pt",
            "labels_test": "dms_train_test_label_mat.pt",
            "model_name": "Parapred",
            "score_type": "cosine",
            "function": calculate_metrics_dms.get_metrics_dms,
            "dataset_type": "dms"
        },
        "seqid_dms": {
            "df_path": "ablangrbd_renameddatasets.parquet",
            "labels_val": "dms_train_val_label_mat.pt",
            "labels_test": "dms_train_test_label_mat.pt",
            "model_name": "SEQID",
            "score_type": "seq_identity",
            "function": calculate_metrics_dms.get_metrics_dms,
            "dataset_type": "dms"
        },
        "cdrh3id_dms": {
            "df_path": "ablangrbd_renameddatasets.parquet",
            "labels_val": "dms_train_val_label_mat.pt",
            "labels_test": "dms_train_test_label_mat.pt",
            "model_name": "CDRH3ID",
            "score_type": "cdrh3_identity",
            "function": calculate_metrics_dms.get_metrics_dms,
            "dataset_type": "dms"
        }
    }
    
    print(f"\n🔧 Configured {len(CONFIGS)} model/dataset/metric combinations:")
    for name, config in CONFIGS.items():
        print(f"  • {name}: {config['model_name']} on {config['dataset_type']} using {config['score_type']}")
    
    # =============================================================================
    # FILTER AVAILABLE CONFIGURATIONS
    # =============================================================================
    
    print(f"\n{'='*60}")
    print("FILTERING AVAILABLE CONFIGURATIONS")
    print(f"{'='*60}")
    
    # Check which configurations can actually run based on available files
    available_configs = {}
    missing_configs = []
    
    for config_name, config in CONFIGS.items():
        files_to_check = [config["df_path"], config["labels_val"], config["labels_test"]]
        missing_files = [f for f in files_to_check if not os.path.exists(f)]
        
        if not missing_files:
            available_configs[config_name] = config
            print(f"✅ {config_name}: Ready to run")
        else:
            missing_configs.append(config_name)
            print(f"❌ {config_name}: Missing files - {missing_files}")
    
    print(f"\n📊 Summary:")
    print(f"  • Available configurations: {len(available_configs)}/{len(CONFIGS)}")
    print(f"  • Missing configurations: {len(missing_configs)}")
    
    if missing_configs:
        print(f"\n⚠️ Configurations that will be skipped: {', '.join(missing_configs)}")
    
    if not available_configs:
        print("❌ No configurations are available to run!")
        sys.exit(1)
    
    # =============================================================================
    # PRE-COMPUTED THRESHOLDS
    # =============================================================================
    
    # Pre-computed thresholds to skip threshold optimization
    PRECOMPUTED_THRESHOLDS = {
        # Existing pre-computed thresholds (keep these values)
        "ablangpdb_sabdab_cosine": {
            "epitope_threshold": 0.5037,
            "antigen_threshold": 0.2697
        },
        "ablangrbd_sabdab_cosine": {
            "epitope_threshold": 0.7912,
            "antigen_threshold": -0.2969
        },
        "ablangpre_sabdab_cosine": {
            "epitope_threshold": 0.6941,
            "antigen_threshold": 0.5851
        },
        "parapred_sabdab_cosine": {
            "epitope_threshold": 0.9985,
            "antigen_threshold": 0.9974
        },
        "seqid_sabdab": {
            "epitope_threshold": 0.6684,
            "antigen_threshold": 0.3380
        },
        "cdrh3id_sabdab": {
            "epitope_threshold": 0.2727,
            "antigen_threshold": 0.0000
        },
        "ablangpdb_dms_cosine": {
            "epitope_threshold": -0.0419
        },
        "ablangrbd_dms_cosine": {
            "epitope_threshold": 0.8493
        },
        "ablangpre_dms_cosine": {
            "epitope_threshold": 0.6608
        },
        "parapred_dms_cosine": {
            "epitope_threshold": 0.9888
        },
        "seqid_dms": {
            "epitope_threshold": 0.6497
        },
        "cdrh3id_dms": {
            "epitope_threshold": 0.1905
        }
    }
    
    use_precomputed = len(PRECOMPUTED_THRESHOLDS) > 0
    if use_precomputed:
        print(f"\n🔄 Using precomputed thresholds for {len(PRECOMPUTED_THRESHOLDS)} configurations")
        for config_name, thresholds in PRECOMPUTED_THRESHOLDS.items():
            print(f"  • {config_name}: {thresholds}")
    else:
        print("🆕 Computing fresh thresholds for all configurations")
    
    # =============================================================================
    # RUN COMPREHENSIVE BENCHMARKS
    # =============================================================================
    
    execution_results = {}
    failed_configs = []
    
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE BENCHMARK EXECUTION")
    print(f"{'='*70}")
    print(f"Total configurations to run: {len(available_configs)}")
    
    for i, (config_name, config) in enumerate(available_configs.items(), 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(available_configs)}] Running: {config_name}")
        print(f"Model: {config['model_name']}, Dataset: {config['dataset_type']}, Score: {config['score_type']}")
        print(f"{'='*70}")
        
        try:
            # Prepare arguments
            args = {
                "df_path": config["df_path"],
                "labels_file_val": config["labels_val"],
                "labels_file_test": config["labels_test"],
                "score_type": config["score_type"],
                "model_name": config["model_name"],
                "output_folder": OUTPUT_FOLDER
            }
            
            # Add precomputed thresholds if available
            if config_name in PRECOMPUTED_THRESHOLDS:
                thresholds = PRECOMPUTED_THRESHOLDS[config_name]
                if config["dataset_type"] == "sabdab":
                    if "epitope_threshold" in thresholds:
                        args["epitope_threshold"] = thresholds["epitope_threshold"]
                    if "antigen_threshold" in thresholds:
                        args["antigen_threshold"] = thresholds["antigen_threshold"]
                elif config["dataset_type"] == "dms":
                    if "epitope_threshold" in thresholds:
                        args["epitope_threshold"] = thresholds["epitope_threshold"]
            
            # Execute the benchmark
            config["function"](**args)
            
            execution_results[config_name] = "✅ Success"
            print(f"\n✅ [{i}/{len(available_configs)}] {config_name} completed successfully!")
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            execution_results[config_name] = error_msg
            failed_configs.append(config_name)
            print(f"\n❌ [{i}/{len(available_configs)}] {config_name} failed: {str(e)}")
            print("Traceback:")
            print(traceback.format_exc())
            continue
    
    print(f"\n{'='*70}")
    print("COMPREHENSIVE BENCHMARK EXECUTION SUMMARY")
    print(f"{'='*70}")
    
    for config_name, result in execution_results.items():
        print(f"{config_name:30} {result}")
    
    successful_configs = len(available_configs) - len(failed_configs)
    print(f"\n📊 Results:")
    print(f"  • Successful: {successful_configs}/{len(available_configs)}")
    print(f"  • Failed: {len(failed_configs)}")
    
    if failed_configs:
        print(f"\n⚠️ Failed configurations: {', '.join(failed_configs)}")
    else:
        print("\n🎉 All available configurations completed successfully!")
    
    # =============================================================================
    # GENERATE COMPREHENSIVE EXCEL REPORT
    # =============================================================================
    
    print(f"\n{'='*70}")
    print("GENERATING COMPREHENSIVE EXCEL REPORT")
    print(f"{'='*70}")
    
    try:
        # Generate summary statistics
        print("\n📊 Generating summary statistics...")
        print_summary_stats(OUTPUT_FOLDER)
        
        # Generate the Excel file
        excel_path = generate_results_excel(
            output_folder=OUTPUT_FOLDER,
            excel_filename=EXCEL_FILENAME
        )
        
        print(f"\n🎉 Comprehensive Excel report generated successfully!")
        print(f"📁 File location: {excel_path}")
        print(f"📏 File size: {os.path.getsize(excel_path):,} bytes")
        
        # Provide usage instructions
        print(f"\n📖 Excel Report Contents:")
        print(f"  • Models as rows (AbLangPDB, AbLangRBD, AbLangPre, etc.)")
        print(f"  • Datasets grouped as column headers (SAbDab, DMS)")
        print(f"  • Metrics: ROC-AUC, Average Precision, F1 Score")
        print(f"  • Best performance: Bold formatting")
        print(f"  • Second best: Italic formatting")
        print(f"  • Values rounded to 4 decimal places")
        
    except Exception as e:
        print(f"❌ Error generating Excel report: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        print("\nDebugging information:")
        print(f"  • Output folder: {OUTPUT_FOLDER}")
        if os.path.exists(OUTPUT_FOLDER):
            print(f"  • Files in folder: {len(os.listdir(OUTPUT_FOLDER))}")
            
            # List summary files found
            import glob
            summary_files = glob.glob(os.path.join(OUTPUT_FOLDER, "*summarymetrics.txt"))
            print(f"  • Summary files found: {len(summary_files)}")
            for f in summary_files[:5]:  # Show first 5
                print(f"    - {os.path.basename(f)}")
            if len(summary_files) > 5:
                print(f"    - ... and {len(summary_files)-5} more")
        else:
            print(f"  • Output folder does not exist!")
    
    # =============================================================================
    # FINAL SUMMARY AND ANALYSIS
    # =============================================================================
    
    print(f"\n{'='*70}")
    print("COMPREHENSIVE PIPELINE COMPLETION SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n🔧 Configuration:")
    print(f"  • Recalculated embeddings: {RECALCULATE_EMBEDDINGS}")
    print(f"  • Used precomputed thresholds: {use_precomputed}")
    print(f"  • Batch size: {BATCH_SIZE}")
    
    print(f"\n📈 Benchmarking Results:")
    print(f"  • Total configurations possible: {len(CONFIGS)}")
    print(f"  • Configurations attempted: {len(available_configs)}")
    print(f"  • Successful runs: {successful_configs}")
    print(f"  • Failed runs: {len(failed_configs)}")
    
    if os.path.exists(os.path.join(OUTPUT_FOLDER, EXCEL_FILENAME)):
        print(f"\n📊 Excel Report:")
        print(f"  • Status: ✅ Generated successfully")
        print(f"  • Location: {os.path.join(OUTPUT_FOLDER, EXCEL_FILENAME)}")
        print(f"  • Ready for analysis and sharing")
    else:
        print(f"\n📊 Excel Report:")
        print(f"  • Status: ❌ Generation failed")
        print(f"  • Check error messages above")
    
    print(f"\n🔬 Models Configured for Benchmarking:")
    all_models = set()
    for config_name, config in CONFIGS.items():
        all_models.add(f"{config['model_name']} ({config['score_type']})")
            
    for model in sorted(all_models):
        print(f"  • {model}")
    
    print(f"\n📊 Datasets Configured:")
    datasets_configured = set()
    for config_name, config in CONFIGS.items():
        datasets_configured.add(config['dataset_type'].upper())
            
    for dataset in sorted(datasets_configured):
        print(f"  • {dataset}")
    
    print(f"\n🎯 Next Steps:")
    print(f"  1. 📊 Open the Excel report for comprehensive performance comparison")
    print(f"  2. 🔍 Identify best-performing models for each dataset")
    print(f"  3. 📈 Analyze performance patterns across different similarity metrics")
    print(f"  4. 📋 Share results with your research team")
    print(f"  5. 📝 Consider additional analyses based on findings")
    
    if failed_configs:
        print(f"\n⚠️ Failed Configurations to Investigate:")
        for config in failed_configs:
            print(f"  • {config}: {execution_results[config]}")
    
    if missing_configs:
        print(f"\n❓ Configurations Not Attempted (Missing Files):")
        for config in missing_configs:
            print(f"  • {config}")
    
    print(f"\n💡 Model Coverage Summary:")
    print(f"  • Total unique models: {len(all_models)}")
    print(f"  • Embedding-based models: AbLangPDB, AbLangRBD, AbLangPre, AbLang2,")
    print(f"    AbLang-Heavy, AntiBERTy, BALM, ESM-2, IgBERT, Parapred")
    print(f"  • Sequence-based models: SEQID, CDRH3ID")
    print(f"  • Total configurations: {len(CONFIGS)}")
    
    print(f"\n🏁 Comprehensive benchmarking pipeline completed!")
    print(f"\n📄 Report: {os.path.join(OUTPUT_FOLDER, EXCEL_FILENAME)}")
    
    # Return success status
    return successful_configs > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
