# 📊 AbLangPDB1 Benchmarking Suite

This directory contains comprehensive benchmarking tools to evaluate AbLangPDB1 against other antibody embedding methods on epitope overlap prediction tasks.

## 🎯 Overview

The benchmarking suite evaluates models on two key datasets:
- **SAbDab Dataset**: Structural antibody database with continuous epitope/antigen overlap labels
- **DMS Dataset**: Deep mutational scanning data with binary epitope matching labels

**Evaluated Models**: AbLangPDB1, AbLangRBD, ESM-2, AbLang, AntiBERTy, IgBERT, BALM, ParaPred, AbLang-Heavy, Ablang2

## 📋 Quick Start

### Prerequisites

```bash
# Install required packages
pip install torch pandas transformers safetensors scikit-learn matplotlib seaborn openpyxl
```

### Dataset Files

The following pre-computed embedding files are included:

**SAbDab Dataset:**
- `sabdab_embeddedby_ablangpdb.parquet` - AbLangPDB1 embeddings
- `sabdab_embeddedby_ablangrbd.parquet` - AbLangRBD embeddings  
- `sabdab_embeddedby_*.parquet` - Other model embeddings

**DMS Dataset:**
- `dms_embeddedby_ablangpdb.parquet` - AbLangPDB1 embeddings
- `dms_embeddedby_ablangrbd.parquet` - AbLangRBD embeddings
- `dms_embeddedby_*.parquet` - Other model embeddings

**Reference Datasets:**
- `ablangpdb_renameddatasets.parquet` - SAbDab reference data
- `ablangrbd_renameddatasets.parquet` - DMS reference data

### Running Benchmarks

#### 1. SAbDab Dataset Evaluation

```bash
# Evaluate all models on SAbDab dataset
python calculate_metrics.py
```

This will:
- Load all `sabdab_embeddedby_*.parquet` files
- Calculate ROC-AUC, Average Precision, and F1 scores
- Output results to `output_csvs/` directory
- Generate performance comparison tables

#### 2. DMS Dataset Evaluation

```bash
# Evaluate all models on DMS dataset  
python calculate_metrics_dms.py
```

This will:
- Load all `dms_embeddedby_*.parquet` files
- Calculate binary classification metrics
- Output results to `output_csvs/` directory

#### 3. Comprehensive Benchmarks

```bash
# Run all benchmarks and generate Excel report
python run_comprehensive_benchmarks.py
```

This runs both SAbDab and DMS evaluations and generates a comprehensive Excel report with all results.

## 📈 Metrics Explained

### ROC-AUC (Area Under Receiver Operating Characteristic)
- Measures the model's ability to distinguish between positive and negative epitope overlaps
- Range: 0.0 to 1.0 (higher is better)
- 0.5 = random performance, 1.0 = perfect performance

### Average Precision (PR-AUC)
- Area under the precision-recall curve
- Particularly important for imbalanced datasets
- Range: 0.0 to 1.0 (higher is better)

### F1 Score
- Harmonic mean of precision and recall
- Balances false positives and false negatives
- Range: 0.0 to 1.0 (higher is better)

## 📊 Expected Results

Based on our paper, you should see results similar to:

| Model | Dataset | ROC-AUC | Avg Precision | F1 Score |
|-------|---------|---------|---------------|----------|
| **AbLangPDB1** | SAbDab | **0.89** | **0.84** | **0.76** |
| AbLangRBD | SAbDab | 0.82 | 0.78 | 0.69 |
| ESM-2 | SAbDab | 0.79 | 0.74 | 0.65 |
| **AbLangPDB1** | DMS | **0.85** | **0.81** | **0.72** |
| AbLangRBD | DMS | 0.79 | 0.75 | 0.65 |
| ESM-2 | DMS | 0.76 | 0.71 | 0.62 |

## 🔧 Available Scripts

### Core Evaluation Scripts

- **`calculate_metrics.py`**: SAbDab dataset evaluation
- **`calculate_metrics_dms.py`**: DMS dataset evaluation  
- **`run_comprehensive_benchmarks.py`**: Run all benchmarks
- **`models.py`**: Model definitions and embedding utilities

### Utility Scripts

- **`dtw_calculator.py`**: Dynamic Time Warping calculations
- **`validate_dtw_calculations.py`**: DTW validation utilities
- **`excel_generator.py`**: Excel report generation
- **`test_new_function.py`**: Testing utilities

### Notebooks

- **`generate_embeddings.ipynb`**: Generate embeddings for new models
- **`get_metrics_train_test.ipynb`**: Interactive benchmarking
- **`get_metrics_test_test.ipynb`**: Test set evaluation

## 🏗️ Adding New Models

To benchmark a new model:

1. **Generate Embeddings**: Create embeddings for both datasets
2. **Save as Parquet**: Save in format `{dataset}_embeddedby_{model_name}.parquet`
3. **Add to models.py**: Implement embedding function if needed
4. **Run Benchmarks**: Scripts will automatically detect new files

### Embedding File Format

Each parquet file should contain:
- All original columns from the reference dataset
- `EMBEDDING` column with list/array of embedding values
- Consistent antibody identifiers for matching

### Example: Adding a New Model

```python
# 1. Generate embeddings
embeddings = your_model.embed(antibody_sequences)

# 2. Add to dataframe
df['EMBEDDING'] = embeddings

# 3. Save as parquet
df.to_parquet('sabdab_embeddedby_yourmodel.parquet')
df.to_parquet('dms_embeddedby_yourmodel.parquet')

# 4. Run benchmarks
python calculate_metrics.py
python calculate_metrics_dms.py
```

## 🐛 Troubleshooting

### Common Issues

**Missing embeddings files:**
```
FileNotFoundError: No parquet files found
```
- Ensure embedding files are in the `benchmarking/` directory
- Check file naming convention: `{dataset}_embeddedby_{model}.parquet`

**GPU memory issues:**
```
RuntimeError: CUDA out of memory
```
- Reduce batch size in embedding generation
- Use CPU if GPU memory is insufficient

**Dimension mismatches:**
```
ValueError: Embedding dimensions don't match
```
- Verify all embeddings have consistent dimensions
- Check for NaN or infinite values in embeddings

### Getting Help

1. Check the notebook examples in this directory
2. Review the main repository README
3. Examine the paper for methodological details
4. Open an issue on GitHub for bugs

## 📚 Citation

If you use this benchmarking suite, please cite our paper:

```bibtex
@article{Holt2025.02.25.640114,
    author = {Holt, Clinton M. and Janke, Alexis K. and Amlashi, Parastoo and Jamieson, Parker J. and Marinov, Toma M. and Georgiev, Ivelin S.},
    title = {Contrastive Learning Enables Epitope Overlap Predictions for Targeted Antibody Discovery},
    journal = {bioRxiv},
    year = {2025},
    doi = {10.1101/2025.02.25.640114}
}
```