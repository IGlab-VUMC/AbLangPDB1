# AbLangPDB1: Contrastive-Learned Antibody Embeddings for Epitope Overlap Predictions

This repository contains the implementation, benchmarking code, and usage examples for **AbLangPDB1**, a fine-tuned antibody language model for generating epitope-information-rich embeddings of antibodies.

## 📄 Publication

> **Contrastive Learning Enables Epitope Overlap Predictions for Targeted Antibody Discovery**  
> Clinton M. Holt, Alexis K. Janke, Parastoo Amlashi, Parker J. Jamieson, Toma M. Marinov, Ivelin S. Georgiev  
> *bioRxiv*, 2025. https://doi.org/10.1101/2025.02.25.640114

Vanderbilt Center for Antibody Therapeutics, Vanderbilt University Medical Center, Nashville, TN 37232, USA

## 🔬 Model Description

**AbLangPDB1** is designed to predict epitope overlap between antibodies by generating high-quality embeddings that capture epitope-specificity information. The model uses contrastive learning on paired heavy and light chain sequences to learn representations where antibodies targeting similar epitopes cluster together in embedding space.

### Architecture

```
Heavy Chain Seq → [AbLang Heavy] → 768-dim → |
                                              | → [Concatenate] → [Mixer Network] → 1536-dim Paired Embedding
Light Chain Seq → [AbLang Light] → 768-dim → |
```

The model processes heavy and light chains independently using pre-trained [AbLang](https://huggingface.co/qilowoq/AbLang_heavy) models, then fuses their embeddings through a custom Mixer network (6 fully connected layers) to produce a unified 1536-dimensional embedding.

## 🎯 Intended Uses

1. **Epitope Classification**: Compare antibodies with unknown epitopes against reference databases to find the most similar epitope targets
2. **Antibody Search**: Identify antibodies with similar epitope specificity in large sequence databases
3. **Therapeutic Discovery**: Find candidate antibodies that may target the same epitope as a reference therapeutic

## 📊 Training Data

- **Source**: 1,909 non-redundant human antibodies from [Structural Antibody Database (SAbDab)](https://doi.org/10.1093/nar/gkt1043)
- **Cutoff Date**: February 19, 2024
- **Antigen Assignment**: Pfam domain-based categorization using [pfam_scan](https://github.com/aziele/pfam_scan)
- **Data Splits**: 80% training, 10% validation, 10% test (clone-group aware splitting)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone this repository
git clone https://github.com/your-username/AbLangPDB1.git
cd AbLangPDB1

# Install dependencies
pip install torch pandas "transformers>=4.30.0" safetensors
```

### 2. Download Model Weights

```bash
# Download the model weights from HuggingFace
curl -L https://huggingface.co/clint-holt/AbLangPDB1/resolve/main/ablangpdb_model.safetensors?download=true -o ablangpdb_model.safetensors

# Alternatively, you can download the entire model repository:
# git clone https://huggingface.co/clint-holt/AbLangPDB1 model_weights
```

### 3. Basic Usage

```python
import torch
import pandas as pd
from transformers import AutoTokenizer
from ablangpaired_model import AbLangPaired, AbLangPairedConfig

# Setup device and load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_config = AbLangPairedConfig(checkpoint_filename="ablangpdb_model.safetensors")
model = AbLangPaired(model_config, device).to(device)
model.eval()

# Load tokenizers (download from HuggingFace if needed)
heavy_tokenizer = AutoTokenizer.from_pretrained("clint-holt/AbLangPDB1", subfolder="heavy_tokenizer")
light_tokenizer = AutoTokenizer.from_pretrained("clint-holt/AbLangPDB1", subfolder="light_tokenizer")

# Prepare your antibody sequences
data = {
    'HC_AA': ["EVQLVESGGGLVQPGGSLRLSCAASGFNLYYYSIHWVRQAPGKGLEWVASISPYSSSTSYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCARGRWYRRALDYWGQGTLVTVSS"],
    'LC_AA': ["DIQMTQSPSSLSASVGDRVTITCRASQSVSSAVAWYQQKPGKAPKLLIYSASSLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQYPYYSSLITFGQGTKVEIK"]
}
df = pd.DataFrame(data)

# Preprocess sequences (add spaces between amino acids)
df["PREPARED_HC_SEQ"] = df["HC_AA"].apply(lambda x: " ".join(list(x)))
df["PREPARED_LC_SEQ"] = df["LC_AA"].apply(lambda x: " ".join(list(x)))

# Tokenize and generate embeddings
h_tokens = heavy_tokenizer(df["PREPARED_HC_SEQ"].tolist(), padding='longest', return_tensors="pt")
l_tokens = light_tokenizer(df["PREPARED_LC_SEQ"].tolist(), padding='longest', return_tensors="pt")

with torch.no_grad():
    embeddings = model(
        h_input_ids=h_tokens['input_ids'].to(device),
        h_attention_mask=h_tokens['attention_mask'].to(device),
        l_input_ids=l_tokens['input_ids'].to(device),
        l_attention_mask=l_tokens['attention_mask'].to(device)
    )

print(f"Generated embeddings shape: {embeddings.shape}")  # Expected: (1, 1536)
```

## 📈 Benchmarking

This repository includes comprehensive benchmarking code to evaluate AbLangPDB1 against other methods:

### Available Scripts

- `benchmarking/calculate_metrics.py`: Evaluation on SAbDab dataset with continuous epitope/antigen labels
- `benchmarking/calculate_metrics_dms.py`: Evaluation on DMS dataset with binary epitope labels
- `benchmarking/get_metrics.ipynb`: Interactive notebook for running benchmarks

### Benchmark Datasets

- **SAbDab Dataset**: Structural antibody database with epitope/antigen overlap labels
- **DMS Dataset**: Deep mutational scanning data with binary epitope matching

### Metrics

- **ROC-AUC**: Area under the receiver operating characteristic curve
- **Average Precision**: Area under the precision-recall curve
- **F1 Score**: Harmonic mean of precision and recall


## ⚠️ Limitations

- **Species**: Optimized for human antibodies; reduced accuracy expected for mouse BCRs
- **Domain Coverage**: Lower performance for antibodies targeting domains not included in training (non-Pfam domains)
- **Custom Architecture**: Requires the provided `ablangpaired_model.py` implementation (not compatible with standard HuggingFace model loading)

## 🔗 Model Weights & Resources

- **HuggingFace Model**: [clint-holt/AbLangPDB1](https://huggingface.co/clint-holt/AbLangPDB1)
- **Direct Download**: 
  ```bash
  curl -L https://huggingface.co/clint-holt/AbLangPDB1/resolve/main/ablangpdb_model.safetensors?download=true -o ablangpdb_model.safetensors
  ```
- **Paper**: [bioRxiv link](https://doi.org/10.1101/2025.02.25.640114)

## 📚 Citation

If you use this model or code in your research, please cite our paper:

```bibtex
@article{Holt2025.02.25.640114,
    author = {Holt, Clinton M. and Janke, Alexis K. and Amlashi, Parastoo and Jamieson, Parker J. and Marinov, Toma M. and Georgiev, Ivelin S.},
    title = {Contrastive Learning Enables Epitope Overlap Predictions for Targeted Antibody Discovery},
    elocation-id = {2025.02.25.640114},
    year = {2025},
    doi = {10.1101/2025.02.25.640114},
    publisher = {Cold Spring Harbor Laboratory},
    URL = {https://www.biorxiv.org/content/early/2025/04/01/2025.02.25.640114},
    eprint = {https://www.biorxiv.org/content/early/2025/04/01/2025.02.25.640114.full.pdf},
    journal = {bioRxiv}
}
```

## 📝 License

This project is licensed under the MIT License.