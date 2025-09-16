#!/usr/bin/env python3
"""
AbLangPDB1 Quick Start Example

This script demonstrates how to use AbLangPDB1 to generate embeddings for antibody sequences.
Run this script after downloading the model weights to verify your installation.

Usage:
    python quick_start_example.py

Requirements:
    - torch
    - pandas  
    - transformers
    - safetensors
    - ablangpdb_model.safetensors (download from HuggingFace)
"""

import torch
import pandas as pd
from transformers import AutoTokenizer
from ablangpaired_model import AbLangPaired, AbLangPairedConfig
import os
import sys

def main():
    print("🧬 AbLangPDB1 Quick Start Example")
    print("=" * 50)
    
    # Check if model weights exist
    model_path = "ablangpdb_model.safetensors"
    if not os.path.exists(model_path):
        print(f"❌ Model weights not found: {model_path}")
        print("\n📥 Download the model weights from HuggingFace:")
        print("🤗 HuggingFace Page: https://huggingface.co/clint-holt/AbLangPDB1")
        print("\n📋 Download options:")
        print("Option 1 (curl):")
        print('  curl -L "https://huggingface.co/clint-holt/AbLangPDB1/resolve/main/ablangpdb_model.safetensors?download=true" -o ablangpdb_model.safetensors')
        print("\nOption 2 (huggingface_hub):")
        print("  pip install huggingface_hub")
        print("  python -c \"from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='clint-holt/AbLangPDB1', filename='ablangpdb_model.safetensors', local_dir='.')\"")
        print("\n💡 Model size: 738MB - Please ensure you have sufficient disk space and internet bandwidth.")
        sys.exit(1)
    
    print(f"✅ Found model weights: {model_path}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Load model
    print("\n🔄 Loading model...")
    try:
        config = AbLangPairedConfig(checkpoint_filename=model_path)
        model = AbLangPaired(config, device).to(device)
        model.eval()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    # Load tokenizers
    print("\n🔄 Loading tokenizers...")
    try:
        heavy_tokenizer = AutoTokenizer.from_pretrained("clint-holt/AbLangPDB1", subfolder="heavy_tokenizer")
        light_tokenizer = AutoTokenizer.from_pretrained("clint-holt/AbLangPDB1", subfolder="light_tokenizer")
        print("✅ Tokenizers loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading tokenizers: {e}")
        print("📡 This requires internet connection to download from HuggingFace")
        sys.exit(1)
    
    # Example antibody sequences
    print("\n🧬 Processing example antibody sequences...")
    
    # Example from the paper (SARS-CoV-2 antibody)
    heavy_chain = "EVQLVESGGGLVQPGGSLRLSCAASGFNLYYYSIHWVRQAPGKGLEWVASISPYSSSTSYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCARGRWYRRALDYWGQGTLVTVSS"
    light_chain = "DIQMTQSPSSLSASVGDRVTITCRASQSVSSAVAWYQQKPGKAPKLLIYSASSLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQYPYYSSLITFGQGTKVEIK"
    
    print(f"Heavy chain: {heavy_chain[:50]}...")
    print(f"Light chain: {light_chain[:50]}...")
    
    # Tokenize sequences (add spaces between amino acids)
    print("\n🔄 Tokenizing sequences...")
    h_tokens = heavy_tokenizer(" ".join(heavy_chain), return_tensors="pt")
    l_tokens = light_tokenizer(" ".join(light_chain), return_tensors="pt")
    
    # Generate embedding
    print("\n🔄 Generating embedding...")
    try:
        with torch.no_grad():
            embedding = model(
                h_input_ids=h_tokens['input_ids'].to(device),
                h_attention_mask=h_tokens['attention_mask'].to(device),
                l_input_ids=l_tokens['input_ids'].to(device),
                l_attention_mask=l_tokens['attention_mask'].to(device)
            )
        
        print(f"✅ Generated embedding shape: {embedding.shape}")
        print(f"📊 Embedding dimensionality: {embedding.shape[1]}")
        print(f"🔢 First 5 embedding values: {embedding[0][:5].tolist()}")
        
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        sys.exit(1)
    
    # Demonstrate batch processing
    print("\n🔄 Demonstrating batch processing with multiple antibodies...")
    
    # Create a small dataset with multiple antibodies
    example_data = {
        'HC_AA': [
            heavy_chain,  # Example 1
            "QVQLVQSGAEVKKPGSSVKVSCKASGGTFSSYWIEWVRQAPGQGLEWMGIIYPILSEGSTKYYNEKFKDRATLSADTSTSTAYMELSSLTSEDTAVYYCARGGAYYGSGYYAMDYWGQGTLVTVSS",  # Example 2
            "EVQLLESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCARYHGGDAMDYWGQGTLVTVSS"  # Example 3
        ],
        'LC_AA': [
            light_chain,  # Example 1
            "EIVLTQSPGTLSLSPGERATLSCRASQSVSSSYLAWYQQKPGQAPRLLIYGASSRATGIPDRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGSSPLTFGAGTKVEIK",  # Example 2
            "DIQMTQSPSSLSASVGDRVTITCRASQSISSWLAWYQQKPGKAPKLLIYKASSLESGVPSRFSGSGSGTEFTLTISSLQPDDFATYYCQQYNSYSYTFGQGTKVEIK"  # Example 3
        ]
    }
    
    df = pd.DataFrame(example_data)
    
    # Preprocess sequences
    df["PREPARED_HC_SEQ"] = df["HC_AA"].apply(lambda x: " ".join(list(x)))
    df["PREPARED_LC_SEQ"] = df["LC_AA"].apply(lambda x: " ".join(list(x)))
    
    # Tokenize batch
    h_tokens_batch = heavy_tokenizer(df["PREPARED_HC_SEQ"].tolist(), padding='longest', return_tensors="pt")
    l_tokens_batch = light_tokenizer(df["PREPARED_LC_SEQ"].tolist(), padding='longest', return_tensors="pt")
    
    # Generate embeddings for batch
    with torch.no_grad():
        embeddings_batch = model(
            h_input_ids=h_tokens_batch['input_ids'].to(device),
            h_attention_mask=h_tokens_batch['attention_mask'].to(device),
            l_input_ids=l_tokens_batch['input_ids'].to(device),
            l_attention_mask=l_tokens_batch['attention_mask'].to(device)
        )
    
    print(f"✅ Generated batch embeddings shape: {embeddings_batch.shape}")
    print(f"📊 Number of antibodies processed: {embeddings_batch.shape[0]}")
    
    # Calculate pairwise similarities
    print("\n🔄 Calculating pairwise similarities...")
    similarities = torch.cosine_similarity(embeddings_batch.unsqueeze(1), embeddings_batch.unsqueeze(0), dim=2)
    
    print("📊 Cosine similarity matrix:")
    for i in range(similarities.shape[0]):
        row = " ".join([f"{similarities[i][j]:.3f}" for j in range(similarities.shape[1])])
        print(f"   Ab{i+1}: [{row}]")
    
    print("\n🎉 Quick start example completed successfully!")
    print("\n📚 Next steps:")
    print("   • See pdb_inference_examples.ipynb for more detailed examples")
    print("   • Check benchmarking/ directory for evaluation scripts")
    print("   • Visit https://huggingface.co/clint-holt/AbLangPDB1 for more resources")

if __name__ == "__main__":
    main()