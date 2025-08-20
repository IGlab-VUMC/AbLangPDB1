"""
Enhanced tokenization and embedding system for antibody sequences.

This module provides improved tokenization that removes the 157 amino acid length restriction
and implements dynamic batch-wise padding for better memory efficiency.
"""

# Standard library imports
import sys
import os
from typing import List, Dict, Tuple

# Data processing imports
import pandas as pd
from tqdm import tqdm

# PyTorch imports
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

# Hugging Face Transformers imports
from transformers import AutoTokenizer

# Local import
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from ablangpaired_model import AbLangPairedConfig


class AntibodySequenceDataset(Dataset):
    """
    Custom Dataset class for antibody sequences that supports dynamic padding.
    """
    
    def __init__(self, df: pd.DataFrame, heavy_tokenizer, light_tokenizer, max_length: int = 159):
        """
        Initialize the dataset.
        
        Args:
            df: DataFrame containing antibody sequences with HC_AA and LC_AA columns
            heavy_tokenizer: Tokenizer for heavy chain sequences
            light_tokenizer: Tokenizer for light chain sequences  
            max_length: Maximum sequence length (default 159 for AbLang)
        """
        self.df = df.copy()
        self.heavy_tokenizer = heavy_tokenizer
        self.light_tokenizer = light_tokenizer
        self.max_length = max_length
        
        # Prepare sequences by adding spaces between amino acids
        self.df["PREPARED_HC_SEQ"] = self.df["HC_AA"].apply(lambda x: " ".join(list(str(x))))
        self.df["PREPARED_LC_SEQ"] = self.df["LC_AA"].apply(lambda x: " ".join(list(str(x))))
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        
        Returns:
            Dictionary containing tokenized sequences and metadata
        """
        row = self.df.iloc[idx]
        
        # Tokenize heavy chain
        h_tokens = self.heavy_tokenizer(
            row["PREPARED_HC_SEQ"],
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_special_tokens_mask=True,
            padding=False  # No padding yet - will be done in collate_fn
        )
        
        # Tokenize light chain  
        l_tokens = self.light_tokenizer(
            row["PREPARED_LC_SEQ"],
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt", 
            return_special_tokens_mask=True,
            padding=False  # No padding yet - will be done in collate_fn
        )
        
        return {
            'h_input_ids': h_tokens['input_ids'].squeeze(0),
            'h_attention_mask': h_tokens['attention_mask'].squeeze(0),
            'h_special_tokens_mask': h_tokens['special_tokens_mask'].squeeze(0),
            'l_input_ids': l_tokens['input_ids'].squeeze(0),
            'l_attention_mask': l_tokens['attention_mask'].squeeze(0), 
            'l_special_tokens_mask': l_tokens['special_tokens_mask'].squeeze(0),
            'original_h_length': len(row["HC_AA"]),
            'original_l_length': len(row["LC_AA"])
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for dynamic padding within batches.
    
    Args:
        batch: List of sample dictionaries from AntibodySequenceDataset
        
    Returns:
        Dictionary containing padded batch tensors
    """
    # Extract all tensors
    h_input_ids = [item['h_input_ids'] for item in batch]
    h_attention_masks = [item['h_attention_mask'] for item in batch]  
    h_special_tokens_masks = [item['h_special_tokens_mask'] for item in batch]
    l_input_ids = [item['l_input_ids'] for item in batch]
    l_attention_masks = [item['l_attention_mask'] for item in batch]
    l_special_tokens_masks = [item['l_special_tokens_mask'] for item in batch]
    
    # Pad sequences to the longest in this batch
    h_input_ids_padded = pad_sequence(h_input_ids, batch_first=True, padding_value=0)
    h_attention_masks_padded = pad_sequence(h_attention_masks, batch_first=True, padding_value=0)
    h_special_tokens_masks_padded = pad_sequence(h_special_tokens_masks, batch_first=True, padding_value=1)
    
    l_input_ids_padded = pad_sequence(l_input_ids, batch_first=True, padding_value=0)
    l_attention_masks_padded = pad_sequence(l_attention_masks, batch_first=True, padding_value=0)
    l_special_tokens_masks_padded = pad_sequence(l_special_tokens_masks, batch_first=True, padding_value=1)
    
    return {
        'h_input_ids': h_input_ids_padded.to(torch.int16),
        'h_attention_mask': h_attention_masks_padded.to(torch.bool),
        'h_special_tokens_mask': h_special_tokens_masks_padded.to(torch.bool),
        'l_input_ids': l_input_ids_padded.to(torch.int16), 
        'l_attention_mask': l_attention_masks_padded.to(torch.bool),
        'l_special_tokens_mask': l_special_tokens_masks_padded.to(torch.bool),
        'original_h_lengths': torch.tensor([item['original_h_length'] for item in batch]),
        'original_l_lengths': torch.tensor([item['original_l_length'] for item in batch])
    }


def tokenize_data(df: pd.DataFrame, model_config: AbLangPairedConfig,
                  batch_size: int = 256, max_length: int = 159,
                  remove_stop_codons: bool = True) -> DataLoader:
    """
    Enhanced tokenization with dynamic batch-wise padding and no length filtering.
    
    Args:
        df: DataFrame containing antibody sequences with HC_AA and LC_AA columns
        model_config: AbLangPairedConfig that tells where to load the tokenizers from
        batch_size: Batch size for DataLoader
        max_length: Maximum sequence length (will be truncated, not filtered)
        remove_stop_codons: Whether to remove sequences with stop codons (*)
        
    Returns:
        DataLoader with tokenized sequences ready for model input
    """
    print(f"Enhanced tokenization starting with {len(df)} sequences...")
    
    # Optional: Remove sequences with stop codons (but keep long sequences)
    if remove_stop_codons:
        original_len = len(df)
        df = df[(~df["HC_AA"].str.contains("\\*", na=False)) & 
                (~df["LC_AA"].str.contains("\\*", na=False))]
        print(f"Removed {original_len - len(df)} sequences with stop codons")
    
    # Load tokenizers
    print("Loading tokenizers...")
    heavy_tokenizer = AutoTokenizer.from_pretrained(
        model_config.heavy_model_id, 
        revision=model_config.heavy_revision
    )
    light_tokenizer = AutoTokenizer.from_pretrained(
        model_config.light_model_id,
        revision=model_config.light_revision  
    )
    
    # Create dataset
    dataset = AntibodySequenceDataset(df, heavy_tokenizer, light_tokenizer, max_length)
    
    # Create DataLoader with custom collate function
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 to avoid multiprocessing issues with tokenizers
    )
    
    print(f"Created DataLoader with {len(dataset)} sequences in {len(dataloader)} batches")
    return dataloader


def fix_unknown_tokens(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fix unknown token issues by converting UNK (token 24) to MASK (token 23).
    
    Args:
        input_ids: Input token IDs
        attention_mask: Attention mask
        
    Returns:
        Tuple of fixed (input_ids, attention_mask)
    """
    # Handle special token 24 (replace with token 23 and set attention mask to False)
    matches = torch.where(input_ids == 24)
    if len(matches[0]) > 0:
        input_ids[matches] = 23
        attention_mask[matches] = False
        
    return input_ids, attention_mask


def embed_dataloader(dataloader: DataLoader, model, device) -> torch.Tensor:
    """
    Enhanced embedding function that works with the new tokenization system.
    
    Args:
        dataloader: DataLoader containing tokenized antibody sequences (from tokenize_data_enhanced)
        model: Trained AbLangPaired model
        device: Device to run inference on (CPU or GPU)
        
    Returns:
        Tensor containing embeddings for all antibodies
    """
    model.to(device)
    model.eval()
    
    # Preallocate tensor for all embeddings  
    num_embeddings = len(dataloader.dataset)
    embedding_dim = 1536
    all_embeds = torch.zeros((num_embeddings, embedding_dim), dtype=torch.float32)
    
    # Generate embeddings batch by batch
    current_batch_index = 0
    print("Now Embedding Antibodies with Enhanced Method")
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Extract tensors from batch dictionary
            h_input_ids = batch['h_input_ids'].to(device)
            h_attention_mask = batch['h_attention_mask'].to(device)
            l_input_ids = batch['l_input_ids'].to(device)
            l_attention_mask = batch['l_attention_mask'].to(device)
            
            # Fix unknown tokens
            h_input_ids, h_attention_mask = fix_unknown_tokens(h_input_ids, h_attention_mask)
            l_input_ids, l_attention_mask = fix_unknown_tokens(l_input_ids, l_attention_mask)
            
            # Forward pass to get embeddings
            embeds = model(
                h_input_ids=h_input_ids,
                h_attention_mask=h_attention_mask,
                l_input_ids=l_input_ids, 
                l_attention_mask=l_attention_mask
            )
            
            # Store embeddings in preallocated tensor
            batch_size = embeds.size(0)
            all_embeds[current_batch_index:current_batch_index + batch_size] = embeds.detach().cpu()
            current_batch_index += batch_size
            
            # Clean up GPU memory
            del h_input_ids, h_attention_mask, l_input_ids, l_attention_mask, embeds
            torch.cuda.empty_cache()
    
    return all_embeds


if __name__ == "__main__":
    """
    Generate AbLangPre embeddings for SAbDab dataset.
    
    This section handles the missing AbLangPre embedding for SAbDab that was causing
    issues in the comprehensive benchmarking pipeline. Uses AbLangPaired with
    use_pretrained=True to load the pretrained AbLang models.
    """
    from ablangpaired_model import AbLangPaired
    
    print("🔄 Generating missing AbLangPre embeddings for SAbDab dataset...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Load the SAbDab dataset
        input_file = "ablangpdb_renameddatasets.parquet"
        output_file = "sabdab_embeddedby_ablangpre.parquet"
        
        print(f"Loading dataset from {input_file}...")
        df = pd.read_parquet(input_file)
        
        if "EMBEDDING" in df.columns:
            df = df.drop(columns=["EMBEDDING"])
            
        print(f"Loaded {len(df)} sequences")
        
        # Set up AbLangPre model using AbLangPaired with use_pretrained=True
        # We can use any model_config since use_pretrained=True will override it
        dummy_model_path = "../../../huggingface/AbLangRBD1/model.safetensors"
        model_config = AbLangPairedConfig(checkpoint_filename=dummy_model_path)
        
        print("🔄 Loading AbLangPre model (use_pretrained=True)...")
        model = AbLangPaired(model_config, device=device, use_pretrained=True)
        
        # Tokenize and embed using the enhanced methods
        print("🔄 Tokenizing sequences...")
        tokenized_dataloader = tokenize_data(df, model_config, batch_size=256)
        
        print("🔄 Generating AbLangPre embeddings...")
        all_embeds = embed_dataloader(tokenized_dataloader, model, device)
        
        # Add embeddings to dataframe
        df['EMBEDDING'] = list(all_embeds.cpu().numpy())
        
        # Save the result
        print(f"💾 Saving embeddings to {output_file}...")
        df.to_parquet(output_file)
        
        print(f"✅ Successfully generated AbLangPre embeddings!")
        print(f"📊 Embedding shape: {all_embeds.shape}")
        print(f"📁 Saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error generating AbLangPre embeddings: {str(e)}")
        import traceback
        traceback.print_exc()