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
import numpy as np
from tqdm import tqdm

import ablang
import ablang2
from antiberty import AntiBERTyRunner

# PyTorch imports
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

# Hugging Face Transformers imports
from transformers import AutoTokenizer, AutoModel, RobertaModel, RobertaTokenizer, BertTokenizer, BertModel


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


def embed_dataloader(dataloader: DataLoader, model, device) -> List[np.ndarray]:
    """
    Enhanced embedding function that works with the new tokenization system.
    
    Args:
        dataloader: DataLoader containing tokenized antibody sequences (from tokenize_data_enhanced)
        model: Trained AbLangPaired model
        device: Device to run inference on (CPU or GPU)
        
    Returns:
        List of numpy arrays containing embeddings for all antibodies (pandas DataFrame compatible)
    """
    model.to(device)
    model.eval()
    
    # Store embeddings as list of numpy arrays
    all_embeddings = []
    
    # Generate embeddings batch by batch
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
            
            # Convert batch embeddings to individual numpy arrays
            batch_numpy = embeds.detach().cpu().numpy()
            for embedding in batch_numpy:
                all_embeddings.append(embedding)
            
            # Clean up GPU memory
            del h_input_ids, h_attention_mask, l_input_ids, l_attention_mask, embeds
            torch.cuda.empty_cache()
    
    return all_embeddings


class BalmEmbedder:
    """
    Embeds antibody sequences using the BALM model.
    """
    def __init__(self, model_directory, device="cpu", batch_size=128):
        self.model_directory = model_directory
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.tokenizer = RobertaTokenizer.from_pretrained(model_directory)
        self.model = RobertaModel.from_pretrained(model_directory).to(self.device)
        self.model.eval()

    def embed(self, df):
        sequences = [f"{hc}</s>{lc}" for hc, lc in zip(df["HC_AA"], df["LC_AA"])]
        
        encoded_input = self.tokenizer(
            sequences,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        dataset = torch.utils.data.TensorDataset(
            encoded_input['input_ids'],
            encoded_input['attention_mask']
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)

        all_embeddings = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                input_ids, attention_mask = batch
                input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Masked Mean Pooling
                last_hidden = outputs.last_hidden_state
                attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                sum_embeddings = torch.sum(last_hidden * attention_mask_expanded, 1)
                sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
                batch_embeddings = sum_embeddings / sum_mask
                
                # L2 Normalization
                normalized_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                all_embeddings.append(normalized_embeddings.cpu().numpy())

        # Return list of numpy arrays for pandas DataFrame compatibility
        embeddings_array = np.vstack(all_embeddings)
        return [embedding for embedding in embeddings_array]

class Ablang2Embedder:
    """
    Embeds antibody sequences using the AbLang2 model.
    """
    def __init__(self, device="cpu", batch_size=256):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model = ablang2.pretrained(model_to_use='ablang2-paired', device=self.device)
        # self.model.eval()

    def embed(self, df):
        sequences = [[hc, lc] for hc, lc in zip(df["HC_AA"].apply(lambda aa: aa[:157]), df["LC_AA"])]
        
        all_embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), self.batch_size)):
                batch_sequences = sequences[i:i+self.batch_size]
                batch_embeddings = self.model(batch_sequences, mode='seqcoding')
                
                # Convert to tensor and normalize
                batch_tensor = torch.from_numpy(batch_embeddings)
                normalized_batch = F.normalize(batch_tensor, p=2, dim=1)
                all_embeddings.append(normalized_batch.numpy())
        
        # Return list of numpy arrays for pandas DataFrame compatibility
        return [embedding for embedding in np.vstack(all_embeddings)]


class Esm2Embedder:
    """
    Embeds antibody sequences using the ESM-2 model.
    """
    def __init__(self, model_name="facebook/esm2_t33_650M_UR50D", device="cpu", batch_size=128):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed(self, df):
        # Using two CLS tokens between chains as requested
        sequences = [f"{hc}{self.tokenizer.cls_token}{self.tokenizer.cls_token}{lc}" for hc, lc in zip(df["HC_AA"], df["LC_AA"])]
        
        all_embeddings = []
        for i in tqdm(range(0, len(sequences), self.batch_size)):
            batch_sequences = sequences[i:i+self.batch_size]
            encoded_input = self.tokenizer(
                batch_sequences,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoded_input)
                
                # Masked Mean Pooling
                last_hidden = outputs.last_hidden_state
                attention_mask = encoded_input['attention_mask']
                attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                sum_embeddings = torch.sum(last_hidden * attention_mask_expanded, 1)
                sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
                batch_embeddings = sum_embeddings / sum_mask

                # L2 Normalization
                normalized_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                all_embeddings.append(normalized_embeddings.cpu().numpy())
        
        # Return list of numpy arrays for pandas DataFrame compatibility
        embeddings_array = np.vstack(all_embeddings)
        return [embedding for embedding in embeddings_array]


class AntibertyEmbedder:
    """
    Embeds antibody heavy chain sequences using the AntiBERTy model.
    """
    def __init__(self, batch_size=256):
        self.runner = AntiBERTyRunner()
        self.batch_size = batch_size

    def embed(self, df):
        sequences = df["HC_AA"].tolist()
        
        all_embeddings = []
        # Process in batches for efficiency
        for i in tqdm(range(0, len(sequences), self.batch_size)):
            batch_sequences = sequences[i:i+self.batch_size]
            embeddings_list = self.runner.embed(batch_sequences)
            
            # Convert list of tensors to numpy arrays and perform mean pooling
            batch_embeddings = []
            for tensor in embeddings_list:
                # The output of AntiBERTyRunner already accounts for padding.
                mean_embedding = tensor.mean(dim=0).cpu().numpy()
                batch_embeddings.append(mean_embedding)
            
            # L2 Normalization for the batch
            batch_array = np.array(batch_embeddings)
            batch_tensor = torch.from_numpy(batch_array)
            normalized_batch = F.normalize(batch_tensor, p=2, dim=1)
            all_embeddings.append(normalized_batch.numpy())
        
        # Return list of numpy arrays for pandas DataFrame compatibility
        embeddings_array = np.vstack(all_embeddings)
        return [embedding for embedding in embeddings_array]


class AblangHeavyEmbedder:
    """
    Embeds antibody heavy chain sequences using the AbLang-Heavy model.
    """
    def __init__(self, device="cpu", batch_size=256):
        self.device = device
        self.model = ablang.pretrained("heavy", device=self.device)
        self.batch_size = batch_size
        self.model.freeze()

    def embed(self, df: pd.DataFrame):
        sequences = df["HC_AA"].apply(lambda aa: aa[:157]).tolist()
        all_embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), self.batch_size)):
                batch_sequences = sequences[i:i+self.batch_size]
                batch_embeddings = self.model(batch_sequences, mode='seqcoding')
                
                # Convert to tensor and normalize
                batch_tensor = torch.from_numpy(batch_embeddings)
                normalized_batch = F.normalize(batch_tensor, p=2, dim=1)
                all_embeddings.append(normalized_batch.numpy())

        # Return list of numpy arrays for pandas DataFrame compatibility
        embeddings_array = np.vstack(all_embeddings)
        return [embedding for embedding in embeddings_array]

class IgbertEmbedder:
    """
    Embeds antibody sequences using the IgBERT model.
    """
    def __init__(self, device="cpu", batch_size=256):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.tokenizer = BertTokenizer.from_pretrained("Exscientia/IgBert", do_lower_case=False)
        self.model = BertModel.from_pretrained("Exscientia/IgBert").to(self.device)
        self.model.eval()

    def embed(self, df):
        # The tokenizer expects space-separated sequences with a [SEP] token
        sequences = [f"{' '.join(hc)} [SEP] {' '.join(lc)}" for hc, lc in zip(df["HC_AA"], df["LC_AA"])]
        
        all_embeddings = []
        for i in tqdm(range(0, len(sequences), self.batch_size)):
            batch_sequences = sequences[i:i+self.batch_size]
            encoded_input = self.tokenizer(
                batch_sequences,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoded_input)
                
                # Masked Mean Pooling (ignoring [CLS], [SEP], and padding)
                last_hidden = outputs.last_hidden_state
                attention_mask = encoded_input['attention_mask']
                
                # To correctly average, we must exclude special tokens from the attention mask
                special_tokens_mask = (
                    (encoded_input['input_ids'] == self.tokenizer.cls_token_id) |
                    (encoded_input['input_ids'] == self.tokenizer.sep_token_id)
                )
                attention_mask[special_tokens_mask] = 0
                
                attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                sum_embeddings = torch.sum(last_hidden * attention_mask_expanded, 1)
                sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
                batch_embeddings = sum_embeddings / sum_mask

                # L2 Normalization
                normalized_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                all_embeddings.append(normalized_embeddings.cpu().numpy())
        
        # Return list of numpy arrays for pandas DataFrame compatibility
        embeddings_array = np.vstack(all_embeddings)
        return [embedding for embedding in embeddings_array]


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
        from time import time
        start_time = time()
        all_embeds = embed_dataloader(tokenized_dataloader, model, device)
        end_time = time()
        print(f"⏱️ Embedding generation took {end_time - start_time:.2f} seconds")
        # Add time per antibody
        print(f"⏱️ Time per antibody: {(end_time - start_time) / len(df):.4f} seconds")
        # Add embeddings to dataframe (embeddings are already in list format)
        df['EMBEDDING'] = all_embeds
        
        # Save the result
        # print(f"💾 Saving embeddings to {output_file}...")
        # df.to_parquet(output_file)
        
        print(f"✅ Successfully generated AbLangPre embeddings!")
        print(f"📊 Number of embeddings: {len(all_embeds)}")
        print(f"📊 Embedding shape per antibody: {all_embeds[0].shape if all_embeds else 'N/A'}")
        print(f"📁 Saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error generating AbLangPre embeddings: {str(e)}")
        import traceback
        traceback.print_exc()