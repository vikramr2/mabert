import torch    # type: ignore
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sys import argv
from sequence_loader import read_unaligned_sequences, extract_sequence_dictionary
from tqdm import tqdm   # type: ignore
import gc
import os
import numpy as np

def embed(sequence):
    """
    Embed a DNA sequence using the DNABERT model.
    
    Args:
        sequence (str): The DNA sequence to embed.
    
    Returns:
        torch.Tensor: The embedding of the sequence.
    """
    sequences = [sequence]

    # Tokenize the sequence
    tokens_ids = tokenizer.batch_encode_plus(
        sequences, 
        return_tensors="pt", 
        padding="max_length", 
        max_length=tokenizer.model_max_length
    )["input_ids"]

    # Move tokens to the appropriate device
    tokens_ids = tokens_ids.to(device)

    # Compute the embeddings
    attention_mask = tokens_ids != tokenizer.pad_token_id
    torch_outs = model(
        tokens_ids,
        attention_mask=attention_mask,
        encoder_attention_mask=attention_mask,
        output_hidden_states=True
    )

    # Compute sequence embeddings
    embeddings = torch_outs['hidden_states'][-1].detach()

    # Add embed dimension axis
    attention_mask = torch.unsqueeze(attention_mask, dim=-1)

    # Compute mean embeddings per sequence
    mean_sequence_embeddings = torch.sum(attention_mask * embeddings, axis=-2) / torch.sum(attention_mask, axis=1)
    return mean_sequence_embeddings.squeeze(0)  # Remove the batch dimension

if __name__ == "__main__":
    if len(argv) < 2:
        raise ValueError("Please provide the path to the unaligned sequence file as a command line argument.")
    
    unaligned_sequence_file = argv[1]
    sequences = read_unaligned_sequences(unaligned_sequence_file)
    
    # Extract names and data from the sequences
    sequence_dict = extract_sequence_dictionary(sequences)

    # Print the number of sequences and the first sequence name and data
    print("Number of sequences:", len(sequence_dict))
    print("First sequence name:", next(iter(sequence_dict)))
    print("First sequence data:", sequence_dict[next(iter(sequence_dict))])

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref")
    model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref")

    # Move model to GPU
    model = model.to(device)

    # Compute the embedding dictionary
    embeddings = {}
    for i, (name, sequence) in enumerate(tqdm(sequence_dict.items())):
        with torch.no_grad():  # Disable gradient computation
            embedding = embed(sequence)
            embeddings[name] = embedding.cpu()  # Move to CPU immediately
        
        # Clear cache every 10 sequences
        if i % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    # Save embeddings to an npy file
    os.makedirs('embeddings', exist_ok=True)
    np.savez('embeddings/embeddings.npz', **embeddings)

    print("Embeddings computed for all sequences.")
