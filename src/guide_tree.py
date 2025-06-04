import torch    # type: ignore
from transformers import AutoTokenizer, AutoModel
from sys import argv
from sequence_loader import read_unaligned_sequences, extract_sequence_dictionary
from tqdm import tqdm   # type: ignore
import os
import gc

def embed(sequence, pooling='mean'):
    """
    Embed a DNA sequence using the DNABERT model.
    
    Args:
        sequence (str): The DNA sequence to embed.
    
    Returns:
        torch.Tensor: The embedding of the sequence.
    """
    inputs = tokenizer(sequence, return_tensors='pt')["input_ids"].to(device)
    hidden_states = model(inputs)[0]  # [1, sequence_length, 768]
    
    if pooling == 'mean':
        embedding = torch.mean(hidden_states[0], dim=0)
    elif pooling == 'max':
        embedding = torch.max(hidden_states[0], dim=0)[0]
    else:
        raise ValueError("Pooling method must be 'mean' or 'max'.")
    
    return embedding

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
    model = AutoModel.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref")

    # Move model to GPU
    model = model.to(device)

    embeddings = {}
    for i, (name, sequence) in enumerate(tqdm(sequence_dict.items())):
        with torch.no_grad():  # Disable gradient computation
            embedding = embed(sequence, pooling='mean')
            embeddings[name] = embedding.cpu()  # Move to CPU immediately
        
        # Clear cache every 10 sequences
        if i % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    print("Embeddings computed for all sequences.")
