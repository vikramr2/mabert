import os
import torch
from transformers import AutoTokenizer, AutoModel

# Set multiple environment variables to disable Flash Attention
os.environ['FLASH_ATTENTION_DISABLE'] = '1'
os.environ['DISABLE_FLASH_ATTN'] = '1'
os.environ['USE_FLASH_ATTENTION'] = '0'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref")
model = AutoModel.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref")

# Move model to device
model = model.to(device)

dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
inputs = tokenizer(dna, return_tensors='pt')["input_ids"].to(device)
hidden_states = model(inputs)[0]

embedding_mean = torch.mean(hidden_states[0], dim=0)
print(embedding_mean.shape)
