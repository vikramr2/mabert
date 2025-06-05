# mabert
A prototype: Multiple sequence Alignment using Bidirectional Encoder Representations from Transformers (MABERT). A BERT encodes the sequences, hierarchical clustering is then used to create a guide tree through which alignment is performed.

1. Compute BERT embeddings for each sequence using the `InstaDeepAI/nucleotide-transformer-500m-human-ref` model
2. Build a guide tree using hierarchical vector clustering
3. Compute alignment along the guide tree using MAFFT

## Setup

To install MABERT, simply run the following:

```bash
git clone https://github.com/vikramr2/mabert.git
cd mabert
pip install -r requirements.txt
```

## Usage

The use can control what step to start from. Just passing in `-s` with a FASTA dataset of unaligned sequences starts the pipeline from scratch. Alternatively, the user can start from the embedding stage by passing in `-e` pointing to an embeddings `.npz` file. Passing in `-t` starts from an existing guide tree.

```
./mabert <OPTIONS>

OPTIONS:
    -s, --sequences FILE    Path to unaligned sequences file (FASTA format)
    -e, --embeddings FILE   Path to embeddings file (NPZ format)
    -t, --tree FILE         Path to guide tree file (Newick format)
    -o, --output FILE       Output aligned sequences file (default: aligned_sequences.fasta)
    -h, --help              Show this help message
```

### Example Commands

**Run from scratch**
```
./mabert -s data/unaligned.txt -o output.fasta
```

**Start from embeddings**
```
./mabert -e embeddings.npz -s data/unaligned.txt -o output.fasta
```

**Start from guide tree**
```
./mabert -t guide_tree.newick -s data/unaligned.txt -o output.fasta
```
