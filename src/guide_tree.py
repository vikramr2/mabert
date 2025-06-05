import numpy as np
from sys import argv
from sklearn.metrics.pairwise import cosine_distances   # type: ignore
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform   # type: ignore


def compute_tree(embeddings):
    """
    Compute agglomerative clustering tree from embeddings using cosine distance
    and save as MAFFT-compatible guide tree.
    
    Args:
        embeddings: Dictionary mapping sequence IDs to embedding vectors
        
    Returns:
        str: Path to the saved guide tree file
    """
    # Extract sequence IDs and embedding vectors
    seq_ids = list(embeddings.keys())
    vectors = np.array(list(embeddings.values()))
    
    print(f"Computing tree for {len(seq_ids)} sequences...")
    
    # Compute cosine distance matrix
    cos_dist_matrix = cosine_distances(vectors)
    
    # Convert to condensed distance matrix for scipy linkage
    condensed_distances = squareform(cos_dist_matrix, checks=False)
    
    # Perform hierarchical clustering using average linkage
    linkage_matrix = linkage(condensed_distances, method='average')
    
    # Convert to tree structure
    tree = to_tree(linkage_matrix, rd=False)
    
    # Generate guide tree in Newick format for MAFFT
    guide_tree_path = "guide_tree.newick"
    newick_string = tree_to_newick(tree, seq_ids)
    
    with open(guide_tree_path, 'w') as f:
        f.write(newick_string + ";\n")
    
    print(f"Guide tree saved to: {guide_tree_path}")
    return guide_tree_path

def tree_to_newick(tree, seq_ids):
    """
    Convert scipy tree to Newick format string.
    
    Args:
        tree: scipy tree object
        seq_ids: list of sequence identifiers
        
    Returns:
        str: Newick format string
    """
    def get_newick_string(node):
        if node.is_leaf():
            # Leaf node - return sequence ID
            return seq_ids[node.id]
        else:
            # Internal node - recursively build string
            left_str = get_newick_string(node.left)
            right_str = get_newick_string(node.right)
            # Include branch length (distance)
            return f"({left_str}:{node.dist:.6f},{right_str}:{node.dist:.6f})"
    
    return get_newick_string(tree)

if __name__ == "__main__":
    if len(argv) < 2:
        raise ValueError("Please provide the path to the embeddings file as a command line argument.")

    # Load embeddings
    embeddings_file = argv[1]
    
    # Load the embeddings - assuming it's a dictionary saved as .npy file
    # If it's a different format, you may need to adjust this loading
    try:
        embeddings = np.load(embeddings_file, allow_pickle=True)
        print(f"Loaded embeddings for {len(embeddings)} sequences")
    except:
        print("Error loading embeddings. Please ensure the file contains a dictionary mapping sequence IDs to vectors.")
        raise
    
    # Compute and save the guide tree
    guide_tree_path = compute_tree(embeddings)

    print(f"Tree saved to {guide_tree_path}")
    