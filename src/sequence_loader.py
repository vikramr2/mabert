from sys import argv

unaligned_sequence_file = argv[1]

def read_unaligned_sequences(file_path):
    sequences = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith('#'):
                sequences.append(line)
    return sequences

def extract_sequence_dictionary(sequences):
    sequence_dict = {}
    for i in range(0, len(sequences), 2):
        name = sequences[i]
        name = name[1:]  # Remove the leading '>'
        data = sequences[i + 1] if i + 1 < len(sequences) else ''
        sequence_dict[name] = data
    return sequence_dict

def main():
    sequences = read_unaligned_sequences(unaligned_sequence_file)

    # Extract names and data from the sequences
    sequence_names = sequences[0::2]
    sequence_data = sequences[1::2]

    # Remove the first character '>' from each name
    sequence_names = [name[1:] for name in sequence_names]

    # Create a dictionary to hold the sequences
    sequence_dict = {name: data for name, data in zip(sequence_names, sequence_data)}

    print("Number of sequences:", len(sequence_dict))
    print("First sequence name:", next(iter(sequence_dict)))
    print("First sequence data:", sequence_dict[next(iter(sequence_dict))])

if __name__ == "__main__":
    main()
