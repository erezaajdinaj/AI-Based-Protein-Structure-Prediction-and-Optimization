#hidrofobic:
#G, A, V, L, I, P, F, W, M
#polar:
#S, T, C, N, Q, Y, K, R, H, D, E
#based on the article from BIOLONGEVITY LABS:
#Polar vs Nonpolar Amino Acids: Classification, Structure, and Research Relevance

VALID = {"A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"}
HYDROPHOBIC = {"G", "A", "V", "L", "I", "P", "F", "W", "M"}
def to_hp(sequence):
    seq = sequence.upper()
    result = ""
    for letter in seq:
        if letter not in VALID:
            raise ValueError(f"Invalid amino acid: {letter}")
        if letter in HYDROPHOBIC:
            result += "H"
        else:
            result += "P"
    return result
protein = input("Enter a protein: ")
print(to_hp(protein))