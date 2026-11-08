def is_valid(coords):
    return len(set(coords)) == len(coords)


if __name__ == "__main__":
    print(is_valid([(0, 0), (1, 0), (1, 1), (0, 1), (0, 2)]))
    print(is_valid([(0, 0), (1, 0), (1, 1), (0, 1), (1, 0)]))
    print(is_valid([(0, 0)]))