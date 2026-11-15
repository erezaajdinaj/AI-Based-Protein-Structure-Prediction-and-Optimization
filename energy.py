DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def energy(hp, coords):
    position = {coord: index for index, coord in enumerate(coords)}
    total = 0
    for i, (x, y) in enumerate(coords):
        if hp[i] != "H":
            continue
        for dx, dy in DIRS:
            j = position.get((x + dx, y + dy))
            if j is not None and j > i + 1 and hp[j] == "H":
                total -= 1
    return total


if __name__ == "__main__":
    print(energy("HPHHP", [(0, 0), (1, 0), (1, 1), (0, 1), (0, 2)]))
    print(energy("HPHPPHH", [(0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (0, 1), (0, 2)]))
    print(energy("HHHH", [(0, 0), (1, 0), (2, 0), (3, 0)]))
    print(energy("HHHH", [(0, 0), (1, 0), (1, 1), (0, 1)]))