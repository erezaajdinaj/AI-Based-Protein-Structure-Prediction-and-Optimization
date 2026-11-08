DIRECTIONS = {
    "R": (1, 0),
    "L": (-1, 0),
    "U": (0, 1),
    "D": (0, -1),
}


def build_coords(moves):
    coords = [(0, 0)]
    for move in moves:
        dx, dy = DIRECTIONS[move]
        x, y = coords[-1]
        coords.append((x + dx, y + dy))
    return coords


if __name__ == "__main__":
    print(build_coords([]))
    print(build_coords(["R"]))
    print(build_coords(["R", "R", "U", "L"]))
    print(build_coords(["U", "U", "R"]))