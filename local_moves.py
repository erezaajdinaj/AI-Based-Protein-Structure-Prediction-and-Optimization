import random

from movement_generator import DIRECTIONS, build_coords
from collision import is_valid


def straight_line(n_residues):
    return ["R"] * (n_residues - 1)


def random_valid_structure(n_residues, rng=None, max_tries=1000):
    rng = rng or random
    for _ in range(max_tries):
        position = (0, 0)
        visited = {position}
        moves = []
        stuck = False
        for _ in range(n_residues - 1):
            options = []
            for letter, (dx, dy) in DIRECTIONS.items():
                nxt = (position[0] + dx, position[1] + dy)
                if nxt not in visited:
                    options.append((letter, nxt))
            if not options:
                stuck = True
                break
            letter, position = rng.choice(options)
            visited.add(position)
            moves.append(letter)
        if not stuck:
            return moves
    return straight_line(n_residues)


def mutate(moves, rng=None):
    rng = rng or random
    new_moves = list(moves)
    if not new_moves:
        return new_moves
    index = rng.randrange(len(new_moves))
    choices = [d for d in DIRECTIONS if d != new_moves[index]]
    new_moves[index] = rng.choice(choices)
    return new_moves


def valid_neighbor(moves, rng=None, max_tries=50):
    rng = rng or random
    invalid = 0
    for _ in range(max_tries):
        candidate = mutate(moves, rng)
        if is_valid(build_coords(candidate)):
            return candidate, invalid
        invalid += 1
    return list(moves), invalid


if __name__ == "__main__":
    rng = random.Random(1)
    line = straight_line(8)
    print(line)
    start = random_valid_structure(8, rng)
    print(start, is_valid(build_coords(start)))
    neighbor, invalid = valid_neighbor(start, rng)
    print(neighbor, is_valid(build_coords(neighbor)), invalid)