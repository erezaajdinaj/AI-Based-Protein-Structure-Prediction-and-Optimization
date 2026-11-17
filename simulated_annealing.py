import math
import random
import time

from movement_generator import build_coords
from energy import energy
from local_moves import random_valid_structure, valid_neighbor


def simulated_annealing(hp, time_limit=5.0, seed=None, t_start=2.0, t_end=0.05):
    rng = random.Random(seed)
    n = len(hp)
    start = time.perf_counter()

    current_moves = random_valid_structure(n, rng)
    current_energy = energy(hp, build_coords(current_moves))
    best_moves = list(current_moves)
    best_energy = current_energy

    explored = 1
    valid = 1
    invalid = 0
    history = [(0.0, best_energy)]

    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= time_limit:
            break
        temperature = t_start * (t_end / t_start) ** (elapsed / time_limit)

        candidate, bad = valid_neighbor(current_moves, rng)
        invalid += bad
        explored += bad
        if candidate == current_moves:
            continue

        explored += 1
        valid += 1
        candidate_energy = energy(hp, build_coords(candidate))
        delta = candidate_energy - current_energy

        if delta <= 0 or rng.random() < math.exp(-delta / temperature):
            current_moves = candidate
            current_energy = candidate_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_moves = list(current_moves)
                history.append((time.perf_counter() - start, best_energy))

    return {
        "moves": best_moves,
        "energy": best_energy,
        "explored": explored,
        "valid": valid,
        "invalid": invalid,
        "time": time.perf_counter() - start,
        "history": history,
    }