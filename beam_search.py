import random
import time

from movement_generator import DIRECTIONS, build_coords
from collision import is_valid
from energy import energy


def allowed_moves(moves):
    if not moves:
        return ["R"]
    if all(move == "R" for move in moves):
        return ["R", "U"]
    return list(DIRECTIONS)


def score(hp, moves):
    coords = build_coords(moves)
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    area = (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1)
    return energy(hp, coords), area


def run_width(hp, width, rng, deadline, stats):
    beam = [[]]
    for _ in range(len(hp) - 1):
        candidates = []
        for moves in beam:
            if time.perf_counter() > deadline:
                return None
            for letter in allowed_moves(moves):
                new_moves = moves + [letter]
                stats["explored"] += 1
                if not is_valid(build_coords(new_moves)):
                    stats["invalid"] += 1
                    continue
                stats["valid"] += 1
                candidates.append((score(hp, new_moves), new_moves))
        if not candidates:
            return None
        rng.shuffle(candidates)
        candidates.sort(key=lambda item: item[0])
        stats["pruned"] += max(0, len(candidates) - width)
        beam = [moves for _, moves in candidates[:width]]
    return beam[0]


def beam_search(hp, time_limit=5.0, seed=None, start_width=10):
    rng = random.Random(seed)
    start = time.perf_counter()
    deadline = start + time_limit
    stats = {"explored": 0, "valid": 0, "invalid": 0, "pruned": 0}

    best_moves = ["R"] * (len(hp) - 1)
    best_energy = energy(hp, build_coords(best_moves))
    history = [(0.0, best_energy)]
    width = start_width
    last_width = 0

    while time.perf_counter() < deadline:
        moves = run_width(hp, width, rng, deadline, stats)
        if moves is None:
            break
        last_width = width
        current_energy = energy(hp, build_coords(moves))
        if current_energy < best_energy:
            best_energy = current_energy
            best_moves = moves
            history.append((time.perf_counter() - start, best_energy))
        width *= 2

    return {
        "moves": best_moves,
        "energy": best_energy,
        "explored": stats["explored"],
        "valid": stats["valid"],
        "invalid": stats["invalid"],
        "pruned": stats["pruned"],
        "width": last_width,
        "time": time.perf_counter() - start,
        "history": history,
    }