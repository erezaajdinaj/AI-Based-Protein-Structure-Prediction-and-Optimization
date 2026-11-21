import random
import time

from movement_generator import DIRECTIONS

DIRS = list(DIRECTIONS.values())


def run_width(hp, width, rng, deadline, stats):
    n = len(hp)
    if n <= 2:
        return ["R"] * (n - 1), 0

    is_h0 = 1 if hp[0] == "H" else 0
    is_h1 = 1 if hp[1] == "H" else 0
    init_state = (
        0,                       # energy
        "R",                     # moves (compact string)
        1, 0,                    # lx, ly
        {(0, 0): 0, (1, 0): 1},  # pos_map
        True,                    # all_r (symmetry breaking)
        0, 1, 0, 0,              # min_x, max_x, min_y, max_y
        is_h1, 0, is_h0 + is_h1  # sum_hx, sum_hy, count_h
    )
    beam = [init_state]

    for idx in range(2, n):
        is_h = hp[idx] == "H"
        is_last = idx == n - 1
        candidates = []

        for p_idx, (en, _, lx, ly, pos_map, all_r,
                    min_x, max_x, min_y, max_y, shx, shy, ch) in enumerate(beam):
            if time.perf_counter() > deadline:
                return None

            allowed = ("R", "U") if all_r else ("R", "L", "U", "D")
            for letter in allowed:
                dx, dy = DIRECTIONS[letter]
                nx, ny = lx + dx, ly + dy
                stats["explored"] += 1

                if (nx, ny) in pos_map:
                    stats["invalid"] += 1
                    continue

                if not is_last:
                    if ((nx + 1, ny) in pos_map and (nx - 1, ny) in pos_map and
                            (nx, ny + 1) in pos_map and (nx, ny - 1) in pos_map):
                        stats["invalid"] += 1
                        continue

                stats["valid"] += 1
                new_en = en
                if is_h:
                    for ddx, ddy in DIRS:
                        j = pos_map.get((nx + ddx, ny + ddy))
                        if j is not None and j < idx - 1 and hp[j] == "H":
                            new_en -= 1

                n_min_x = nx if nx < min_x else min_x
                n_max_x = nx if nx > max_x else max_x
                n_min_y = ny if ny < min_y else min_y
                n_max_y = ny if ny > max_y else max_y
                area = (n_max_x - n_min_x + 1) * (n_max_y - n_min_y + 1)

                if is_h:
                    n_shx = shx + nx
                    n_shy = shy + ny
                    n_ch = ch + 1
                    h_dist = abs(nx - (n_shx / n_ch)) + abs(ny - (n_shy / n_ch)) if n_ch > 1 else 0.0
                else:
                    n_shx, n_shy, n_ch, h_dist = shx, shy, ch, 0.0

                candidates.append((
                    new_en, area, h_dist, rng.random(),
                    p_idx, letter, nx, ny,
                    n_min_x, n_max_x, n_min_y, n_max_y,
                    n_shx, n_shy, n_ch
                ))

        if not candidates:
            return None

        candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        stats["pruned"] += max(0, len(candidates) - width)

        new_beam = []
        for (new_en, _, _, _, p_idx, letter, nx, ny,
             n_min_x, n_max_x, n_min_y, n_max_y, n_shx, n_shy, n_ch) in candidates[:width]:
            _, moves, _, _, pos_map, all_r, *_ = beam[p_idx]
            new_pos_map = pos_map.copy()
            new_pos_map[(nx, ny)] = idx
            new_beam.append((
                new_en, moves + letter, nx, ny, new_pos_map,
                all_r and (letter == "R"),
                n_min_x, n_max_x, n_min_y, n_max_y,
                n_shx, n_shy, n_ch
            ))
        beam = new_beam

    return list(beam[0][1]), beam[0][0]


def beam_search(hp, time_limit=5.0, seed=None, start_width=10):
    rng = random.Random(seed)
    start = time.perf_counter()
    deadline = start + time_limit
    stats = {"explored": 0, "valid": 0, "invalid": 0, "pruned": 0}

    best_moves = ["R"] * (len(hp) - 1)
    best_energy = 0
    history = [(0.0, best_energy)]
    width = start_width
    last_width = 0

    while time.perf_counter() < deadline:
        result = run_width(hp, width, rng, deadline, stats)
        if result is None:
            break
        moves, current_energy = result
        last_width = width
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