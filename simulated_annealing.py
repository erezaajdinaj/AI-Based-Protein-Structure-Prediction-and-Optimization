import math
import random
import time

from energy import energy

DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
DIR_TO_CHAR = {(1, 0): "R", (-1, 0): "L", (0, 1): "U", (0, -1): "D"}


def coords_to_moves(coords):
    moves = []
    for i in range(len(coords) - 1):
        dx = coords[i + 1][0] - coords[i][0]
        dy = coords[i + 1][1] - coords[i][1]
        moves.append(DIR_TO_CHAR[(dx, dy)])
    return moves


def random_valid_coords(n, rng, max_tries=1000):
    for _ in range(max_tries):
        coords = [(0, 0)]
        visited = {(0, 0)}
        stuck = False
        for _ in range(n - 1):
            x, y = coords[-1]
            opts = [(x + dx, y + dy) for dx, dy in DIRS if (x + dx, y + dy) not in visited]
            if not opts:
                stuck = True
                break
            nxt = rng.choice(opts)
            visited.add(nxt)
            coords.append(nxt)
        if not stuck:
            return coords
    return [(i, 0) for i in range(n)]


def pull_move(coords, occupied, rng):
    n = len(coords)
    if n < 3:
        return None
    i = rng.randint(1, n - 2)
    step = 1 if rng.random() < 0.5 else -1

    xi, yi = coords[i]
    x_next, y_next = coords[i + step]
    dx_b = x_next - xi
    dy_b = y_next - yi

    perp = [(-dy_b, dx_b), (dy_b, -dx_b)]
    rng.shuffle(perp)
    for px, py in perp:
        L = (x_next + px, y_next + py)
        if L in occupied:
            continue
        C = (xi + px, yi + py)
        if 0 <= i - step < n and coords[i - step] == C:
            new_coords = list(coords)
            new_coords[i] = L
            return new_coords
        if C not in occupied:
            new_coords = list(coords)
            new_coords[i] = L
            k = i - step
            prev_old = (xi, yi)
            new_coords[k] = C
            while True:
                k_next = k - step
                if k_next < 0 or k_next >= n:
                    break
                if abs(new_coords[k][0] - new_coords[k_next][0]) + abs(new_coords[k][1] - new_coords[k_next][1]) == 1:
                    break
                old_k = coords[k]
                new_coords[k_next] = prev_old
                prev_old = old_k
                k = k_next
            return new_coords
    return None


def pivot_move(coords, rng):
    n = len(coords)
    if n < 3:
        return None
    i = rng.randint(1, n - 2)
    px, py = coords[i]
    rot = rng.choice((1, 2, 3))
    new_coords = list(coords)

    if i < n // 2:
        static = set(coords[i:])
        for k in range(i):
            dx, dy = coords[k][0] - px, coords[k][1] - py
            if rot == 1:
                nx, ny = px - dy, py + dx
            elif rot == 2:
                nx, ny = px - dx, py - dy
            else:
                nx, ny = px + dy, py - dx
            if (nx, ny) in static:
                return None
            new_coords[k] = (nx, ny)
        return new_coords
    else:
        static = set(coords[:i + 1])
        for k in range(i + 1, n):
            dx, dy = coords[k][0] - px, coords[k][1] - py
            if rot == 1:
                nx, ny = px - dy, py + dx
            elif rot == 2:
                nx, ny = px - dx, py - dy
            else:
                nx, ny = px + dy, py - dx
            if (nx, ny) in static:
                return None
            new_coords[k] = (nx, ny)
        return new_coords


def end_move(coords, occupied, rng):
    if len(coords) < 2:
        return None
    if rng.random() < 0.5:
        ax, ay = coords[1]
        opts = [(ax + dx, ay + dy) for dx, dy in DIRS if (ax + dx, ay + dy) not in occupied]
        if opts:
            new_coords = list(coords)
            new_coords[0] = rng.choice(opts)
            return new_coords
    else:
        ax, ay = coords[-2]
        opts = [(ax + dx, ay + dy) for dx, dy in DIRS if (ax + dx, ay + dy) not in occupied]
        if opts:
            new_coords = list(coords)
            new_coords[-1] = rng.choice(opts)
            return new_coords
    return None


def propose_move(coords, occupied, rng):
    r = rng.random()
    if r < 0.65:
        return pull_move(coords, occupied, rng)
    elif r < 0.85:
        return pivot_move(coords, rng)
    else:
        return end_move(coords, occupied, rng)


def simulated_annealing(hp, time_limit=5.0, seed=None, t_start=1.5, t_end=0.15, n_cycles=5):
    rng = random.Random(seed)
    n = len(hp)
    start = time.perf_counter()

    if n <= 2:
        return {
            "moves": ["R"] * (n - 1),
            "energy": 0,
            "explored": 1,
            "valid": 1,
            "invalid": 0,
            "time": time.perf_counter() - start,
            "history": [(0.0, 0)],
        }

    deadline = start + time_limit
    cycle_len = time_limit / n_cycles

    current_coords = random_valid_coords(n, rng)
    occupied = set(current_coords)
    current_energy = energy(hp, current_coords)

    best_coords = list(current_coords)
    best_energy = current_energy

    explored = 1
    valid = 1
    invalid = 0
    history = [(0.0, best_energy)]

    while True:
        now = time.perf_counter()
        if now >= deadline:
            break

        progress = ((now - start) % cycle_len) / cycle_len
        temperature = t_start * (t_end / t_start) ** progress

        candidate = propose_move(current_coords, occupied, rng)
        explored += 1
        if candidate is None:
            invalid += 1
            continue

        valid += 1
        candidate_energy = energy(hp, candidate)
        delta = candidate_energy - current_energy

        if delta <= 0 or rng.random() < math.exp(-delta / temperature):
            current_coords = candidate
            occupied = set(current_coords)
            current_energy = candidate_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_coords = list(current_coords)
                history.append((time.perf_counter() - start, best_energy))

    return {
        "moves": coords_to_moves(best_coords),
        "energy": best_energy,
        "explored": explored,
        "valid": valid,
        "invalid": invalid,
        "time": time.perf_counter() - start,
        "history": history,
    }