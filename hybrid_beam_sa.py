import math
import random
import time

from movement_generator import DIRECTIONS

DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
DIR_TO_CHAR = {(1, 0): "R", (-1, 0): "L", (0, 1): "U", (0, -1): "D"}


def fast_energy(h_indices, coords):
    h_pos = {coords[i]: i for i in h_indices}
    total = 0
    for i in h_indices:
        x, y = coords[i]
        for dx, dy in DIRS:
            j = h_pos.get((x + dx, y + dy))
            if j is not None and j > i + 1:
                total -= 1
    return total


def moves_to_coords(moves):
    x, y = 0, 0
    coords = [(0, 0)]
    for m in moves:
        dx, dy = DIRECTIONS[m]
        x += dx
        y += dy
        coords.append((x, y))
    return coords


def coords_to_moves(coords):
    return [
        DIR_TO_CHAR[(coords[i + 1][0] - coords[i][0], coords[i + 1][1] - coords[i][1])]
        for i in range(len(coords) - 1)
    ]


def run_beam_seed(hp, width, rng, deadline, stats, top_k=4):
    n = len(hp)
    if n <= 2:
        return [(0, [(0, 0), (1, 0)][:n])]

    is_h0 = 1 if hp[0] == "H" else 0
    is_h1 = 1 if hp[1] == "H" else 0
    init_state = (
        0, "R", 1, 0,
        {(0, 0): 0, (1, 0): 1},
        True,
        0, 1, 0, 0,
        is_h1, 0, is_h0 + is_h1
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

    # Select top_k diverse structures from the final beam
    picked = []
    seen_prefix = set()
    prefix_len = max(1, (n - 1) // 3)
    for state in beam:
        en, moves_str = state[0], state[1]
        sig = moves_str[:prefix_len]
        if sig not in seen_prefix or not picked:
            seen_prefix.add(sig)
            picked.append((en, moves_to_coords(moves_str)))
            if len(picked) >= top_k:
                break
    idx_fill = 1
    while len(picked) < top_k and idx_fill < len(beam):
        st = beam[idx_fill]
        picked.append((st[0], moves_to_coords(st[1])))
        idx_fill += 1
    return picked


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


def hybrid_beam_sa(hp, time_limit=5.0, seed=None, beam_fraction=0.25, max_beam_width=1280):
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
            "pruned": 0,
            "width": 0,
            "time": time.perf_counter() - start,
            "history": [(0.0, 0)],
        }

    deadline = start + time_limit
    beam_deadline = start + time_limit * beam_fraction
    stats = {"explored": 0, "valid": 0, "invalid": 0, "pruned": 0}
    h_indices = [i for i, c in enumerate(hp) if c == "H"]

    # Phase 1: Fast Beam Search Seeding
    seeds_pool = [(0, [(i, 0) for i in range(n)])]
    best_energy = 0
    best_coords = [(i, 0) for i in range(n)]
    history = [(0.0, 0)]
    width = 20
    last_width = 0

    while time.perf_counter() < beam_deadline and width <= max_beam_width:
        res = run_beam_seed(hp, width, rng, beam_deadline, stats, top_k=4)
        if res is None:
            break
        last_width = width
        seeds_pool = res
        if res[0][0] < best_energy:
            best_energy = res[0][0]
            best_coords = list(res[0][1])
            history.append((time.perf_counter() - start, best_energy))
        width *= 2

    # Phase 2: Parallel Tempering (4 Replicas with Pull / Pivot / End)
    temps = [0.20, 0.45, 0.90, 1.60]
    n_rep = len(temps)
    replicas = []
    for r in range(n_rep):
        en, coords = seeds_pool[r % len(seeds_pool)]
        replicas.append([list(coords), set(coords), en])

    step_count = 0
    while time.perf_counter() < deadline:
        for r in range(n_rep):
            coords, occupied, cur_en = replicas[r]
            p = rng.random()
            if p < 0.68:
                cand = pull_move(coords, occupied, rng)
            elif p < 0.85:
                cand = pivot_move(coords, rng)
            else:
                cand = end_move(coords, occupied, rng)

            stats["explored"] += 1
            if cand is None:
                stats["invalid"] += 1
                continue

            stats["valid"] += 1
            cand_en = fast_energy(h_indices, cand)
            delta = cand_en - cur_en

            if delta <= 0 or rng.random() < math.exp(-delta / temps[r]):
                replicas[r][0] = cand
                replicas[r][1] = set(cand)
                replicas[r][2] = cand_en
                if cand_en < best_energy:
                    best_energy = cand_en
                    best_coords = list(cand)
                    history.append((time.perf_counter() - start, best_energy))

        step_count += 1
        if step_count % 50 == 0:
            for r in range(n_rep - 1):
                e1, e2 = replicas[r][2], replicas[r + 1][2]
                t1, t2 = temps[r], temps[r + 1]
                diff = (1.0 / t1 - 1.0 / t2) * (e1 - e2)
                if diff >= 0 or rng.random() < math.exp(diff):
                    replicas[r], replicas[r + 1] = replicas[r + 1], replicas[r]

    return {
        "moves": coords_to_moves(best_coords),
        "energy": best_energy,
        "explored": stats["explored"],
        "valid": stats["valid"],
        "invalid": stats["invalid"],
        "pruned": stats["pruned"],
        "width": last_width,
        "time": time.perf_counter() - start,
        "history": history,
    }