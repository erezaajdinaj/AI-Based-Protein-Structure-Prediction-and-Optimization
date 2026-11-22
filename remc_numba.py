import math
import time
import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(fn):
        return fn

DIRS_X = np.array([1, -1, 0, 0], dtype=np.int32)
DIRS_Y = np.array([0, 0, 1, -1], dtype=np.int32)
DIR_TO_CHAR = {(1, 0): "R", (-1, 0): "L", (0, 1): "U", (0, -1): "D"}


@njit
def _seed_numba(seed):
    np.random.seed(seed)


@njit
def calc_energy(hp_arr, h_indices, xs, ys, grid):
    en = 0
    for idx in range(len(h_indices)):
        i = h_indices[idx]
        x = xs[i]
        y = ys[i]
        j = grid[x + 1, y]
        if j > i + 1 and hp_arr[j] == 1:
            en -= 1
        j = grid[x - 1, y]
        if j > i + 1 and hp_arr[j] == 1:
            en -= 1
        j = grid[x, y + 1]
        if j > i + 1 and hp_arr[j] == 1:
            en -= 1
        j = grid[x, y - 1]
        if j > i + 1 and hp_arr[j] == 1:
            en -= 1
    return en


@njit
def recenter_replica(xs, ys, grid, n, gsize):
    min_x = xs[0]
    max_x = xs[0]
    min_y = ys[0]
    max_y = ys[0]
    for i in range(1, n):
        if xs[i] < min_x:
            min_x = xs[i]
        if xs[i] > max_x:
            max_x = xs[i]
        if ys[i] < min_y:
            min_y = ys[i]
        if ys[i] > max_y:
            max_y = ys[i]
    if min_x >= 4 and max_x <= gsize - 5 and min_y >= 4 and max_y <= gsize - 5:
        return
    cx = (min_x + max_x) // 2
    cy = (min_y + max_y) // 2
    target = gsize // 2
    sx = target - cx
    sy = target - cy
    for i in range(n):
        grid[xs[i], ys[i]] = -1
    for i in range(n):
        xs[i] += sx
        ys[i] += sy
        grid[xs[i], ys[i]] = i


@njit
def run_chunk(hp_arr, h_indices, n, gsize, n_rep, xs_all, ys_all, grids, energies,
              temps, temp_order, best_x, best_y, best_en,
              n_sweeps, swap_interval, sweep_offset):
    explored = 0
    valid = 0
    invalid = 0
    new_x = np.empty(n, dtype=np.int32)
    new_y = np.empty(n, dtype=np.int32)

    for sw in range(n_sweeps):
        do_recenter = ((sweep_offset + sw) % 20 == 0)
        for t_idx in range(n_rep):
            r = temp_order[t_idx]
            temp = temps[t_idx]
            xs = xs_all[r]
            ys = ys_all[r]
            grid = grids[r]
            cur_en = energies[r]

            if do_recenter:
                recenter_replica(xs, ys, grid, n, gsize)

            explored += 1
            mv = np.random.random()

            if mv < 0.72:
                i = np.random.randint(1, n - 1)
                step = 1 if np.random.random() < 0.5 else -1
                xi = xs[i]
                yi = ys[i]
                xn = xs[i + step]
                yn = ys[i + step]
                dxb = xn - xi
                dyb = yn - yi

                if np.random.random() < 0.5:
                    px1, py1 = -dyb, dxb
                    px2, py2 = dyb, -dxb
                else:
                    px1, py1 = dyb, -dxb
                    px2, py2 = -dyb, dxb

                found = False
                lx = 0
                ly = 0
                cx = 0
                cy = 0
                corner_only = False

                for side in range(2):
                    px = px1 if side == 0 else px2
                    py = py1 if side == 0 else py2
                    tlx = xn + px
                    tly = yn + py
                    if tlx < 2 or tlx >= gsize - 2 or tly < 2 or tly >= gsize - 2:
                        continue
                    if grid[tlx, tly] != -1:
                        continue
                    tcx = xi + px
                    tcy = yi + py
                    if tcx < 2 or tcx >= gsize - 2 or tcy < 2 or tcy >= gsize - 2:
                        continue
                    prev_i = i - step
                    if 0 <= prev_i < n and xs[prev_i] == tcx and ys[prev_i] == tcy:
                        lx, ly = tlx, tly
                        corner_only = True
                        found = True
                        break
                    if grid[tcx, tcy] == -1:
                        lx, ly = tlx, tly
                        cx, cy = tcx, tcy
                        corner_only = False
                        found = True
                        break

                if not found:
                    invalid += 1
                    continue

                valid += 1
                if corner_only:
                    grid[xi, yi] = -1
                    xs[i] = lx
                    ys[i] = ly
                    grid[lx, ly] = i
                    cand_en = calc_energy(hp_arr, h_indices, xs, ys, grid)
                    delta = cand_en - cur_en
                    if delta <= 0 or np.random.random() < math.exp(-delta / temp):
                        energies[r] = cand_en
                        if cand_en < best_en:
                            best_en = cand_en
                            for k in range(n):
                                best_x[k] = xs[k]
                                best_y[k] = ys[k]
                    else:
                        grid[lx, ly] = -1
                        xs[i] = xi
                        ys[i] = yi
                        grid[xi, yi] = i
                else:
                    for k in range(n):
                        new_x[k] = xs[k]
                        new_y[k] = ys[k]
                    new_x[i] = lx
                    new_y[i] = ly
                    k = i - step
                    prev_old_x = xi
                    prev_old_y = yi
                    new_x[k] = cx
                    new_y[k] = cy
                    last_moved = k
                    while True:
                        k_next = k - step
                        if k_next < 0 or k_next >= n:
                            break
                        if abs(new_x[k] - new_x[k_next]) + abs(new_y[k] - new_y[k_next]) == 1:
                            break
                        old_kx = xs[k]
                        old_ky = ys[k]
                        new_x[k_next] = prev_old_x
                        new_y[k_next] = prev_old_y
                        prev_old_x = old_kx
                        prev_old_y = old_ky
                        k = k_next
                        last_moved = k

                    start_m = i if i < last_moved else last_moved
                    end_m = last_moved if i < last_moved else i
                    for m in range(start_m, end_m + 1):
                        grid[xs[m], ys[m]] = -1
                    for m in range(start_m, end_m + 1):
                        grid[new_x[m], new_y[m]] = m
                    cand_en = calc_energy(hp_arr, h_indices, new_x, new_y, grid)
                    delta = cand_en - cur_en
                    if delta <= 0 or np.random.random() < math.exp(-delta / temp):
                        for m in range(start_m, end_m + 1):
                            xs[m] = new_x[m]
                            ys[m] = new_y[m]
                        energies[r] = cand_en
                        if cand_en < best_en:
                            best_en = cand_en
                            for m in range(n):
                                best_x[m] = xs[m]
                                best_y[m] = ys[m]
                    else:
                        for m in range(start_m, end_m + 1):
                            grid[new_x[m], new_y[m]] = -1
                        for m in range(start_m, end_m + 1):
                            grid[xs[m], ys[m]] = m

            elif mv < 0.88:
                i = np.random.randint(1, n - 1)
                px = xs[i]
                py = ys[i]
                rot = np.random.randint(1, 4)
                if i < n // 2:
                    start_m = 0
                    end_m = i - 1
                else:
                    start_m = i + 1
                    end_m = n - 1

                ok = True
                for m in range(start_m, end_m + 1):
                    dx = xs[m] - px
                    dy = ys[m] - py
                    if rot == 1:
                        nx = px - dy
                        ny = py + dx
                    elif rot == 2:
                        nx = px - dx
                        ny = py - dy
                    else:
                        nx = px + dy
                        ny = py - dx
                    if nx < 2 or nx >= gsize - 2 or ny < 2 or ny >= gsize - 2:
                        ok = False
                        break
                    occ = grid[nx, ny]
                    if occ != -1 and (occ < start_m or occ > end_m):
                        ok = False
                        break
                    new_x[m] = nx
                    new_y[m] = ny

                if not ok:
                    invalid += 1
                    continue

                valid += 1
                for m in range(start_m, end_m + 1):
                    grid[xs[m], ys[m]] = -1
                for m in range(start_m, end_m + 1):
                    grid[new_x[m], new_y[m]] = m
                for m in range(n):
                    if m < start_m or m > end_m:
                        new_x[m] = xs[m]
                        new_y[m] = ys[m]

                cand_en = calc_energy(hp_arr, h_indices, new_x, new_y, grid)
                delta = cand_en - cur_en
                if delta <= 0 or np.random.random() < math.exp(-delta / temp):
                    for m in range(start_m, end_m + 1):
                        xs[m] = new_x[m]
                        ys[m] = new_y[m]
                    energies[r] = cand_en
                    if cand_en < best_en:
                        best_en = cand_en
                        for m in range(n):
                            best_x[m] = xs[m]
                            best_y[m] = ys[m]
                else:
                    for m in range(start_m, end_m + 1):
                        grid[new_x[m], new_y[m]] = -1
                    for m in range(start_m, end_m + 1):
                        grid[xs[m], ys[m]] = m

            else:
                if np.random.random() < 0.5:
                    idx_e = 0
                    ax = xs[1]
                    ay = ys[1]
                else:
                    idx_e = n - 1
                    ax = xs[n - 2]
                    ay = ys[n - 2]
                ox = xs[idx_e]
                oy = ys[idx_e]
                d_start = np.random.randint(0, 4)
                found = False
                nx = 0
                ny = 0
                for off in range(4):
                    d = (d_start + off) & 3
                    tx = ax + DIRS_X[d]
                    ty = ay + DIRS_Y[d]
                    if 2 <= tx < gsize - 2 and 2 <= ty < gsize - 2 and grid[tx, ty] == -1:
                        nx = tx
                        ny = ty
                        found = True
                        break
                if not found:
                    invalid += 1
                    continue

                valid += 1
                grid[ox, oy] = -1
                xs[idx_e] = nx
                ys[idx_e] = ny
                grid[nx, ny] = idx_e
                cand_en = calc_energy(hp_arr, h_indices, xs, ys, grid)
                delta = cand_en - cur_en
                if delta <= 0 or np.random.random() < math.exp(-delta / temp):
                    energies[r] = cand_en
                    if cand_en < best_en:
                        best_en = cand_en
                        for m in range(n):
                            best_x[m] = xs[m]
                            best_y[m] = ys[m]
                else:
                    grid[nx, ny] = -1
                    xs[idx_e] = ox
                    ys[idx_e] = oy
                    grid[ox, oy] = idx_e

        total_sw = sweep_offset + sw + 1
        if total_sw % swap_interval == 0:
            parity = (total_sw // swap_interval) & 1
            t_idx = parity
            while t_idx + 1 < n_rep:
                r1 = temp_order[t_idx]
                r2 = temp_order[t_idx + 1]
                e1 = energies[r1]
                e2 = energies[r2]
                t1 = temps[t_idx]
                t2 = temps[t_idx + 1]
                diff = (1.0 / t1 - 1.0 / t2) * (e1 - e2)
                if diff >= 0.0 or np.random.random() < math.exp(diff):
                    temp_order[t_idx] = r2
                    temp_order[t_idx + 1] = r1
                t_idx += 2

    return best_en, explored, valid, invalid


def coords_to_moves(xs, ys):
    return [
        DIR_TO_CHAR[(int(xs[i + 1] - xs[i]), int(ys[i + 1] - ys[i]))]
        for i in range(len(xs) - 1)
    ]


_WARMED_UP = False


def _warmup():
    global _WARMED_UP
    if _WARMED_UP:
        return
    hp_arr = np.array([1, 0, 1, 0, 1, 1], dtype=np.int8)
    h_indices = np.array([0, 2, 4, 5], dtype=np.int32)
    n = 6
    gsize = 24
    n_rep = 2
    xs_all = np.zeros((n_rep, n), dtype=np.int32)
    ys_all = np.zeros((n_rep, n), dtype=np.int32)
    grids = np.full((n_rep, gsize, gsize), -1, dtype=np.int16)
    energies = np.zeros(n_rep, dtype=np.int32)
    c = gsize // 2
    for r in range(n_rep):
        for i in range(n):
            xs_all[r, i] = c - n // 2 + i
            ys_all[r, i] = c
            grids[r, xs_all[r, i], ys_all[r, i]] = i
    temps = np.array([0.2, 1.5], dtype=np.float64)
    temp_order = np.arange(n_rep, dtype=np.int32)
    best_x = xs_all[0].copy()
    best_y = ys_all[0].copy()
    _seed_numba(1)
    run_chunk(hp_arr, h_indices, n, gsize, n_rep, xs_all, ys_all, grids, energies,
              temps, temp_order, best_x, best_y, 0, 10, 5, 0)
    _WARMED_UP = True


def remc_numba(hp, time_limit=5.0, seed=None, n_replicas=20, t_min=0.14, t_max=2.20,
               swap_interval=100, chunk_sweeps=1500):
    _warmup()
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

    actual_seed = int(seed) if seed is not None else int(time.time() * 1000) % (2**31 - 1)
    _seed_numba(actual_seed)

    hp_arr = np.array([1 if ch == "H" else 0 for ch in hp], dtype=np.int8)
    h_indices = np.array([i for i, ch in enumerate(hp) if ch == "H"], dtype=np.int32)
    gsize = 2 * n + 12
    xs_all = np.zeros((n_replicas, n), dtype=np.int32)
    ys_all = np.zeros((n_replicas, n), dtype=np.int32)
    grids = np.full((n_replicas, gsize, gsize), -1, dtype=np.int16)
    energies = np.zeros(n_replicas, dtype=np.int32)

    center = gsize // 2
    for r in range(n_replicas):
        for i in range(n):
            xs_all[r, i] = center - n // 2 + i
            ys_all[r, i] = center
            grids[r, xs_all[r, i], ys_all[r, i]] = i
        energies[r] = calc_energy(hp_arr, h_indices, xs_all[r], ys_all[r], grids[r])

    temps = np.array(
        [t_min * (t_max / t_min) ** (i / (n_replicas - 1)) for i in range(n_replicas)],
        dtype=np.float64,
    )
    temp_order = np.arange(n_replicas, dtype=np.int32)
    best_x = xs_all[0].copy()
    best_y = ys_all[0].copy()
    best_en = int(energies[0])

    explored = 0
    valid = 0
    invalid = 0
    sweep_offset = 0
    history = [(0.0, best_en)]
    deadline = start + time_limit

    while time.perf_counter() < deadline:
        new_best, exp_c, val_c, inv_c = run_chunk(
            hp_arr, h_indices, n, gsize, n_replicas, xs_all, ys_all, grids, energies,
            temps, temp_order, best_x, best_y, best_en,
            chunk_sweeps, swap_interval, sweep_offset,
        )
        explored += int(exp_c)
        valid += int(val_c)
        invalid += int(inv_c)
        sweep_offset += chunk_sweeps

        if new_best < best_en:
            best_en = int(new_best)
            history.append((time.perf_counter() - start, best_en))

    return {
        "moves": coords_to_moves(best_x, best_y),
        "energy": best_en,
        "explored": explored,
        "valid": valid,
        "invalid": invalid,
        "time": time.perf_counter() - start,
        "history": history,
    }