import random
import time

from energy import energy

HEADINGS = ["R", "U", "L", "D"]
STEP = [(1, 0), (0, 1), (-1, 0), (0, -1)]
TURN = (0, 1, -1)


def decode(genes):
    heading = 0
    x, y = 1, 0
    coords = [(0, 0), (1, 0)]
    visited = {(0, 0), (1, 0)}
    moves = ["R"]
    for gene in genes:
        heading = (heading + TURN[gene]) % 4
        dx, dy = STEP[heading]
        x += dx
        y += dy
        if (x, y) in visited:
            return None
        visited.add((x, y))
        coords.append((x, y))
        moves.append(HEADINGS[heading])
    return moves, coords


def random_genome(n_genes, rng, max_tries=1000):
    for _ in range(max_tries):
        genes = []
        heading = 0
        position = (1, 0)
        visited = {(0, 0), (1, 0)}
        stuck = False
        for _ in range(n_genes):
            options = []
            for gene in (0, 1, 2):
                h = (heading + TURN[gene]) % 4
                nxt = (position[0] + STEP[h][0], position[1] + STEP[h][1])
                if nxt not in visited:
                    options.append((gene, h, nxt))
            if not options:
                stuck = True
                break
            gene, heading, position = rng.choice(options)
            visited.add(position)
            genes.append(gene)
        if not stuck:
            return genes
    return [0] * n_genes


def select(population, size, rng):
    contenders = rng.sample(population, min(size, len(population)))
    return min(contenders, key=lambda item: item[0])[1]


def crossover(parent_a, parent_b, rng, max_tries=10):
    invalid = 0
    if len(parent_a) < 2:
        return list(parent_a), invalid
    for _ in range(max_tries):
        cut = rng.randint(1, len(parent_a) - 1)
        child = parent_a[:cut] + parent_b[cut:]
        if decode(child) is not None:
            return child, invalid
        invalid += 1
    return list(parent_a), invalid


def mutate_valid(genes, rng, max_tries=30):
    invalid = 0
    if not genes:
        return list(genes), invalid
    for _ in range(max_tries):
        child = list(genes)
        index = rng.randrange(len(child))
        child[index] = rng.choice([g for g in (0, 1, 2) if g != child[index]])
        if decode(child) is not None:
            return child, invalid
        invalid += 1
    return list(genes), invalid


def genetic_algorithm(hp, time_limit=5.0, seed=None, population_size=100,
                      mutation_rate=0.9, elite=2, tournament=3,
                      dedupe=True, local_steps=0):
    rng = random.Random(seed)
    n = len(hp)
    start = time.perf_counter()

    if n <= 2:
        return {"moves": ["R"] * (n - 1), "energy": 0, "explored": 1, "valid": 1,
                "invalid": 0, "duplicates": 0, "generations": 0,
                "time": time.perf_counter() - start, "history": [(0.0, 0)]}

    n_genes = n - 2
    deadline = start + time_limit
    explored = 0
    valid = 0
    invalid = 0
    duplicates = 0

    population = []
    seen = set()
    for _ in range(population_size):
        genes = random_genome(n_genes, rng)
        if dedupe and tuple(genes) in seen:
            continue
        seen.add(tuple(genes))
        moves, coords = decode(genes)
        population.append((energy(hp, coords), genes))
        explored += 1
        valid += 1
    population.sort(key=lambda item: item[0])

    best_energy, best_genes = population[0]
    best_genes = list(best_genes)
    history = [(0.0, best_energy)]
    generations = 0

    while time.perf_counter() < deadline:
        new_population = [(e, list(g)) for e, g in population[:elite]]
        seen = {tuple(g) for _, g in new_population}
        attempts = 0

        while len(new_population) < population_size and time.perf_counter() < deadline:
            attempts += 1
            if attempts > population_size * 50:
                break
            if attempts > population_size * 10:
                genes = random_genome(n_genes, rng)
            else:
                parent_a = select(population, tournament, rng)
                parent_b = select(population, tournament, rng)
                genes, bad = crossover(parent_a, parent_b, rng)
                invalid += bad
                explored += bad
                if rng.random() < mutation_rate:
                    genes, bad = mutate_valid(genes, rng)
                    invalid += bad
                    explored += bad

            key = tuple(genes)
            if dedupe and key in seen:
                duplicates += 1
                continue
            seen.add(key)

            moves, coords = decode(genes)
            current_energy = energy(hp, coords)
            for _ in range(local_steps):
                candidate, bad = mutate_valid(genes, rng)
                invalid += bad
                explored += bad + 1
                valid += 1
                candidate_moves, candidate_coords = decode(candidate)
                candidate_energy = energy(hp, candidate_coords)
                if candidate_energy <= current_energy:
                    genes, current_energy = candidate, candidate_energy

            explored += 1
            valid += 1
            new_population.append((current_energy, genes))
            if current_energy < best_energy:
                best_energy = current_energy
                best_genes = list(genes)
                history.append((time.perf_counter() - start, best_energy))

        population = sorted(new_population, key=lambda item: item[0])
        generations += 1

    moves, coords = decode(best_genes)
    return {
        "moves": moves,
        "energy": best_energy,
        "explored": explored,
        "valid": valid,
        "invalid": invalid,
        "duplicates": duplicates,
        "generations": generations,
        "time": time.perf_counter() - start,
        "history": history,
    }