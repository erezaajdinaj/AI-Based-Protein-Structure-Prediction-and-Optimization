import random
import time

from movement_generator import build_coords
from collision import is_valid
from energy import energy
from local_moves import random_valid_structure, valid_neighbor


def select(population, size, rng):
    contenders = rng.sample(population, min(size, len(population)))
    return min(contenders, key=lambda item: item[0])[1]


def crossover(parent_a, parent_b, rng, max_tries=10):
    length = len(parent_a)
    invalid = 0
    if length < 2:
        return list(parent_a), invalid
    for _ in range(max_tries):
        cut = rng.randint(1, length - 1)
        child = parent_a[:cut] + parent_b[cut:]
        if is_valid(build_coords(child)):
            return child, invalid
        invalid += 1
    return list(parent_a), invalid


def genetic_algorithm(hp, time_limit=5.0, seed=None, population_size=100,
                      mutation_rate=0.3, elite=2, tournament=3):
    rng = random.Random(seed)
    n = len(hp)
    start = time.perf_counter()

    explored = 0
    valid = 0
    invalid = 0

    population = []
    for _ in range(population_size):
        moves = random_valid_structure(n, rng)
        population.append((energy(hp, build_coords(moves)), moves))
        explored += 1
        valid += 1
    population.sort(key=lambda item: item[0])

    best_energy = population[0][0]
    best_moves = list(population[0][1])
    history = [(0.0, best_energy)]
    generations = 0

    while time.perf_counter() - start < time_limit:
        new_population = [(e, list(m)) for e, m in population[:elite]]
        while len(new_population) < population_size:
            parent_a = select(population, tournament, rng)
            parent_b = select(population, tournament, rng)
            child, bad = crossover(parent_a, parent_b, rng)
            invalid += bad
            explored += bad
            if rng.random() < mutation_rate:
                child, bad = valid_neighbor(child, rng)
                invalid += bad
                explored += bad
            explored += 1
            valid += 1
            new_population.append((energy(hp, build_coords(child)), child))
        population = sorted(new_population, key=lambda item: item[0])
        generations += 1

        if population[0][0] < best_energy:
            best_energy = population[0][0]
            best_moves = list(population[0][1])
            history.append((time.perf_counter() - start, best_energy))

    return {
        "moves": best_moves,
        "energy": best_energy,
        "explored": explored,
        "valid": valid,
        "invalid": invalid,
        "generations": generations,
        "time": time.perf_counter() - start,
        "history": history,
    }