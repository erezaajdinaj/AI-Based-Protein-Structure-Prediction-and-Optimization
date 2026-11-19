import csv
import tracemalloc

from hill_climbing import hill_climbing
from simulated_annealing import simulated_annealing
from genetic_algorithm import genetic_algorithm
from beam_search import beam_search

TEST_SEQUENCES = {
    "10_Length": "HHPPHHPPHP",
    "20_Benchmark": "HPHPPHHPHPPHPHHPPHPH",
    "24_Benchmark": "HHPPHPPHPPHPPHPPHPPHPPHH",
    "30_Length": "HHPPHHPPHHPPHHPPHHPPHHPPHHPPHH",
    "36_Benchmark": "PPPHHPPHHPPPPPHHHHHHHPPHHPPPPHHPPHPP",
    "48_Benchmark": "PPHPPHHPPHHPPPPPHHHHHHHHHHPPPPPPHHPPHHPPHPPHHHHH",
    "50_Benchmark": "HHPHPHPHPHHHHPHPPPHPPPHPPPPHPPPHPPPHPHHHHPHPHPHPHH",
}

SEEDS = [1, 2, 3, 4, 5]
TIME_LIMIT = 20.0

#SEEDS=[1]
#TIME_LIMIT=2.0

ALGORITHMS = [
    ("Hill_Climbing", hill_climbing),
    ("Simulated_Annealing", simulated_annealing),
    ("Genetic_Algorithm", genetic_algorithm),
    ("Beam_Search", beam_search)
]


def run_experiments(filename="experiment_results1.csv", history_filename="convergence_history.csv"):
    print("Starting experiments. This will take some time...")

    with open(filename, mode="w", newline="") as file, \
            open(history_filename, mode="w", newline="") as history_file:
        writer = csv.writer(file)
        history_writer = csv.writer(history_file)
        writer.writerow([
            "Sequence_Name", "Length", "Algorithm", "Seed",
            "Best_Energy", "Time_Sec", "Memory_MB",
            "Explored", "Valid", "Invalid",
            "Pruned", "Pruning_Percent", "Generations"
        ])
        history_writer.writerow(["Sequence_Name", "Algorithm", "Seed", "Time_Sec", "Best_Energy"])

        for seq_name, seq in TEST_SEQUENCES.items():
            n = len(seq)
            print(f"\n--- Testing Sequence: {seq_name} (Length: {n}) ---")

            for seed in SEEDS:
                for algo_name, algo_func in ALGORITHMS:
                    print(f"Running {algo_name} (Seed: {seed})...", end="", flush=True)

                    tracemalloc.start()

                    result = algo_func(seq, time_limit=TIME_LIMIT, seed=seed)

                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()

                    memory_mb = peak / (1024 * 1024)

                    energy = result.get("energy", 0)
                    explored = result.get("explored", 0)
                    valid = result.get("valid", 0)
                    invalid = result.get("invalid", 0)
                    time_taken = result.get("time", 0)
                    pruned = result.get("pruned", 0)
                    generations = result.get("generations", 0)

                    pruning_percent = 0.0
                    if pruned > 0 and valid > 0:
                        pruning_percent = (pruned / valid) * 100

                    writer.writerow([
                        seq_name, n, algo_name, seed,
                        energy, round(time_taken, 3), round(memory_mb, 4),
                        explored, valid, invalid,
                        pruned, round(pruning_percent, 2), generations
                    ])

                    for t, e in result.get("history", []):
                        history_writer.writerow([seq_name, algo_name, seed, round(t, 3), e])

                    file.flush()
                    history_file.flush()

                    print(f" Done! (Energy: {energy}, RAM: {round(memory_mb, 2)} MB)")

    print(f"\nExperiments completed! Data saved to '{filename}' and '{history_filename}'.")


if __name__ == "__main__":
    run_experiments()