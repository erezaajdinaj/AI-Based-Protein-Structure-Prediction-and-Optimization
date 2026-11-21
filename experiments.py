import argparse
import csv
import importlib
import os
import tracemalloc

TEST_SEQUENCES = {
    "10_Length": "HHPPHHPPHP",
    "20_Benchmark": "HPHPPHHPHPPHPHHPPHPH",
    "24_Benchmark": "HHPPHPPHPPHPPHPPHPPHPPHH",
    "30_Length": "HHPPHHPPHHPPHHPPHHPPHHPPHHPPHH",
    "36_Benchmark": "PPPHHPPHHPPPPPHHHHHHHPPHHPPPPHHPPHPP",
    "48_Benchmark": "PPHPPHHPPHHPPPPPHHHHHHHHHHPPPPPPHHPPHHPPHPPHHHHH",
    "50_Benchmark": "HHPHPHPHPHHHHPHPPPHPPPHPPPPHPPPHPPPHPHHHHPHPHPHPHH",
}

ALGORITHMS = {
    "Hill_Climbing": ("hill_climbing", "hill_climbing"),
    "Simulated_Annealing": ("simulated_annealing", "simulated_annealing"),
    "Genetic_Algorithm": ("genetic_algorithm", "genetic_algorithm"),
    "Beam_Search": ("beam_search", "beam_search"),
    "Hybrid_Beam_SA": ("hybrid_beam_sa", "hybrid_beam_sa"),
    "REMC_Numba": ("remc_numba", "remc_numba"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run the experiments and save them as CSV files")
    parser.add_argument("--algos", default="all",
                        help="comma separated names, for example Genetic_Algorithm (default: all four)")
    parser.add_argument("--label", default=None,
                        help="name written in the Algorithm column (only when one algorithm is run)")
    parser.add_argument("--sequences", default="all",
                        help="comma separated sequence names, for example 36_Benchmark,50_Benchmark")
    parser.add_argument("--seeds", default="1,2,3,4,5", help="comma separated seeds")
    parser.add_argument("--time-limit", type=float, default=20.0, help="seconds per run")
    parser.add_argument("--out", default=".", help="output folder, for example results/ga_v2")
    return parser.parse_args()


def load_algorithm(name):
    module_name, function_name = ALGORITHMS[name]
    return getattr(importlib.import_module(module_name), function_name)


def run_experiments(args):
    names = list(ALGORITHMS) if args.algos == "all" else args.algos.split(",")
    for name in names:
        if name not in ALGORITHMS:
            raise SystemExit(f"Unknown algorithm '{name}'. Choose from: {', '.join(ALGORITHMS)}")
    if args.label and len(names) != 1:
        raise SystemExit("--label can be used only when exactly one algorithm is run")

    sequence_names = list(TEST_SEQUENCES) if args.sequences == "all" else args.sequences.split(",")
    for name in sequence_names:
        if name not in TEST_SEQUENCES:
            raise SystemExit(f"Unknown sequence '{name}'. Choose from: {', '.join(TEST_SEQUENCES)}")

    seeds = [int(s) for s in args.seeds.split(",")]
    algorithms = [(args.label or name, load_algorithm(name)) for name in names]

    os.makedirs(args.out, exist_ok=True)
    results_path = os.path.join(args.out, "experiment_results.csv")
    history_path = os.path.join(args.out, "convergence_history.csv")
    total = len(sequence_names) * len(seeds) * len(algorithms)
    print(f"Starting {total} runs of {args.time_limit} s each. Output folder: {args.out}")

    done = 0
    with open(results_path, mode="w", newline="") as results_file, \
            open(history_path, mode="w", newline="") as history_file:
        writer = csv.writer(results_file)
        history_writer = csv.writer(history_file)
        writer.writerow([
            "Sequence_Name", "Length", "Algorithm", "Seed",
            "Best_Energy", "Time_Sec", "Memory_MB",
            "Explored", "Valid", "Invalid",
            "Pruned", "Pruning_Percent", "Beam_Width", "Generations", "Duplicates",
        ])
        history_writer.writerow(["Sequence_Name", "Algorithm", "Seed", "Time_Sec", "Best_Energy"])

        for seq_name in sequence_names:
            seq = TEST_SEQUENCES[seq_name]
            print(f"\n--- {seq_name} (length {len(seq)}) ---")

            for seed in seeds:
                for algo_name, algo_func in algorithms:
                    done += 1
                    print(f"[{done}/{total}] {algo_name} (seed {seed})...", end="", flush=True)

                    tracemalloc.start()
                    result = algo_func(seq, time_limit=args.time_limit, seed=seed)
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()

                    memory_mb = peak / (1024 * 1024)
                    valid = result.get("valid", 0)
                    pruned = result.get("pruned", 0)
                    pruning_percent = 0.0
                    if pruned > 0 and valid > 0:
                        pruning_percent = pruned / valid * 100

                    writer.writerow([
                        seq_name, len(seq), algo_name, seed,
                        result.get("energy", 0), round(result.get("time", 0), 3), round(memory_mb, 4),
                        result.get("explored", 0), valid, result.get("invalid", 0),
                        pruned, round(pruning_percent, 2),
                        result.get("width", ""), result.get("generations", 0), result.get("duplicates", ""),
                    ])
                    for t, e in result.get("history", []):
                        history_writer.writerow([seq_name, algo_name, seed, round(t, 3), e])

                    results_file.flush()
                    history_file.flush()
                    print(f" energy {result.get('energy', 0)}, memory {round(memory_mb, 2)} MB")

    print(f"\nDone. Saved '{results_path}' and '{history_path}'.")


if __name__ == "__main__":
    run_experiments(parse_args())