import argparse
import tracemalloc

from hp_model import to_hp
from hill_climbing import hill_climbing
from hybrid_beam_sa import hybrid_beam_sa
from simulated_annealing import simulated_annealing
from genetic_algorithm import genetic_algorithm
from beam_search import beam_search
from movement_generator import build_coords
from protein import draw

ALGORITHMS = {
    "hill": hill_climbing,
    "sa": simulated_annealing,
    "ga": genetic_algorithm,
    "beam": beam_search,
    "hybrid": hybrid_beam_sa,
}


def parse_args():
    parser = argparse.ArgumentParser(description="2D HP protein folding")
    parser.add_argument("--seq", required=True, help="sequence")
    parser.add_argument("--hp", action="store_true", help="sequence is already written as H/P")
    parser.add_argument("--algo", choices=list(ALGORITHMS), default="hill", help="algorithm")
    parser.add_argument("--time-limit", type=float, default=10.0, help="time budget in seconds")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--save", default=None, help="custom path for the figure (default: results folder)")
    parser.add_argument("--no-show", action="store_true", help="save the figure without opening a window")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.hp:
        hp = args.seq.upper()
        if set(hp) - {"H", "P"}:
            print("Error: with --hp the sequence must contain only H and P")
            return
    else:
        try:
            hp = to_hp(args.seq)
        except ValueError as error:
            print("Error:", error)
            return

    tracemalloc.start()
    result = ALGORITHMS[args.algo](hp, args.time_limit, args.seed)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    coords = build_coords(result["moves"])

    print("Algorithm:", args.algo)
    print("HP:", hp)
    print("Length:", len(hp))
    print("Energy:", result["energy"])
    print("Explored:", result["explored"])
    print("Valid:", result["valid"])
    print("Invalid:", result["invalid"])
    if "pruned" in result:
        print("Pruned:", result["pruned"])
        print("Final beam width:", result["width"])
    if "generations" in result:
        print("Generations:", result["generations"])
    print("Time:", round(result["time"], 2))
    print("Memory (MB):", round(peak / (1024 * 1024), 2))

    seed_text = "noseed" if args.seed is None else f"seed{args.seed}"
    save_path = args.save or f"results/{args.algo}_{len(hp)}_{seed_text}_E{result['energy']}.png"
    print("Figure saved to:", save_path)
    draw(hp, coords, result["energy"], save_path, args.algo, not args.no_show)


if __name__ == "__main__":
    main()