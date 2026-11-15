import argparse

from hp_model import to_hp
from hill_climbing import hill_climbing
from movement_generator import build_coords
from protein import draw


def parse_args():
    parser = argparse.ArgumentParser(description="2D HP protein folding")
    parser.add_argument("--seq", required=True, help="amino acid sequence")
    parser.add_argument("--time-limit", type=float, default=10.0, help="time budget in seconds")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        hp = to_hp(args.seq)
    except ValueError as error:
        print("Error:", error)
        return

    result = hill_climbing(hp, args.time_limit, args.seed)
    coords = build_coords(result["moves"])

    print("Sequence:", args.seq.upper())
    print("HP:", hp)
    print("Length:", len(hp))
    print("Energy:", result["energy"])
    print("Explored:", result["explored"])
    print("Valid:", result["valid"])
    print("Invalid:", result["invalid"])
    print("Time:", round(result["time"], 2))
    draw(hp, coords, result["energy"])


if __name__ == "__main__":
    main()