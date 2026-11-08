import argparse

from hp_model import to_hp
from movement_generator import build_coords
from collision import is_valid
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

    moves = ["R"] * (len(hp) - 1)
    coords = build_coords(moves)

    print("Sequence:", args.seq.upper())
    print("HP:", hp)
    print("Length:", len(hp))
    print("Time limit:", args.time_limit)
    print("Seed:", args.seed)
    valid = is_valid(coords)
    print("Valid:", valid)

    if valid:
        draw(hp, coords)


if __name__ == "__main__":
    main()