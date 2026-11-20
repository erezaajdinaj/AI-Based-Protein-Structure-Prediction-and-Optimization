import os

import matplotlib.pyplot as plt


def contacts(seq, coords):
    pairs = []
    for i in range(len(coords)):
        for j in range(i + 2, len(coords)):
            if seq[i] == "H" and seq[j] == "H":
                dist = abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1])
                if dist == 1:
                    pairs.append((i, j))
    return pairs


def draw(seq, coords, energy=None, save_path=None, label=None, show=True):
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    pairs = contacts(seq, coords)

    fig, ax = plt.subplots()
    ax.plot(xs, ys, color="gray", linewidth=2, zorder=1)
    for i, j in pairs:
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], "--", color="green", linewidth=3, zorder=2)
    for i, (x, y) in enumerate(coords):
        color = "red" if seq[i] == "H" else "blue"
        edge = "black" if i == 0 else color
        width = 4 if i == 0 else 1
        ax.scatter(x, y, s=400, color=color, edgecolors=edge, linewidths=width, zorder=3)
        ax.text(x, y, seq[i], ha="center", va="center", color="white", zorder=4)
    ax.set_xticks(range(min(xs) - 1, max(xs) + 2))
    ax.set_yticks(range(min(ys) - 1, max(ys) + 2))
    ax.grid(True)
    ax.set_aspect("equal")

    parts = []
    if label:
        parts.append(label)
    if energy is not None:
        parts.append(f"Energy = {energy}")
    parts.append(f"H-H contacts: {len(pairs)}")
    ax.set_title(" | ".join(parts))

    fig.tight_layout()
    if save_path:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)