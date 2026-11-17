import matplotlib.pyplot as plt

#proteina = "HPHPPHH"
#coords = [(0,0), (1,0), (2,0), (2,1), (1,1),(0,1), (0,2)]

def contacts(seq, coords):
    pairs = []
    for i in range(len(coords)):
        for j in range(i + 2, len(coords)):
            if seq[i] == "H" and seq[j] == "H":
                dist = abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1])
                if dist == 1:
                    pairs.append((i, j))
    return pairs


def draw(seq, coords, energy=None):
    min_x = min(c[0] for c in coords)
    min_y = min(c[1] for c in coords)
    coords = [(x - min_x, y - min_y) for x, y in coords]
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
    if energy is not None:
        ax.set_title(f"Energy = {energy}  (H-H contacts: {len(pairs)})")
    plt.show()