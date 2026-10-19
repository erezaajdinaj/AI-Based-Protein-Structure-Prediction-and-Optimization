import matplotlib.pyplot as plt

proteina = "HPHPPHH"
coords = [(0,0), (1,0), (2,0), (2,1), (1,1),(0,1), (0,2)]
fig,ax = plt.subplots()
xs = [c[0] for c in coords]
ys = [c[1] for c in coords]
ax.plot(xs,ys, color='red', linewidth=2)
for (i), (x,y) in enumerate(coords):
    ngjyra="red" if proteina[i] == "H" else "blue"
    ax.scatter(x,y, s=400, color=ngjyra, zorder=3)
    ax.text(x, y, proteina[i], ha="center", va="center", color="white", zorder=4)
ax.set_xticks(range(min(xs) - 1, max(xs) + 2))
ax.set_yticks(range(min(ys) - 1, max(ys) + 2))
ax.grid(True)
ax.set_aspect("equal")
plt.show()