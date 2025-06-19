import matplotlib.pyplot as plt
import numpy as np

# Define the 2D ice shelf geometry
x = np.linspace(0, 10, 100)
top_edge = (
    np.sin(x) * 0.5 + 1.5 + np.random.uniform(-0.2, 0.2, size=x.shape)
)  # Jagged top edge
bottom_edge = np.ones_like(x) * 0.5 + np.cos(x) * 0.1  # Smoother bottom edge

# Sharp interior features (crevasses)
crevasse_x = [2, 4, 6, 8]
crevasse_depth = [1.2, 1.3, 1.1, 1.4]

# Plot the geometry
plt.figure(figsize=(10, 5))
plt.plot(x, top_edge, label="Top Edge (Fractures)", color="blue")
plt.plot(x, bottom_edge, label="Bottom Edge (Ocean Interface)", color="green")

# Add crevasses
for cx, cy in zip(crevasse_x, crevasse_depth):
    plt.plot(
        [cx, cx],
        [cy, bottom_edge[np.argmin(np.abs(x - cx))]],
        color="red",
        linestyle="--",
        label="Crevasse" if cx == crevasse_x[0] else "",
    )

# Detached icebergs
iceberg_x = [9, 10]
iceberg_y = [0.7, 0.6]
plt.scatter(iceberg_x, iceberg_y, color="purple", label="Detached Icebergs")

# Plot settings
plt.title("2D Representation of an Ice Shelf Geometry")
plt.xlabel("Horizontal Distance (arbitrary units)")
plt.ylabel("Vertical Distance (arbitrary units)")
plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
plt.legend()
plt.grid()
plt.show()
