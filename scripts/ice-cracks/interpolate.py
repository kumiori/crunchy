import numpy as np
import matplotlib.pyplot as plt

# from scipy.interpolate import griddata
# from scipy.interpolate import RegularGridInterpolator
from PIL import Image

import json

filename = "./src/larsen_coordinates.json"
with open(filename) as f:
    data = json.load(f)

thickness_image_path = "./src/larsen_thickness_map.png"
thickness_image = Image.open(thickness_image_path).convert("L")  # Convert to grayscale
thickness_array = np.array(thickness_image)

# Normalize the image data to [0, 1] (if not already normalized)
thickness_array = thickness_array / 255.0

x, y = data["x"], data["y"]
z = np.zeros_like(x)  # Create a z array with the same size as x and y
# Define the mesh points for interpolation (normalize to [0, 1])

dof_coordinates = np.column_stack(
    (x - np.min(x), y - np.min(y))
)  # Shift to positive domain

dof_coordinates[:, 0] /= np.ptp(x)  # Normalize x to [0, 1]
dof_coordinates[:, 1] /= np.ptp(y)  # Normalize y to [0, 1]
# Define your x, y, and field_values data (assume they're already available)

# Create a grid for interpolation
grid_x, grid_y = np.mgrid[
    min(x) : max(x) : 100j,  # 100 points in the x direction
    min(y) : max(y) : 100j,  # 100 points in the y direction
]

interpolator = RegularGridInterpolator(
    (grid_x, grid_y), thickness_array, bounds_error=False, fill_value=0.0
)
# Interpolate the thickness data onto the mesh points
field_values = interpolator(dof_coordinates)


# Interpolate the field values onto the grid
grid_z = griddata(
    points=(x, y), values=field_values, xi=(grid_x, grid_y), method="linear"
)

# Plot the interpolated field
plt.figure(figsize=(8, 6))
plt.contourf(grid_x, grid_y, grid_z, levels=50, cmap="viridis")  # Contour plot
plt.colorbar(label="Thickness (normalized)")  # Add color bar
plt.scatter(
    x, y, color="red", s=10, label="Points"
)  # Add original points for reference
plt.title("Interpolated Thickness Field")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()
