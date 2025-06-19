from PIL import Image
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import pyvista as pv

from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import json

pv.start_xvfb()

filename = "./larsen_coordinates.json"
with open(filename) as f:
    data = json.load(f)

x, y = data["x"], data["y"]
z = np.zeros_like(x)  # Create a z array with the same size as x and y

# Load the thickness image
thickness_image_path = "./larsen_thickness_map.png"
thickness_image = Image.open(thickness_image_path).convert("L")  # Convert to grayscale
thickness_array = np.array(thickness_image)

# Normalize the image data to [0, 1] (if not already normalized)
thickness_array = thickness_array / 255.0

# Define the grid for the thickness image (normalized to [0, 1])
image_resolution = thickness_array.shape
x_grid = np.linspace(0, 1, image_resolution[1])  # Columns
y_grid = np.linspace(0, 1, image_resolution[0])  # Rows

# display the thickness image
plt.figure(figsize=(8, 6))
plt.imshow(thickness_array, cmap="viridis")
# plt.show()
plt.savefig("_thickness_image.png")
plt.close()

# Create an interpolator for the image data
interpolator = RegularGridInterpolator(
    (y_grid, x_grid), thickness_array, bounds_error=False, fill_value=0.0
)

# Define the mesh points for interpolation (normalize to [0, 1])
dof_coordinates = np.column_stack((x, y))  # Use raw x and y directly

# Scale `dof_coordinates` to [0, 1] to match the grid
dof_coordinates[:, 0] = (dof_coordinates[:, 0] - min(x)) / (max(x) - min(x))
dof_coordinates[:, 1] = (dof_coordinates[:, 1] - min(y)) / (max(y) - min(y))

field_values = interpolator(dof_coordinates)

__import__("pdb").set_trace()

# Create a pyvista PolyData object for visualization
points_3d = np.column_stack((x, y, z))  # Create a 3D array

# Create the PolyData object
polygon_pv = pv.PolyData(points_3d)

polygon_pv["Thickness"] = field_values  # Add thickness as point data
# Visualize the scalar field on the mesh
plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(polygon_pv, scalars="Thickness", cmap="viridis", point_size=10)
# plotter.add_scalar_bar(title="Thickness (normalized)", vertical=True)
plotter.screenshot("thickness_field.png")  # Save to file


# Create a grid for interpolation
grid_x, grid_y = np.mgrid[
    min(x) : max(x) : 100j,  # 100 points in the x direction
    min(y) : max(y) : 100j,  # 100 points in the y direction
]


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
plt.savefig("interpolated_thickness_field.png")
# plt.show()
