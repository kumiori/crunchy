import importlib.util
import numpy as np
import matplotlib.pyplot as plt
import gmsh
import json

filename = "./src/larsen_coordinates.json"
with open(filename) as f:
    data = json.load(f)

x, y = data["x"], data["y"]
data = {"x": x, "y": y}
# with open("larsen_coordinates.json", "w") as f:
#     json.dump(data, f, indent=4)
# __import__("pdb").set_trace()
# Execute the file to retrieve the list of points
# spec = importlib.util.spec_from_file_location("larsen_points", file_path)
# larsen_points = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(larsen_points)

# x = np.array(larsen_points.x)
# y = np.array(larsen_points.y)

# Plot the points with matplotlib
plt.figure(figsize=(8, 6))
plt.plot(x, y, "bo-", label="Points")
for i, (xi, yi) in enumerate(zip(x, y)):
    plt.text(xi, yi, str(i), fontsize=8, ha="right", color="red")

scale_length_km = 100 * 1.60934  # Convert 100 miles to km

# Approximate the horizontal line length based on the plot's scale
# x_start = min(x) + 0.1 * (
#     max(x) - min(x)
# )  # Start slightly to the right of the left edge
# x_end = (
#     x_start + scale_length_km / 1000
# )  # Adjust scale for the plot's unit (e.g., km to plot units)
y_scale = min(y) - 0.02 * (
    max(y) - min(y)
)  # Place the scale line slightly below the lowest point
x_start, x_end = -0.9, -0.65 * 1.60934
# Add the scale line
plt.plot(
    [x_start, x_end],
    [y_scale, y_scale],
    "k-",
    linewidth=2,
    label="~100 km",
)
plt.text(
    (x_start + x_end) / 2,
    y_scale - 0.04,
    "~100 km",
    fontsize=10,
    ha="center",
    color="black",
)

plt.title("Point Plot of Larsen Points")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.axis("equal")
plt.grid(True)
plt.legend()
# plt.show()
plt.savefig("larsen_points.png")


# Create a Python script for gmsh to generate a 2D mesh
def create_gmsh_2d_mesh(
    points_x, points_y, output_file="larsen_mesh.msh", mesh_size=0.1
):
    gmsh.initialize()
    gmsh.model.add("Larsen Ice Shelf")

    # Add points to gmsh
    point_tags = []
    for i, (px, py) in enumerate(zip(points_x, points_y)):
        tag = gmsh.model.geo.addPoint(px, py, 0, meshSize=mesh_size)
        point_tags.append(tag)

    # Create a closed loop using the points
    line_tags = []
    for i in range(len(point_tags) - 1):
        line_tags.append(gmsh.model.geo.addLine(point_tags[i], point_tags[i + 1]))
    line_tags.append(gmsh.model.geo.addLine(point_tags[-1], point_tags[0]))

    # Create a curve loop and a plane surface
    curve_loop = gmsh.model.geo.addCurveLoop(line_tags)
    surface = gmsh.model.geo.addPlaneSurface([curve_loop])

    # Add physical groups
    gmsh.model.geo.addPhysicalGroup(2, [surface], tag=1)  # Tag 1 for the surface
    gmsh.model.geo.addPhysicalGroup(1, line_tags, tag=2)  # Tag 2 for the boundary lines

    # Synchronize and generate the mesh
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    # Save the mesh
    gmsh.write(output_file)
    gmsh.finalize()


from shapely.geometry import Polygon
from shapely.geometry import LineString
from matplotlib.patches import Polygon as MplPolygon

# Create a Shapely polygon from the points
polygon = Polygon(zip(x, y))

# Plot the polygon using Matplotlib
plt.figure(figsize=(8, 6))
plt.gca().add_patch(
    MplPolygon(
        list(polygon.exterior.coords),
        closed=True,
        edgecolor="blue",
        facecolor="lightblue",
        alpha=0.5,
    )
)

# Annotate the plot
plt.plot(x, y, "ro", label="Vertices")  # Plot the vertices
for i, (xi, yi) in enumerate(zip(x, y)):
    plt.text(xi, yi, str(i), fontsize=8, ha="right", color="black")

# Formatting
plt.title("Shapely Polygon Representation of Larsen Points")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.axis("equal")
plt.grid(False)
plt.legend()
# plt.show()
plt.savefig("larsen_polygon.png")
# Generate the 2D mesh for the Larsen points
# exit()

from dolfinx.io import gmshio, XDMFFile
from mpi4py import MPI


for meshsize in [0.1, 0.05, 0.01]:
    create_gmsh_2d_mesh(x, y, f"meshes/larsen_mesh_{meshsize}.msh", meshsize)
    # Load the mesh using dolfinx
    with XDMFFile(MPI.COMM_WORLD, f"meshes/larsen_mesh_{meshsize}.xdmf", "w") as xdmf:
        # with gmshio.read_from_msh(
        #     f"meshes/larsen_mesh_{meshsize}.msh", comm=MPI.COMM_WORLD
        # ) as mesh:
        #     xdmf.write_mesh(mesh)
        #     xdmf.write_meshtags(mesh)
        #     # xdmf.write_field(mesh.geometry.x, "coordinates")

        mesh, cell_tags, facet_tags = gmshio.read_from_msh(
            f"meshes/larsen_mesh_{meshsize}.msh", comm=MPI.COMM_WORLD
        )
        xdmf.write_mesh(mesh)
        if facet_tags is not None:
            xdmf.write_meshtags(
                facet_tags, mesh.geometry
            )  # Write facet tags to the XDMF file
        # Similarly, save cell tags if needed
        if cell_tags is not None:
            xdmf.write_meshtags(
                cell_tags, mesh.geometry
            )  # Write cell tags to the XDMF file
