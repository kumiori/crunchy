from dolfinx.mesh import create_rectangle
from mpi4py import MPI


def create_2d_mesh(dimensions=(1, 1), resolution=(10, 10)):
    """Generate a simple rectangular mesh."""
    return create_rectangle(
        MPI.COMM_WORLD, points=((0.0, 0.0), dimensions), n=resolution, cell_type=2
    )


from mpi4py import MPI
import random
import math
import gmsh


def generate_holes(num_holes, R, lc, min_radius=None, max_radius=None):
    """
    Generate random positions and radii for holes inside a circle.

    Parameters:
        num_holes (int): Number of holes to generate.
        R (float): Radius of the outer circle.
        lc (float): Base length scale.
        min_radius (float, optional): Minimum hole radius. Defaults to `lc`.
        max_radius (float, optional): Maximum hole radius. Defaults to `5 * lc`.

    Returns:
        List of tuples: [(x, y, radius), ...] representing hole positions and radii.
    """
    if min_radius is None:
        min_radius = lc
    if max_radius is None:
        max_radius = 5 * lc

    holes = [
        (
            random.uniform(-R + max_radius, R - max_radius),  # x-position
            random.uniform(-R + max_radius, R - max_radius),  # y-position
            random.uniform(min_radius, max_radius),  # radius
        )
        for _ in range(num_holes)
    ]
    return holes


def validate_holes(holes, R):
    """
    Validate hole positions and radii to ensure they are within bounds
    and do not intersect with each other.

    Parameters:
        holes (list of tuples): [(x, y, radius), ...] representing hole positions and radii.
        R (float): Radius of the outer circle.

    Returns:
        bool: True if all holes are valid, False otherwise.
    """
    for i, (x1, y1, r1) in enumerate(holes):
        # Check that the hole does not touch the boundary of the circle
        if math.sqrt(x1**2 + y1**2) + r1 > R:
            print(f"Error: Hole at ({x1}, {y1}) with radius {r1} touches the boundary.")
            return False

        # Check for intersections with other holes
        for j, (x2, y2, r2) in enumerate(holes):
            if i != j:  # Skip self-comparison
                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if distance < r1 + r2:  # Holes overlap
                    print(
                        f"Error: Hole at ({x1}, {y1}) with radius {r1} intersects "
                        f"with hole at ({x2}, {y2}) with radius {r2}."
                    )
                    return False

    # All checks passed
    return True


def mesh_circle_with_holes_gmshapi(
    name,
    R,
    lc,
    tdim,
    num_holes,
    hole_radius=None,
    hole_positions=None,
    refinement_factor=0.1,
    order=1,
    msh_file=None,
    comm=MPI.COMM_WORLD,
):
    """
    Create a 2D circle mesh with holes using the Python API of Gmsh.
    """
    # Perform Gmsh work only on rank = 0
    if comm.rank == 0:
        # Initialise gmsh and set options
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 6)

        model = gmsh.model()
        model.add(name)
        model.setCurrent(name)

        # Define main circle
        p0 = model.geo.addPoint(0.0, 0.0, 0, lc)
        p1 = model.geo.addPoint(R, 0.0, 0, lc)
        p2 = model.geo.addPoint(0.0, R, 0.0, lc)
        p3 = model.geo.addPoint(-R, 0, 0, lc)
        p4 = model.geo.addPoint(0, -R, 0, lc)
        c1 = model.geo.addCircleArc(p1, p0, p2)
        c2 = model.geo.addCircleArc(p2, p0, p3)
        c3 = model.geo.addCircleArc(p3, p0, p4)
        c4 = model.geo.addCircleArc(p4, p0, p1)
        outer_circle = model.geo.addCurveLoop([c1, c2, c3, c4])

        # Generate holes if not provided
        if hole_positions is None or hole_radius is None:
            while True:
                holes = generate_holes(num_holes, R, lc)
                if validate_holes(holes, R):
                    break
            hole_positions = [(x, y) for x, y, _ in holes]
            hole_radius = [r for _, _, r in holes]

        # Add holes to the model
        hole_loops = []
        for (hx, hy), hr in zip(hole_positions, hole_radius):
            h0 = model.geo.addPoint(hx, hy, 0, lc * refinement_factor)
            h1 = model.geo.addPoint(hx + hr, hy, 0, lc * refinement_factor)
            h2 = model.geo.addPoint(hx, hy + hr, 0, lc * refinement_factor)
            h3 = model.geo.addPoint(hx - hr, hy, 0, lc * refinement_factor)
            h4 = model.geo.addPoint(hx, hy - hr, 0, lc * refinement_factor)
            hc1 = model.geo.addCircleArc(h1, h0, h2)
            hc2 = model.geo.addCircleArc(h2, h0, h3)
            hc3 = model.geo.addCircleArc(h3, h0, h4)
            hc4 = model.geo.addCircleArc(h4, h0, h1)
            hole_loop = model.geo.addCurveLoop([hc1, hc2, hc3, hc4])
            hole_loops.append(hole_loop)

        # Define the surface with holes
        plane_surface = model.geo.addPlaneSurface([outer_circle] + hole_loops)

        # Synchronize and generate mesh
        model.geo.synchronize()
        surface_entities = [model[1] for model in model.getEntities(tdim)]
        model.addPhysicalGroup(tdim, surface_entities, tag=5)
        model.setPhysicalName(tdim, 5, "Film surface")

        gmsh.model.mesh.setOrder(order)
        model.mesh.generate(tdim)

        # Optional: Write msh file
        if msh_file is not None:
            gmsh.write(msh_file)

        gmsh.finalize()

    return gmsh.model if comm.rank == 0 else None, tdim


def mesh_circle_with_holes_gmshapi_old(
    name,
    R,
    lc,
    tdim,
    num_holes,
    hole_radius,
    hole_positions=None,
    refinement_factor=0.1,
    order=1,
    msh_file=None,
    comm=MPI.COMM_WORLD,
):
    """
    Create 2d circle mesh with holes using the Python API of Gmsh.
    """
    # Perform Gmsh work only on rank = 0
    if comm.rank == 0:
        import gmsh

        # Initialise gmsh and set options
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 6)

        model = gmsh.model()
        model.add("CircleWithHoles")
        model.setCurrent("CircleWithHoles")

        # Define main circle
        p0 = model.geo.addPoint(0.0, 0.0, 0, lc)
        p1 = model.geo.addPoint(R, 0.0, 0, lc)
        p2 = model.geo.addPoint(0.0, R, 0.0, lc)
        p3 = model.geo.addPoint(-R, 0, 0, lc)
        p4 = model.geo.addPoint(0, -R, 0, lc)
        c1 = model.geo.addCircleArc(p1, p0, p2)
        c2 = model.geo.addCircleArc(p2, p0, p3)
        c3 = model.geo.addCircleArc(p3, p0, p4)
        c4 = model.geo.addCircleArc(p4, p0, p1)
        outer_circle = model.geo.addCurveLoop([c1, c2, c3, c4])

        # Define holes
        hole_points = []
        hole_loops = []
        hole_tags = []

        if hole_positions is None:
            # Generate random positions for the holes
            hole_positions = [
                (
                    random.uniform(-R + hole_radius, R - hole_radius),
                    random.uniform(-R + hole_radius, R - hole_radius),
                )
                for _ in range(num_holes)
            ]

        if hole_radius is None:
            # Generate random radii for the holes
            hole_radius = [random.uniform(lc, 5 * lc) for _ in range(num_holes)]

        for i, (hx, hy) in enumerate(hole_positions):
            h0 = model.geo.addPoint(hx, hy, 0, lc * refinement_factor)
            h1 = model.geo.addPoint(hx + hole_radius, hy, 0, lc * refinement_factor)
            h2 = model.geo.addPoint(hx, hy + hole_radius, 0, lc * refinement_factor)
            h3 = model.geo.addPoint(hx - hole_radius, hy, 0, lc * refinement_factor)
            h4 = model.geo.addPoint(hx, hy - hole_radius, 0, lc * refinement_factor)
            hc1 = model.geo.addCircleArc(h1, h0, h2)
            hc2 = model.geo.addCircleArc(h2, h0, h3)
            hc3 = model.geo.addCircleArc(h3, h0, h4)
            hc4 = model.geo.addCircleArc(h4, h0, h1)
            hole_loop = model.geo.addCurveLoop([hc1, hc2, hc3, hc4])
            hole_loops.append(hole_loop)
            # Mark hole boundaries as separate physical groups
            hole_boundary_tag = model.addPhysicalGroup(
                1, [hc1, hc2, hc3, hc4], tag=i + 10
            )
            hole_tags.append(hole_boundary_tag)
            model.setPhysicalName(1, hole_boundary_tag, f"Hole {i}")

        # Define the surface with holes
        plane_surface = model.geo.addPlaneSurface([outer_circle] + hole_loops)

        model.geo.synchronize()
        surface_entities = [model[1] for model in model.getEntities(tdim)]
        model.addPhysicalGroup(tdim, surface_entities, tag=5)
        model.setPhysicalName(tdim, 5, "Film surface")

        gmsh.model.mesh.setOrder(order)
        model.mesh.generate(tdim)

        # Optional: Write msh file
        if msh_file is not None:
            gmsh.write(msh_file)

        gmsh.finalize()

    return gmsh.model if comm.rank == 0 else None, tdim
