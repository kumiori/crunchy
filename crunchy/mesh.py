from dolfinx.mesh import create_rectangle
from mpi4py import MPI
from numpy import pi


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
            print(
                f"Warning: Hole at ({x1:.2f}, {y1:.2f}) with radius {r1:.2f} touches the boundary. Retrying..."
            )
            return False

        # Check for intersections with other holes
        for j, (x2, y2, r2) in enumerate(holes):
            if i != j:  # Skip self-comparison
                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if distance < r1 + r2:  # Holes overlap
                    print(
                        f"Warning: Hole at ({x1:.2f}, {y1:.2f}) with radius {r1:.2f} intersects "
                        f"with hole at ({x2:.2f}, {y2:.2f}) with radius {r2:.2f}. Retrying..."
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
        hole_tags = []

        for i, ((hx, hy), hr) in enumerate(zip(hole_positions, hole_radius)):
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

            hole_boundary_tag = model.addPhysicalGroup(
                1, [hc1, hc2, hc3, hc4], tag=i + 10
            )
            hole_tags.append(hole_boundary_tag)
            model.setPhysicalName(1, hole_boundary_tag, f"Hole {i}")

        # Define the surface with holes
        plane_surface = model.geo.addPlaneSurface([outer_circle] + hole_loops)

        # Synchronize and generate mesh
        model.geo.synchronize()
        assert len(model.getEntities(2)) > 0, "No 2D entities found."

        surface_entities = [model[1] for model in model.getEntities(tdim)]
        model.addPhysicalGroup(tdim, surface_entities, tag=666)
        model.setPhysicalName(tdim, 666, "Domain surface")

        for dim in range(4):  # Loop over dimensions 0 (points) to 3 (volumes)
            entities = gmsh.model.getEntities(dim)
            print(f"Entities of dimension {dim}: {entities}")

        gmsh.model.mesh.setOrder(order)
        gmsh.option.setNumber("Mesh.Optimize", 2)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        model.mesh.generate(tdim)

        # Optional: Write msh file
        if msh_file is not None:
            gmsh.write(msh_file)

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


def mesh_matrix_fiber_reinforced(comm, geom_parameters):
    """
    Generate a square domain with a circular hole using Gmsh.

    Parameters:
        geom_parameters: dict
            Dictionary containing geometric parameters:
            - "L"  : Length of the square domain
            - "R"  : Radius of the circular hole
        lc: float
            Characteristic length for meshing.
    """
    gmsh.initialize()
    gmsh.model.add("Square_with_Hole")

    # Geometric parameters
    L = geom_parameters["L"]
    R = geom_parameters["R_inner"]
    lc = geom_parameters["lc"]
    tdim = geom_parameters["geometric_dimension"]
    # Use the Open Cascade Kernel (OCC)
    rect = gmsh.model.occ.addRectangle(-L / 2, -L / 2, 0, L, L, tag=1)  # Square
    hole = gmsh.model.occ.addDisk(0, 0, 0, R, R, tag=2)  # Circular hole
    # Boolean cut: Subtract the hole from the rectangle
    domain_with_hole = gmsh.model.occ.cut([(2, rect)], [(2, hole)])
    gmsh.model.occ.synchronize()  # Sync CAD with GMSH
    domain_surfaces = domain_with_hole[0]  # Extract the remaining surface entity

    hole_boundary = gmsh.model.getBoundary(domain_surfaces, oriented=False)
    # Assign a uniform mesh size to all surfaces
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)

    # Define physical groups for boundary conditions
    domain_tag = gmsh.model.addPhysicalGroup(
        2, [domain_with_hole[0][0][1]], name="Domain"
    )
    gmsh.model.setPhysicalName(2, domain_tag, "Domain")
    hole_edges = [edge[1] for edge in hole_boundary]  # Extract edge IDs
    hole_tag = gmsh.model.addPhysicalGroup(1, hole_edges)
    gmsh.model.setPhysicalName(1, hole_tag, "Hole_Boundary")
    outer_boundary = gmsh.model.getBoundary([(2, rect)], oriented=False)

    top_edges = []
    bottom_edges = []
    for edge in outer_boundary:
        edge_id = edge[1]  # Get the tag of the edge
        com = gmsh.model.occ.getCenterOfMass(1, edge_id)  # Get edge centroid
        y_com = com[1]  # Extract Y coordinate
        if abs(y_com - (L / 2)) < 1e-6:  # Top boundary
            top_edges.append(edge_id)
        elif abs(y_com + (L / 2)) < 1e-6:  # Bottom boundary
            bottom_edges.append(edge_id)

    top_tag = gmsh.model.addPhysicalGroup(1, top_edges)
    gmsh.model.setPhysicalName(1, top_tag, "Top_Boundary")
    bottom_tag = gmsh.model.addPhysicalGroup(1, bottom_edges)
    gmsh.model.setPhysicalName(1, bottom_tag, "Bottom_Boundary")

    # Generate and save the mesh
    gmsh.model.mesh.generate(tdim)

    # Save mesh to MSH file
    msh_filename = "square_with_hole.msh"
    gmsh.write(msh_filename)

    # Finalize Gmsh
    return gmsh.model if comm.rank == 0 else None, tdim


def mesh_disk_with_rhomboidal_hole(comm=MPI.COMM_WORLD, geom_parameters=None):
    """
    Creates a 2D disk with a rhomboidal hole using Gmsh.

    Args:
        comm (MPI.Comm): MPI communicator.
        geom_parameters (dict): Dictionary containing geometric parameters:
            - 'R_outer': Outer disk radius.
            - 'angle': Angle of the rhomboid in degrees.
            - 'axis': Axis length of the rhomboid.
            - 'lc': Mesh element size.
            - 'a': Half-width of the refined symmetric region (default 7 * lc).

    Returns:
        None (Gmsh model is created and can be meshed/exported).
    """
    if geom_parameters is None:
        geom_parameters = {
            "R_outer": 1.0,  # Outer disk radius
            "angle": 45.0,  # Angle of the rhomboid in degrees
            "axis": 0.5,  # Axis length of the rhomboid
            "lc": 0.05,  # Mesh element size
            "a": None,  # Half-width of the refined region (-a < x < a)
        }

    R_outer = geom_parameters["R_outer"]
    angle = geom_parameters["angle"]
    axis = geom_parameters["axis"]
    lc = geom_parameters["lc"]
    a = geom_parameters["a"] if geom_parameters["a"] is not None else 7 * lc

    if comm.rank == 0:
        import warnings
        import gmsh
        from numpy import tan, pi

        warnings.filterwarnings("ignore")

        # Initialize gmsh
        gmsh.initialize()
        gmsh.model.add("DiskWithRhomboidalHole")

        # Create outer circle (disk boundary)
        outer_circle = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, R_outer, R_outer, tag=1)

        # Create rhomboidal hole
        angle_rad = angle * (pi / 180.0)  # Convert angle to radians
        dx = axis * 0.5
        dy = axis * 0.5 * (1 / tan(angle_rad / 2))  # Adjust for rhomboid shape
        print(f"dx = {dx}, dy = {dy}")

        rhomboid_points = [
            gmsh.model.occ.addPoint(-dx, 0, 0),
            gmsh.model.occ.addPoint(0, -dy, 0),
            gmsh.model.occ.addPoint(dx, 0, 0),
            gmsh.model.occ.addPoint(0, dy, 0),
        ]

        rhomboid_lines = [
            gmsh.model.occ.addLine(rhomboid_points[0], rhomboid_points[1]),
            gmsh.model.occ.addLine(rhomboid_points[1], rhomboid_points[2]),
            gmsh.model.occ.addLine(rhomboid_points[2], rhomboid_points[3]),
            gmsh.model.occ.addLine(rhomboid_points[3], rhomboid_points[0]),
        ]

        rhomboid_loop = gmsh.model.occ.addCurveLoop(rhomboid_lines)
        rhomboid_surface = gmsh.model.occ.addPlaneSurface([rhomboid_loop])

        cut_entities, _ = gmsh.model.occ.cut(
            [(2, outer_circle)], [(2, rhomboid_surface)]
        )
        surface_tag = cut_entities[0][1]  # Extract tag from result

        # Synchronize before meshing
        gmsh.model.occ.synchronize()

        # Define physical groups
        gmsh.model.addPhysicalGroup(2, [surface_tag], tag=1)
        gmsh.model.setPhysicalName(2, 1, "DiskDomain")
        gmsh.model.addPhysicalGroup(2, [rhomboid_surface], tag=2)
        gmsh.model.setPhysicalName(2, 2, "RhomboidHole")

        # Mesh settings
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)

        refinement_field = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(
            refinement_field, "VIn", lc / 3
        )  # Finer mesh inside
        gmsh.model.mesh.field.setNumber(
            refinement_field, "VOut", lc
        )  # Coarser mesh outside
        gmsh.model.mesh.field.setNumber(refinement_field, "XMin", -a)
        gmsh.model.mesh.field.setNumber(refinement_field, "XMax", a)
        gmsh.model.mesh.field.setNumber(refinement_field, "YMin", -R_outer)
        gmsh.model.mesh.field.setNumber(refinement_field, "YMax", R_outer)
        gmsh.model.mesh.field.setAsBackgroundMesh(refinement_field)

        gmsh.model.mesh.generate(2)

        # Save mesh to file
        gmsh.write("disc_with_rhomboidal_hole.msh")

        print("Mesh created and saved as 'disc_with_rhomboidal_hole.msh'")

        # Finalize gmsh
        # gmsh.finalize()

        tdim = 2

    return gmsh.model if comm.rank == 0 else None, tdim


def mesh_rect_with_rhomboidal_hole(comm=MPI.COMM_WORLD, geom_parameters=None):
    """
    Creates a 2D rectangular mesh with a rhomboidal hole using Gmsh.

    Args:
        comm (MPI.Comm): MPI communicator.
        geom_parameters (dict): Dictionary containing geometric parameters:
            - 'Lx': Length of the rectangle along the x-axis.
            - 'Ly': Length of the rectangle along the y-axis.
            - 'angle': Angle of the rhomboid in degrees.
            - 'axis': Axis length of the rhomboid.
            - 'lc': Mesh element size.

    Returns:
        None (Gmsh model is created and can be meshed/exported).
    """
    if geom_parameters is None:
        geom_parameters = {
            "Lx": 2.0,  # Length of the rectangle along the x-axis
            "Ly": 1.0,  # Length of the rectangle along the y-axis
            "angle": 45.0,  # Angle of the rhomboid in degrees
            "axis": 0.5,  # Axis length of the rhomboid
            "lc": 0.05,  # Mesh element size
        }

    Lx = geom_parameters["Lx"]
    Ly = geom_parameters["Ly"]
    angle = geom_parameters["angle"]
    axis = geom_parameters["axis"]
    lc = geom_parameters["lc"]

    if comm.rank == 0:
        import warnings
        import gmsh
        from numpy import tan, pi

        warnings.filterwarnings("ignore")

        # Initialize gmsh
        gmsh.initialize()
        gmsh.model.add("RectWithRhomboidalHole")

        # Create outer rectangle
        rect = gmsh.model.occ.addRectangle(-Lx / 2, -Ly / 2, 0, Lx, Ly, tag=1)

        # Create rhomboidal hole
        angle_rad = angle * (pi / 180.0)  # Convert angle to radians
        dx = axis * 0.5
        dy = axis * 0.5 * (1 / tan(angle_rad / 2))  # Adjust for rhomboid shape

        rhomboid_points = [
            gmsh.model.occ.addPoint(-dx, 0, 0),
            gmsh.model.occ.addPoint(0, -dy, 0),
            gmsh.model.occ.addPoint(dx, 0, 0),
            gmsh.model.occ.addPoint(0, dy, 0),
        ]

        rhomboid_lines = [
            gmsh.model.occ.addLine(rhomboid_points[0], rhomboid_points[1]),
            gmsh.model.occ.addLine(rhomboid_points[1], rhomboid_points[2]),
            gmsh.model.occ.addLine(rhomboid_points[2], rhomboid_points[3]),
            gmsh.model.occ.addLine(rhomboid_points[3], rhomboid_points[0]),
        ]

        rhomboid_loop = gmsh.model.occ.addCurveLoop(rhomboid_lines)
        rhomboid_surface = gmsh.model.occ.addPlaneSurface([rhomboid_loop])

        cut_entities, _ = gmsh.model.occ.cut([(2, rect)], [(2, rhomboid_surface)])
        surface_tag = cut_entities[0][1]  # Extract tag from result

        # Synchronize before meshing
        gmsh.model.occ.synchronize()

        # Define physical groups
        gmsh.model.addPhysicalGroup(2, [surface_tag], tag=1)
        gmsh.model.setPhysicalName(2, 1, "RectDomain")
        gmsh.model.addPhysicalGroup(2, [rhomboid_surface], tag=2)
        gmsh.model.setPhysicalName(2, 2, "RhomboidHole")

        # Mesh settings
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)

        gmsh.model.mesh.generate(2)

        # Save mesh to file
        gmsh.write("rect_with_rhomboidal_hole.msh")

        print("Mesh created and saved as 'rect_with_rhomboidal_hole.msh'")

        # Finalize gmsh
        # gmsh.finalize()
    return gmsh.model if comm.rank == 0 else None, 2


def create_extended_rectangle(comm=MPI.COMM_WORLD, geom_parameters=None):
    """
    Create a mesh of a rectangle (the main domain) embedded in an extended rectangle.
    The extension is a border of width ext around the main domain.
    The inner and outer parts are separated by the common interface (embedded as a curve in the mesh).

    Parameters
    ----------
    geom_parameters : dict
        Dictionary with the following keys:
            "L" : float
                Length (width) of the main domain (x-direction)
            "H" : float
                Height of the main domain (y-direction)
            "ext" : float
                Extension length added to each side of the main domain.
            "lc" : float
                Mesh size
            "tdim" : int
                Geometric dimension (should be 2)

    Returns
    -------
    msh_filename : str
        Name of the saved MSH file.
    outer_tag, inner_tag : tuple
        A tuple with the physical group tags for the extended (outer) region and for the inner (main) domain.
    """
    gmsh.initialize()
    gmsh.model.add("extended_rectangle")
    if geom_parameters is None:
        raise ValueError("geom_parameters must be provided.")

    L = geom_parameters["L"]
    H = geom_parameters["H"]
    ext = geom_parameters["ext"]
    lc = geom_parameters["lc"]
    tdim = geom_parameters["tdim"]

    # Create the outer (extended) rectangle.
    # We center the main domain at the origin; the outer rectangle extends ext beyond the main domain.
    # Outer rectangle bounds:
    #   x from -L/2 - ext to L/2 + ext, y from -H/2 - ext to H/2 + ext.
    outer = gmsh.model.occ.addRectangle(
        -L / 2,
        -H / 2 - ext,
        0,
        L,
        H + 2 * ext,
        tag=1,
        # -L / 2 - ext, -H / 2 - ext, 0, L + 2 * ext, H + 2 * ext, tag=1
    )

    # Create the inner (main) domain rectangle.
    inner = gmsh.model.occ.addRectangle(-L / 2, -H / 2, 0, L, H, tag=2)

    # Fragment the outer surface with the inner rectangle.
    # This operation splits the outer rectangle into two regions: one corresponding to the inner domain
    # and one corresponding to the outer extension.
    # The fragment command guarantees that the common boundary is preserved as an interface.
    fragments = gmsh.model.occ.fragment([(2, outer)], [(2, inner)])
    gmsh.model.occ.synchronize()

    # After the fragment, there will be several surfaces.
    # We need to identify which one corresponds to the inner domain and which ones form the extension.
    # A simple approach is to use the center of mass. We expect the inner domain to have its center inside the original inner rectangle.

    # Get all surfaces (entities of dim 2)
    surfaces = gmsh.model.getEntities(dim=2)

    top_ext = []
    bottom_ext = []
    for dim, tag in surfaces:
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        print(com[0], com[1])
        if com[1] > lc:
            top_ext.append(tag)
        elif com[1] < -lc:
            bottom_ext.append(tag)

    print("Top extension surfaces:", top_ext)
    print("Bottom extension surfaces:", bottom_ext)

    edge_occurrences = {}  # Dictionary to store, for each edge, a list of surfaces (by their tag) it belongs to.
    surface_boundaries = {}  # Dictionary to store boundaries for each surface.

    for entity in surfaces:
        dim, tag = entity
        boundary_edges = gmsh.model.getBoundary([entity], oriented=False)
        surface_boundaries[tag] = boundary_edges

        # Loop over each boundary edge for the surface.
        for be in boundary_edges:
            # be is a tuple, e.g. (1, edge_tag)
            edge_tag = be[1]
            if edge_tag in edge_occurrences:
                edge_occurrences[edge_tag].append(tag)
            else:
                edge_occurrences[edge_tag] = [tag]

    internal_edges = [
        edge_tag
        for edge_tag, surf_tags in edge_occurrences.items()
        if len(surf_tags) > 1
    ]

    print("Surface boundaries:")
    for s_tag, b_edges in surface_boundaries.items():
        print(f"Surface {s_tag}: edges {[be[1] for be in b_edges]}")

    print("Internal interface edge tags:", internal_edges)

    # Retrieve the boundaries (edges) of the top and bottom extension surfaces.
    top_boundary_edges = gmsh.model.getBoundary(
        [(2, tag) for tag in top_ext], oriented=False
    )
    bottom_boundary_edges = gmsh.model.getBoundary(
        [(2, tag) for tag in bottom_ext], oriented=False
    )
    # Extract the edge tags.
    top_edge_tags = [
        edge[1] for edge in top_boundary_edges if edge[1] not in internal_edges
    ]
    bottom_edge_tags = [
        edge[1] for edge in bottom_boundary_edges if edge[1] not in internal_edges
    ]
    # Create physical groups for the top and bottom boundaries (these will be used for applying Dirichlet BCs).
    top_bound_tag = gmsh.model.addPhysicalGroup(1, top_edge_tags, tag=21)
    gmsh.model.setPhysicalName(1, top_bound_tag, "Top_Boundary")
    bottom_bound_tag = gmsh.model.addPhysicalGroup(1, bottom_edge_tags, tag=22)
    gmsh.model.setPhysicalName(1, bottom_bound_tag, "Bottom_Boundary")
    print("Top boundary edges:", top_edge_tags)
    print("Bottom boundary edges:", bottom_edge_tags)

    inner_surfs = []  # List for the inner domain surface(s)
    outer_surfs = []  # List for the extension surfaces

    # Loop through each surface, get its center of mass, and classify it.
    for dim, tag in surfaces:
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        # Check if the center is inside the inner rectangle bounds.
        if (-L / 2 <= com[0] <= L / 2) and (-H / 2 <= com[1] <= H / 2):
            inner_surfs.append(tag)
        else:
            outer_surfs.append(tag)

    # To make sure the internal interface is embedded (i.e. the same nodes appear on both subdomains),
    # use the OCC boolean fragmentation to split the overall domain.
    # Here we create physical groups so that we can later impose different material properties or boundary conditions.
    inner_tag = gmsh.model.addPhysicalGroup(2, inner_surfs, tag=10)
    gmsh.model.setPhysicalName(2, inner_tag, "Main_Domain")
    outer_tag = gmsh.model.addPhysicalGroup(2, outer_surfs, tag=11)
    gmsh.model.setPhysicalName(2, outer_tag, "Extended_Domain")

    # Set the mesh size everywhere (only one characteristic length lc used)
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)

    # Generate the mesh
    gmsh.model.mesh.generate(tdim)

    # Optionally save the mesh
    msh_filename = "extended_rectangle_with_interface.msh"
    gmsh.write(msh_filename)

    # gmsh.finalize()

    return gmsh.model if comm.rank == 0 else None, 2
