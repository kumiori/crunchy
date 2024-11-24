from dolfinx.fem import FunctionSpace, Function


def map_field_to_function_space(field, mesh):
    """Map a 2D field to a FEniCS function space."""
    V = FunctionSpace(mesh, ("CG", 1))
    func = Function(V)
    # Interpolation logic here...
    return func
