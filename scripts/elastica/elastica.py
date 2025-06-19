from dolfinx import mesh, fem, nls
import ufl
from mpi4py import MPI
import numpy as np
import basix
import petsc4py.PETSc as PETSc

# 1. Create a 1D mesh for the parameter `s`
mesh = mesh.create_interval(MPI.COMM_WORLD, 50, [0.0, 1.0])
P2 = basix.ufl.element(
    "Lagrange", mesh.basix_cell(), degree=2
)  # Quadratic Lagrange elements
V = fem.functionspace(mesh, P2)
W = fem.functionspace(mesh, P2)

t = fem.Constant(mesh, np.array([0.0], dtype=PETSc.ScalarType))

# Mixed space: (x, y, lambda)
MixedElement = basix.ufl.mixed_element([P2, P2, P2])
M = fem.functionspace(mesh, MixedElement)

# 2. Define trial and test functions
# u = ufl.TrialFunction(M)  # [x(s), y(s), lambda(s)]
u = fem.Function(M)  # [x(s), y(s), lambda(s)]
v = ufl.TestFunction(M)
u_k = fem.Function(M)  # Current solution

x, y, lam = ufl.split(u)
vx, vy, vlam = ufl.split(v)

# 3. Define geometric quantities
dx_ds = ufl.Dx(x, 0)
dy_ds = ufl.Dx(y, 0)
d2x_ds2 = ufl.Dx(dx_ds, 0)
d2y_ds2 = ufl.Dx(dy_ds, 0)
dx = ufl.Measure("dx", mesh)

# Curvature
kappa = (d2y_ds2 * dx_ds - d2x_ds2 * dy_ds) / (ufl.sqrt(dx_ds**2 + dy_ds**2) + 1e-8)

# 4. Energy functional
bending_energy = 1.0 / 2.0 * kappa**2 * dx
constraint = lam * (dx_ds**2 + dy_ds**2 - 1)

F = ufl.derivative(bending_energy, u, v)

# 5. Boundary conditions

Vx, _ = M.sub(0).collapse()  # For x component
Vy, _ = M.sub(1).collapse()  # For y component

boundary_dofs_x = fem.locate_dofs_geometrical(
    Vx, lambda x: np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], 1))
)
boundary_dofs_y = fem.locate_dofs_geometrical(
    Vy, lambda x: np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], 1))
)
bcs = [
    fem.dirichletbc(np.array(0.0, dtype=PETSc.ScalarType), boundary_dofs_x, Vx),
    fem.dirichletbc(np.array(0.0, dtype=PETSc.ScalarType), boundary_dofs_y, Vy),
]
# 6. Nonlinear problem
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver


problem = NonlinearProblem(F, u_k, bcs=bcs)
solver = NewtonSolver(MPI.COMM_WORLD, problem)

# Solver options
# solver.convergence_criterion = "incremental"
solver.rtol = 1e-8

# 7. Solve the problem
u_k.x.array[:] = 0  # Initial guess
solver.solve(u_k)

# 8. Extract solution
x_sol, y_sol, lambda_sol = u_k.split()

# 9. Visualization or output
import matplotlib.pyplot as plt

x_values = x_sol.x.array
y_values = y_sol.x.array
plt.plot(x_values, y_values, label="Elastica")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
