"""
Heat equation solver using FEniCSx.

Implements a parametric diffusion problem on Omega = (-1, 1)^2:

    du/dt - div(K(x, y; theta) grad u) = f(x, y)    in Omega x (0, T]
    u = 1                                             on Gamma_0
    u = 0                                             on Gamma_1
    u = 0                                             at t = 0

where K(x, y; theta) is a Gaussian permeability field parameterized by
theta = [mu_x, mu_y, var_x, var_y], and f(x, y) = x + y.

The boundary is split as:
    Gamma_0: left edge (x = -1, y <= 0) and bottom edge (y = -1, x <= 0)
    Gamma_1: remainder of boundary

Time discretization: backward Euler (implicit).
Spatial discretization: P1 finite elements via FEniCSx.

Adapted from the FEniCSx tutorial by Jørgen S. Dokken:
    https://jsdokken.com/dolfinx-tutorial/chapter2/heat_equation.html

Modifications from the tutorial:
    - Spatially varying permeability K(x, y; theta) replacing constant diffusivity
    - Parametric Gaussian permeability controlled by theta = [mu_x, mu_y, var_x, var_y]
    - Split inhomogeneous Dirichlet boundary conditions (p=1 on Gamma_0, p=0 on Gamma_1)
    - Structured mesh generation via gmsh
    - Class-based interface for use as a forecast operator in ensemble filtering

Dependencies:
    dolfinx, ufl, petsc4py, mpi4py, gmsh, rbnicsx, numpy
"""

import math

import dolfinx
import dolfinx.fem
import dolfinx.io
import dolfinx.mesh
import gmsh
import mpi4py.MPI
import numpy as np
import numpy.typing
import petsc4py.PETSc
import rbnicsx.backends
import ufl
from dolfinx.fem import Constant, Function, FunctionSpace
from dolfinx.fem.petsc import LinearProblem


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------

def create_mesh(dx: float):
    """
    Create a structured quadrilateral mesh on Omega = (-1, 1)^2 using gmsh.

    The boundary is tagged as:
        Tag 1: Gamma_0 (bottom edge, y = -1)
        Tag 2: Left and right edges
        Tag 3: Top edge (y = 1)

    Parameters
    ----------
    dx : float
        Target mesh size (smaller = finer mesh).

    Returns
    -------
    mesh, subdomains, boundaries : dolfinx mesh objects
    """
    gmsh.initialize()
    gmsh.model.add("heat_equation")

    p0 = gmsh.model.geo.addPoint(-1.0, -1.0, 0.0, dx)
    p1 = gmsh.model.geo.addPoint( 1.0, -1.0, 0.0, dx)
    p2 = gmsh.model.geo.addPoint( 1.0,  1.0, 0.0, dx)
    p3 = gmsh.model.geo.addPoint(-1.0,  1.0, 0.0, dx)

    l0 = gmsh.model.geo.addLine(p0, p1)
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p0)

    loop = gmsh.model.geo.addCurveLoop([l0, l1, l2, l3])
    surf = gmsh.model.geo.addPlaneSurface([loop])

    # Structured (transfinite) mesh
    n = int(2 / dx) + 1
    for line in [l0, l1, l2, l3]:
        gmsh.model.geo.mesh.setTransfiniteCurve(line, n)
    gmsh.model.geo.mesh.setTransfiniteSurface(surf)
    gmsh.model.geo.mesh.setRecombine(2, surf)

    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(1, [l0], 1)
    gmsh.model.addPhysicalGroup(1, [l1, l3], 2)
    gmsh.model.addPhysicalGroup(1, [l2], 3)
    gmsh.model.addPhysicalGroup(2, [surf], 1)
    gmsh.model.mesh.generate(2)

    mesh, subdomains, boundaries = dolfinx.io.gmshio.model_to_mesh(
        gmsh.model, comm=mpi4py.MPI.COMM_WORLD, rank=0, gdim=2
    )
    gmsh.finalize()
    return mesh, subdomains, boundaries


# ---------------------------------------------------------------------------
# Solver class
# ---------------------------------------------------------------------------

class HeatEquationProblem:
    """
    Parametric heat equation solver on Omega = (-1, 1)^2.

    The permeability field is a Gaussian bump:

        K(x, y; theta) = 1 / (2 pi sqrt(var_x) sqrt(var_y))
                         * exp(-0.5 * ((x - mu_x)^2 / var_x
                                     + (y - mu_y)^2 / var_y))

    where theta = [mu_x, mu_y, var_x, var_y].

    Usage
    -----
    # Create solver
    problem = HeatEquationProblem(mesh, dt=0.02, T=2.0)

    # Solve full trajectory from t=0
    solutions, times = problem.solve_trajectory(theta=[0, 0, 1, 1])

    # Or step-by-step (for use inside a filtering loop)
    problem.set_initial_condition_from_array(particle)
    uh = problem.solve_one_step(theta=[0, 0, 1, 1], t_np1=dt)
    state = problem.get_current_state_array()
    """

    def __init__(self, mesh, dt: float, T: float):
        self.mesh = mesh
        self.dt = float(dt)
        self.T = float(T)

        # P1 finite element space
        self.V = FunctionSpace(mesh, ("Lagrange", 1))

        # Trial and test functions
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)

        # Solution at current and previous time step
        self.u = Function(self.V)
        self.u.name = "u"

        self.u_prev = Function(self.V)
        self.u_prev.name = "u_prev"

        # Initial condition: p(x, y, 0) = 0
        self.u_prev.interpolate(
            lambda x: np.zeros((1, x.shape[1]), dtype=petsc4py.PETSc.ScalarType)
        )

        # Symbolic parameters for K(x, y; theta)
        # theta = [mu_x, mu_y, var_x, var_y]
        self.mu_symb = rbnicsx.backends.SymbolicParameters(mesh, shape=(4,))

        x_coord = ufl.SpatialCoordinate(mesh)
        K = (
            1.0
            / (2.0 * math.pi
               * ufl.sqrt(self.mu_symb[2])
               * ufl.sqrt(self.mu_symb[3]))
            * ufl.exp(
                -0.5 * (
                    (x_coord[0] - self.mu_symb[0]) ** 2 / self.mu_symb[2]
                    + (x_coord[1] - self.mu_symb[1]) ** 2 / self.mu_symb[3]
                )
            )
        )

        # Forcing function f(x, y) = x + y (time-independent in this problem)
        self.f = Function(self.V)
        self._update_rhs()

        # Boundary conditions
        self.bcs = self._build_boundary_conditions()

        # Backward Euler weak form:
        #   (u^{n+1}, v)/dt + a(K; u^{n+1}, v) = (u^n, v)/dt + (f, v)
        dt_const = Constant(mesh, petsc4py.PETSc.ScalarType(self.dt))
        self.a_form = (
            ufl.inner(u, v) / dt_const * ufl.dx
            + ufl.inner(K * ufl.grad(u), ufl.grad(v)) * ufl.dx
        )
        self.L_form = (
            ufl.inner(self.u_prev, v) / dt_const * ufl.dx
            + ufl.inner(self.f, v) * ufl.dx
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_boundary_conditions(self):
        """
        Dirichlet BCs:
            Gamma_0 (left lower + bottom left): u = 1
            Gamma_1 (remainder):                u = 0
        """
        fdim = self.mesh.topology.dim - 1

        def gamma0(x):
            on_left_lower  = np.logical_and(np.isclose(x[0], -1.0), x[1] <= 0.0)
            on_bottom_left = np.logical_and(np.isclose(x[1], -1.0), x[0] <= 0.0)
            return np.logical_or(on_left_lower, on_bottom_left)

        def gamma1(x):
            on_left_upper   = np.logical_and(np.isclose(x[0], -1.0), x[1] > 0.0)
            on_bottom_right = np.logical_and(np.isclose(x[1], -1.0), x[0] > 0.0)
            on_right        = np.isclose(x[0], 1.0)
            on_top          = np.isclose(x[1], 1.0)
            return np.logical_or.reduce((on_left_upper, on_bottom_right, on_right, on_top))

        facets0 = dolfinx.mesh.locate_entities_boundary(self.mesh, fdim, gamma0)
        facets1 = dolfinx.mesh.locate_entities_boundary(self.mesh, fdim, gamma1)

        dofs0 = dolfinx.fem.locate_dofs_topological(self.V, fdim, facets0)
        dofs1 = dolfinx.fem.locate_dofs_topological(self.V, fdim, facets1)

        u0 = Function(self.V)
        u0.interpolate(lambda x: np.full((1, x.shape[1]), 1.0, dtype=petsc4py.PETSc.ScalarType))

        u1 = Function(self.V)
        u1.interpolate(lambda x: np.full((1, x.shape[1]), 0.0, dtype=petsc4py.PETSc.ScalarType))

        return [
            dolfinx.fem.dirichletbc(u0, dofs0),
            dolfinx.fem.dirichletbc(u1, dofs1),
        ]

    def _update_rhs(self):
        """Set forcing term f(x, y) = x + y."""
        def f_eval(x):
            vals = np.zeros((1, x.shape[1]), dtype=petsc4py.PETSc.ScalarType)
            vals[0, :] = x[0] + x[1]
            return vals
        self.f.interpolate(f_eval)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def set_parameters(self, theta: np.ndarray):
        """
        Set theta = [mu_x, mu_y, var_x, var_y].
        var_x and var_y must be strictly positive.
        """
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (4,):
            raise ValueError("theta must have shape (4,).")
        if theta[2] <= 0 or theta[3] <= 0:
            raise ValueError("var_x and var_y (theta[2], theta[3]) must be positive.")
        self.mu_symb.value[:] = theta

    def set_initial_condition_from_array(self, values: np.ndarray):
        """Set u_prev from a 1-D array of nodal values."""
        with self.u_prev.vector.localForm() as loc:
            loc[:] = values

    def set_initial_condition_zero(self):
        """Reset u_prev = 0 (problem initial condition)."""
        self.u_prev.interpolate(
            lambda x: np.zeros((1, x.shape[1]), dtype=petsc4py.PETSc.ScalarType)
        )

    def get_current_state_array(self) -> np.ndarray:
        """Return a copy of the current nodal solution as a 1-D numpy array."""
        return self.u_prev.x.array.copy()

    def solve_one_step(self, theta: np.ndarray, t_np1: float) -> Function:
        """
        Advance one backward-Euler step from u_prev to u at time t_{n+1}.

        Parameters
        ----------
        theta : array-like, shape (4,)
            Parameter vector [mu_x, mu_y, var_x, var_y].
        t_np1 : float
            Target time (used to update boundary conditions if time-dependent).

        Returns
        -------
        uh : dolfinx.fem.Function
            Solution at t_{n+1}. u_prev is also updated in place.
        """
        self.set_parameters(theta)

        problem = LinearProblem(
            self.a_form,
            self.L_form,
            u=self.u,
            bcs=self.bcs,
        )

        # Direct LU solver (MUMPS if available)
        ksp = problem.solver
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        try:
            pc.setFactorSolverType("mumps")
        except Exception:
            pass

        uh = problem.solve()
        uh.name = "u"

        self.u.x.array[:]      = uh.x.array[:]
        self.u_prev.x.array[:] = uh.x.array[:]

        return uh

    def solve_trajectory(self, theta: np.ndarray):
        """
        Solve on [0, T] starting from u = 0, returning all time steps.

        Parameters
        ----------
        theta : array-like, shape (4,)
            Parameter vector [mu_x, mu_y, var_x, var_y].

        Returns
        -------
        solutions : list of dolfinx.fem.Function
        times     : list of float
        """
        self.set_initial_condition_zero()
        self.set_parameters(theta)

        nsteps = int(round(self.T / self.dt))
        solutions, times = [], []

        for n in range(1, nsteps + 1):
            t = n * self.dt
            uh = self.solve_one_step(theta, t)

            uh_copy = Function(self.V)
            uh_copy.x.array[:] = uh.x.array[:]
            uh_copy.name = f"u_{n}"

            solutions.append(uh_copy)
            times.append(t)

        return solutions, times
