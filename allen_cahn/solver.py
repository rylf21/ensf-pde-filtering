"""
Allen-Cahn equation solver using FEniCSx (IMEX scheme).

Solves the Allen-Cahn equation on Omega = (-0.5, 0.5)^2:

    dphi/dt = alpha^2 * Delta(phi) - M(phi) * (phi^3 - phi)

with homogeneous Neumann boundary conditions and a random
initial condition phi_0 ~ U(-0.9, 0.9).

Time discretization: implicit--explicit (IMEX) scheme
    - Diffusion term: treated implicitly (backward Euler)
    - Nonlinear reaction term: treated explicitly

This gives a linear system at each time step

Spatial discretization: Q1 finite elements on a structured
quadrilateral mesh via FEniCSx.

Three mobility cases are supported :
    Case 1: M = 1 + xi_t              (constant, with optional noise)
    Case 2: M = max(1 + xi_t, 0)      (solution-independent stochastic)
    Case 3: M = max(1 - phi^2 + xi_t, 0) (solution-dependent stochastic)

where xi_t is an optional stochastic perturbation with std xi_sigma.

Note: Only Case 1 (constant mobility) is used in the thesis experiments.
Cases 2 and 3 are included for completeness; extensions to those cases
follow directly from the same solver interface.

Dependencies:
    dolfinx, ufl, petsc4py, mpi4py, rbnicsx, numpy
"""

import numpy as np
from mpi4py import MPI

import ufl
from dolfinx import fem, mesh, io
from dolfinx.fem.petsc import LinearProblem
import rbnicsx.backends
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------

def create_mesh(Nx: int = 128, Ny: int = 128, comm=None):
    """
    Create a structured Q1 quadrilateral mesh on Omega = (-0.5, 0.5)^2.

    Parameters
    ----------
    Nx, Ny : int
        Number of cells in x and y directions. Default 128x128.
    comm : MPI communicator, optional

    Returns
    -------
    domain : dolfinx mesh
    V : dolfinx FunctionSpace (Q1)
    """
    if comm is None:
        comm = MPI.COMM_WORLD

    domain = mesh.create_rectangle(
        comm,
        [np.array([-0.5, -0.5]), np.array([0.5, 0.5])],
        [Nx, Ny],
        mesh.CellType.quadrilateral,
    )
    V = fem.FunctionSpace(domain, ("Q", 1))
    return domain, V


# ---------------------------------------------------------------------------
# Solver class
# ---------------------------------------------------------------------------

class AllenCahnIMEX:
    """
    Semi-implicit (IMEX) Allen-Cahn solver on Omega = (-0.5, 0.5)^2.

    The weak form discretized in time reads:

        (phi^{n+1} - phi^n)/dt = alpha^2 * Delta(phi^{n+1})
                                 - M^n * (phi^n)^3 + M^n * phi^n

    Rearranging into the standard linear system a(phi^{n+1}, v) = L(v):

        a(u, v) = (1/dt) * u*v*dx + alpha^2 * grad(u).grad(v)*dx
        L(v)    = (1/dt) * phi^n*v*dx - M^n*((phi^n)^3 - phi^n)*v*dx

    Homogeneous Neumann BCs are imposed weakly (no boundary term needed).

    The LHS matrix is assembled once and reused at every time step,
    which is the key computational advantage of the IMEX approach.

    Usage
    -----
    domain, V = create_mesh(Nx=128, Ny=128)
    problem = AllenCahnIMEX(V, dt=0.01, alpha=0.01, mobility_case=1)

    # Set initial condition
    rng = np.random.default_rng(42)
    phi0 = 1.8 * rng.random(V.dofmap.index_map.size_local) - 0.9
    problem.set_initial_condition(phi0)

    # Run full trajectory
    solution, times, energies = problem.solve(Nstep=1000)

    # Or step-by-step (for filtering loop)
    problem.set_initial_condition(particle)
    phi_next, t, E = problem.step()
    """

    def __init__(
        self,
        V,
        dt: float,
        alpha: float,
        mobility_case: int = 1,
        xi_sigma: float = 0.0,
        noise_is_spatial: bool = True,
        comm=None,
        pc: str = "ilu",
    ):
        """
        Parameters
        ----------
        V : dolfinx FunctionSpace
        dt : float
            Time step size.
        alpha : float
            Interface width parameter (epsilon in the Allen--Cahn equation).
            Thesis experiments use alpha = 0.01.
        mobility_case : int
            1 = constant (default), 2 = solution-independent stochastic,
            3 = solution-dependent stochastic.
        xi_sigma : float
            Standard deviation of stochastic mobility perturbation.
            Set to 0.0 for deterministic mobility (Case 1 in thesis).
        noise_is_spatial : bool
            If True, xi_t is a spatial field; if False, a scalar.
        pc : str
            PETSc preconditioner type. Default 'ilu'.
        """
        self.comm          = MPI.COMM_WORLD if comm is None else comm
        self.V             = V
        self.dt            = float(dt)
        self.alpha         = float(alpha)
        self.mobility_case = int(mobility_case)
        self.xi_sigma      = float(xi_sigma)
        self.noise_is_spatial = bool(noise_is_spatial)
        self._t            = 0.0

        # Solution fields
        self.phi      = fem.Function(V, name="phi")
        self.phi_prev = fem.Function(V, name="phi_prev")

        # Mobility field (updated each step)
        self._M = fem.Function(V, name="M")
        self._M.x.array[:] = 1.0
        self._M.x.scatter_forward()

        # Variational forms
        u  = ufl.TrialFunction(V)
        v  = ufl.TestFunction(V)
        dx = ufl.dx(domain=V.mesh)
        self._dx = dx

        # LHS: time-invariant, assembled once
        a = (
            (1.0 / self.dt) * u * v * dx
            + (self.alpha ** 2) * ufl.dot(ufl.grad(u), ufl.grad(v)) * dx
        )

        # RHS: updated each step via phi_prev and _M
        L = (
            (1.0 / self.dt) * self.phi_prev * v * dx
            - self._M * (self.phi_prev ** 3 - self.phi_prev) * v * dx
        )

        petsc_opts = {
            "ksp_type": "cg",
            "pc_type": pc,
            "ksp_rtol": 1e-8,
        }
        self._problem = fem.petsc.LinearProblem(
            a, L, u=self.phi, petsc_options=petsc_opts
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_mobility(self):
        """Update mobility field M for the current time step."""
        arr = self._M.x.array

        if self.xi_sigma > 0.0:
            xi = (
                self.xi_sigma * np.random.randn(arr.size)
                if self.noise_is_spatial
                else self.xi_sigma * float(np.random.randn())
            )
        else:
            xi = 0.0

        if self.mobility_case == 1:
            arr[:] = 1.0 + xi

        elif self.mobility_case == 2:
            arr[:] = np.maximum(1.0 + xi, 0.0)

        elif self.mobility_case == 3:
            phi_arr = self.phi_prev.x.array
            arr[:] = np.maximum(1.0 - phi_arr ** 2 + xi, 0.0)

        else:
            raise ValueError("mobility_case must be 1, 2, or 3.")

        self._M.x.scatter_forward()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def set_initial_condition(self, callback_or_array, scatter: bool = True):
        """
        Set phi_prev (and phi) to the given initial condition.

        Parameters
        ----------
        callback_or_array : callable, np.ndarray, or dolfinx.fem.Function
            - callable: interpolated via FEniCSx interpolate
            - ndarray:  assigned directly to phi_prev.x.array
            - Function: array is copied
        """
        if isinstance(callback_or_array, fem.Function):
            self.phi_prev.x.array[:] = callback_or_array.x.array
        elif callable(callback_or_array):
            self.phi_prev.interpolate(callback_or_array)
        else:
            self.phi_prev.x.array[:] = np.asarray(callback_or_array)

        if scatter:
            self.phi_prev.x.scatter_forward()

        self.phi.x.array[:] = self.phi_prev.x.array
        self.phi.x.scatter_forward()

    def get_state_array(self) -> np.ndarray:
        """Return a copy of the current phi as a 1-D numpy array."""
        n_owned = self.V.dofmap.index_map.size_local
        return self.phi_prev.x.array[:n_owned].copy()

    def step(self):
        """
        Advance one IMEX time step.

        Returns
        -------
        phi_out : dolfinx.fem.Function
            Solution at t_{n+1}.
        t : float
            Current time after the step.
        energy : float
            Discrete energy at t_{n+1}.
        """
        self.phi.x.array[:] = self.phi_prev.x.array
        self.phi.x.scatter_forward()

        self._update_mobility()
        self._problem.solve()
        self.phi.x.scatter_forward()

        self.phi_prev.x.array[:] = self.phi.x.array
        self.phi_prev.x.scatter_forward()

        self._t += self.dt

        phi_out = fem.Function(self.V)
        phi_out.x.array[:] = self.phi.x.array
        phi_out.name = "phi"

        return phi_out, self._t, self.compute_energy(phi_out)

    def compute_energy(self, phi: fem.Function) -> float:
        """
        Compute the discrete energy:

            E(phi) = integral_Omega [ (alpha^2/2)|grad phi|^2
                                    + (1/4)(phi^2 - 1)^2 ] dx
        """
        grad_term = 0.5 * self.alpha ** 2 * ufl.inner(
            ufl.grad(phi), ufl.grad(phi)
        )
        bulk_term = 0.25 * (phi ** 2 - 1.0) ** 2
        E_form  = fem.form((grad_term + bulk_term) * self._dx)
        E_local = fem.assemble_scalar(E_form)
        return float(self.comm.allreduce(E_local, op=MPI.SUM))

    def solve(self, Nstep: int):
        """
        Run the full trajectory for Nstep time steps from the current
        initial condition.

        Parameters
        ----------
        Nstep : int
            Number of time steps.

        Returns
        -------
        solution  : list of dolfinx.fem.Function
        times     : list of float
        energies  : np.ndarray, shape (Nstep,)
        """
        times, solution, energies = [], [], []

        # Store initial state
        phi0 = fem.Function(self.V)
        phi0.x.array[:] = self.phi_prev.x.array
        phi0.name = "phi"
        solution.append(phi0)
        times.append(self._t)

        for _ in tqdm(range(Nstep), desc="Allen-Cahn solve"):
            phi_n, t_n, E_n = self.step()
            solution.append(phi_n)
            times.append(t_n)
            energies.append(E_n)

        return solution, times, np.array(energies)
