# EnSF for PDE-Governed Systems — FEniCSx Solvers and Filtering Experiments

This repository contains the FEniCSx solvers and Ensemble Score Filter (EnSF)
filtering experiments developed for Chapter 3 of the dissertation:

> López Fajardo, R.Y. *Model–Data Integration in Complex Dynamical Systems:
> Applications in Materials Synthesis.* Florida State University, 2026.

The code accompanies two state estimation case studies: a parametric heat
equation and the Allen--Cahn phase-field equation, both solved with FEniCSx
and coupled with the EnSF data assimilation framework.

---

## Repository structure

```
ensf-pde-filtering/
├── README.md                        ← this file
├── .gitignore                       ← excludes data/ and figures/ from git
├── data/                            ← generated datasets (not tracked by git)
│   └── .gitkeep                     ← keeps folder in repo when data is absent
├── figures/                         ← generated figures (not tracked by git)
│   └── .gitkeep                     ← keeps folder in repo when figures are absent
├── heat_equation/
│   ├── solver.py                    ← HeatEquationProblem class
│   ├── ensf.py                      ← reverse_SDE and score likelihood (shared)
│   ├── 01_run_experiment.ipynb      ← EnSF for 100%/25%/5% observations
│   └── 02_visualize.ipynb           ← thesis figures
└── allen_cahn/
    ├── solver.py                    ← AllenCahnIMEX class
    ├── 00_generate_truth.ipynb      ← reference trajectory and observations
    ├── 01_run_experiment.ipynb      ← EnSF for 100%/70% observations
    └── 02_visualize.ipynb           ← thesis figures
```

The EnSF implementation in `heat_equation/ensf.py` is shared by both chapters.
Run notebooks in numbered order within each folder; generate the heat equation
data and Allen--Cahn truth first before running the filtering experiments.

---

## Problems

### Heat equation

Parametric diffusion on $\Omega = (-1,1)^2$ with spatially varying Gaussian
permeability field $K(x,y;\theta)$:

$$\frac{\partial u}{\partial t} - \nabla \cdot (K(x,y;\theta)\, \nabla u) = f(x,y)$$

with split Dirichlet boundary conditions and observation model
$Y_t = \arctan(X_t) + \varepsilon_t$.

### Allen--Cahn equation

Phase-field grain coarsening on $\Omega = (-0.5, 0.5)^2$:

$$\frac{\partial \phi}{\partial t} = \alpha^2 \Delta\phi - M(\phi)(\phi^3 - \phi)$$

with homogeneous Neumann boundary conditions, random initial condition
$\phi_0 \sim U(-0.9, 0.9)$, and observation model $Y_t = \arctan(X_t) + \varepsilon_t$.
Case 1 (constant mobility, $M=1$) is the setting reported in the thesis.

---

## Scope and limitations

The heat equation experiments cover 100%, 25%, and 5% observation densities.
The Allen--Cahn experiments cover 100% and 70% observation densities (Case 1).

The **10% Allen--Cahn case** with LETKF comparison is not included here.
The full experiment, including the LETKF baseline, is available in the
companion repository by Huynh, T.:
https://github.com/Toanhuynh997/StateEst_PDEs/tree/main/AllenCahn

The Allen--Cahn results in this repository use a plain $\arctan$ observation
operator, consistent with the problem formulation in Huynh et al. (2026).
The filtering experiments were developed as part of the collaboration leading
to that publication; this repository shares the FEniCSx solver and the
author's filtering notebooks, which use a simplified implementation of the
observation operator. Results may therefore differ from the thesis figures,
which were produced using the full collaboration codebase.

---

## Dependencies and environment

FEniCSx is notoriously sensitive to version mismatches — DOLFINx, UFL,
Basix, and FFCx must be installed as a matched set. The experiments in
this repository were developed and tested with the following versions:

| Package | Version |
|---------|---------|
| `dolfinx` | 0.7.0.0 |
| `ufl` | 2022.3.0.dev0 |
| `basix` | 0.7.0.0 |
| `ffcx` | 0.7.0.dev0 |
| `petsc4py` | 3.15.1 |
| `mpi4py` | 3.1.3 |
| `gmsh` | 4.11.1 |
| `numpy` | 1.26.4 |
| `h5py` | 3.6.0 |
| `matplotlib` | 3.7.0 |
| `tqdm` | 4.64.1 |

`rbnicsx` is required for the heat equation solver only
(`heat_equation/solver.py`). It is not available on conda-forge and must
be installed from source:

```bash
pip install git+https://github.com/RBniCS/RBniCSx.git
```

**Installation (recommended):**

Follow the official FEniCSx installation instructions for your platform:
https://fenicsproject.org/download/

Once FEniCSx is installed, additionally install RBniCSx (required for the
heat equation solver only):

```bash
pip install git+https://github.com/RBniCS/RBniCSx.git
```

> **Note:** If you are using a newer version of DOLFINx, the API has
> changed in several places. Known breaking changes after 0.7.x include:
> `fem.FunctionSpace` → `fem.functionspace` (lowercase),
> updated `LinearProblem` import paths, and changes to
> `x.scatter_forward()` behavior. The code has not been tested with
> versions beyond 0.7.0.

---

## Attribution

### Filtering algorithm

The Ensemble Score Filter (EnSF) implemented in `heat_equation/ensf.py` is
based on:

> Bao, F., Zhang, Z., & Zhang, G. (2024). *An ensemble score filter for
> tracking high-dimensional nonlinear dynamical systems.* Computer Methods
> in Applied Mechanics and Engineering, 432, 117447.
> https://doi.org/10.1016/j.cma.2024.117447

### Heat equation solver

The FEniCSx solver structure (backward Euler time discretization, variational
form, `LinearProblem` usage) is adapted from:

> Dokken, J.S. *FEniCSx Tutorial*, Chapter 2 — The Heat Equation.
> https://jsdokken.com/dolfinx-tutorial/chapter2/heat_equation.html
> Licensed under CC BY 4.0.

Modifications: spatially varying Gaussian permeability field $K(x,y;\theta)$,
parametric interface via `rbnicsx.backends.SymbolicParameters`, split
inhomogeneous Dirichlet boundary conditions, and a class-based interface
for use as a forecast operator inside an ensemble filtering loop.

### Allen--Cahn solver and experiment setup

The grain coarsening experiment setup — domain $\Omega = (-0.5, 0.5)^2$,
interfacial width $\varepsilon = 0.01$, $T = 10$, spatial mesh $h = 1/128$,
time step $\Delta t = 0.01$, filter time step $\Delta t_\text{filter} = T/250$,
initial condition $\phi_0 \sim U(-0.9, 0.9)$, and three mobility cases —
follows the numerical experiments of:

> Huynh, P.T., López Fajardo, R.Y., Zhang, G., Ju, L., & Bao, F. (2026).
> *A score-based diffusion model approach for adaptive learning of stochastic
> partial differential equation solutions.* Journal of Computational Physics,
> 556, 114814. https://doi.org/10.1016/j.jcp.2026.114814

Note: the reference solution in that paper uses a second-order BDF scheme
(BDF2) with central finite differences, which preserves the discrete maximum
bound principle and unconditional energy stability. The solver in this
repository uses a first-order IMEX scheme implemented in FEniCSx, which is a
simpler alternative suitable for the filtering forecast step. The IMEX time
discretization follows:

> Tang, T., & Yang, J. (2016). *Implicit-explicit scheme for the Allen-Cahn
> equation preserves the maximum principle.* Journal of Computational
> Mathematics, 34(5), 451--461.
> https://www.jstor.org/stable/45151391

Initial implementation attempted to follow the FEniCS Project community forum:

> *Correct implementation of Allen-Cahn.* FEniCS Project Discourse, 2023.
> https://fenicsproject.discourse.group/t/correct-implementation-of-allencahn/13487

That implementation requires the `dolfinx_mpc` library for periodic boundary
conditions, which could not be installed due to version incompatibilities.
The solver was subsequently reimplemented with homogeneous Neumann boundary
conditions using the IMEX scheme above, with AI coding assistance
(ChatGPT/Claude), using the general FEniCSx time-stepping structure from:

> Dokken, J.S. *FEniCSx Tutorial*, Chapter 2 — The Heat Equation.
> https://jsdokken.com/dolfinx-tutorial/chapter2/heat_equation.html
> Licensed under CC BY 4.0.
