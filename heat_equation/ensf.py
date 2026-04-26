"""
Ensemble Score Filter (EnSF) utilities.

Implements the score-based reverse SDE used in the analysis step of the
Ensemble Score Filter (EnSF) introduced in:

    Bao, F., Zhang, Z., & Zhang, G. (2024). "An ensemble score filter for
    tracking high-dimensional nonlinear dynamical systems."
    https://doi.org/10.1016/j.cma.2024.117447

The filter replaces the standard Kalman or importance-weight analysis step
with a reverse-time stochastic differential equation (SDE) driven by a
score-based diffusion model. This allows non-Gaussian posterior
distributions to be represented without importance sampling.

Usage
-----
    from ensf import reverse_SDE, make_score_likelihood_arctan

    # Define score likelihood for arctan observation model
    score_fn = make_score_likelihood_arctan(obs, observed_mask, obs_sigma)

    # Run analysis step
    state_particles = reverse_SDE(
        x0=state_particles,
        ensemble_size=ensemble_size,
        state_dim=state_dim,
        eps_alpha=0.05,
        score_likelihood=score_fn,
        time_steps=300,
    )
"""

import numpy as np


# ---------------------------------------------------------------------------
# SDE schedule functions
# ---------------------------------------------------------------------------

def _alpha(t: float, eps_alpha: float) -> float:
    """Forward process mean scaling: alpha(t) = 1 - (1 - eps_alpha) * t."""
    return 1.0 - (1.0 - eps_alpha) * t


def _drift(t: float, eps_alpha: float) -> float:
    """Drift coefficient b(t) = d/dt log(alpha(t))."""
    return -(1.0 - eps_alpha) / (1.0 - (1.0 - eps_alpha) * t)


def _diffusion(t: float, eps_alpha: float) -> float:
    """
    Diffusion coefficient g(t) = sqrt(d(beta_t^2)/dt - 2 b(t) beta_t^2),
    where beta_t^2 = t (variance schedule).
    """
    return np.sqrt(1.0 - 2.0 * _drift(t, eps_alpha) * t)

def _sigma_sq(t: float) -> float:
    """Variance schedule: sigma^2(t) = t."""
    return t


# ---------------------------------------------------------------------------
# Core reverse SDE sampler
# ---------------------------------------------------------------------------

def reverse_SDE(
    x0: np.ndarray,
    ensemble_size: int,
    state_dim: int,
    eps_alpha: float,
    score_likelihood=None,
    time_steps: int = 300,
) -> np.ndarray:
    """
    Generate posterior samples via the reverse-time SDE (EnSF analysis step).

    Starting from a standard Gaussian, runs the reverse diffusion process
    guided by the prior ensemble x0 and an optional likelihood score function.

    Parameters
    ----------
    x0 : np.ndarray, shape (ensemble_size, state_dim)
        Prior ensemble (output of forecast/prediction step).
    ensemble_size : int
        Number of ensemble members.
    state_dim : int
        Dimension of the state space.
    eps_alpha : float
        Small positive constant controlling the forward diffusion endpoint.
        Typical value: 0.05.
    score_likelihood : callable or None
        Function score_likelihood(xt, t) -> np.ndarray of shape
        (ensemble_size, state_dim), returning the score of the likelihood
        at diffusion pseudo-time t. If None, only the prior is used
        (no data assimilation).
    time_steps : int
        Number of Euler steps for the reverse SDE. Higher values give more
        accurate sampling at greater computational cost. Typical: 300.

    Returns
    -------
    xt : np.ndarray, shape (ensemble_size, state_dim)
        Posterior ensemble after the analysis step.
    """
    dt = 1.0 / time_steps

    # Initialize from standard Gaussian
    xt = np.random.randn(ensemble_size, state_dim)
    t = 1.0

    for _ in range(time_steps):
        alpha_t  = _alpha(t, eps_alpha)
        sigma2_t = _sigma_sq(t)
        g        = _diffusion(t, eps_alpha)
        b        = _drift(t, eps_alpha)

        # Prior score: nabla_xt log p(xt | x0) ~ -(xt - alpha_t * x0) / sigma2_t
        prior_score = (xt - alpha_t * x0) / sigma2_t

        if score_likelihood is not None:
            likelihood_score = score_likelihood(xt, t)
            total_score = prior_score - likelihood_score
        else:
            total_score = prior_score

        # Reverse Euler-Maruyama step
        xt += (
            -dt * (b * xt + g**2 * total_score)
            + np.sqrt(dt) * g * np.random.randn(ensemble_size, state_dim)
        )

        t -= dt

    return xt


# ---------------------------------------------------------------------------
# Score likelihood factory for arctan observation model
# ---------------------------------------------------------------------------

def make_score_likelihood_arctan(
    obs: np.ndarray,
    observed_mask: np.ndarray,
    obs_sigma: float,
):
    """
    Build a score likelihood function for the arctan observation model:

        Y = arctan(X) + epsilon,    epsilon ~ N(0, obs_sigma^2 I)

    The score of the likelihood with respect to X is:

        nabla_X log p(Y | X) = -(arctan(X) - Y) / obs_sigma^2 * 1/(1 + X^2)

    A linear damping factor g(tau) = 1 - tau is applied to gradually
    introduce the likelihood as the reverse SDE progresses from tau=1 to 0.

    Parameters
    ----------
    obs : np.ndarray, shape (state_dim,)
        Full observation vector at the current assimilation time.
        Entries at unobserved locations are ignored.
    observed_mask : np.ndarray of bool, shape (state_dim,)
        True at observed degrees of freedom.
    obs_sigma : float
        Observation noise standard deviation.

    Returns
    -------
    score_fn : callable
        score_fn(xt, tau) -> np.ndarray, shape (ensemble_size, state_dim)
    """
    def score_fn(xt: np.ndarray, tau: float) -> np.ndarray:
        score = np.zeros_like(xt)
        resid = np.arctan(xt[:, observed_mask]) - obs[observed_mask]
        jac   = 1.0 / (1.0 + xt[:, observed_mask] ** 2)
        score[:, observed_mask] = -(resid / obs_sigma**2) * jac
        return (1.0 - tau) * score

    return score_fn
