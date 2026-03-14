"""
Continuous time linear Gaussian model
"""

from functools import partial

from chex import PRNGKey, Array
import jax
from jax import numpy as jnp
from jax.scipy.stats import norm


def random_corr_chol(key: PRNGKey, dim: int, jitter: float = 1e-6):
    """
    Generate a random correlation matrix R and its Cholesky factor L
    such that R = L @ L.T.

    Parameters
    ----------
    key:    Random key.
    dim:    Dimension of the correlation matrix.
    jitter: Small diagonal shift to ensure strict positive definiteness.

    Returns
    -------
    L: Lower-triangular Cholesky factor of R.
    R: Random correlation matrix with diag(R) = 1.
    """
    M = jax.random.normal(key, (dim, dim))

    # SPD by construction
    C = M @ M.T + jitter * jnp.eye(dim)
    C = 0.5 * (C + C.T)

    # Rescale to correlation matrix
    d = jnp.sqrt(jnp.diag(C))
    R = C / (d[:, None] * d[None, :])
    R = 0.5 * (R + R.T)

    L = jnp.linalg.cholesky(R)
    return L, R


@partial(jax.jit, static_argnums=(3))
def get_data(key: PRNGKey, phi: float, sigma: float, dim: int, dT: Array):
    """
    Produce continuous time LGSSM data where
        dX_t = -phi X_t dt + sigma dW_t
        Y_k | X_{t_k} ~ N(X_{t_k}, I)
    where dT[k] = t_{k+1} - t_k.

    Parameters
    ----------

    Returns
    -------
    xs: (T, dim) Latent states x_0, ..., x_{T-1}
    ys: (T, dim) Observations y_0, ..., y_{T-1}
    As: (T, 1) Persistence for each timestep
    Qs: (T, 1) STD for each timestep 
    """

    init_key, sampling_key = jax.random.split(key)
    T = dT.shape[0]  # number of timesteps rather than horizon time

    x0 = (sigma / jnp.sqrt(2 * phi)) * jax.random.normal(init_key, (dim,))
    eps_xs, eps_ys = jax.random.normal(sampling_key, (2, T, dim))

    # continuous time dynamics
    As = jnp.exp(-phi * dT)
    Qs = sigma * jnp.sqrt((1.0 - jnp.exp(-2.0 * phi * dT)) / (2.0 * phi))

    def body(x_k, inps):
        eps_x, eps_y, At, Qt = inps
        y_k = x_k + eps_y
        x_kp1 = At * x_k + Qt * eps_x
        return x_kp1, (x_k, y_k)
    
    _, (xs, ys) = jax.lax.scan(body, x0, (eps_xs, eps_ys, As, Qs))
    return xs, ys, As, Qs


@partial(jnp.vectorize, signature="(n),(n)->()")
def log_potential(x, y):
    val = norm.logpdf(y, x)
    return jnp.sum(val)


def log_likelihood(x, y):
    return jnp.sum(log_potential(x, y))


def log_pdf(xs, ys, sigma):
    def _logpdf(zs):
        out = jnp.sum(norm.logpdf(zs[0], scale=sigma))
        out += jnp.sum(norm.logpdf(zs[1:], zs[:-1], sigma))
        out += jnp.sum(norm.logpdf(zs, ys))
        return out

    return jnp.vectorize(_logpdf, signature="(T,d)->()")(xs)









# @partial(jax.jit, static_argnums=(4,))
# def get_dynamics(nu, phi, tau, rho, dim):
#     F = phi * jnp.eye(dim)
#     Q, P0 = stationary_covariance(phi, tau, rho, dim)
#     mu = nu * jnp.ones((dim,))
#     b = mu + F @ mu
#     return mu, P0, F, Q, b


# @partial(jax.jit, static_argnums=(3,))
# def stationary_covariance(phi, tau, rho, dim):
#     U = tau * rho * jnp.ones((dim, dim))
#     U = U.at[np.diag_indices(dim)].set(tau)
#     vec_U = jnp.reshape(U, (dim ** 2, 1))
#     vec_U_star = vec_U / (1 - phi ** 2)
#     U_star = jnp.reshape(vec_U_star, (dim, dim))
#     return U, U_star
