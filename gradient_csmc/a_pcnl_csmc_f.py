"""
Implements the Particle-aPCNL kernel of the paper.
"""
from functools import partial
from typing import Callable, Union, Any

import jax
from chex import Array, PRNGKey
from jax import numpy as jnp
from jax.scipy.linalg import solve_triangular

from gradient_csmc.csmc import kernel as csmc_kernel
from gradient_csmc.utils.math import mvn_logpdf

_STABLE_ZERO = 1e-15


def get_kernel(mu0: Array, P0: Array, r0: Callable,
               mut: Union[Callable, tuple[Callable, Any]], Qs,
               rt: Union[Callable, tuple[Callable, Any]],
               resampling_func: Callable, ancestor_move_func: Callable, N: int,
               backward: bool = False
               ):
    """
    Constructor for the Particle-aPCNL kernel.

    Parameters
    ----------
    mu0:
        Initial distribution mean vector.
    P0:
        Initial distribution covariance matrix.
    r0:
        Initial distribution log-density.
    mut:
        Transition function. Either a callable or a tuple of a callable and its parameters.
    Qs:
        Array of covariance matrices. The first element is the covariance at time 0: p(x0) = N(m0, Qs[0]).
        The rest is the covariance matrix of transition functions: p(xt | xt-1) = N(xt | µ(xt-1), Qs[t]).
    rt:
        Potential function. Either a callable or a tuple of a callable and its parameters.
    resampling_func:
        Resampling function.
    ancestor_move_func:
        Ancestor move function.
    N:
        Number of particles.
    backward:
        Whether to run the backward sampling kernel.

    Returns
    -------
    kernel: Callable
        APCNL-CSMC kernel function.

    """
    chol_P0 = jnp.linalg.cholesky(P0)
    chol_Qs = jnp.linalg.cholesky(Qs)

    inv_chol_P0 = solve_triangular(chol_P0, jnp.eye(P0.shape[-1]), lower=True)
    inv_chol_Qs = jax.vmap(lambda z: solve_triangular(z, jnp.eye(Qs.shape[-1]), lower=True))(chol_Qs)

    Qs_ = jnp.insert(Qs, 0, P0, axis=0)
    get_proposal_params = make_proposal(Qs_)

    mut, mut_params = mut if isinstance(mut, tuple) else (mut, None)
    rt, rt_params = rt if isinstance(rt, tuple) else (rt, None)

    val_and_grad_r0 = jnp.vectorize(jax.value_and_grad(r0), signature='(d)->(),(d)')
    val_and_grad_rt = jnp.vectorize(jax.value_and_grad(rt, argnums=1), signature='(d),(d)->(),(d)', excluded=(2,))

    def kernel(key: PRNGKey, x_star: Array, b_star: Array, ells: Array, deltas: Array):
        """
        Implements the Particle-aPCNL kernel of the paper.

        Parameters
        ----------
        key: PRNGKey
            JAX PRNGKey.
        x_star:
            Reference trajectory.
        b_star:
            Reference ancestor indices.
        ells:
            Proposal variance scaling.
        deltas:
            Gradient step-size.

        Returns
        -------
        x_star: Array
            New reference trajectory.
        b_star: Array
            New reference ancestor indices.

        """
        ###############################
        #        HOUSEKEEPING         #
        ###############################
        betas = 2 / (2 + ells)
        pCs, chols_prop, chols_inv_prop = get_proposal_params(betas)
        # jax.debug.print("pCS shape = {}", pCs.shape)

        T, d_x = x_star.shape
        key_csmc, _, key_aux = jax.random.split(key, 3)

        _, grad_log_w_star_0 = val_and_grad_r0(x_star[0])
        _, grad_log_w_star = jax.vmap(val_and_grad_rt)(x_star[:-1], x_star[1:], rt_params)
        grad_log_w_star = jnp.insert(grad_log_w_star, 0, grad_log_w_star_0, axis=0)

        chol_Cs = jnp.insert(chol_Qs, 0, chol_P0, axis=0)
        drift = 0.5 * deltas[:, None] * jax.vmap(lambda C, g: C @ g)(pCs, grad_log_w_star)
 
        eps_aux = jax.random.normal(key_aux, shape=(T, d_x))
        noise = jax.vmap(lambda delta, chol, eps: jnp.sqrt(0.5 * delta) * (chol @ eps))(deltas, chol_Cs, eps_aux)
        aux_vars = x_star + drift + noise

        ###############################################
        #       Proposal and weight functions         #
        ###############################################

        spec_M0_rvs = partial(M0_rvs, params=(aux_vars[0], mu0, betas[0], chols_prop[0]))
        spec_M0_logpdf = partial(M0_logpdf, params=(aux_vars[0], mu0, betas[0], chols_inv_prop[0]))
        spec_Mt_rvs = partial(Mt_rvs, mut=mut)
        spec_Mt_logpdf = partial(Mt_logpdf, mut=mut)
        spec_Gamma_0 = partial(Gamma_0, val_and_grad_r0=val_and_grad_r0, params=(aux_vars[0], mu0, pCs[0], 
                                                                                 inv_chol_P0, deltas[0], ells[0]))
        spec_Gamma_t = partial(Gamma_t, val_and_grad_rt=val_and_grad_rt, mut=mut)

        Mt_params = (aux_vars[1:], betas[1:], chols_prop[1:], chols_inv_prop[1:]), mut_params
        Mt = spec_Mt_rvs, spec_Mt_logpdf, Mt_params
        Gamma_t_params = aux_vars[1:], deltas[1:], ells[1:], pCs[1:], inv_chol_Qs, mut_params, rt_params
        Gamma_t_plus_params = spec_Gamma_t, Gamma_t_params
        M0 = spec_M0_rvs, spec_M0_logpdf

        ###########################
        #       Call CSMC         #
        ###########################
        return csmc_kernel(key_csmc, x_star, b_star, M0, spec_Gamma_0, Mt, Gamma_t_plus_params, resampling_func,
                           ancestor_move_func, N, backward)

    return kernel

def make_proposal(Qs):
    """
    Note: Assume preconditioning matrix = Q

    Qs: Array (T, d, d) of covariances matrices
    """
    def _get_proposal_params(beta, Q):
        Q_prop = (1 - beta) * Q
        chol_Q_prop = jnp.linalg.cholesky(Q_prop)
        chol_Q_prop_inv = solve_triangular(chol_Q_prop, jnp.eye(Q.shape[0]), lower=True)
        return Q, chol_Q_prop, chol_Q_prop_inv

    def get_proposal_params(betas):
        return jax.vmap(_get_proposal_params)(betas, Qs)
    
    return get_proposal_params


def M0_rvs(key, N, params):
    u, mu0, beta, chol_prop = params
    dx = mu0.shape[0]
    mean_0 = (1 - beta) * mu0
    mean_0 = (beta * u) + mean_0
    eps = jax.random.normal(key, (N, dx))
    # jax.debug.print("mean_0: {}", mean_0)
    # jax.debug.print("P0: {}", chol_prop @ chol_prop.T)
    return mean_0[None, :] + eps @ chol_prop.T


def M0_logpdf(x, params):
    u, mu0, beta, chol_prop_inv = params
    mean_0 = (1 - beta) * mu0
    mean_0 = (beta * u) + mean_0
    return mvn_logpdf(x, mean_0, None, chol_inv=chol_prop_inv, constant=False)


def Mt_rvs(key, x_prev, params, mut):
    (u, beta, chol, _), mu_params = params
    N_, dx = x_prev.shape
    mean_prior = (1 - beta) * mut(x_prev, mu_params)
    mean_t = (beta * u[None, :]) + mean_prior
    eps = jax.random.normal(key, (N_, dx))
    return mean_t + eps @ chol.T


def Mt_logpdf(x_prev, x, params, mut):
    (u, beta, chol, chol_inv), mu_params = params
    mean_prior = (1 - beta) * mut(x_prev, mu_params)
    mean_t = (beta * u[None, :]) + mean_prior
    return mvn_logpdf(x, mean_t, None, chol_inv=chol_inv, constant=False)


def Gamma_0(x, val_and_grad_r0, params):
    u, mu0, pC, inv_chol_P0, delta, ell = params
    out = mvn_logpdf(x, mu0, None, chol_inv=inv_chol_P0, constant=False)

    r0_val, r0_grad = val_and_grad_r0(x)   
    out += r0_val
    
    correction_term = mvn_logpdf(x + 0.5 * delta * (r0_grad @ pC.T), u, None, chol_inv=jnp.sqrt(2.0 / ell)*inv_chol_P0, constant=False)
    out += correction_term
    return out


def Gamma_t(x_prev, x, params, val_and_grad_rt, mut):
    u, delta, ell, pC, inv_chol_Q, mu_params, rt_params_here = params
    out = mvn_logpdf(x, mut(x_prev, mu_params), None, chol_inv=inv_chol_Q, constant=False)

    rt_val, rt_grad = val_and_grad_rt(x_prev, x, rt_params_here)
    out += rt_val
    
    correction_term =  mvn_logpdf(x + 0.5 * delta * (rt_grad @ pC.T), u, None, chol_inv=jnp.sqrt(2.0 / ell)*inv_chol_Q, constant=False)
    out += correction_term
    return out


if __name__ == "__main__":
    Qs = jnp.stack([jnp.eye(10) for _ in range(10)])
    betas = 2 / jnp.ones(10)
    get_proposal_params = make_proposal(Qs)
    pCs, chols_prop, chols_inv_prop = get_proposal_params(betas)
    jax.debug.print("pCS shape = {}", pCs.shape)