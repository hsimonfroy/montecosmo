from jax import numpy as jnp, random as jr, grad, hessian, vmap, jvp, jit, vjp, linearize, lax
from functools import partial

def cov_x_from_pot_x_y(pot_fn, x, y, method='exact', chunk_size=None, eps_diag=1e-9):
    """
    Compute marginal covariance matrix on x from potential function U at point x and y.
    Use that if H = nabla^2 U(x,y) = [A B; B^T D] then Cov_x = (A - B D^-1 B^T)^-1.
    By assuming D is diagonal, the computation is performed without forming B or full y-Hessian,
    allowing for large y dimension.
    
    Args:
      pot_fn: callable (x, y) -> scalar potential U(x, y)
      x: jnp.array, shape (m,)
      y: jnp.array, shape (n,)
      method: 'exact' or 'hutchinson' to compute diag(D)
      eps_diag: small regularization added to diagonal D before inversion for numerical stability

    Returns:
      cov_x: jnp.array shape (m, m)
      schur: the Schur complement matrix A - B D^-1 B^T (m x m)
    """
    m = x.shape[0]
    n = y.shape[0]

    # 1) compute A explicitly (m x m)
    A = hessian(pot_fn, argnums=0)(x, y)  # shape (m,m)

    # 2) compute diag(D)
    if chunk_size is None:
        chunk_size = n
    if method == 'exact':
        diag_D = diag_in_chunks(partial(pot_fn, x), y, chunk_size=chunk_size) # NOTE: JAX>=0.4.32 for batch_size
        # diag_D = diag_exact(partial(pot_fn, x), y, chunk_size=chunk_size)
    elif method == 'hutchinson':
        diag_D = diag_hutchinson(partial(pot_fn, x), y, n_probes=chunk_size, seed=42)
    diag_Dinv = 1.0 / (diag_D + eps_diag)  # shape (n,) regularized

    # 3) Define function that computes (B D^-1 B^T) v for arbitrary v in R^m
    def BDinvBT_matvec(v):
        # v: shape (m,) compute u = B^T v = grad_y of scalar s = v^T grad_x(x,y)
        scalar_s = lambda xx, yy: jnp.vdot(grad(pot_fn, argnums=0)(xx, yy), v)
        u = grad(scalar_s, argnums=1)(x, y)      # shape (n,)
        u_scaled = u * diag_Dinv                # shape (n,)
        scalar_t = lambda xx, yy: jnp.vdot(grad(pot_fn, argnums=1)(xx, yy), u_scaled)
        result = grad(scalar_t, argnums=0)(x, y)  # shape (m,)
        return result

    # 4) Build Schur complement matrix, symmetrize it for safety, invert to get Sigma_x
    BDinvBT = vmap(BDinvBT_matvec)(jnp.eye(m)).T  # shape (m, m) where each column is for e_i
    schur = A - BDinvBT
    schur = (schur + schur.T) / 2
    cov_x = jnp.linalg.inv(schur)
    return cov_x, schur

def diag_in_chunks(pot_fn, y, chunk_size=64):
    def body(_, ids):
        # For each index k in idxs, form unit basis e_k and jvp
        def per_k(k):
            e = jnp.zeros_like(y).at[k].set(1.0)
            # More efficient than Hvp for diagonal term
            _, jvp_out = jvp(lambda yy: jvp(pot_fn, (yy,), (e,))[1], (y,), (e,))
            return jvp_out
            # _, hvp = linearize(grad(pot_fn), y)   # hvp(v) = H @ v
            # return hvp(e)[k]
        return None, vmap(per_k)(ids)

    n = y.shape[0]
    n_chunks = (n + chunk_size - 1) // chunk_size
    ids = jnp.pad(jnp.arange(n), (0, n_chunks * chunk_size - n))
    ids = jnp.stack(jnp.split(ids, n_chunks))
    _, diag = lax.scan(body, None, ids)
    return diag.reshape(-1)[:n]

def diag_exact(pot_fn, y, chunk_size=64):
    def fn(idx):
        # For each index k in idxs, form unit basis e_k and jvp
        e = jnp.zeros_like(y).at[idx].set(1.0)
        # More efficient than Hvp for diagonal term
        _, jvp_out = jvp(lambda yy: jvp(pot_fn, (yy,), (e,))[1], (y,), (e,))
        return jvp_out
    
    return lax.map(fn, jnp.arange(y.shape[0]), batch_size=chunk_size) # NOTE: JAX>=0.4.32 for batch_size

def diag_hutchinson(pot_fn, y, n_probes=64, seed=42):
    if isinstance(seed, int):
        seed = jr.key(seed)
    seeds = jr.split(seed, n_probes)
    _, hvp = linearize(grad(pot_fn), y)

    def body(diag, seed):
        r = jr.rademacher(seed, y.shape, dtype=float)
        Hr = hvp(r)
        return diag + r * Hr / n_probes, None

    diag, _ = lax.scan(body, jnp.zeros_like(y), seeds)
    return diag







# from jax import random as jr, jacobian
# def make_test_pot(m, n):
#     # simple Gaussian-like test with coupling x^T M z and independent priors
#     key = jr.key(0)
#     M = jr.normal(key, (m, n)) * 0.01
#     cov = jr.normal(key, (m, m))
#     Q = cov @ cov.T + jnp.eye(m) # x precision
#     R = jnp.eye(n) * jnp.abs(jr.normal(key, (n,)))  # z precision (diagonal)
#     def pot(x, z):
#         prior_x = 0.5 * (x @ Q @ x)
#         prior_z = 0.5 * (z @ R @ z)
#         cross = 2 * jnp.dot(x, (M @ z))  # linear coupling in exponent
#         return prior_x + prior_z + cross
#     return pot

# m, n = 3, 6
# pot_test = make_test_pot(m, n)
# x = jnp.zeros(m).astype(float) * 0.1
# z = jnp.zeros(n).astype(float) * 0.2

# Sigma_x_est, S = cov_x_from_pot_x_y(pot_test, x, z)

# # For small dims compute full Hessian and invert directly to compare:
# full_H = hessian(lambda xy: pot_test(xy[:m], xy[m:]), argnums=0)(jnp.concatenate([x, z])) 
# # Build full Hessian via block hessians:
# A_full = hessian(pot_test, argnums=0)(x, z)
# B_full = jacobian(grad(pot_test, argnums=0), argnums=1)(x, z)  # shape (m, n)
# D_full = hessian(pot_test, argnums=1)(x, z)
# H_full = jnp.block([[A_full, B_full],
#                     [B_full.T, D_full]])
# # Invert full H and read Sigma_x from top-left block of H^{-1}
# Hinv = jnp.linalg.inv(H_full)
# Sigma_x_true = Hinv[:m, :m]

# print("Sigma_x_est\n", Sigma_x_est)
# print("Sigma_x_true\n", Sigma_x_true)
# print("Spectra\n", jnp.linalg.eigvalsh(Sigma_x_true))
# print("Max abs diff:", jnp.max(jnp.abs(Sigma_x_est - Sigma_x_true)))
