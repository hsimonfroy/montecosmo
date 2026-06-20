import os; os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='1.' # NOTE: jax preallocates GPU (default 75%)
from pathlib import Path
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from jax import numpy as jnp, random as jr, config as jconfig, devices as jdevices, jit, vmap, pmap, grad, debug, tree, lax, value_and_grad
jconfig.update("jax_enable_x64", True)
print(jdevices())

save_dir = Path("/pscratch/sd/h/hsimfroy/png/abacus_c0_i0_z0.8_lrg/optim") # Perlmutter
load_dir = Path("/pscratch/sd/h/hsimfroy/png/abacus_c0_i0_z0.8_lrg/load/") # Perlmutter




# from jax.nn import softplus
# from numpyro.contrib.tfp.distributions import SinhArcsinh

# def noise_dist(delta, a, c, t):   # a, c, t are small coeff vectors
#     loc       = delta                      # or delta_g_det
#     scale     = softplus(a)
#     skewness  =          c
#     tailweight= jnp.exp( t)
#     return SinhArcsinh(loc=loc, scale=scale,
#                        skewness=skewness, tailweight=tailweight)


from jax.nn import softplus
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.util import validate_sample, promote_shapes


class SinhArcsinh(dist.Distribution):
    """Sinh-arcsinh distribution (Jones & Pewsey 2009), standalone NumPyro implementation.

    Matches tfp.distributions.SinhArcsinh (Normal base): if Z ~ Normal(0, 1), then
        Y = loc + scale * sinh((arcsinh(Z) + skewness) * tailweight).
    - skewness > 0 (resp. < 0) skews right (resp. left).
    - tailweight > 1 (resp. < 1) gives heavier (resp. lighter) tails than Normal.
    - skewness=0, tailweight=1 recovers Normal(loc, scale).
    """
    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "skewness": constraints.real,
        "tailweight": constraints.positive,
    }
    support = constraints.real
    reparametrized_params = ["loc", "scale", "skewness", "tailweight"]

    def __init__(self, loc=0.0, scale=1.0, skewness=0.0, tailweight=1.0, *, validate_args=None):
        batch_shape = lax.broadcast_shapes(
            jnp.shape(loc), jnp.shape(scale), jnp.shape(skewness), jnp.shape(tailweight))
        self.loc, self.scale, self.skewness, self.tailweight = promote_shapes(
            loc, scale, skewness, tailweight)
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        z = jr.normal(key, shape)
        w = jnp.sinh((jnp.arcsinh(z) + self.skewness) * self.tailweight)
        return self.loc + self.scale * w

    @validate_sample
    def log_prob(self, value):
        w = (value - self.loc) / self.scale
        # Map back to the standard normal variate Z (inverse transform).
        z = jnp.sinh(jnp.arcsinh(w) / self.tailweight - self.skewness)
        # log N(z) + log|dz/dw| + log|dw/dvalue|,
        # with |dz/dw| = sqrt(1+z^2)/(tailweight*sqrt(1+w^2)).
        return (-0.5 * jnp.log(2 * jnp.pi)
                - 0.5 * z ** 2
                + 0.5 * jnp.log1p(z ** 2)
                - jnp.log(self.tailweight)
                - 0.5 * jnp.log1p(w ** 2)
                - jnp.log(self.scale))




# ---- Plot examples of SinhArcsinh (loc=0) against a Gaussian reference ----
def _pdf(d, x):
    return jnp.exp(d.log_prob(x))

x = jnp.linspace(-6, 6, 1000)
gauss = dist.Normal(0.0, 1.0)  # reference Gaussian

panels = [
    ("Skewness  (scale=1, tailweight=1)", "skewness",
     [(s, dict(scale=1., skewness=s, tailweight=1.)) for s in (-1., -0.5, 0., 0.5, 1.)]),
    ("Tailweight  (scale=1, skewness=0)", "tailweight",
     [(t, dict(scale=1., skewness=0., tailweight=t)) for t in (0.5, 0.7, 1., 1.5, 2.)]),
    ("Scale  (skewness=0.5, tailweight=1.2)", "scale",
     [(sc, dict(scale=sc, skewness=0.5, tailweight=1.2)) for sc in (0.5, 1., 1.5, 2.)]),
]

fig, axs = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4))
for ax, (title, label, cases) in zip(np.atleast_1d(axs), panels):
    for val, kw in cases:
        ax.plot(x, _pdf(SinhArcsinh(loc=0.0, **kw), x), label=f"{label}={val:g}")
    ax.plot(x, _pdf(gauss, x), "k--", lw=1.5, label="Gaussian(0,1)")
    ax.set_title(title); ax.set_xlabel("x"); ax.set_ylabel("pdf"); ax.legend(fontsize=8)

fig.tight_layout()
out = Path(__file__).with_name("sinharcsinh.png")
fig.savefig(out, dpi=150)
print("saved", out)
