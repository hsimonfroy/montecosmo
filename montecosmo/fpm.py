from collections.abc import Callable
# from typing import ClassVar
# from typing_extensions import TypeAlias

# from equinox.internal import ω
# from jaxtyping import ArrayLike, Float, PyTree

# from diffrax._custom_types import Args, BoolScalarLike, DenseInfo, RealScalarLike, VF
# from diffrax import LocalLinearInterpolation, RESULTS, AbstractTerm, AbstractSolver
# import jax_cosmo as jc
# from jax import lax




# from jaxpm.pm import pm_forces
# from jax import numpy as jnp
# from jaxpm.growth import E, growth_factor as Gp, Gf, dGfa, growth_rate
# import jax_cosmo as jc
# from jax import lax
# from diffrax import ODETerm
# from diffrax._custom_types import RealScalarLike


# def gp(cosmo, a):
#     r""" Derivative of D1 against a

#     Parameters
#     ----------
#     cosmo: dict
#       Cosmology dictionary.

#     a : array_like
#        Scale factor.

#     Returns
#     -------
#     Scalar float Tensor : the derivative of D1 against a.

#     Notes
#     -----

#     The expression for :math:`gp(a)` is:

#     .. math::
#         gp(a)=\frac{dD1}{da}= D'_{1norm}/a
#     """
#     f1 = growth_rate(cosmo, a)
#     g1 = Gp(cosmo, a)
#     D1f = f1 * g1 / a
#     return D1f


# def diffrax_ode(cosmo, mesh_shape, paint_absolute_pos=True, halo_size=0, sharding=None):
#     def nbody_ode(a, state, args):
#         """
#         state is a tuple (position, velocities)
#         """
#         pos, vel = state

#         forces = (
#             pm_forces(
#                 pos,
#                 mesh_shape=mesh_shape,
#                 # paint_absolute_pos=paint_absolute_pos,
#                 # halo_size=halo_size,
#                 # sharding=sharding,
#             )
#             * 1.5
#             * cosmo.Omega_m
#         )

#         # Computes the update of position (drift)
#         dpos = 1.0 / (a**3 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * vel

#         # Computes the update of velocity (kick)
#         dvel = 1.0 / (a**2 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * forces

#         return jnp.stack([dpos, dvel])

#     return nbody_ode


# def symplectic_ode(mesh_shape, paint_absolute_pos=True, halo_size=0, sharding=None):
#     def drift(a, vel, args):
#         """
#         state is a tuple (position, velocities)
#         """
#         cosmo = args
#         # Computes the update of position (drift)
#         dpos = 1 / (a**3 * E(cosmo, a)) * vel

#         return dpos

#     def kick(a, pos, args):
#         """
#         state is a tuple (position, velocities)
#         """
#         # Computes the update of velocity (kick)
#         cosmo = args

#         forces = (
#             pm_forces(
#                 pos,
#                 mesh_shape=mesh_shape,
#                 # paint_absolute_pos=paint_absolute_pos,
#                 # halo_size=halo_size,
#                 # sharding=sharding,
#             )
#             * 1.5
#             * cosmo.Omega_m
#         )

#         # Computes the update of velocity (kick)
#         dvel = 1.0 / (a**2 * E(cosmo, a)) * forces

#         return dvel

#     return kick, drift


# class LeapFrogODETerm(ODETerm):
#     def contr(self, t0: RealScalarLike, t1: RealScalarLike, **kwargs) -> RealScalarLike:

#         action = kwargs.get("action", "D")  # Action type: 'D' for Drift, 'K' for Kick
#         cosmo = kwargs.get("cosmo", None)
#         t0t1 = (t0 * t1) ** 0.5  # Geometric mean of t0 and t1

#         if cosmo is None:
#             return 0.0

#         if action == "D":  # Drift case
#             return (Gp(cosmo, t1) - Gp(cosmo, t0)) / gp(cosmo, t0t1)

#         elif action == "FK":
#             return (Gf(cosmo, t0t1) - Gf(cosmo, t0)) / dGfa(cosmo, t0)

#         elif action == "K":  # Kick case
#             last_kick_cond = kwargs.get(
#                 "cond", False
#             )  # True for last kick, False for double kick

#             # Dynamic conditions for double kick or last kick
#             def double_kick(t0, t1, t0t1):
#                 # Two kicks combined
#                 t2 = 2 * t1 - t0  # Next time step t2 for the second kick
#                 t1t2 = (t1 * t2) ** 0.5  # Intermediate scale factor
#                 return (Gf(cosmo, t1)   - Gf(cosmo, t0t1)) / dGfa(cosmo, t1) + (
#                         Gf(cosmo, t1t2) - Gf(cosmo, t1))   / dGfa(cosmo, t1)  # fmt: skip

#             def single_kick(t0, t1, t0t1):
#                 # Single kick for the final step
#                 return (Gf(cosmo, t1) - Gf(cosmo, t0t1)) / dGfa(cosmo, t1)

#             return lax.cond(last_kick_cond, single_kick, double_kick, t0, t1, t0t1)

#         else:
#             raise ValueError(f"Unknown action type: {action}")



















# _ErrorEstimate: TypeAlias = None

# Ya: TypeAlias = PyTree[Float[ArrayLike, "y"]]
# Yb: TypeAlias = PyTree[Float[ArrayLike, "y"]]

# _SolverState: TypeAlias = tuple[Ya, Yb]

# class EfficientLeapFrog(AbstractSolver):
#     """
#     Efficient LeapFrog

#     Symplectic method. Does not support adaptive step sizing. Uses 1st order local
#     linear interpolation for dense/ts output.
#     """
#     initial_t0 : RealScalarLike
#     final_t1: RealScalarLike
#     cosmo: jc.Cosmology  # Declares cosmology object as a data member
#     term_structure: ClassVar = (AbstractTerm, AbstractTerm)
#     interpolation_cls: ClassVar[Callable[..., LocalLinearInterpolation]] = (
#         LocalLinearInterpolation
#     )

#     def order(self, terms):
#         return 2

#     def init(
#         self,
#         terms: tuple[LeapFrogODETerm, LeapFrogODETerm],
#         t0: RealScalarLike,
#         t1: RealScalarLike,
#         y0: tuple[Ya, Yb],
#         args: Args,
#     ) -> _SolverState:
#         term_1, _ = terms
#         y0_1, y0_2 = y0

#         # Compute forces (kick update)
#         control = term_1.contr(t0, t1, action="FK", cosmo=self.cosmo)
#         y1_2 = (y0_2**ω + term_1.vf_prod(t0, y0_1, args, control) ** ω).ω

#         return (y0_1, y1_2)


#     def step(
#         self,
#         terms: tuple[LeapFrogODETerm, LeapFrogODETerm],
#         t0: RealScalarLike,
#         t1: RealScalarLike,
#         y0: tuple[Ya, Yb],
#         args: Args,
#         solver_state: _SolverState,
#         made_jump: BoolScalarLike,
#     ) -> tuple[tuple[Ya, Yb], _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
#         del made_jump

#         term_1, term_2 = terms
#         y0_1, y0_2 = lax.cond(t0 == self.initial_t0, lambda _ : solver_state , lambda _ : y0, None)
#         t0t1 = (t0 * t1) ** 0.5

#         # Drift
#         control1 = term_2.contr(t0, t1, action="D", cosmo=self.cosmo)
#         y1_1 = (y0_1**ω + term_2.vf_prod(t0t1, y0_2, args, control1) ** ω).ω

#         # Double kick or last kick
#         control2 = term_1.contr(
#             t0, t1, action="K", cosmo=self.cosmo, cond=(t1 == self.final_t1)
#         )
#         y1_2 = (y0_2**ω + term_1.vf_prod(t1, y1_1, args, control2) ** ω).ω

#         y1 = (y1_1, y1_2)
#         dense_info = dict(y0=y0, y1=y1)
#         return y1, None, dense_info, solver_state, RESULTS.successful

#     def func(
#         self,
#         terms: tuple[AbstractTerm, AbstractTerm],
#         t0: RealScalarLike,
#         y0: tuple[Ya, Yb],
#         args: Args,
#     ) -> VF:
#         term_1, term_2 = terms
#         y0_1, y0_2 = y0
#         f1 = term_1.vf(t0, y0_2, args)
#         f2 = term_2.vf(t0, y0_1, args)
#         return f1, f2
    








