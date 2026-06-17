"""Regulated-dynamics (stochastic-isokinetic) integrator of Abreu & Tuckerman.

Implements the NVT "regulated dynamics" of Abreu & Tuckerman (Eur. Phys. J. B
2021, 94, 231; J. Chem. Theory Comput. 2020, 16, 7314) as an OpenMM
CustomIntegrator. This is the Hamiltonian reformulation of the stochastic
isokinetic Nose-Hoover (SIN(R)) method:

    H = sum_a  m_a c_a^2 ln cosh(p_a / (m_a c_a))  +  U(r),    c_a = sqrt(L kT / m_a)

so the velocity is a *bounded* nonlinear function of momentum,

    v_a = c_a tanh(p_a / (m_a c_a)),     |v_a| < c_a = sqrt(L kT / m_a),

which prevents kinetic energy from piling up in any single degree of freedom and
thereby suppresses the RESPA resonances that otherwise cap MD time steps at a few
fs. Each Cartesian DOF carries one Nose-Hoover-Langevin (NHL) thermostat:

    dr = v(p) dt
    dp = (F - u p) dt
    du = [(p v(p) - kT)/Q] dt - gamma u dt + sqrt(2 gamma kT / Q) dW

The configurational distribution sampled is exactly canonical, rho(r) ~ exp(-U/kT)
(the velocity distribution is NOT Maxwell-Boltzmann -- this scheme is for
configurational sampling / free energies, not dynamical properties).

Parameters (paper defaults, NVT solvation-FE example):
    L                 : 1 < L <= 4 (resonance-optimal); larger -> closer to MB but
                        worse resonance avoidance. Default 2.
    characteristic_time tau : thermostat time, default 10 fs; then
                        Q = kT tau^2,  gamma = 1/tau.

Integrator: single-time-scale "middle" (BAOAB-like) splitting
    B (force half-kick) - A (drift half) - O (NHL thermostat, middle) - A - B
The thermostat O is itself split OU(half) - NHdrift(half) - p-scale - NHdrift(half)
- OU(half). RESPA force-group splitting (for the very large steps) is a later
extension; this single-scale version already gives the bounded-velocity sampling
and is what we validate first.

Limitation: this version assumes an UNCONSTRAINED system (the paper uses fully
flexible force fields, e.g. SPC-Fw). Constraints would need the regulated velocity
threaded through the constraint solver, which is not handled here yet.
"""

import openmm
from openmm import unit

from .units import KB


def assign_respa_force_groups(system, three_scale=True):
    """Assign force groups for RESPA with regulated dynamics.

    three_scale=True (the paper's scheme for PME systems): bonded/other -> group 0
    (fast inner), nonbonded DIRECT space -> group 1 (middle), nonbonded RECIPROCAL
    space (PME) -> group 2 (slow outer). The reciprocal space is smooth and
    expensive, so it goes on the large outer step.

    three_scale=False (two scale): nonbonded -> group 1 (slow), everything else ->
    group 0 (fast).
    """
    for force in system.getForces():
        if "Nonbonded" in force.__class__.__name__:
            force.setForceGroup(1)
            if three_scale and hasattr(force, "setReciprocalSpaceForceGroup"):
                force.setReciprocalSpaceForceGroup(2)
        else:
            force.setForceGroup(0)


def make_regulated_integrator(temperature=298.15, timestep_fs=4.0, L=2.0,
                              characteristic_time_fs=10.0, respa=None,
                              system=None):
    """Build the NVT regulated-dynamics CustomIntegrator (see module docstring).

    ``respa`` controls multiple-time-scale (RESPA) force splitting:
      None / []        : single time scale (the whole force `f` on the outer step).
      [n_inner]        : 2-scale -- nonbonded (group 1, slow, outer) + everything
                         else (group 0, fast) run n_inner inner substeps.
      [n_inner,n_mid]  : 3-scale -- bonded (group 0, fast, innermost) / nonbonded
                         direct space (group 1, middle, n_mid middle substeps per
                         outer) / nonbonded PME RECIPROCAL space (group 2, slow,
                         outer). This is the paper's scheme: the smooth, expensive
                         long-range force takes the large outer step.
    respa[i] is the number of substeps of the next-faster scale per step of scale
    (i+1). Pass ``system`` (mutated: force groups assigned) whenever respa is set.

    Returns an openmm.CustomIntegrator; init the per-DOF momentum with
    init_regulated_momenta() after the Context velocities are set.
    """
    kT = (KB * (temperature * unit.kelvin)).value_in_unit(unit.kilojoule_per_mole)
    tau = characteristic_time_fs * 1e-3                      # ps
    Q = kT * tau ** 2                                        # thermostat mass
    gamma = 1.0 / tau                                        # friction (1/ps)
    respa = list(respa or [])
    n_scales = len(respa) + 1                                # force groups 0..n_scales-1
    if respa:
        if system is None:
            raise ValueError("respa requires `system` to assign force groups")
        assign_respa_force_groups(system, three_scale=(n_scales >= 3))

    ci = openmm.CustomIntegrator(timestep_fs * unit.femtosecond)
    ci.addGlobalVariable("kT", kT)
    ci.addGlobalVariable("Lreg", L)
    ci.addGlobalVariable("Q", Q)
    ci.addGlobalVariable("gamma", gamma)
    ci.addPerDofVariable("p", 0.0)        # momentum (the integrated variable)
    ci.addPerDofVariable("u", 0.0)        # NHL thermostat velocity, one per DOF
    ci.addPerDofVariable("c", 0.0)        # c_a = sqrt(L kT / m), constant per DOF
    ci.addPerDofVariable("vreg", 0.0)     # regulated velocity v(p)

    ci.addUpdateContextState()
    ci.addComputePerDof("c", "sqrt(Lreg*kT/m)")

    def kick(group, h):                                    # B : p += h * F_group
        fterm = "f" if n_scales == 1 else f"f{group}"
        ci.addComputePerDof("p", f"p + {h}*{fterm}")

    def drift(h):                                          # A : x += h v(p)
        ci.addComputePerDof("vreg", "c*tanh(p/(m*c))")
        ci.addComputePerDof("x", f"x + {h}*vreg")

    def thermostat(h):                                    # O (middle), timestep h
        ci.addComputePerDof("u", f"exp(-0.5*gamma*{h})*u + sqrt((1-exp(-gamma*{h}))*kT/Q)*gaussian")
        ci.addComputePerDof("vreg", "c*tanh(p/(m*c))")
        ci.addComputePerDof("u", f"u + 0.5*{h}*(p*vreg - kT)/Q")
        ci.addComputePerDof("p", f"p*exp(-u*{h})")
        ci.addComputePerDof("vreg", "c*tanh(p/(m*c))")
        ci.addComputePerDof("u", f"u + 0.5*{h}*(p*vreg - kT)/Q")
        ci.addComputePerDof("u", f"exp(-0.5*gamma*{h})*u + sqrt((1-exp(-gamma*{h}))*kT/Q)*gaussian")

    # Recursive RESPA: propagate force group `level` (and all faster) over time h.
    def build(level, h):
        if level == 0:                                    # innermost: B A O A B
            kick(0, f"0.5*{h}")
            drift(f"0.5*{h}")
            thermostat(h)
            drift(f"0.5*{h}")
            kick(0, f"0.5*{h}")
        else:
            n = respa[level - 1]                           # substeps of (level-1)
            kick(level, f"0.5*{h}")
            for _ in range(n):
                build(level - 1, f"({h}/{n})")
            kick(level, f"0.5*{h}")

    build(n_scales - 1, "dt")

    # keep the Context velocity = regulated velocity (for KE / reporters)
    ci.addComputePerDof("vreg", "c*tanh(p/(m*c))")
    ci.addComputePerDof("v", "vreg")
    return ci


def init_regulated_momenta(context, integrator):
    """Initialize the integrator's per-DOF momentum p from the Context velocities.

    Uses the exact map p = m c artanh(v/c) where |v|<c, clamped just inside the
    bound for the rare thermal velocity that exceeds c; the thermostat relaxes any
    residual mismatch within a few tau.
    """
    import numpy as np
    state = context.getState(getVelocities=True)
    v = np.array(state.getVelocities().value_in_unit(
        unit.nanometer / unit.picosecond))
    system = context.getSystem()
    m = np.array([system.getParticleMass(i).value_in_unit(unit.dalton)
                  for i in range(system.getNumParticles())])
    kT = integrator.getGlobalVariableByName("kT")
    L = integrator.getGlobalVariableByName("Lreg")
    c = np.sqrt(L * kT / m)[:, None]                         # (N,1)
    ratio = np.clip(v / c, -0.999, 0.999)
    p = (m[:, None] * c) * np.arctanh(ratio)
    integrator.setPerDofVariableByName("p", p)
