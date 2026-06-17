"""Experimental AFED-like dynamic-lambda sampler (toy prototype).

Idea
----
Treat the alchemical coordinate lambda as an auxiliary dynamical variable with
its own (fictitious) mass, friction and thermostat, evolving in a potential of
mean force coupled to the physical system. One *master* lambda in [0, 1] drives
both alchemical context parameters through a smooth schedule:

    lambda_electrostatics = smoothstep(lambda)
    lambda_sterics        = smoothstep(lambda)

(Optionally a two-stage map discharges electrostatics before decoupling sterics;
see master_to_alchemical.)

Outer loop, per block:
    1. Set OpenMM context parameters from the current lambda.
    2. Run N MD steps for the atomic coordinates (thermostatted at T by OpenMM).
    3. Estimate dU/dlambda by finite difference of the potential energy.
    4. Advance lambda one BAOAB step of the extended-system Langevin dynamics,
       thermostatted at a HIGH temperature T_s with a LARGE fictitious mass:
           m_lambda * d2(lambda)/dt2 = -dU/dlambda - dV_barrier/dlambda
                                       - dV_bias/dlambda
                                       - gamma * m_lambda * d(lambda)/dt + noise(T_s)
    5. Reflect lambda at the [0, 1] boundaries; record trajectory.

This is d-AFED / TAMD applied to the alchemical coordinate (cf. constant-pH
lambda-dynamics, JCTC 2024, 10.1021/acs.jctc.4c00704): a large lambda mass
adiabatically decouples lambda from the atoms, so the mean force on lambda is
the true free-energy gradient at the physical temperature T, while a high
lambda temperature T_s >> T accelerates barrier crossing. A BarrierPotential
centered at lambda=0.5 separates the two physical end states (lambda=0
decoupled, lambda=1 coupled). The physical-temperature free energy is recovered
by reweighting the lambda histogram with the HIGH temperature:

    A(lambda) = -k_B T_s ln P(lambda) - V_barrier(lambda) - V_bias(lambda)

(see reweight_free_energy). The optional AdaptiveBias additionally flattens the
landscape; replace it with ABF/OPES without touching the driver.

This is a *prototype for methodology development*, not a converged estimator.
"""

from dataclasses import dataclass, field

import numpy as np
import openmm
from openmm import unit
from scipy.special import erf

from . import alchemy
from .integrators import make_integrator
from .units import KB


# --------------------------------------------------------------------------- #
# master-lambda -> (elec, steric) maps
# --------------------------------------------------------------------------- #

def master_to_alchemical(master_lambda, mode="combined"):
    """Map master lambda in [0,1] to (lambda_elec, lambda_steric).

    mode="combined": both follow smoothstep(lambda).
    mode="staged":   lambda in [0,0.5] discharges electrostatics (steric=1);
                     lambda in [0.5,1] decouples sterics (elec=0).
    Note: lambda=1 -> fully coupled, lambda=0 -> fully decoupled, matching the
    fixed-window convention.
    """
    lam = float(np.clip(master_lambda, 0.0, 1.0))
    if mode == "combined":
        s = alchemy.smoothstep(lam)
        return s, s
    elif mode == "staged":
        if lam >= 0.5:
            # Upper half: sterics fully on, electrostatics ramping.
            elec = alchemy.smoothstep((lam - 0.5) / 0.5)
            return elec, 1.0
        else:
            # Lower half: electrostatics off, sterics ramping.
            steric = alchemy.smoothstep(lam / 0.5)
            return 0.0, steric
    raise ValueError(f"unknown mode {mode!r}")


def _smoothstep_prime(x):
    """d/dx smoothstep(x) = 6x(1-x) on [0,1], zero outside."""
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return 6.0 * x * (1.0 - x)


def schedule_slopes(master_lambda, mode="combined"):
    """Analytic (d lambda_elec/d lambda, d lambda_steric/d lambda) for the map.

    These are the chain-rule factors that convert the alchemical parameter
    derivatives dU/d(lambda_elec), dU/d(lambda_steric) returned by OpenMM into
    the master-lambda derivative dU/d(lambda).
    """
    lam = float(np.clip(master_lambda, 0.0, 1.0))
    if mode == "combined":
        sp = _smoothstep_prime(lam)
        return sp, sp
    elif mode == "staged":
        if lam >= 0.5:
            # elec = smoothstep((lam-0.5)/0.5); chain rule gives the extra 1/0.5.
            return _smoothstep_prime((lam - 0.5) / 0.5) / 0.5, 0.0
        else:
            return 0.0, _smoothstep_prime(lam / 0.5) / 0.5
    raise ValueError(f"unknown mode {mode!r}")


# --------------------------------------------------------------------------- #
# two-region (relative A->B) master map: one tau drives BOTH region lambdas
# --------------------------------------------------------------------------- #

# Suffixed global-parameter names for the two named alchemical regions.
TWO_REGION_PARAMS = ("lambda_electrostatics_A", "lambda_sterics_A",
                     "lambda_electrostatics_B", "lambda_sterics_B")

# Buffered clamp on the master coordinate when mapping to PHYSICAL lambda: tau is
# reflected in [-0.5, 1.5] but the applied lambda is held within this buffered
# range so softcore stays near its valid endpoints (prevents large-timestep NaN)
# while keeping a real gradient through the histogram basins [0,0.2] and [0.8,1].
LAM_LO, LAM_HI = -0.2, 1.2


def master_to_two_region(tau, mode="staged"):
    """Map a master coordinate tau in [0,1] to the four region lambdas.

    tau = 0 -> region A fully coupled, region B a ghost (decoupled).
    tau = 1 -> region A a ghost, region B fully coupled.
    A's coupling decreases with tau (lambda_A driven by 1-tau), B's increases
    (lambda_B driven by tau); each side is staged (discharge electrostatics
    before decoupling sterics) through master_to_alchemical. Returns the dict
    apply_two_region_lambdas consumes. tau=0->1 is therefore the A->B process,
    so the free energy A(1)-A(0) is ddG_hyd(A->B) = dG_hyd(B) - dG_hyd(A).
    """
    eA, sA = master_to_alchemical(1.0 - tau, mode)
    eB, sB = master_to_alchemical(tau, mode)
    return {"lambda_electrostatics_A": eA, "lambda_sterics_A": sA,
            "lambda_electrostatics_B": eB, "lambda_sterics_B": sB}


def two_region_slopes(tau, mode="staged"):
    """d(region lambda)/d tau for the four globals (chain-rule factors).

    Region A is driven by (1-tau), so its slopes carry a minus sign; region B is
    driven by tau directly. Used to convert OpenMM's per-parameter energy
    derivatives into dU/d(tau).
    """
    seA, ssA = schedule_slopes(1.0 - tau, mode)
    seB, ssB = schedule_slopes(tau, mode)
    return {"lambda_electrostatics_A": -seA, "lambda_sterics_A": -ssA,
            "lambda_electrostatics_B": seB, "lambda_sterics_B": ssB}


def register_alchemical_derivatives(
        system, targets=("lambda_electrostatics", "lambda_sterics")):
    """Enable analytic dU/d(alchemical lambda) on an alchemical System.

    OpenMMTools puts the alchemical lambdas on Custom{Nonbonded,Bond}Forces,
    which support addEnergyParameterDerivative. After this, a Context built from
    the system answers getState(getParameterDerivatives=True) with the exact
    energy derivatives -- no finite differencing of the (huge) total energy.
    Idempotent: skips parameters already registered on a force.

    ``targets`` are the global-parameter names to register; the default is the
    single-region pair, but a two-region (relative) system passes the four
    suffixed names (lambda_electrostatics_A, lambda_sterics_A, ..._B).
    """
    for force in system.getForces():
        if not hasattr(force, "addEnergyParameterDerivative"):
            continue
        if not hasattr(force, "getNumGlobalParameters"):
            continue
        globals_ = {force.getGlobalParameterName(i)
                    for i in range(force.getNumGlobalParameters())}
        already = {force.getEnergyParameterDerivativeName(i)
                   for i in range(force.getNumEnergyParameterDerivatives())}
        for name in targets:
            if name in globals_ and name not in already:
                force.addEnergyParameterDerivative(name)


def analytic_dUdlambda(context, lam, mode):
    """Exact dU/d(master lambda) via OpenMM energy parameter derivatives (kJ/mol).

    Requires register_alchemical_derivatives() to have been called on the system.
    The context's alchemical parameters must already be set for `lam`.
    """
    derivs = context.getState(getParameterDerivatives=True
                              ).getEnergyParameterDerivatives()
    dU_de = derivs["lambda_electrostatics"]   # kJ/mol per unit lambda_elec
    dU_ds = derivs["lambda_sterics"]          # kJ/mol per unit lambda_steric
    se, ss = schedule_slopes(lam, mode)
    return dU_de * se + dU_ds * ss


# --------------------------------------------------------------------------- #
# modular bias
# --------------------------------------------------------------------------- #

class AdaptiveBias:
    """Histogram-based adaptive bias placeholder (WHAM-lite / metadynamics-lite).

    After `burnin` updates, the negative of the running free-energy estimate
    F(lambda) = -kT ln P(lambda) is applied as a bias to flatten sampling.
    The interface (value, gradient, update) is what an ABF/OPES replacement
    would also implement.
    """

    def __init__(self, nbins=25, kT_kJ=2.5, burnin=200, fill=1.0):
        self.nbins = nbins
        self.kT_kJ = kT_kJ
        self.burnin = burnin
        self.fill = fill  # kJ/mol pseudocount strength
        self.edges = np.linspace(0.0, 1.0, nbins + 1)
        self.centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.counts = np.zeros(nbins, dtype=np.float64)
        self.n_updates = 0
        self._bias = np.zeros(nbins, dtype=np.float64)

    def update(self, lam):
        idx = min(int(lam * self.nbins), self.nbins - 1)
        self.counts[idx] += 1.0
        self.n_updates += 1
        if self.n_updates >= self.burnin:
            self._recompute_bias()

    def _recompute_bias(self):
        p = self.counts + self.fill
        p /= p.sum()
        F = -self.kT_kJ * np.log(p)            # free-energy estimate (kJ/mol)
        self._bias = -(F - F.mean())           # bias ~ -F, mean-zero

    def value(self, lam):
        """Bias potential Vbias(lambda) in kJ/mol (linear interpolation)."""
        return float(np.interp(lam, self.centers, self._bias))

    def gradient(self, lam, h=1e-3):
        """dVbias/dlambda via central finite difference (kJ/mol)."""
        a = self.value(min(lam + h, 1.0))
        b = self.value(max(lam - h, 0.0))
        return (a - b) / (min(lam + h, 1.0) - max(lam - h, 0.0))

    def pmf_estimate(self):
        """Return (centers, F_kJ) free-energy estimate from the histogram."""
        p = self.counts + self.fill
        p /= p.sum()
        F = -self.kT_kJ * np.log(p)
        return self.centers, F - F.min()


class NullBias:
    """No-op bias (unbiased extended-system dynamics)."""

    def update(self, lam):
        pass

    def value(self, lam):
        return 0.0

    def gradient(self, lam, h=1e-3):
        return 0.0

    def pmf_estimate(self):
        return np.array([]), np.array([])


# --------------------------------------------------------------------------- #
# barrier potential on lambda (d-AFED / constant-pH style)
# --------------------------------------------------------------------------- #

class BarrierPotential:
    """Double-well + barrier + soft-wall potential on lambda (Eq. 3).

    Implements Eq. 3 of Hong et al., JCTC 2024, 20, 10010 (constant-pH AFED),
    after Aho et al. (GROMACS constant pH, JCTC 2022, 18, 6148) / Donnini:

        V(lam) = -k [ exp(-(lam-1-b)^2 / 2a^2) + exp(-(lam+b)^2 / 2a^2) ]
                 + d   exp(-(lam-0.5)^2 / 2s^2)
                 + 0.5 w [ (1 - erf(r(lam+m))) + (1 + erf(r(lam-1-m))) ]

    - the two negative Gaussians are wells just outside lambda=0 and lambda=1
      that bias lambda toward the physical end states;
    - the positive central Gaussian shapes the barrier top at lambda=0.5;
    - the two complementary error functions are SOFT CONFINING WALLS just
      outside [0, 1] (these replace hard reflection).

    Amplitudes (k, d) set the barrier height; the remaining ``shape`` parameters
    (a, b, s, w, r, m) fix the geometry. Use ``from_height`` to calibrate the
    potential to a requested barrier height by solving Eq. 3 iteratively (the
    exact Aho et al. procedure, SI Eqs. S1-S6). The whole potential is removed
    analytically during reweighting, so it changes only sampling efficiency,
    not the recovered free energy.
    """

    def __init__(self, k=0.0, d=0.0, a=0.05, b=0.0, s=0.30,
                 w=1000.0, r=13.5, m=0.2019):
        self.k = float(k)
        self.d = float(d)
        self.a = float(a)
        self.b = float(b)
        self.s = float(s)
        self.w = float(w)
        self.r = float(r)
        self.m = float(m)

    def value(self, lam):
        """Barrier potential V(lambda) in kJ/mol (scalar or array)."""
        a2 = 2.0 * self.a ** 2
        s2 = 2.0 * self.s ** 2
        wells = (np.exp(-((lam - 1.0 - self.b) ** 2) / a2)
                 + np.exp(-((lam + self.b) ** 2) / a2))
        center = np.exp(-((lam - 0.5) ** 2) / s2)
        walls = 0.5 * self.w * (
            (1.0 - erf(self.r * (lam + self.m)))
            + (1.0 + erf(self.r * (lam - 1.0 - self.m))))
        return -self.k * wells + self.d * center + walls

    def gradient(self, lam):
        """dV/dlambda in kJ/mol (scalar or array)."""
        a2 = self.a ** 2
        s2 = self.s ** 2
        e1 = np.exp(-((lam - 1.0 - self.b) ** 2) / (2.0 * a2))
        e2 = np.exp(-((lam + self.b) ** 2) / (2.0 * a2))
        e3 = np.exp(-((lam - 0.5) ** 2) / (2.0 * s2))
        d_wells = -e1 * (lam - 1.0 - self.b) / a2 - e2 * (lam + self.b) / a2
        d_center = -e3 * (lam - 0.5) / s2
        # d/dlam erf(r x) = r (2/sqrt(pi)) exp(-(r x)^2)
        d_walls = self.w * self.r / np.sqrt(np.pi) * (
            np.exp(-(self.r * (lam - 1.0 - self.m)) ** 2)
            - np.exp(-(self.r * (lam + self.m)) ** 2))
        return -self.k * d_wells + self.d * d_center + d_walls

    def realized_height(self, grid=20001):
        """Numerically realized barrier height: V(top) - V(well minimum), kJ/mol.

        The well minimum is located in the physical basin near lambda=0 and the
        barrier top in the central region; by left-right symmetry the lambda=1
        basin is equivalent. Matches Aho's V(0.5) - Min_lambda(V) definition.
        """
        lam = np.linspace(-0.5, 1.5, grid)
        V = self.value(lam)
        vmin = V[(lam >= -0.3) & (lam <= 0.3)].min()
        vtop = V[(lam >= 0.35) & (lam <= 0.65)].max()
        return float(vtop - vmin)

    @classmethod
    def from_height(cls, height, sigma0=0.02, eps=1e-5, max_iter=100000,
                    grid_n=20001, s=0.30, w=1000.0, r=13.5, m=0.2019,
                    min_barrier=0.45):
        """Calibrate Eq. 3 to a desired barrier ``height`` (kJ/mol), iteratively.

        Exact procedure of Aho et al. (GROMACS constant pH, JCTC 2022, 18, 6148),
        SI Eqs. S1-S6 (= Eq. 3 of Hong et al. JCTC 2024): the central amplitude
        is fixed at d = height/2 and the shape parameters s, w, r, m are fixed;
        the well depth k, position b and width a are refined until the left-well
        Boltzmann-weighted mean position x0 -> 0 (Eqs. S3-S4) and its dispersion
        sigma -> sigma0 = 0.02 (Eqs. S5-S6), with k set each step so the total
        barrier equals `height` (Eq. S2). Reproduces the published Table S1
        parameters. Barriers below `min_barrier` kJ/mol are forced to zero
        (walls only). Returns a configured BarrierPotential.
        """
        if height < min_barrier:
            return cls(k=0.0, d=0.0, a=0.05, b=0.0, s=s, w=w, r=r, m=m)

        d = 0.5 * height
        k, a, b = 0.5 * height, 0.05, -0.1
        lam = np.linspace(-0.5, 1.5, grid_n)
        left = lam < 0.5
        # central Gaussian and walls are fixed during the iteration; precompute.
        fixed = (d * np.exp(-((lam - 0.5) ** 2) / (2.0 * s * s))
                 + 0.5 * w * ((1.0 - erf(r * (lam + m)))
                              + (1.0 + erf(r * (lam - 1.0 - m)))))

        def potential(k, a, b):
            wells = (np.exp(-((lam - 1.0 - b) ** 2) / (2.0 * a * a))
                     + np.exp(-((lam + b) ** 2) / (2.0 * a * a)))
            return -k * wells + fixed

        def left_basin_stats(V):
            mask = left & (V < 0.0)
            wgt = np.exp(-V[mask])
            sw = wgt.sum()
            x0 = float((lam[mask] * wgt).sum() / sw)
            sig = float(np.sqrt(((lam[mask] - x0) ** 2 * wgt).sum() / sw))
            return x0, sig

        x0 = 1.0
        sig = sigma0
        for _ in range(max_iter):
            k = k + (0.5 * height + potential(k, a, b).min())   # S2
            x0, _ = left_basin_stats(potential(k, a, b))        # S4
            b = b + 0.01 * x0                                   # S3
            _, sig = left_basin_stats(potential(k, a, b))       # S6
            a = a / (1.0 + 0.01 * (sig - sigma0) / sigma0)      # S5
            if abs(x0) < eps and abs((sig - sigma0) / sigma0) < eps:
                break
        return cls(k=k, d=d, a=a, b=b, s=s, w=w, r=r, m=m)


# --------------------------------------------------------------------------- #
# the dynamic-lambda driver
# --------------------------------------------------------------------------- #

@dataclass
class DynamicLambdaResult:
    times_ps: np.ndarray
    lambdas: np.ndarray
    dUdlambda: np.ndarray
    energies_kJ: np.ndarray
    bias_kJ: np.ndarray
    barrier_kJ: np.ndarray = None
    lambda_temperature: float = None
    bias: object = None
    barrier: object = None
    config: dict = field(default_factory=dict)


def _reflect(x, v, lo=0.0, hi=1.0):
    """Reflect coordinate x with velocity v into [lo, hi] (elastic walls).

    Confinement is normally handled by the soft wall terms of BarrierPotential
    (Eq. 3); this is only a wide hard safety net (default [0, 1]) so a runaway
    timestep cannot send lambda to infinity. Call with generous bounds.
    """
    span = hi - lo
    while x < lo or x > hi:
        if x < lo:
            x = 2.0 * lo - x
            v = -v
        elif x > hi:
            x = 2.0 * hi - x
            v = -v
        if span <= 0.0:
            break
    return x, v


def estimate_dUdlambda(context, alchemical_state, lam, mode, h=1e-3):
    """Finite-difference estimate of dU/dlambda at fixed coordinates (kJ/mol).

    U is evaluated at lambda+h and lambda-h (mapped through the schedule) on the
    *current* configuration; the configuration is not propagated here.
    """
    lp = min(lam + h, 1.0)
    lm = max(lam - h, 0.0)

    le, ls = master_to_alchemical(lp, mode)
    alchemy.apply_lambdas(context, alchemical_state, le, ls)
    up = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole)

    le, ls = master_to_alchemical(lm, mode)
    alchemy.apply_lambdas(context, alchemical_state, le, ls)
    um = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole)

    return (up - um) / (lp - lm)


def run_dynamic_lambda(
    solvated,
    alchemical_system,
    alchemical_state,
    platform,
    properties,
    temperature=298.15,
    lambda_temperature=1000.0,  # T_s: HIGH auxiliary temperature for lambda (K)
    timestep_fs=2.0,
    friction_per_ps=1.0,
    md_steps_per_block=50,
    n_blocks=2000,
    lambda_mass=200.0,       # LARGE fictitious mass for adiabatic separation
    lambda_friction=5.0,     # 1/ps
    lambda_dt_ps=None,       # defaults to md_steps_per_block * timestep
    mode="combined",
    bias=None,
    barrier=None,
    lambda0=0.5,
    minimize_first=True,
    seed=None,
    progress=True,
    use_analytic_dudl=True,
    integrator_type="langevin",
    mts_inner_steps=None,
):
    """Run the toy dynamic-lambda simulation (d-AFED / TAMD on lambda).

    lambda is thermostatted at the HIGH temperature ``lambda_temperature``
    (T_s) with a LARGE ``lambda_mass`` for adiabatic separation; ``barrier`` is
    a BarrierPotential separating the end states. Recover the free energy with
    reweight_free_energy (which removes the barrier and uses T_s).

    Returns a DynamicLambdaResult with the lambda trajectory and diagnostics.
    """
    rng = np.random.default_rng(seed)
    kT_kJ = (KB * (temperature * unit.kelvin)).value_in_unit(unit.kilojoule_per_mole)
    # lambda thermostat runs at the HIGH temperature T_s (this is the d-AFED knob).
    kT_s_kJ = (KB * (lambda_temperature * unit.kelvin)).value_in_unit(
        unit.kilojoule_per_mole)

    if bias is None:
        bias = NullBias()
    if barrier is None:
        # walls-only barrier: soft confinement, no inter-state barrier.
        barrier = BarrierPotential()
    if lambda_dt_ps is None:
        lambda_dt_ps = md_steps_per_block * timestep_fs * 1e-3  # ps

    # Physical-system integrator/context (thermostats the atoms).
    if use_analytic_dudl:
        # Enable exact dU/d(lambda_elec/steric) before building the Context.
        register_alchemical_derivatives(alchemical_system)

    # Plain Langevin, or RESPA/MTS (nonbonded once per outer step, bonded on a
    # ~1 fs inner step) so HMR systems can run at 3-4 fs. make_integrator assigns
    # the force groups on alchemical_system, so it must precede Context creation.
    integrator = make_integrator(
        alchemical_system, temperature, friction_per_ps, timestep_fs,
        kind=integrator_type, mts_inner_steps=mts_inner_steps,
    )
    context = openmm.Context(alchemical_system, integrator, platform, properties)
    if seed is not None:
        integrator.setRandomNumberSeed(int(seed))
    context.setPositions(solvated.positions)

    if minimize_first:
        le, ls = master_to_alchemical(lambda0, mode)
        alchemy.apply_lambdas(context, alchemical_state, le, ls)
        openmm.LocalEnergyMinimizer.minimize(context, maxIterations=200)
    # Seed the initial velocities too, so a run is fully reproducible from `seed`.
    if seed is not None:
        context.setVelocitiesToTemperature(temperature * unit.kelvin, int(seed))
    else:
        context.setVelocitiesToTemperature(temperature * unit.kelvin)

    # Extended-system (lambda) state.
    lam = float(lambda0)
    # Sample initial lambda velocity from Maxwell-Boltzmann at T_s for lambda_mass.
    vlam = rng.normal(0.0, np.sqrt(kT_s_kJ / lambda_mass))
    dt = lambda_dt_ps
    gamma = lambda_friction
    c1 = np.exp(-gamma * dt)
    # O-step noise amplitude uses the HIGH temperature T_s (lambda thermostat).
    c2 = np.sqrt((1.0 - c1 * c1) * kT_s_kJ / lambda_mass)

    times = np.empty(n_blocks)
    lam_traj = np.empty(n_blocks)
    dudl_traj = np.empty(n_blocks)
    energy_traj = np.empty(n_blocks)
    bias_traj = np.empty(n_blocks)
    barrier_traj = np.empty(n_blocks)

    for b in range(n_blocks):
        # (1) set context from lambda, (2) propagate atoms.
        le, ls = master_to_alchemical(lam, mode)
        alchemy.apply_lambdas(context, alchemical_state, le, ls)
        integrator.step(md_steps_per_block)

        # (3) forces on lambda at the new configuration.
        if use_analytic_dudl:
            # Exact dU/dlambda from OpenMM parameter derivatives (no finite
            # differencing of the total energy; immune to single-precision
            # cancellation that loses the electrostatic gradient).
            state = context.getState(getEnergy=True, getParameterDerivatives=True)
            energy = state.getPotentialEnergy().value_in_unit(
                unit.kilojoule_per_mole)
            derivs = state.getEnergyParameterDerivatives()
            se, ss = schedule_slopes(lam, mode)
            dUdl = (derivs["lambda_electrostatics"] * se
                    + derivs["lambda_sterics"] * ss)
        else:
            state = context.getState(getEnergy=True)
            energy = state.getPotentialEnergy().value_in_unit(
                unit.kilojoule_per_mole)
            dUdl = estimate_dUdlambda(context, alchemical_state, lam, mode)
            # restore the actual lambda mapping after the finite-difference probe
            alchemy.apply_lambdas(context, alchemical_state, le, ls)

        f_lambda = -dUdl - barrier.gradient(lam) - bias.gradient(lam)

        # (4) BAOAB step for the 1-D lambda coordinate.
        vlam += 0.5 * dt * f_lambda / lambda_mass            # B
        lam += 0.5 * dt * vlam                               # A
        vlam = c1 * vlam + c2 * rng.standard_normal()        # O
        lam += 0.5 * dt * vlam                               # A
        # recompute force at new lambda is skipped (forces updated next block);
        # apply the second half-kick with the same f_lambda for stability.
        vlam += 0.5 * dt * f_lambda / lambda_mass            # B

        # (5) confinement is by the BarrierPotential soft walls (Eq. 3); only a
        #     wide hard reflection guards against a runaway step (numerics).
        lam, vlam = _reflect(lam, vlam, lo=-0.5, hi=1.5)

        bias.update(lam)

        times[b] = (b + 1) * dt
        lam_traj[b] = lam
        dudl_traj[b] = dUdl
        energy_traj[b] = energy
        bias_traj[b] = bias.value(lam)
        barrier_traj[b] = barrier.value(lam)

        if progress and (b + 1) % max(1, n_blocks // 10) == 0:
            print(f"  block {b+1}/{n_blocks}  lambda={lam:.3f}")

    del context, integrator
    return DynamicLambdaResult(
        times_ps=times,
        lambdas=lam_traj,
        dUdlambda=dudl_traj,
        energies_kJ=energy_traj,
        bias_kJ=bias_traj,
        barrier_kJ=barrier_traj,
        lambda_temperature=lambda_temperature,
        bias=bias,
        barrier=barrier,
        config=dict(
            temperature=temperature, lambda_temperature=lambda_temperature,
            mode=mode, lambda_mass=lambda_mass,
            lambda_friction=lambda_friction, n_blocks=n_blocks,
            md_steps_per_block=md_steps_per_block,
        ),
    )


# --------------------------------------------------------------------------- #
# the two-region (relative A->B) dynamic-lambda driver
# --------------------------------------------------------------------------- #

def _apply_two_region(context, params):
    """Set the four suffixed region-lambda globals on a Context."""
    for name, val in params.items():
        context.setParameter(name, float(val))


def _morph_dudtau(context, params_fn, tau, morph_params, h=1e-3):
    """dU/dtau contribution of the core charge-morph globals, by finite difference.

    The morph globals drive a NonbondedForce charge offset; OpenMM reads their
    energy-parameter derivative back as 0 on the available platforms (the analytic
    derivative through a parameter offset is not implemented), so use a cheap
    central difference. Each morph global = 1 - tau, so d/dtau = -dU/d(morph).
    Restores the context to the tau state before returning.
    """
    base = params_fn(tau)
    up, dn = dict(base), dict(base)
    for mp in morph_params:
        up[mp] = base[mp] + h
        dn[mp] = base[mp] - h
    _apply_two_region(context, up)
    e_up = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole)
    _apply_two_region(context, dn)
    e_dn = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole)
    _apply_two_region(context, base)            # restore tau state
    return -(e_up - e_dn) / (2.0 * h)           # d(morph)/dtau = -1


def run_dynamic_lambda_two_region(
    setup,
    platform,
    properties,
    temperature=298.15,
    lambda_temperature=2000.0,   # T_s default (validated for the two-region/hybrid path)
    timestep_fs=2.0,
    friction_per_ps=1.0,
    md_steps_per_block=50,
    n_blocks=2000,
    lambda_mass=200.0,
    lambda_friction=5.0,
    lambda_dt_ps=None,
    mode="staged",
    bias=None,
    barrier=None,
    tau0=0.5,
    minimize_first=True,
    seed=None,
    progress=True,
    integrator_type="langevin",
    mts_inner_steps=None,
    integrator_factory=None,
):
    """d-AFED / TAMD on a master tau that drives a two-region (A->B) system.

    ``integrator_factory``: optional callable(system) -> (integrator, init_fn) to
    use a custom MD integrator for the atoms (e.g. the regulated/SIN(R) large-step
    integrator). init_fn(context, integrator) is invoked once after velocities are
    set (used to initialize the regulated momenta). When None, the built-in
    make_integrator (langevin/mts) is used.

    The dual-topology ``setup`` (relative_setup.TwoLigandSetup) carries the four
    suffixed region-lambda globals. A single master tau in [0,1] drives both
    regions (master_to_two_region); tau is propagated as an extended-system
    Langevin coordinate at the HIGH temperature ``lambda_temperature`` with a
    LARGE ``lambda_mass`` (adiabatic separation), exactly as run_dynamic_lambda
    does for the absolute case. dU/d(tau) is the exact analytic chain-rule sum of
    the four registered parameter derivatives. Recover ddG_hyd(A->B) with
    reweight_free_energy on the tau trajectory: A(1) - A(0).

    Returns a DynamicLambdaResult (``lambdas`` is the tau trajectory).
    """
    rng = np.random.default_rng(seed)
    kT_s_kJ = (KB * (lambda_temperature * unit.kelvin)).value_in_unit(
        unit.kilojoule_per_mole)

    if bias is None:
        bias = NullBias()
    if barrier is None:
        barrier = BarrierPotential()
    if lambda_dt_ps is None:
        lambda_dt_ps = md_steps_per_block * timestep_fs * 1e-3  # ps

    # Core charge-morph globals (single-topology only; empty for dual topology).
    # Each is driven 1->0 as tau 0->1 alongside the region lambdas (constant
    # chain-rule slope d/dtau = -1 in the analytic dU/dtau).
    morph_params = sorted(p for p in getattr(setup, "lambda_parameters", [])
                          if p.startswith("lambda_core"))

    def _params(t):
        # Clamp the master coordinate to a buffered [LAM_LO, LAM_HI] for the
        # PHYSICAL lambda mapping. tau itself is reflected in [-0.5, 1.5] (the
        # basin tails), but applying the full extrapolated lambda drives the
        # softcore potential far past its endpoints; on a large explicit-solvent
        # system a single such step at a big (SIN(R)) timestep produces NaN. A
        # small buffer (+/-0.2) past [0,1] keeps a proper lambda-gradient through
        # the histogram basins ([0,0.2] and [0.8,1]) -- so they are not pinned
        # flat at the endpoints -- while cutting off the extreme excursions that
        # blow up. In-range dynamics and the validated small-system results are
        # unaffected.
        tc = LAM_LO if t < LAM_LO else (LAM_HI if t > LAM_HI else t)
        p = master_to_two_region(tc, mode)
        for mp in morph_params:
            p[mp] = 1.0 - tc
        return p

    # Exact dU/d(region lambda) for the four suffixed globals (analytic). Core-morph
    # globals are handled separately by finite difference (_morph_dudtau) since
    # NonbondedForce offset-parameter derivatives read back as 0.
    register_alchemical_derivatives(setup.system, targets=TWO_REGION_PARAMS)

    init_fn = None
    if integrator_factory is not None:
        integrator, init_fn = integrator_factory(setup.system)
    else:
        integrator = make_integrator(
            setup.system, temperature, friction_per_ps, timestep_fs,
            kind=integrator_type, mts_inner_steps=mts_inner_steps,
        )
    context = openmm.Context(setup.system, integrator, platform, properties)
    if seed is not None:
        try:
            integrator.setRandomNumberSeed(int(seed))
        except Exception:
            pass
    context.setPositions(setup.positions)

    if minimize_first:
        _apply_two_region(context, _params(tau0))
        openmm.LocalEnergyMinimizer.minimize(context, maxIterations=200)
    if seed is not None:
        context.setVelocitiesToTemperature(temperature * unit.kelvin, int(seed))
    else:
        context.setVelocitiesToTemperature(temperature * unit.kelvin)
    if init_fn is not None:
        init_fn(context, integrator)

    tau = float(tau0)
    vtau = rng.normal(0.0, np.sqrt(kT_s_kJ / lambda_mass))
    dt = lambda_dt_ps
    gamma = lambda_friction
    c1 = np.exp(-gamma * dt)
    c2 = np.sqrt((1.0 - c1 * c1) * kT_s_kJ / lambda_mass)

    times = np.empty(n_blocks)
    tau_traj = np.empty(n_blocks)
    dudl_traj = np.empty(n_blocks)
    energy_traj = np.empty(n_blocks)
    bias_traj = np.empty(n_blocks)
    barrier_traj = np.empty(n_blocks)

    for b in range(n_blocks):
        _apply_two_region(context, _params(tau))
        integrator.step(md_steps_per_block)

        state = context.getState(getEnergy=True, getParameterDerivatives=True)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        derivs = state.getEnergyParameterDerivatives()
        slopes = two_region_slopes(tau, mode)
        dUdl = sum(derivs[name] * slope for name, slope in slopes.items())
        if morph_params:
            dUdl += _morph_dudtau(context, _params, tau, morph_params)

        f_tau = -dUdl - barrier.gradient(tau) - bias.gradient(tau)

        vtau += 0.5 * dt * f_tau / lambda_mass            # B
        tau += 0.5 * dt * vtau                            # A
        vtau = c1 * vtau + c2 * rng.standard_normal()     # O
        tau += 0.5 * dt * vtau                            # A
        vtau += 0.5 * dt * f_tau / lambda_mass            # B

        tau, vtau = _reflect(tau, vtau, lo=-0.5, hi=1.5)
        bias.update(tau)

        times[b] = (b + 1) * dt
        tau_traj[b] = tau
        dudl_traj[b] = dUdl
        energy_traj[b] = energy
        bias_traj[b] = bias.value(tau)
        barrier_traj[b] = barrier.value(tau)

        if progress and (b + 1) % max(1, n_blocks // 10) == 0:
            print(f"  block {b+1}/{n_blocks}  tau={tau:.3f}")

    del context, integrator
    return DynamicLambdaResult(
        times_ps=times,
        lambdas=tau_traj,
        dUdlambda=dudl_traj,
        energies_kJ=energy_traj,
        bias_kJ=bias_traj,
        barrier_kJ=barrier_traj,
        lambda_temperature=lambda_temperature,
        bias=bias,
        barrier=barrier,
        config=dict(
            temperature=temperature, lambda_temperature=lambda_temperature,
            mode=mode, lambda_mass=lambda_mass, lambda_friction=lambda_friction,
            n_blocks=n_blocks, md_steps_per_block=md_steps_per_block,
            engine="two_region",
        ),
    )


# --------------------------------------------------------------------------- #
# d-AFED reweighting: recover the physical free energy along lambda
# --------------------------------------------------------------------------- #

def reweight_free_energy(lambdas, lambda_temperature, nbins=25,
                         barrier=None, bias=None, fill=1.0, drop_frac=0.0):
    """Recover the physical-temperature free energy A(lambda) by reweighting.

    Because lambda is thermostatted at the HIGH temperature T_s (with a large
    mass so its mean force is the free-energy gradient at the physical T), its
    marginal distribution is

        P(lambda) ~ exp( -(A(lambda) + V_barrier(lambda) + V_bias(lambda)) / kT_s ).

    Inverting and removing the applied barrier/bias gives the d-AFED estimate

        A(lambda) = -kT_s ln P(lambda) - V_barrier(lambda) - V_bias(lambda),

    shifted so min(A) = 0. ``lambda_temperature`` (T_s, kelvin) is the SAME high
    temperature used for the lambda thermostat in run_dynamic_lambda.

    Returns (centers, A_kJ).
    """
    kT_s_kJ = (KB * (lambda_temperature * unit.kelvin)).value_in_unit(
        unit.kilojoule_per_mole)
    lam = np.asarray(lambdas, dtype=float)
    if drop_frac > 0.0:
        lam = lam[int(drop_frac * lam.size):]
    # lambda may sit just outside [0,1] in the soft-wall region; fold the
    # near-endpoint population into the physical end states for the histogram.
    lam = np.clip(lam, 0.0, 1.0)

    edges = np.linspace(0.0, 1.0, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts, _ = np.histogram(lam, bins=edges)

    p = counts.astype(np.float64) + fill   # pseudocount avoids log(0) in empty bins
    p /= p.sum()
    A = -kT_s_kJ * np.log(p)               # -kT_s ln P(lambda)

    if barrier is not None:
        A -= np.array([barrier.value(c) for c in centers])
    if bias is not None:
        A -= np.array([bias.value(c) for c in centers])

    A -= A.min()
    return centers, A


def ti_free_energy(lambdas, dUdlambda, nbins=25):
    """Thermodynamic-integration free energy from the d-AFED mean force.

        dG(0->1) = integral_0^1 <dU/dtau>_tau dtau

    Bins the recorded (tau, dU/dtau) samples, averages the gradient in each bin to
    get the mean force, and integrates (trapezoid). Unlike the histogram estimator
    this has NO -kT_s ln P term, so it neither blows up on sparsely sampled bins nor
    depends on T_s (the mean force at a given tau is T_s-independent). Empty bins
    (tau never visited) ARE interpolated across from the visited ones -- TI cannot
    invent data where tau never went, so n_empty is returned as a health flag.

    Returns (centers, mean_force_kJ, dG_kJ, n_empty).
    """
    lam = np.clip(np.asarray(lambdas, dtype=float), 0.0, 1.0)
    g = np.asarray(dUdlambda, dtype=float)
    edges = np.linspace(0.0, 1.0, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(lam, edges) - 1, 0, nbins - 1)
    mean_force = np.full(nbins, np.nan)
    for i in range(nbins):
        m = idx == i
        if m.any():
            mean_force[i] = g[m].mean()
    visited = ~np.isnan(mean_force)
    n_empty = int((~visited).sum())
    mf = mean_force.copy()
    if n_empty and visited.any():
        mf[~visited] = np.interp(centers[~visited], centers[visited],
                                 mean_force[visited])
    dG = float(np.trapz(mf, centers))
    return centers, mean_force, dG, n_empty


def basin_delta_f(centers, A_kJ, temperature, edge=0.2):
    """End-state free-energy difference from a PMF, by Boltzmann-summing each basin.

    Defines the lambda=0 end state as the whole tau window [0, edge] and lambda=1 as
    [1 - edge, 1] (default edge=0.2 -> [0,0.2] and [0.8,1]), and collapses each window
    to a single free energy by summing the Boltzmann weights of its bins at the
    PHYSICAL temperature (the reweighted A(tau) is the physical PMF):

        A(state) = -kT * ln( sum_{bins in window} exp(-A_bin / kT) )
        ddG      = A(lambda=1) - A(lambda=0)

    This is more robust than reading the single outermost bin: it averages over the
    basin and is dominated by its well rather than one (often sparsely sampled)
    boundary bin. Returns ddG in kJ/mol.
    """
    kT = (KB * (temperature * unit.kelvin)).value_in_unit(unit.kilojoule_per_mole)
    centers = np.asarray(centers, dtype=float)
    A = np.asarray(A_kJ, dtype=float)
    lo = centers <= edge + 1e-9
    hi = centers >= 1.0 - edge - 1e-9
    if not lo.any() or not hi.any():
        raise ValueError(f"no bins in a lambda basin for edge={edge}")
    A0 = -kT * np.log(np.sum(np.exp(-A[lo] / kT)))
    A1 = -kT * np.log(np.sum(np.exp(-A[hi] / kT)))
    return A1 - A0
