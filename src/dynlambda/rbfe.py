"""Relative binding free energy (RBFE) driver and pluggable leg runners.

Thermodynamic cycle (sign convention in free_energy.SIGN_RELATIVE_BINDING):
    ddG_bind(A->B) = dG_complex(A->B) - dG_solvent(A->B)

The two legs are each an alchemical A->B transformation -- one in the bound
complex, one in water. They are run by a FreeEnergyLegRunner so the same cycle
works with the internal fixed-window engine now and a dynamic-lambda engine
later (Stage 9). No adaptive bias is used anywhere; the dynamic-lambda leg, when
implemented, uses the existing AFED histogram-reweighting estimator.
"""

from abc import ABC, abstractmethod

from .free_energy import FreeEnergyResult, relative_binding_from_legs


class FreeEnergyLegRunner(ABC):
    """Runs one alchemical leg and returns a FreeEnergyResult.

    A 'leg' is a single environment (complex or solvent) transformed A->B (or a
    solute decoupled, for absolute hydration). Implementations differ only in the
    sampler; the leg setup and analysis contract are shared.
    """

    @abstractmethod
    def run_leg(self, system_setup, lambda_schedule, output_dir):
        """Return a FreeEnergyResult for this leg."""
        raise NotImplementedError


class FixedWindowLegRunner(FreeEnergyLegRunner):
    """Fixed-lambda windows + MBAR (the validated estimator in this repo)."""

    def __init__(self, temperature=298.15, equil_steps=2500, prod_steps=10000,
                 sample_interval=500, leg_type="complex", method="MBAR"):
        self.temperature = temperature
        self.equil_steps = equil_steps
        self.prod_steps = prod_steps
        self.sample_interval = sample_interval
        self.leg_type = leg_type
        self.method = method

    def run_leg(self, system_setup, lambda_schedule, output_dir):
        # Delegates to the internal hybrid-topology fixed-window sampler.
        from .relative_setup import run_fixed_window_leg
        return run_fixed_window_leg(
            system_setup, lambda_schedule, output_dir,
            temperature=self.temperature,
            equil_steps=self.equil_steps,
            prod_steps=self.prod_steps,
            sample_interval=self.sample_interval,
            leg_type=self.leg_type,
        )


class DynamicLambdaLegRunner(FreeEnergyLegRunner):
    """Placeholder for the AFED-style dynamic-lambda leg (Stage 9).

    NOT yet implemented for relative legs. When implemented it reuses the
    existing AFED machinery in dynamic_lambda.py -- *not* an adaptive bias:
      * lambda is an extended dynamical variable thermostatted at T_s with a
        large mass (adiabatic separation);
      * dU/dlambda comes from analytic energy parameter derivatives
        (register_alchemical_derivatives / schedule_slopes);
      * the free energy is recovered by HISTOGRAM REWEIGHTING of P(lambda)
        (reweight_free_energy): A(lambda) = -kT_s ln P(lambda) - V_barrier.
    """

    def run_leg(self, system_setup, lambda_schedule, output_dir):
        raise NotImplementedError(
            "DynamicLambdaLegRunner is a placeholder. To implement a relative "
            "dynamic-lambda leg, reuse dynamic_lambda.run_dynamic_lambda on the "
            "hybrid-topology system: drive a single master lambda as an extended "
            "variable (large mass, T_s thermostat), take dU/dlambda from the "
            "analytic parameter derivatives, and recover dG by histogram "
            "reweighting (reweight_free_energy). No adaptive bias.")


def combine_rbfe(complex_result, solvent_result, name=None):
    """ddG_bind = dG_complex - dG_solvent (delegates to the cycle math)."""
    return relative_binding_from_legs(complex_result, solvent_result, name=name)


def run_rbfe(ligand_a, ligand_b, complex_input, solvent_input, mapping=None,
             backend="internal", lambda_schedule=None, output_dir=None,
             config=None):
    """Run both legs of an A->B RBFE and combine to ddG_bind.

    backend:
      "internal"        -> FixedWindowLegRunner (this repo's engine)
      "openfe_optional" -> OpenFE if importable, else fall back to "internal"
      a FreeEnergyLegRunner instance -> used directly (handy for tests)

    Returns the rbfe FreeEnergyResult. The per-leg results are attached in
    metadata["complex_leg"] / ["solvent_leg"].
    """
    import os
    config = config or {}

    runner = _resolve_backend(backend, config)

    if isinstance(runner, str) and runner == "openfe":
        from .openfe_backend import run_rbfe_openfe
        return run_rbfe_openfe(ligand_a, ligand_b, complex_input, solvent_input,
                               mapping=mapping, output_dir=output_dir,
                               config=config)

    odir = output_dir or "."
    complex_leg = runner.run_leg(
        complex_input, lambda_schedule, os.path.join(odir, "complex"))
    solvent_leg = runner.run_leg(
        solvent_input, lambda_schedule, os.path.join(odir, "solvent"))

    result = combine_rbfe(complex_leg, solvent_leg,
                          name=f"{ligand_a}->{ligand_b}")
    result.metadata["complex_leg"] = complex_leg.as_dict()
    result.metadata["solvent_leg"] = solvent_leg.as_dict()
    result.metadata["backend"] = "internal"
    return result


def _resolve_backend(backend, config):
    if isinstance(backend, FreeEnergyLegRunner):
        return backend
    if backend == "internal":
        return FixedWindowLegRunner(
            leg_type="complex",
            equil_steps=config.get("equil_steps", 2500),
            prod_steps=config.get("prod_steps", 10000),
            sample_interval=config.get("sample_interval", 500),
            temperature=config.get("temperature", 298.15))
    if backend == "openfe_optional":
        try:
            import openfe  # noqa: F401
            return "openfe"
        except ImportError:
            return FixedWindowLegRunner(
                leg_type="complex",
                equil_steps=config.get("equil_steps", 2500),
                prod_steps=config.get("prod_steps", 10000),
                sample_interval=config.get("sample_interval", 500),
                temperature=config.get("temperature", 298.15))
    raise ValueError(f"unknown backend {backend!r}")
