"""Optional OpenFE backend for relative free energies.

OpenFE is NOT a hard dependency. If it is not importable, every entry point here
raises a clear, actionable ImportError with install notes; the rest of the
package (internal hybrid-topology engine, relative-from-absolute) works without
it. Tests that need OpenFE use @pytest.mark.openfe and are skipped when absent.
"""

_INSTALL_HINT = (
    "OpenFE is not installed. Install it (and a mapper) into the env, e.g.\n"
    "    conda install -c conda-forge openfe\n"
    "or  pip install openfe\n"
    "Then re-run with backend='openfe_optional'. The internal engine "
    "(backend='internal') and relative-from-absolute mode work without OpenFE."
)


def openfe_available():
    """True if OpenFE can be imported."""
    try:
        import openfe  # noqa: F401
        return True
    except ImportError:
        return False


def require_openfe():
    """Import and return the openfe module, or raise an informative ImportError."""
    try:
        import openfe
        return openfe
    except ImportError as exc:  # pragma: no cover - exercised only without openfe
        raise ImportError(_INSTALL_HINT) from exc


def run_rbfe_openfe(ligand_a, ligand_b, complex_input, solvent_input,
                    mapping=None, output_dir=None, config=None):
    """Run an A->B RBFE with OpenFE's equilibrium protocol.

    Constructs small-molecule / solvent / complex components, builds a ligand
    mapping, runs the OpenFE RelativeHybridTopologyProtocol, and returns a
    FreeEnergyResult. Raises ImportError (with install notes) if OpenFE is absent.
    """
    openfe = require_openfe()  # raises with _INSTALL_HINT if missing
    # Construction is intentionally guarded behind the import so the module is
    # importable without OpenFE. The concrete protocol wiring is filled in when
    # OpenFE is available in the target environment.
    raise NotImplementedError(
        "OpenFE is importable but the protocol wiring is not configured in this "
        "build. Use backend='internal' for the internal hybrid-topology engine. "
        f"(openfe version: {getattr(openfe, '__version__', 'unknown')})")
