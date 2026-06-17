"""OpenMM platform selection utilities.

On Apple Silicon macOS there is no CUDA. The fastest generally available
platform is OpenCL (which maps onto the Metal/GPU stack); we fall back to CPU
when OpenCL is unavailable or fails to initialize.
"""

import openmm


def print_platforms():
    """Print every OpenMM platform the current build exposes, with speed."""
    n = openmm.Platform.getNumPlatforms()
    print(f"OpenMM {openmm.__version__}: {n} platform(s) available")
    for i in range(n):
        p = openmm.Platform.getPlatform(i)
        print(f"  [{i}] {p.getName():8s} speed={p.getSpeed():.1f}")


def get_platform_by_name(name):
    """Return the OpenMM Platform with the given name (e.g. 'OpenCL', 'CPU')."""
    return openmm.Platform.getPlatformByName(name)


def _platform_names():
    return {
        openmm.Platform.getPlatform(i).getName()
        for i in range(openmm.Platform.getNumPlatforms())
    }


def get_fastest_platform(prefer_gpu=True, verbose=True):
    """Select the best available platform.

    Preference order on macOS Apple Silicon:
        OpenCL  (GPU via Metal)  ->  CPU
    CUDA is intentionally never selected here.

    Returns (platform, properties_dict).
    """
    available = _platform_names()

    if prefer_gpu and "OpenCL" in available:
        name = "OpenCL"
        # Single precision is the right default for consumer GPUs.
        properties = {"Precision": "single"}
    elif "CPU" in available:
        name = "CPU"
        properties = {}
    else:
        # Last resort: whatever the build offers, fastest first.
        plats = sorted(
            (openmm.Platform.getPlatform(i)
             for i in range(openmm.Platform.getNumPlatforms())),
            key=lambda p: p.getSpeed(),
            reverse=True,
        )
        name = plats[0].getName()
        properties = {}

    platform = openmm.Platform.getPlatformByName(name)
    if verbose:
        print(f"Selected platform: {name} (speed={platform.getSpeed():.1f})")
        if properties:
            print(f"  properties: {properties}")
    return platform, properties


if __name__ == "__main__":
    print_platforms()
    get_fastest_platform()
