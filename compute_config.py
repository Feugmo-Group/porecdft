"""Global compute-backend configuration for porecdft.

A single :class:`ComputeConfig` instance controls every numerical backend in
the package — Warp GPU kernels (Vext), JAX/XLA device (FMT / aWBII / solver),
and the NumPy accumulation dtype.  Load it once at the top of every run script
from a Hydra / OmegaConf config file and pass it down to every compute function.

Typical use
-----------
In a Hydra application::

    # conf/compute.yaml
    use_warp:   false
    warp_device: cpu
    jax_device:  cpu
    dtype:       float64

    # run.py
    @hydra.main(config_path="conf", config_name="config")
    def main(cfg):
        from porecdft.compute_config import ComputeConfig
        compute = ComputeConfig.from_omegaconf(cfg.compute)
        compute.apply_jax_device()   # ← one call wires up JAX globally
        ...
        vext = build_vext_on_grid(..., compute=compute)
        result = jax_solver.run(..., compute=compute)

CLI override to switch to GPU for one run::

    python run.py compute.use_warp=true compute.warp_device=cuda:0 \\
                  compute.jax_device=gpu  compute.dtype=float32
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ComputeConfig:
    """Unified backend settings propagated to all compute functions.

    Attributes
    ----------
    use_warp : bool
        Enable NVIDIA Warp kernels for Vext (LJ, Coulomb, Morse) and the
        Boltzmann orientation average.  Silently ignored if ``warp-lang`` is
        not installed.
    warp_device : str
        Warp device string passed to every ``wp.launch`` call, e.g. ``"cpu"``
        or ``"cuda:0"``.  Must match a device returned by
        ``warp.get_devices()``.
    jax_device : str
        JAX platform name for :func:`apply_jax_device`.  Accepted values:
        ``"cpu"``, ``"gpu"``, ``"tpu"``.  Controls which device JAX places new
        arrays on — including the FMT / aWBII FFTs and the Anderson / FIRE2
        solver arrays.
    dtype : str
        Floating-point precision for NumPy accumulation arrays and Warp output
        casts.  Either ``"float32"`` or ``"float64"`` (default).  JAX compute
        is unaffected (JAX uses float32 unless ``jax_enable_x64=True``).
    """

    use_warp:    bool = False
    warp_device: str  = "cpu"
    jax_device:  str  = "cpu"
    dtype:       str  = "float64"

    # ------------------------------------------------------------------ #
    # Convenience properties                                               #
    # ------------------------------------------------------------------ #

    @property
    def np_dtype(self) -> type:
        """Return the numpy dtype class (``np.float32`` or ``np.float64``)."""
        return np.float32 if self.dtype == "float32" else np.float64

    # ------------------------------------------------------------------ #
    # JAX global device setup                                              #
    # ------------------------------------------------------------------ #

    def apply_jax_device(self) -> None:
        """Set JAX's global default device to :attr:`jax_device`.

        Call **once** at the top of every run script, before any JAX arrays
        are created.  After this call, all JAX operations (FMT/aWBII
        convolutions, PC-SAFT functional, Anderson-on-JAX solver) land on the
        requested device automatically — no per-call device annotation needed.

        Example
        -------
        ::

            compute = ComputeConfig(jax_device="gpu", use_warp=True,
                                    warp_device="cuda:0", dtype="float32")
            compute.apply_jax_device()   # JAX FFTs now run on CUDA
        """
        import jax
        try:
            devices = jax.devices(self.jax_device)
        except RuntimeError as exc:
            raise RuntimeError(
                f"ComputeConfig.apply_jax_device: JAX could not find device "
                f"'{self.jax_device}'. Available platforms: "
                f"{[d.platform for d in jax.devices()]}."
            ) from exc
        jax.config.update("jax_default_device", devices[0])

    def enable_jax_x64(self) -> None:
        """Enable 64-bit mode in JAX (must be called before first JAX op).

        Required when ``dtype="float64"`` and you want JAX intermediate
        computations (not just NumPy) to use float64.  Has no effect if
        ``JAX_ENABLE_X64=1`` is already set in the environment.
        """
        import jax
        jax.config.update("jax_enable_x64", True)

    # ------------------------------------------------------------------ #
    # Factory helpers                                                       #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_omegaconf(cls, cfg) -> "ComputeConfig":
        """Build from a Hydra / OmegaConf DictConfig node.

        Parameters
        ----------
        cfg : DictConfig
            The ``compute`` sub-config, e.g. ``hydra_cfg.compute``.
        """
        from omegaconf import OmegaConf
        d = OmegaConf.to_container(cfg, resolve=True)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def cpu_float64(cls) -> "ComputeConfig":
        """Sensible default for interactive / CI use: CPU, float64, no Warp."""
        return cls(use_warp=False, warp_device="cpu",
                   jax_device="cpu", dtype="float64")

    @classmethod
    def cuda_float32(cls) -> "ComputeConfig":
        """Typical GPU production setting: CUDA, float32, Warp on cuda:0."""
        return cls(use_warp=True, warp_device="cuda:0",
                   jax_device="gpu", dtype="float32")

    def __repr__(self) -> str:
        return (f"ComputeConfig(use_warp={self.use_warp}, "
                f"warp_device={self.warp_device!r}, "
                f"jax_device={self.jax_device!r}, "
                f"dtype={self.dtype!r})")
