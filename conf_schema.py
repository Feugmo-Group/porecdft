"""Hydra / OmegaConf structured config schemas for porecdft.

Every parameter that a run script can tune is declared here as a typed
dataclass.  Hydra merges these defaults with the YAML overrides in ``conf/``
and with any CLI flags, giving a single ``cfg`` object that is validated at
startup.

Usage in a run script::

    import hydra
    from omegaconf import DictConfig
    from porecdft.conf_schema import register_configs

    register_configs()   # call once before @hydra.main

    @hydra.main(config_path="../conf", config_name="config", version_base="1.3")
    def main(cfg: DictConfig) -> None:
        from porecdft.compute_config import ComputeConfig
        compute = ComputeConfig.from_omegaconf(cfg.compute)
        compute.apply_jax_device()
        ...

CLI overrides (examples)::

    python run.py compute.warp_device=cuda:0 compute.use_warp=true
    python run.py solver=fire2
    python run.py run.temperature_K=77 run.p_min_bar=1e-4
    python run.py vext.spacing=0.5 vext.n_orient=200
    python run.py fluid=co2 host=zif8
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# ─────────────────────────────────────────────────────────────────────────────
# Compute backend
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ComputeSchema:
    use_warp:    bool = False
    warp_device: str  = "cpu"
    jax_device:  str  = "cpu"
    dtype:       str  = "float64"


# ─────────────────────────────────────────────────────────────────────────────
# External-potential grid builder
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VextSchema:
    spacing:          float = 0.5         # Å — grid resolution
    pbc_supercell:    List[int] = field(default_factory=lambda: [3, 3, 3])
    n_orient:         int   = 20          # Fibonacci orientations
    averaging:        str   = "boltzmann" # "boltzmann" | "arithmetic"
    v_reject_below_K: float = -10000.0   # (K) reject contact orientations
    v_cap_above_K:    float = 5000.0     # (K) wall cap
    access_factor:    float = 5.0        # access mask: V < access_factor * T_K


# ─────────────────────────────────────────────────────────────────────────────
# Anderson mixing solver
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AndersonSchema:
    _target_: str  = "porecdft.solver.anderson.anderson_solve"
    m:             int   = 6      # history depth
    beta:          float = 0.15   # mixing parameter
    max_iter:      int   = 2000
    tol:           float = 0.1    # L∞ tolerance in log-density space
    safeguard_alpha: float = 0.02
    picard_warmup: int   = 100
    step_clip:     float = 0.5
    log_clip:      float = 15.0


@dataclass
class Fire2Schema:
    _target_: str  = "porecdft.solver.fire2.fire2_solve"
    dt_start:      float = 0.02
    dt_max:        float = 0.5
    N_min:         int   = 5
    f_inc:         float = 1.1
    f_dec:         float = 0.5
    alpha_start:   float = 0.1
    f_alpha:       float = 0.99
    max_iter:      int   = 2000
    tol:           float = 0.1


# ─────────────────────────────────────────────────────────────────────────────
# Run conditions (temperature, pressure sweep, output)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunSchema:
    temperature_K: float = 298.0
    p_min_bar:     float = 0.1
    p_max_bar:     float = 50.0
    n_pressure:    int   = 13
    p_log_space:   bool  = True       # True = log-spaced, False = linear
    output_dir:    str   = "outputs"
    cache_dir:     str   = ""         # "" → same dir as run.py
    save_figure:   bool  = True
    figure_dpi:    int   = 180


# ─────────────────────────────────────────────────────────────────────────────
# Fluid: single-component
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FluidSchema:
    name:         str   = "methane"   # key in tutorials/data/pcsaft_fluids.json
    cutoff_lj:    float = 15.0        # Å — LJ cutoff for this fluid
    ff_dat:       str   = ""          # "" → use host_ff dat (same file for LJ)
    forcefield:   str   = "dreiding"  # "dreiding" | "uff" | "custom"


# ─────────────────────────────────────────────────────────────────────────────
# Host framework
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HostSchema:
    cif:          str   = ""          # path relative to tutorials/data/structures/
    ff_dat:       str   = "DREIDING.dat"
    charge_method: str  = "zero"      # "zero" | "ddec6" | "repeat" | "qeq"


# ─────────────────────────────────────────────────────────────────────────────
# Top-level composed schema
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PorecdftConfig:
    compute: ComputeSchema = field(default_factory=ComputeSchema)
    vext:    VextSchema    = field(default_factory=VextSchema)
    solver:  AndersonSchema = field(default_factory=AndersonSchema)
    run:     RunSchema     = field(default_factory=RunSchema)
    fluid:   FluidSchema   = field(default_factory=FluidSchema)
    host:    HostSchema    = field(default_factory=HostSchema)


# ─────────────────────────────────────────────────────────────────────────────
# ConfigStore registration
# ─────────────────────────────────────────────────────────────────────────────

def register_configs() -> None:
    """Register all structured configs in Hydra's ConfigStore.

    Call once at the top of any run script that uses ``@hydra.main``,
    **before** the decorator is applied::

        from porecdft.conf_schema import register_configs
        register_configs()

        @hydra.main(config_path="../conf", config_name="config", version_base="1.3")
        def main(cfg): ...
    """
    from hydra.core.config_store import ConfigStore
    cs = ConfigStore.instance()

    # Root schema — merges all groups
    cs.store(name="base_config", node=PorecdftConfig)

    # Config-group schemas — these are the defaults each group YAML extends
    cs.store(group="compute", name="base_compute", node=ComputeSchema)
    cs.store(group="vext",    name="base_vext",    node=VextSchema)
    cs.store(group="solver",  name="anderson",     node=AndersonSchema)
    cs.store(group="solver",  name="fire2",        node=Fire2Schema)
    cs.store(group="run",     name="base_run",     node=RunSchema)
    cs.store(group="fluid",   name="base_fluid",   node=FluidSchema)
    cs.store(group="host",    name="base_host",    node=HostSchema)
