"""mprm — Método de Projeção Multiescala para Navier-Stokes incompressível.

Pacote modular com submódulos:
  mprm.problem    — SideBC, FlowProblem, construtores (NumPy puro)
  mprm.transport  — mac_transport, mac_wstar, mac_divergence (NumPy puro)
  mprm.downscaler — MACDownscaler (SciPy, sem Firedrake)
  mprm.projector  — MPrMProjector (Firedrake)
  mprm.forces     — ChannelFinePoisson, obstacle_forces, strouhal (NumPy/SciPy)
  mprm.flow       — run_flow, run_cavity, run_von_karman
  mprm.viz        — make_frame_saver, make_gif
  mprm.benchmarks.hh — hh_benchmark_mac
"""

# ── Problem (geometry, BCs) ──────────────────────────────────────────────
from mprm.problem import (
    SideBC,
    FlowProblem,
    square_cylinder,
    cavity_problem,
    von_karman_problem,
    assert_solid_interior_to_subdomains,
)

# ── Transport (MAC staggered grid) ────────────────────────────────────────
from mprm.transport import (
    solid_faces,
    mac_transport,
    mac_wstar,
    mac_divergence,
)

# ── Downscaler ─────────────────────────────────────────────────────────────
from mprm.downscaler import MACDownscaler

# ── Projector (MRCM) ──────────────────────────────────────────────────────
from mprm.projector import MPrMProjector

# ── Forces + diagnostics ──────────────────────────────────────────────────
from mprm.forces import (
    ChannelFinePoisson,
    DiagnosticPressureSolver,
    obstacle_forces,
    strouhal,
)

# ── Flow (time loops) ─────────────────────────────────────────────────────
from mprm.flow import (
    mprm_projection_step,
    FineProjector,
    run_flow,
    run_cavity,
    run_von_karman,
)

# ── Visualization ─────────────────────────────────────────────────────────
from mprm.viz import make_frame_saver, make_gif

# ── Benchmarks ────────────────────────────────────────────────────────────
from mprm.benchmarks.hh import (
    hh_benchmark_mac,
    olshanskii_u_np,
    hh_p_np,
    hh_build_w,
)
