"""Abstração de problema (contorno + obstáculo) — MPrM.

Define SideBC, FlowProblem e construtores para cavidade e von Kármán.
Puro NumPy — não importa Firedrake.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Union
import numpy as np
import os, sys

MRCM_REPO = os.path.expanduser("~/mrcm/mrcm-firedrake")
if MRCM_REPO not in sys.path:
    sys.path.insert(0, MRCM_REPO)
from mrcm.problem import BoundaryData

# Abstração de problema (contorno + obstáculo) — compartilhada pelo transporte
# MAC, pelo projetor MRCM e pelo downscaler. A cavidade é um caso particular.
# ===========================================================================

from dataclasses import dataclass          # noqa: E402
from typing import Callable, Optional, Union  # noqa: E402

_SIDE_KINDS = ("wall", "slip", "inflow", "outflow")


@dataclass
class SideBC:
    """Condição de um lado externo do domínio MAC.

    kind:
      "wall"    — u·n = 0, no-slip tangencial (velocity = veloc. da parede);
      "slip"    — u·n = 0, free-slip (gradiente tangencial nulo);
      "inflow"  — u·n = profile (perfil normal prescrito), tangencial = velocity;
      "outflow" — contorno aberto (Dirichlet p=0 na projeção; grad. nulo no
                  transporte).
    velocity: velocidade tangencial da parede (float ou callable(coord)->float).
    profile:  perfil de u·n na entrada (float ou callable(coord)->float).
    """
    kind: str
    velocity: Optional[Union[float, Callable]] = None
    profile: Optional[Union[float, Callable]] = None

    def __post_init__(self):
        if self.kind not in _SIDE_KINDS:
            raise ValueError(f"SideBC.kind inválido: {self.kind!r} "
                             f"(use um de {_SIDE_KINDS})")
        if self.kind == "inflow" and self.profile is None:
            raise ValueError("SideBC(kind='inflow') exige 'profile'.")

    def tangential(self, coord=0.0) -> float:
        """Velocidade tangencial da parede (0 se não especificada)."""
        v = self.velocity
        if v is None:
            return 0.0
        return float(v(coord)) if callable(v) else float(v)


@dataclass
class FlowProblem:
    """Descrição declarativa do problema de escoamento na malha MAC.

    Nx,Ny células; domínio [0,Lx]×[0,Ly]; viscosidade nu; `sides` mapeia
    'left'/'right'/'top'/'bottom' -> SideBC; `solid` é uma máscara (Nx,Ny)
    booleana de células do obstáculo (None = sem obstáculo); `force` é uma
    força de corpo (fx,fy) opcional.
    """
    Nx: int
    Ny: int
    Lx: float
    Ly: float
    nu: float
    sides: dict
    solid: Optional[np.ndarray] = None
    force: Optional[tuple] = None

    def __post_init__(self):
        for s in ("left", "right", "top", "bottom"):
            if s not in self.sides:
                raise ValueError(f"FlowProblem.sides sem o lado '{s}'.")
            if not isinstance(self.sides[s], SideBC):
                raise TypeError(f"sides['{s}'] deve ser SideBC.")
        if self.solid is not None:
            self.solid = np.asarray(self.solid, dtype=bool)
            if self.solid.shape != (self.Nx, self.Ny):
                raise ValueError(
                    f"solid.shape {self.solid.shape} != (Nx,Ny) "
                    f"({self.Nx},{self.Ny}).")

    def side(self, name: str) -> SideBC:
        return self.sides[name]

    @property
    def hx(self) -> float:
        return self.Lx / self.Nx

    @property
    def hy(self) -> float:
        return self.Ly / self.Ny

    @property
    def has_outflow(self) -> bool:
        return any(bc.kind == "outflow" for bc in self.sides.values())

    @property
    def is_impermeable(self) -> bool:
        return (self.solid is None and
                all(bc.kind in ("wall", "slip")
                    for bc in self.sides.values()))


def square_cylinder(Nx, Ny, Lx, Ly, D, xc, yc) -> np.ndarray:
    """Máscara booleana (Nx,Ny) de um cilindro QUADRADO de lado D centrado
    em (xc,yc): célula sólida se seu centro está dentro do quadrado."""
    hx, hy = Lx / Nx, Ly / Ny
    xcell = (np.arange(Nx) + 0.5) * hx
    ycell = (np.arange(Ny) + 0.5) * hy
    X, Y = np.meshgrid(xcell, ycell, indexing="ij")
    return (np.abs(X - xc) <= D / 2.0) & (np.abs(Y - yc) <= D / 2.0)


def cavity_problem(N, Re, lid_velocity=1.0) -> FlowProblem:
    """Cavidade quadrada unitária com tampa deslizante (Lref=1; nu=U/Re)."""
    sides = {
        "top": SideBC("wall", velocity=lid_velocity),
        "bottom": SideBC("wall", velocity=0.0),
        "left": SideBC("wall", velocity=0.0),
        "right": SideBC("wall", velocity=0.0),
    }
    return FlowProblem(Nx=N, Ny=N, Lx=1.0, Ly=1.0,
                       nu=lid_velocity / Re, sides=sides)


def von_karman_problem(Lx, Ly, Nx, Ny, D, xc, yc, U_in, Re,
                       inflow="parabolic", walls="slip",
                       inflow_norm="max",
                       outflow="pressure") -> FlowProblem:
    """Escoamento em torno de um cilindro quadrado (von Kármán). Re = U_in*D/nu.

    inflow: "uniform" (u=U_in constante) ou "parabolic" (perfil de canal).
    inflow_norm: "max" (pico = U_in, conforme artigo Lages §3.5) ou "mean"
                 (média = U_in, comportamento anterior).
    walls:  "wall" (no-slip) ou "slip" (free-slip) para top/bottom.
    outflow: "pressure" (Dirichlet p=0). Demais valores reservados p/ extensão.
    """
    if outflow != "pressure":
        raise ValueError("Apenas outflow='pressure' é suportado.")
    nu = U_in * D / Re
    if inflow == "uniform":
        profile = float(U_in)
    elif inflow == "parabolic":
        Ly_c = float(Ly)
        if inflow_norm == "max":
            # pico = U_in (‖u_∞‖ = U_in, como no artigo)
            def profile(y):
                return 4.0 * U_in * y * (Ly_c - y) / (Ly_c ** 2)
        else:  # "mean" — média = U_in (comportamento anterior)
            def profile(y):
                return 6.0 * U_in * y * (Ly_c - y) / (Ly_c ** 2)
    else:
        raise ValueError(f"inflow inválido: {inflow!r}")
    sides = {
        "left": SideBC("inflow", profile=profile),
        "right": SideBC("outflow"),
        "top": SideBC(walls),
        "bottom": SideBC(walls),
    }
    solid = square_cylinder(Nx, Ny, Lx, Ly, D, xc, yc)
    return FlowProblem(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, nu=nu,
                       sides=sides, solid=solid)


def assert_solid_interior_to_subdomains(solid, cfg, strict=True):
    """Garante que o obstáculo não atravessa interfaces de subdomínio.

    Se `strict=True` (default), levanta ValueError. Se `strict=False`, apenas
    emite warning — útil para exploração/visualização com geometria arbitrária,
    mas os resultados perto das interfaces contaminadas serão menos precisos.
    """
    import warnings
    if solid is None or not solid.any():
        return
    nx, ny = cfg.nlocx, cfg.nlocy
    xs = np.where(solid.any(axis=1))[0]
    ys = np.where(solid.any(axis=0))[0]
    imin, imax = xs.min(), xs.max() + 1
    jmin, jmax = ys.min(), ys.max() + 1
    msg = None
    for i in range(1, cfg.Ndomx):
        if imin < i * nx < imax:
            msg = (f"Obstáculo atravessa a interface vertical x-cel={i*nx} "
                   f"(bbox x=[{imin},{imax})).")
    for j in range(1, cfg.Ndomy):
        if jmin < j * ny < jmax:
            msg2 = (f"Obstáculo atravessa a interface horizontal y-cel={j*ny} "
                    f"(bbox y=[{jmin},{jmax})).")
            msg = msg + " " + msg2 if msg else msg2
    if msg:
        if strict:
            raise ValueError(msg + " Alinhe o obstáculo ao interior de um "
                             "subdomínio ou use strict=False.")
        else:
            warnings.warn(msg + " Resultados perto das interfaces contaminadas "
                          "serão menos precisos.")


def _impermeable_problem(cfg) -> "FlowProblem":
    """FlowProblem impermeável (4 paredes, sem obstáculo) coerente com o cfg —
    o comportamento retrocompatível do projetor/downscaler."""
    sides = {s: SideBC("wall", velocity=0.0)
             for s in ("left", "right", "top", "bottom")}
    return FlowProblem(Nx=cfg.Nglobx, Ny=cfg.Ngloby,
                       Lx=cfg.Lx, Ly=cfg.Ly, nu=0.0, sides=sides)


def _darcy_boundary_from_problem(problem) -> dict:
    """Mapeia os lados do FlowProblem para BoundaryData do Darcy da projeção
    (spec §4): entrada/paredes -> flux=0 (no-flow forte); saída -> pressure=0."""
    def _zero(x, y):
        from firedrake import Constant
        return Constant(0.0)
    bnd = {}
    for s, bc in problem.sides.items():
        if bc.kind == "outflow":
            bnd_val = BoundaryData(pressure=_zero)
        else:                       # wall, slip, inflow -> v·n = 0
            bnd_val = BoundaryData(flux=_zero)
        bnd[s] = bnd_val
    return bnd


# ===========================================================================
