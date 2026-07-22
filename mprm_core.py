"""
mprm_core — Método de Projeção Multiescala (MPrM) para Navier-Stokes incompressível.

Implementa o algoritmo de Lages, "The multiscale projection method for the
solution of incompressible Navier-Stokes equations" (Algoritmo 1), usando o
pacote `mrcm` (~/mrcm/mrcm-firedrake) como solver multiescala (MRCM,
Guiraldello et al. 2018) do passo de projeção de Chorin (1968):

  1. w* = uⁿ − δt ∇·(uⁿuⁿ) + δt ν ∇²uⁿ            (transporte explícito, MAC)
  2. v = δt ∇p,  ∇·v = ∇·w*,  v·n = w*·n = 0       (projeção ⇒ Darcy com K=δt,
                                                     resolvido pelo MRCM)
  3. downscaling: média dos fluxos de interface (Eq. 21) + problemas locais de
     Neumann (Eq. 22) na malha MAC ⇒ ṽ conservativo na escala fina
  4. uⁿ⁺¹ = w* − ṽ                                  (Eq. 23)

O transporte (passo 1) e o downscaling (passo 3) usam a malha deslocada
(staggered/MAC) do artigo (§2.6); o passo de projeção usa os componentes do
mrcm (BDM₁–DG₀ em triângulos) — desvio de discretização documentado no
notebook. Todo o estágio offline (MsBF, fatorações locais, matriz de
interface) é construído uma única vez e reutilizado a cada passo de tempo.

Suporte a paralelismo: as classes aceitam um comunicador MPI; cada rank
possui um subconjunto de subdomínios (round-robin, como no executor mpi do
mrcm / Jaramillo et al. 2022). Com comm de tamanho 1 o código degenera para o
caminho sequencial usado no notebook.
"""

from __future__ import annotations

import os
import sys
import numpy as np

MRCM_REPO = os.path.expanduser("~/mrcm/mrcm-firedrake")
if MRCM_REPO not in sys.path:
    sys.path.insert(0, MRCM_REPO)

from mrcm.config import MRCMConfig                                   # noqa: E402
from mrcm.problem import DarcyProblem, BoundaryData                  # noqa: E402
from mrcm.topology import Skeleton, DOFManager, mark_sides           # noqa: E402
from mrcm.interface_space import make_interface_space                # noqa: E402
from mrcm.robin import make_robin                                    # noqa: E402
from mrcm.local_form import build_local_form                         # noqa: E402
from mrcm.local_solver import LocalSolver, build_subdomain_workspace # noqa: E402
from mrcm.global_assembly import GlobalAssembler, compute_subdomain_triplets  # noqa: E402
from mrcm.reconstructor import reconstruct_subdomain                 # noqa: E402


# ===========================================================================
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


def assert_solid_interior_to_subdomains(solid, cfg):
    """Garante que o obstáculo não atravessa interfaces de subdomínio:
    nenhuma face de interface (colunas i*nlocx, linhas j*nlocy) pode separar
    fluido de sólido de forma inconsistente — exigimos que cada célula sólida
    e suas vizinhas na direção normal à interface estejam no mesmo subdomínio.
    Critério prático: nenhuma interface interna corta o bounding-box do sólido.
    """
    if solid is None or not solid.any():
        return
    nx, ny = cfg.nlocx, cfg.nlocy
    xs = np.where(solid.any(axis=1))[0]
    ys = np.where(solid.any(axis=0))[0]
    imin, imax = xs.min(), xs.max() + 1        # [imin, imax) em células
    jmin, jmax = ys.min(), ys.max() + 1
    for i in range(1, cfg.Ndomx):
        if imin < i * nx < imax:
            raise ValueError(
                f"Obstáculo atravessa a interface vertical x-cel={i*nx} "
                f"(bbox x=[{imin},{imax})). Alinhe o obstáculo ao interior "
                f"de um subdomínio (spec §5.3, R4).")
    for j in range(1, cfg.Ndomy):
        if jmin < j * ny < jmax:
            raise ValueError(
                f"Obstáculo atravessa a interface horizontal y-cel={j*ny} "
                f"(bbox y=[{jmin},{jmax})).")


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
# Transporte explícito na malha MAC (artigo §2.6, Eqs. 8/24-29)
# ===========================================================================
#
# Convenções: malha cartesiana Nx×Ny em [0,Lx]×[0,Ly].
#   u : (Nx+1, Ny)  — componente x nas faces verticais  (i, j+1/2)
#   v : (Nx, Ny+1)  — componente y nas faces horizontais (i+1/2, j)
# Índice x primeiro. Paredes impermeáveis: u[0,:]=u[Nx,:]=v[:,0]=v[:,Ny]=0.
# Condições tangenciais via células fantasma: parede com velocidade
# tangencial U_w ⇒ ghost = 2·U_w − interior (não-deslizamento).


def solid_faces(solid):
    """Máscaras de face MAC tocadas por célula sólida.

    fx (Nx+1, Ny): face-u sólida se célula à esquerda OU à direita é sólida.
    fy (Nx, Ny+1): face-v sólida se célula abaixo OU acima é sólida.
    """
    Nx, Ny = solid.shape
    fx = np.zeros((Nx + 1, Ny), dtype=bool)
    fx[:-1, :] |= solid
    fx[1:, :] |= solid
    fy = np.zeros((Nx, Ny + 1), dtype=bool)
    fy[:, :-1] |= solid
    fy[:, 1:] |= solid
    return fx, fy


def _tangential_ghost(interior, side_bc: SideBC):
    """Célula-fantasma tangencial de um lado: no-slip (wall/inflow) =>
    2*V_t - interior; free-slip/aberto (slip/outflow) => interior."""
    if side_bc.kind in ("wall", "inflow"):
        return 2.0 * side_bc.tangential() - interior
    return interior          # slip, outflow: gradiente tangencial nulo


def mac_transport(u, v, problem: "FlowProblem", dt):
    """Velocidade tentativa w* = uⁿ + δt(−∇·(uu) + ν∇²u + f) na malha MAC,
    para um `FlowProblem` arbitrário (contorno + obstáculo).

    Convecção upwind (donor-cell) + difusão central (§2.6 do artigo). A
    cavidade é o caso `cavity_problem(...)` e reproduz o estêncil antigo
    bit-a-bit. Entrada/paredes/obstáculo tratados via fantasmas + máscara
    de faces sólidas.
    """
    Nx = u.shape[0] - 1
    Ny = v.shape[1] - 1
    hx, hy, nu = problem.hx, problem.hy, problem.nu
    S = problem.sides

    # ---- fantasmas tangenciais (u em j: bottom/top; v em i: left/right) ----
    ug = np.empty((Nx + 1, Ny + 2)); ug[:, 1:-1] = u
    ug[:, 0] = _tangential_ghost(u[:, 0], S["bottom"])
    ug[:, -1] = _tangential_ghost(u[:, -1], S["top"])
    vg = np.empty((Nx + 2, Ny + 1)); vg[1:-1, :] = v
    vg[0, :] = _tangential_ghost(v[0, :], S["left"])
    vg[-1, :] = _tangential_ghost(v[-1, :], S["right"])

    w1 = u.copy(); w2 = v.copy()

    # ---- componente x (faces internas i=1..Nx-1) ----
    ue = 0.5 * (u[1:, :] + u[:-1, :])
    Fuu = np.where(ue >= 0.0, u[:-1, :], u[1:, :]) * ue
    dudx_conv = (Fuu[1:, :] - Fuu[:-1, :]) / hx
    v_at_corner = 0.5 * (vg[1:-2, :] + vg[2:-1, :])
    u_upw_y = np.where(v_at_corner >= 0.0, ug[1:-1, :-1], ug[1:-1, 1:])
    Fuv = u_upw_y * v_at_corner
    duvdy = (Fuv[:, 1:] - Fuv[:, :-1]) / hy
    lap_u = ((ug[2:, 1:-1] - 2.0 * ug[1:-1, 1:-1] + ug[:-2, 1:-1]) / hx**2
             + (ug[1:-1, 2:] - 2.0 * ug[1:-1, 1:-1] + ug[1:-1, :-2]) / hy**2)
    fbx = 0.0 if problem.force is None else problem.force[0]
    w1[1:-1, :] = u[1:-1, :] + dt * (-dudx_conv - duvdy + nu * lap_u + fbx)

    # ---- componente y (faces internas j=1..Ny-1) ----
    vc = 0.5 * (v[:, 1:] + v[:, :-1])
    Fvv = np.where(vc >= 0.0, v[:, :-1], v[:, 1:]) * vc
    dvdy_conv = (Fvv[:, 1:] - Fvv[:, :-1]) / hy
    u_at_corner = 0.5 * (ug[:, 1:-2] + ug[:, 2:-1])
    v_upw_x = np.where(u_at_corner >= 0.0, vg[:-1, 1:-1], vg[1:, 1:-1])
    Fvu = v_upw_x * u_at_corner
    duvdx = (Fvu[1:, :] - Fvu[:-1, :]) / hx
    lap_v = ((vg[2:, 1:-1] - 2.0 * vg[1:-1, 1:-1] + vg[:-2, 1:-1]) / hx**2
             + (vg[1:-1, 2:] - 2.0 * vg[1:-1, 1:-1] + vg[1:-1, :-2]) / hy**2)
    fby = 0.0 if problem.force is None else problem.force[1]
    w2[:, 1:-1] = v[:, 1:-1] + dt * (-duvdx - dvdy_conv + nu * lap_v + fby)

    # ---- faces normais de contorno por lado ----
    _apply_normal_bc_x(w1, S["left"], hy, Ny, side="left")
    _apply_normal_bc_x(w1, S["right"], hy, Ny, side="right")
    _apply_normal_bc_y(w2, S["bottom"], hx, Nx, side="bottom")
    _apply_normal_bc_y(w2, S["top"], hx, Nx, side="top")

    # ---- obstáculo: zera faces tocadas por célula sólida (u=0 no sólido) ----
    if problem.solid is not None:
        fx, fy = solid_faces(problem.solid)
        w1[fx] = 0.0
        w2[fy] = 0.0

    return w1, w2


def _profile_on_faces(profile, coords):
    """Avalia um perfil de entrada (float ou callable) nos centros de face."""
    if callable(profile):
        return np.array([profile(c) for c in coords])
    return np.full(len(coords), float(profile))


def _apply_normal_bc_x(w1, side_bc: SideBC, hy, Ny, side):
    """u·n nas faces x=0 (side='left') ou x=Lx (side='right')."""
    idx = 0 if side == "left" else -1
    ycen = (np.arange(Ny) + 0.5) * hy
    if side_bc.kind in ("wall", "slip"):
        w1[idx, :] = 0.0
    elif side_bc.kind == "inflow":
        w1[idx, :] = _profile_on_faces(side_bc.profile, ycen)
    elif side_bc.kind == "outflow":
        w1[idx, :] = w1[1 if side == "left" else -2, :]  # gradiente nulo


def _apply_normal_bc_y(w2, side_bc: SideBC, hx, Nx, side):
    """u·n nas faces y=0 (side='bottom') ou y=Ly (side='top')."""
    idx = 0 if side == "bottom" else -1
    xcen = (np.arange(Nx) + 0.5) * hx
    if side_bc.kind in ("wall", "slip"):
        w2[:, idx] = 0.0
    elif side_bc.kind == "inflow":
        w2[:, idx] = _profile_on_faces(side_bc.profile, xcen)
    elif side_bc.kind == "outflow":
        w2[:, idx] = w2[:, 1 if side == "bottom" else -2]


def mac_wstar(u, v, hx, hy, dt, nu, lid_velocity=1.0):
    """RETROCOMPATÍVEL — cavidade. Constrói o FlowProblem da cavidade e
    delega a `mac_transport`. Mantido para `run_mprm.py`, notebook etc."""
    Nx = u.shape[0] - 1
    Ny = v.shape[1] - 1
    sides = {
        "top": SideBC("wall", velocity=lid_velocity),
        "bottom": SideBC("wall", velocity=0.0),
        "left": SideBC("wall", velocity=0.0),
        "right": SideBC("wall", velocity=0.0),
    }
    prob = FlowProblem(Nx=Nx, Ny=Ny, Lx=Nx * hx, Ly=Ny * hy,
                       nu=nu, sides=sides)
    return mac_transport(u, v, prob, dt)


def mac_divergence(w1, w2, hx, hy):
    """∇·w por célula (FV exato na malha MAC): (Nx, Ny)."""
    return (w1[1:, :] - w1[:-1, :]) / hx + (w2[:, 1:] - w2[:, :-1]) / hy


# ===========================================================================
# Projetor MRCM — estágio offline/online (Algoritmo 1 do artigo)
# ===========================================================================

class MPrMProjector:
    """Passo de projeção do MPrM resolvido pelo MRCM.

    Offline (construtor): malhas/espaços locais, MsBF + caches B1/B2
    (via `mrcm.local_solver.build_subdomain_workspace`), fatoração local
    persistente (`LocalSolver`), montador global e operadores de extração
    de traço de fluxo nas interfaces.

    Online (`project(divw)`): parte particular por subdomínio (fonte
    ∇·w* célula a célula), sistema global de interface, reconstrução e
    média dos fluxos finos nas interfaces (Eq. 21 do artigo).

    Paralelismo: subdomínios distribuídos round-robin entre os ranks de
    `comm` (isub % size), como no executor mpi do mrcm.
    """

    def __init__(self, cfg: MRCMConfig, dt: float, comm=None,
                 problem=None, verbose=False, eps_solid=1e-6):
        from firedrake import (Constant, FunctionSpace, TestFunction,
                               TestFunctions, Function, FacetNormal,
                               VectorFunctionSpace, SpatialCoordinate,
                               assemble, dot)
        from mrcm.topology import SIDE_TO_MARKER

        if comm is None:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
        self.comm = comm
        self.rank, self.size = comm.rank, comm.size
        self.cfg = cfg
        self.dt = float(dt)
        self.verbose = verbose
        self.eps_solid = float(eps_solid)

        # problem=None => cavidade impermeável (comportamento retrocompatível)
        if problem is None:
            problem = _impermeable_problem(cfg)
        self.problem = problem
        if problem.solid is not None:
            assert_solid_interior_to_subdomains(problem.solid, cfg)

        self.skel = Skeleton.cartesian(cfg.Ndomx, cfg.Ndomy)
        self.dofm = DOFManager(self.skel.NGamma, cfg.ndofsU, cfg.ndofsP)
        self.my_subs = list(range(self.rank, cfg.Nsubd, self.size))

        dt_c = self.dt
        eps = self.eps_solid
        solid = problem.solid
        hx_f = cfg.Hx / cfg.nlocx
        hy_f = cfg.Hy / cfg.nlocy
        self.hx_f, self.hy_f = hx_f, hy_f

        def _K(mesh):
            from firedrake import (Constant, Function, FunctionSpace,
                                   VectorFunctionSpace, SpatialCoordinate)
            if solid is None:
                return Constant(dt_c)
            Kf = FunctionSpace(mesh, "DG", 0)
            xc = Function(VectorFunctionSpace(mesh, "DG", 0)).interpolate(
                SpatialCoordinate(mesh))
            xcd = xc.dat.data_ro
            ic = np.clip(np.floor(xcd[:, 0] / hx_f).astype(int),
                         0, cfg.Nglobx - 1)
            jc = np.clip(np.floor(xcd[:, 1] / hy_f).astype(int),
                         0, cfg.Ngloby - 1)
            k = Function(Kf)
            k.dat.data[:] = np.where(solid[ic, jc], dt_c * eps, dt_c)
            return k

        def _invK(mesh):
            from firedrake import Constant, Function
            if solid is None:
                return Constant(1.0 / dt_c)
            K = _K(mesh)
            ik = Function(K.function_space())
            ik.dat.data[:] = 1.0 / K.dat.data_ro
            return ik

        def _zero(x, y):
            from firedrake import Constant
            return Constant(0.0)

        self.problem_darcy = DarcyProblem(
            Lx=cfg.Lx, Ly=cfg.Ly,
            permeability=_K, inv_permeability=_invK,
            source=_zero,
            boundary=_darcy_boundary_from_problem(problem),
        )

        robin = make_robin(cfg.robin_variant, cfg.alpha, *cfg.robin_L_xy(),
                           cfg.robin_clamp_eps, cfg.robin_extreme_beta)
        ispace = make_interface_space(cfg.basis_kind, cfg.degreeU, cfg.degreeP)

        NunkU = self.skel.NGamma * cfg.ndofsU
        self.ws = {}          # isub -> workspace (mesh, V, Q, MsBF, caches)
        self.solvers = {}     # isub -> LocalSolver persistente (LU offline)
        self.src_fn = {}      # isub -> Function DG (fonte ∇·w*)
        self.L_forms = {}     # isub -> forma RHS  -f q dx
        self.cell_map = {}    # isub -> (ic, jc) índices globais de célula p/ dof DG
        self.trace = {}       # (isub, side) -> dict(form, idx, len, order, sign)
        self.out_trace = {}   # (isub, side_name) -> dict p/ fluxo de saída
        self.uh = {}          # isub -> Function(V) para extração de traço

        for isub in self.my_subs:
            ix, iy = isub % cfg.Ndomx, isub // cfg.Ndomx
            flag_side, internal_sides = mark_sides(ix, iy, cfg.Ndomx, cfg.Ndomy)
            edge_l = {s: int(self.skel.ieGt[self.skel.el2edge[isub, s]])
                      for s in internal_sides}
            task = {"cfg_dict": cfg.as_dict(), "isub": isub,
                    "NunkU": NunkU, "edge_l": edge_l,
                    "problem": self.problem_darcy}
            # Offline: MsBF + caches B1/B2
            ws = build_subdomain_workspace(task)
            self.ws[isub] = ws
            mesh, V, Q = ws["mesh"], ws["V"], ws["Q"]
            W = V * Q

            # Solver local persistente (fatoração LU offline; artigo §2.7).
            beta_expr = robin.beta_for_subdomain(isub, mesh)
            a_form, dx_m, ds_m, _, _, n_facet = build_local_form(
                W, mesh, _invK(mesh), beta_expr, internal_sides,
                quadrature_degree=cfg.final_quadrature_degree(),
                facet_quadrature_degree=cfg.final_facet_quadrature_degree(),
            )
            # BCs nativas do mrcm (no-flow forte nos lados externos com fluxo)
            from mrcm.local_form import build_flux_dirichlet_bcs
            bcs = build_flux_dirichlet_bcs(W, mesh, flag_side,
                                           problem=self.problem_darcy)
            self.solvers[isub] = LocalSolver(
                W, a_form, bcs,
                use_iterative=(cfg.local_solver_mode == "iterative"), cfg=cfg)

            # Fonte DG por subdomínio + mapa célula fina global -> dof DG.
            f_i = Function(Q)
            self.src_fn[isub] = f_i
            _, q_test = TestFunctions(W)
            self.L_forms[isub] = -f_i * q_test * dx_m
            Xc = VectorFunctionSpace(mesh, "DG", 0)
            xc = Function(Xc).interpolate(SpatialCoordinate(mesh))
            xcd = xc.dat.data_ro
            assert cfg.mixed_order == 0, "fonte por célula requer DG0"
            ic = np.floor(xcd[:, 0] / hx_f).astype(int)
            jc = np.floor(xcd[:, 1] / hy_f).astype(int)
            self.cell_map[isub] = (np.clip(ic, 0, cfg.Nglobx - 1),
                                   np.clip(jc, 0, cfg.Ngloby - 1))

            # Operadores de traço: fluxo médio por face fina de cada lado interno.
            T = FunctionSpace(mesh, "HDiv Trace", 0)
            tau = TestFunction(T)
            uh = Function(V)
            self.uh[isub] = uh
            for s in internal_sides:
                marker = SIDE_TO_MARKER[s]
                ln = assemble(tau * ds_m(marker)).dat.data_ro.copy()
                idx = np.nonzero(np.abs(ln) > 1e-14)[0]
                xm = assemble(SpatialCoordinate(mesh)[0] * tau
                              * ds_m(marker)).dat.data_ro[idx] / ln[idx]
                ym = assemble(SpatialCoordinate(mesh)[1] * tau
                              * ds_m(marker)).dat.data_ro[idx] / ln[idx]
                coord = ym if s in (1, 3) else xm     # lados verticais: y
                order = np.argsort(coord)
                sign = +1.0 if s in (1, 2) else -1.0  # right/top: n = +e
                self.trace[(isub, s)] = {
                    "form": dot(uh, n_facet) * tau * ds_m(marker),
                    "idx": idx, "len": ln[idx], "order": order, "sign": sign,
                }

            # Traço de fluxo nas faces de SAÍDA (lado outflow externo).
            for s_name, marker in (("left", 1), ("right", 2),
                                   ("bottom", 3), ("top", 4)):
                if problem.side(s_name).kind != "outflow":
                    continue
                flag_idx = {"bottom": 0, "right": 1, "top": 2, "left": 3}[s_name]
                if flag_side[flag_idx]:      # só bordas externas reais
                    continue
                ln = assemble(tau * ds_m(marker)).dat.data_ro.copy()
                idx = np.nonzero(np.abs(ln) > 1e-14)[0]
                if idx.size == 0:
                    continue
                cm = (assemble(SpatialCoordinate(mesh)[1] * tau * ds_m(marker))
                      .dat.data_ro[idx] / ln[idx]) if s_name in ("left", "right") \
                    else (assemble(SpatialCoordinate(mesh)[0] * tau * ds_m(marker))
                          .dat.data_ro[idx] / ln[idx])
                order = np.argsort(cm)
                sign = +1.0 if s_name in ("right", "top") else -1.0
                self.out_trace[(isub, s_name)] = {
                    "form": dot(uh, n_facet) * tau * ds_m(marker),
                    "idx": idx, "len": ln[idx], "order": order, "sign": sign}

        # Montador global
        self.assembler = None
        if self.rank == 0:
            self.assembler = GlobalAssembler(
                cfg, self.skel, self.dofm, ispace, robin,
                basis_per_sides=None, n_facets=None, dss=None, meshes=None,
                Vs=None, Qs=None, face_mass={}, cofunc_flux={}, cofunc_pres={})

        self.last_X = None
        self.last_pT = {}
        self.last_uT = {}
        self.last_boundary_fluxes = {}

    # -- online ------------------------------------------------------------

    def project(self, divw):
        """Resolve a projeção MRCM para a fonte ∇·w* (array Nglobx×Ngloby).

        Retorna `fluxes`: dict com os fluxos médios (Eq. 21) por interface,
          ("v", i, iy) -> array nlocy  (interface vertical entre colunas i-1|i
                                        de subdomínios, orientação +x)
          ("h", ix, j) -> array nlocx  (interface horizontal, orientação +y)
        disponível de forma idêntica em todos os ranks.
        """
        from firedrake import assemble
        cfg = self.cfg
        comm = self.comm

        # 1. Parte particular (fonte atualizada) — paralelo, sem comunicação.
        particular = {}
        for isub in self.my_subs:
            ic, jc = self.cell_map[isub]
            self.src_fn[isub].dat.data[:] = divw[ic, jc]
            particular[isub] = self.solvers[isub].solve(self.L_forms[isub])

        # 2. Triplets locais -> sistema global de interface no rank 0.
        my_trip = {isub: compute_subdomain_triplets(
            cfg, self.skel, self.dofm, isub,
            particular[isub], self.ws[isub]["homogeneous"],
            self.ws[isub]["face_mass"], self.ws[isub]["cofunc_flux"],
            self.ws[isub]["cofunc_pres"]) for isub in self.my_subs}
        gathered = comm.gather(my_trip, root=0)
        if self.rank == 0:
            trip = {}
            for d in gathered:
                trip.update(d)
            _, _, X = self.assembler.assemble_from_triplets(trip)
        else:
            X = None
        X = comm.bcast(X, root=0)
        self.last_X = X

        # 3. Reconstrução distribuída + traços de fluxo nas interfaces.
        my_traces = {}
        for isub in self.my_subs:
            pT, uT = reconstruct_subdomain(
                particular[isub], self.ws[isub]["homogeneous"], X,
                self.ws[isub]["V"], self.ws[isub]["Q"],
                sparsity_cutoff=cfg.reconstruction_sparsity_cutoff)
            self.last_pT[isub], self.last_uT[isub] = pT, uT
            self.uh[isub].assign(uT)
            for (jsub, s), tr in self.trace.items():
                if jsub != isub:
                    continue
                vals = assemble(tr["form"]).dat.data_ro[tr["idx"]]
                flux = tr["sign"] * (vals / tr["len"])[tr["order"]]
                my_traces[(isub, s)] = flux

        # 4. Média dos fluxos (Eq. 21) — allgather (arrays pequenos: nloc
        #    valores por lado de interface).
        all_tr = {}
        for d in comm.allgather(my_traces):
            all_tr.update(d)

        fluxes = {}
        for iy in range(cfg.Ndomy):          # interfaces verticais
            for i in range(1, cfg.Ndomx):
                sl = (i - 1) + iy * cfg.Ndomx   # subdomínio à esquerda
                sr = i + iy * cfg.Ndomx
                fluxes[("v", i, iy)] = 0.5 * (all_tr[(sl, 1)] + all_tr[(sr, 3)])
        for j in range(1, cfg.Ndomy):        # interfaces horizontais
            for ix in range(cfg.Ndomx):
                sb = ix + (j - 1) * cfg.Ndomx   # subdomínio abaixo
                st = ix + j * cfg.Ndomx
                fluxes[("h", ix, j)] = 0.5 * (all_tr[(sb, 2)] + all_tr[(st, 0)])

        # fluxos de saída (v·n reconstruído nas faces externas outflow)
        my_out = {}
        for isub in self.my_subs:
            self.uh[isub].assign(self.last_uT[isub])
            for (jsub, s_name), tr in self.out_trace.items():
                if jsub != isub:
                    continue
                vals = assemble(tr["form"]).dat.data_ro[tr["idx"]]
                my_out[(isub, s_name)] = tr["sign"] * (vals / tr["len"])[tr["order"]]
        self.last_boundary_fluxes = {}
        for d in comm.allgather(my_out):
            self.last_boundary_fluxes.update(d)

        return fluxes


# ===========================================================================
# Downscaling MAC (Eqs. 21-23 do artigo) — conservação na escala fina
# ===========================================================================

class MACDownscaler:
    """Problemas locais de Neumann (Eq. 22) na malha MAC, por subdomínio.

    Matriz FV (5 pontos, K=δt) idêntica para todos os subdomínios: montada e
    fatorada UMA vez (offline). O sistema é singular (Neumann puro); o RHS é
    balanceado (compatibilizado) e a pressão pinada na célula 0. O defeito de
    compatibilidade por subdomínio (violação da continuidade média de fluxo
    após a média da Eq. 21) é reportado em `last_defect`.
    """

    def __init__(self, cfg: MRCMConfig, dt: float):
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        self.cfg = cfg
        self.dt = float(dt)
        n, m = cfg.nlocx, cfg.nlocy
        hx, hy = cfg.Hx / n, cfg.Hy / m
        self.hx, self.hy = hx, hy

        def lid(i, j):
            return i + j * n

        rows, cols, vals = [], [], []
        for j in range(m):
            for i in range(n):
                c = lid(i, j)
                diag = 0.0
                for (di, dj, d, ln) in ((1, 0, hx, hy), (-1, 0, hx, hy),
                                        (0, 1, hy, hx), (0, -1, hy, hx)):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < n and 0 <= jj < m:
                        w = dt * ln / d
                        rows.append(c); cols.append(lid(ii, jj)); vals.append(w)
                        diag -= w
                rows.append(c); cols.append(c); vals.append(diag)
        A = sp.csr_matrix((vals, (rows, cols)), shape=(n * m, n * m))
        # pino da pressão na célula 0 (remoção do modo constante)
        A = A.tolil()
        A[0, :] = 0.0
        A[0, 0] = 1.0
        self.lu = spla.splu(A.tocsc())
        self._A_base = A.tocsc()          # sem sólido (referência p/ máscara)
        self._masked_lu = {}              # solid_loc.tobytes() -> splu
        self.last_defect = 0.0

    def _build_masked_lu(self, solid_loc):
        """Fatoração FV para um padrão de sólido: células sólidas viram
        p=0 (linha identidade, desacopladas); faces sólidas somem do estêncil
        (no-flow). Cacheada por padrão."""
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        n, m = self.cfg.nlocx, self.cfg.nlocy
        dt, hx, hy = self.dt, self.hx, self.hy
        solid = solid_loc.reshape(n, m)

        def lid(i, j):
            return i + j * n

        rows, cols, vals = [], [], []
        for j in range(m):
            for i in range(n):
                c = lid(i, j)
                if solid[i, j]:
                    rows.append(c); cols.append(c); vals.append(1.0)
                    continue
                diag = 0.0
                for (di, dj, d, ln) in ((1, 0, hx, hy), (-1, 0, hx, hy),
                                        (0, 1, hy, hx), (0, -1, hy, hx)):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < n and 0 <= jj < m and not solid[ii, jj]:
                        w = dt * ln / d
                        rows.append(c); cols.append(lid(ii, jj))
                        vals.append(w); diag -= w
                rows.append(c); cols.append(c); vals.append(diag)
        A = sp.csr_matrix((vals, (rows, cols)), shape=(n * m, n * m)).tolil()
        # pino: primeira célula de FLUIDO
        pin = int(np.argmin(solid.flatten(order="F")))
        A[pin, :] = 0.0; A[pin, pin] = 1.0
        return spla.splu(A.tocsc()), pin

    def _get_masked(self, solid_loc):
        key = solid_loc.tobytes()
        if key not in self._masked_lu:
            self._masked_lu[key] = self._build_masked_lu(solid_loc)
        return self._masked_lu[key]

    def solve_patch(self, divw_loc, fW, fE, fS, fN, solid_loc=None):
        """Resolve o Neumann local (Eq. 22) e devolve ṽ (MAC).

        Se `solid_loc` (n,m bool) for dado, células sólidas são excluídas
        (p=0) e faces sólidas ficam com ṽ·n=0.
        """
        cfg, dt = self.cfg, self.dt
        n, m = cfg.nlocx, cfg.nlocy
        hx, hy = self.hx, self.hy

        rhs = (divw_loc * hx * hy).copy()
        rhs[0, :] += fW * hy
        rhs[-1, :] -= fE * hy
        rhs[:, 0] += fS * hx
        rhs[:, -1] -= fN * hx

        if solid_loc is None:
            b = rhs.flatten(order="F")
            defect = b.sum()
            self.last_defect = max(self.last_defect,
                                   abs(defect) / (hx * hy * n * m))
            b -= defect / b.size
            b[0] = 0.0
            p = self.lu.solve(b).reshape((n, m), order="F")
            solid = np.zeros((n, m), dtype=bool)
        else:
            solid = solid_loc.reshape(n, m)
            lu, pin = self._get_masked(solid)
            rhs[solid] = 0.0                       # fonte nula no sólido
            b = rhs.flatten(order="F")
            fluid = ~solid.flatten(order="F")
            nfluid = int(fluid.sum())
            defect = b[fluid].sum()
            self.last_defect = max(self.last_defect,
                                   abs(defect) / (hx * hy * nfluid))
            b[fluid] -= defect / nfluid            # compatibiliza só no fluido
            b[~fluid] = 0.0
            b[pin] = 0.0
            p = lu.solve(b).reshape((n, m), order="F")
            p[solid] = 0.0

        vt1 = np.empty((n + 1, m))
        vt1[1:-1, :] = dt * (p[1:, :] - p[:-1, :]) / hx
        vt1[0, :], vt1[-1, :] = fW, fE
        vt2 = np.empty((n, m + 1))
        vt2[:, 1:-1] = dt * (p[:, 1:] - p[:, :-1]) / hy
        vt2[:, 0], vt2[:, -1] = fS, fN

        if solid_loc is not None:
            fx = np.zeros((n + 1, m), bool); fx[:-1, :] |= solid; fx[1:, :] |= solid
            fy = np.zeros((n, m + 1), bool); fy[:, :-1] |= solid; fy[:, 1:] |= solid
            vt1[fx] = 0.0
            vt2[fy] = 0.0
        return vt1, vt2


# ===========================================================================
# Passo completo do MPrM e laços temporais
# ===========================================================================

def _apply_external_normals_after_step(u, v, problem):
    """Após uⁿ⁺¹=w*−ṽ: paredes/slip impõem u·n=0; entrada e saída preservam
    o que veio de w*−ṽ (entrada: ṽ·n=0 ⇒ u·n=perfil; saída: corrigido)."""
    S = problem.sides
    if S["left"].kind in ("wall", "slip"):
        u[0, :] = 0.0
    if S["right"].kind in ("wall", "slip"):
        u[-1, :] = 0.0
    if S["bottom"].kind in ("wall", "slip"):
        v[:, 0] = 0.0
    if S["top"].kind in ("wall", "slip"):
        v[:, -1] = 0.0


def mprm_projection_step(w1, w2, projector: "MPrMProjector",
                         downscaler: "MACDownscaler"):
    """Passos 2-4 do MPrM. Lê `projector.problem` (obstáculo + saída).
    Assinatura inalterada — retrocompatível."""
    cfg = projector.cfg
    comm = projector.comm
    problem = projector.problem
    n, m = cfg.nlocx, cfg.nlocy
    hx, hy = downscaler.hx, downscaler.hy
    solid = problem.solid

    divw = mac_divergence(w1, w2, hx, hy)
    if solid is not None:
        divw = divw.copy(); divw[solid] = 0.0
    fluxes = projector.project(divw)
    out_fluxes = getattr(projector, "last_boundary_fluxes", {})

    u = w1.copy(); v = w2.copy()
    zeros_m, zeros_n = np.zeros(m), np.zeros(n)
    my_patch = {}
    for isub in projector.my_subs:
        ix, iy = isub % cfg.Ndomx, isub // cfg.Ndomx
        ox, oy = ix * n, iy * m
        fW = fluxes.get(("v", ix, iy), zeros_m)
        fE = fluxes.get(("v", ix + 1, iy), zeros_m)
        fS = fluxes.get(("h", ix, iy), zeros_n)
        fN = fluxes.get(("h", ix, iy + 1), zeros_n)
        # fluxos externos de SAÍDA (v·n reconstruído nas faces outflow)
        if (isub, "right") in out_fluxes:
            fE = out_fluxes[(isub, "right")]
        if (isub, "left") in out_fluxes:
            fW = out_fluxes[(isub, "left")]
        if (isub, "top") in out_fluxes:
            fN = out_fluxes[(isub, "top")]
        if (isub, "bottom") in out_fluxes:
            fS = out_fluxes[(isub, "bottom")]
        solid_loc = None
        if solid is not None:
            sl = solid[ox:ox + n, oy:oy + m]
            solid_loc = sl if sl.any() else None
        vt1, vt2 = downscaler.solve_patch(
            divw[ox:ox + n, oy:oy + m], fW, fE, fS, fN, solid_loc=solid_loc)
        my_patch[isub] = (vt1, vt2)

    for d in comm.allgather(my_patch):
        for isub, (vt1, vt2) in d.items():
            ix, iy = isub % cfg.Ndomx, isub // cfg.Ndomx
            ox, oy = ix * n, iy * m
            u[ox:ox + n + 1, oy:oy + m] = w1[ox:ox + n + 1, oy:oy + m] - vt1
            v[ox:ox + n, oy:oy + m + 1] = w2[ox:ox + n, oy:oy + m + 1] - vt2

    # contorno externo por lado (impermeável => 0; saída/entrada preservam w*)
    _apply_external_normals_after_step(u, v, problem)
    if solid is not None:
        fx, fy = solid_faces(solid)
        u[fx] = 0.0; v[fy] = 0.0
    return u, v


class FineProjector:
    """Referência: projeção pela malha fina global (Poisson MAC 5 pontos,
    Neumann homogêneo, pino de pressão) — o 'fine mesh solver' do artigo."""

    def __init__(self, Nx, Ny, hx, hy, dt):
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        self.Nx, self.Ny = Nx, Ny
        self.hx, self.hy = hx, hy
        self.dt = float(dt)

        def lid(i, j):
            return i + j * Nx

        rows, cols, vals = [], [], []
        for j in range(Ny):
            for i in range(Nx):
                c = lid(i, j)
                diag = 0.0
                for (di, dj, d, ln) in ((1, 0, hx, hy), (-1, 0, hx, hy),
                                        (0, 1, hy, hx), (0, -1, hy, hx)):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < Nx and 0 <= jj < Ny:
                        w = dt * ln / d
                        rows.append(c); cols.append(lid(ii, jj)); vals.append(w)
                        diag -= w
                rows.append(c); cols.append(c); vals.append(diag)
        A = sp.csr_matrix((vals, (rows, cols)), shape=(Nx * Ny, Nx * Ny)).tolil()
        A[0, :] = 0.0
        A[0, 0] = 1.0
        self.lu = spla.splu(A.tocsc())

    def step(self, w1, w2):
        dt, hx, hy = self.dt, self.hx, self.hy
        divw = mac_divergence(w1, w2, hx, hy)
        b = (divw * hx * hy).flatten(order="F")
        b -= b.sum() / b.size
        b[0] = 0.0
        p = self.lu.solve(b).reshape((self.Nx, self.Ny), order="F")
        u = w1.copy()
        v = w2.copy()
        u[1:-1, :] -= dt * (p[1:, :] - p[:-1, :]) / hx
        v[:, 1:-1] -= dt * (p[:, 1:] - p[:, :-1]) / hy
        u[0, :] = u[-1, :] = 0.0
        v[:, 0] = v[:, -1] = 0.0
        return u, v


def _seed_inflow(u, w1_0, problem):
    """Semeia a face de entrada no campo inicial (evita transiente de partida
    com entrada nula)."""
    if problem.side("left").kind == "inflow":
        u = u.copy(); u[0, :] = w1_0[0, :]
    return u


def _sample_probe(u, v, problem, px, py):
    """v-velocidade interpolada no ponto (px,py) — sonda de esteira p/ Strouhal."""
    hx, hy = problem.hx, problem.hy
    j = min(int(py / hy), problem.Ny - 1)
    i = min(int(px / hx), problem.Nx - 1)
    return 0.5 * (v[i, j] + v[i, j + 1])


def run_flow(problem, Ndom, nloc, T, dt=None, method="mprm",
             degreeU=1, degreeP=1, alpha=1.0, basis_kind="legendre_P",
             eps_solid=1e-6, comm=None, verbose=True, log_every=200,
             frame_callback=None, frame_every=0, probes=None,
             obstacle_metrics=False, obstacle_metrics_every=1,
             U_ref=1.0, D=1.0):
    """Laço temporal genérico do MPrM para um FlowProblem. Cavidade e von
    Kármán compartilham este caminho. Devolve dict(u,v,dt,nsteps,divmax,
    defect,t_wall[,cl_history,probe_history])."""
    import time
    if comm is None:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD

    Nx, Ny = problem.Nx, problem.Ny
    hx, hy = problem.hx, problem.hy
    if dt is None:
        umax = max(U_ref, 1e-12)
        nu = max(problem.nu, 1e-12)
        h_min = min(hx, hy)
        dt = 0.4 * min(h_min / umax, h_min * h_min / (4.0 * nu))
    nsteps = int(np.ceil(T / dt))

    if method == "mprm":
        cfg = MRCMConfig(Ndomx=Ndom, Ndomy=Ndom, nlocx=nloc, nlocy=nloc,
                         degreeU=degreeU, degreeP=degreeP, alpha=alpha,
                         basis_kind=basis_kind, Lx=problem.Lx, Ly=problem.Ly,
                         mixed_family="BDM", mixed_order=0,
                         executor_kind="sequential")
        projector = MPrMProjector(cfg, dt, comm=comm,
                                  problem=problem, eps_solid=eps_solid)
        downscaler = MACDownscaler(cfg, dt)
    else:
        fine = ChannelFinePoisson(problem, dt)
        projector = None

    u = np.zeros((Nx + 1, Ny)); v = np.zeros((Nx, Ny + 1))
    # inicializa a entrada em w* (campo inicial = perfil de entrada nas faces)
    w1_0, w2_0 = mac_transport(u, v, problem, dt)
    u = _seed_inflow(u, w1_0, problem)

    cl_hist = []
    probe_hist = [] if probes else None

    # Cache da fatoração LU da pressão diagnóstica (T5)
    diag_solver = None
    if obstacle_metrics and comm.rank == 0:
        diag_solver = DiagnosticPressureSolver(problem)

    t0 = time.perf_counter()
    u_prev, v_prev = u.copy(), v.copy()  # para termo transiente nas forças
    for step in range(nsteps):
        w1, w2 = mac_transport(u, v, problem, dt)
        if method == "mprm":
            u, v = mprm_projection_step(w1, w2, projector, downscaler)
        else:
            u, v = fine.step(w1, w2)

        if comm.rank == 0:
            if obstacle_metrics and problem.solid is not None and step % obstacle_metrics_every == 0:
                p = diag_solver.solve(u, v)
                cd, cl = obstacle_forces(u, v, p, problem, U_ref, D,
                                         u_prev=u_prev, v_prev=v_prev, dt=dt)
                cl_hist.append((dt * (step + 1), cd, cl))
            if probes:
                probe_hist.append(
                    [dt * (step + 1)] + [_sample_probe(u, v, problem, px, py)
                                         for (px, py) in probes])
        if frame_every > 0 and frame_callback is not None and comm.rank == 0:
            if step == 0 or (step + 1) % frame_every == 0:
                frame_callback(step, dt * (step + 1), u, v, hx, hy)
        if verbose and comm.rank == 0 and (step + 1) % log_every == 0:
            dv = np.abs(mac_divergence(u, v, hx, hy))
            dv = dv[~problem.solid].max() if problem.solid is not None else dv.max()
            print(f"  passo {step+1}/{nsteps}  t={dt*(step+1):.3f}  "
                  f"max|div u| = {dv:.2e}", flush=True)
        u_prev, v_prev = u.copy(), v.copy()
    t_wall = time.perf_counter() - t0

    dvall = np.abs(mac_divergence(u, v, hx, hy))
    divmax = float(dvall[~problem.solid].max() if problem.solid is not None
                   else dvall.max())
    out = {"u": u, "v": v, "dt": dt, "nsteps": nsteps, "N": Nx,
           "t_wall": t_wall, "divmax": divmax,
           "defect": (downscaler.last_defect if method == "mprm" else 0.0)}
    if cl_hist:
        out["cl_history"] = np.array(cl_hist)     # colunas: t, Cd, Cl
    if probe_hist is not None:
        out["probe_history"] = np.array(probe_hist)
    return out


def run_cavity(Re=100.0, Ndom=3, nloc=16, T=8.0, dt=None, lid_velocity=1.0,
               degreeU=1, degreeP=1, alpha=1.0, method="mprm",
               basis_kind="nodal_Q",
               comm=None, verbose=True, log_every=200,
               frame_callback=None, frame_every=0):
    """Cavidade com tampa deslizante (wrapper sobre run_flow). Assinatura e
    retorno preservados para compatibilidade com testes/notebook."""
    N = Ndom * nloc
    problem = cavity_problem(N, Re, lid_velocity=lid_velocity)
    if dt is None:
        hx = 1.0 / N
        dt = 0.4 * min(hx / lid_velocity, hx * hx / (4.0 * (lid_velocity / Re)))
    return run_flow(problem, Ndom, nloc, T, dt=dt, method=method,
                    degreeU=degreeU, degreeP=degreeP, alpha=alpha,
                    basis_kind=basis_kind, comm=comm, verbose=verbose,
                    log_every=log_every, frame_callback=frame_callback,
                    frame_every=frame_every, U_ref=lid_velocity)


def run_von_karman(Lx, Ly, D, xc, yc, U_in=1.0, Re=100.0,
                   Ndom=None, nloc=None, degreeU=1, degreeP=3, alpha=1.0,
                   basis_kind="legendre_P", T=None, dt=None, method="mprm",
                   inflow="parabolic", walls="slip", inflow_norm="max",
                   eps_solid=1e-6,
                   comm=None, verbose=True, log_every=200,
                   probes=None, frame_callback=None, frame_every=0,
                   obstacle_metrics_every=1):
    """Escoamento em torno de um cilindro quadrado (von Kármán, artigo §3.5).

    Malha global Nx=Ndom_x*nloc, Ny=Ndom_y*nloc — aqui usamos Ndom (por
    direção) e nloc iguais; ajuste `nloc`/`Ndom` para caber o domínio
    retangular (Nx/Ny ≈ Lx/Ly). Espaços default (degreeU=1,degreeP=3)
    seguem a recomendação do artigo p/ escoamento transiente (pressão rica).
    Devolve o dict de run_flow + cl_history (t,Cd,Cl) e probe_history.
    """
    assert Ndom is not None and nloc is not None, "Ndom e nloc obrigatórios"
    Nx = Ny = Ndom * nloc
    # ajusta a razão de células ao domínio (mantém subdomínios quadrados em H)
    problem = von_karman_problem(Lx, Ly, Nx, Ny, D, xc, yc, U_in, Re,
                                 inflow=inflow, walls=walls,
                                 inflow_norm=inflow_norm)
    if T is None:
        T = 40.0 * D / U_in
    if probes is None:
        probes = [(xc + 3.0 * D, yc)]      # sonda de esteira default
    return run_flow(problem, Ndom, nloc, T, dt=dt, method=method,
                    degreeU=degreeU, degreeP=degreeP, alpha=alpha,
                    basis_kind=basis_kind, eps_solid=eps_solid, comm=comm,
                    verbose=verbose, log_every=log_every, probes=probes,
                    obstacle_metrics=True, obstacle_metrics_every=obstacle_metrics_every,
                    U_ref=U_in, D=D,
                    frame_callback=frame_callback, frame_every=frame_every)


# ===========================================================================
# Referência fina mascarada + diagnósticos (von Kármán)
# ===========================================================================

class ChannelFinePoisson:
    """Poisson MAC global mascarado para o passo de projeção de referência.

    Entrada/paredes/slip: Neumann homogêneo (v·n=0). Saída: Dirichlet p=0.
    Obstáculo: células sólidas excluídas (p=0), faces sólidas no-flow.
    Serve (a) como referência fina de von Kármán e (b) como fonte de pressão
    física para as forças.
    """

    def __init__(self, problem, dt):
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        self.problem = problem
        self.dt = float(dt)
        Nx, Ny = problem.Nx, problem.Ny
        hx, hy = problem.hx, problem.hy
        self.Nx, self.Ny, self.hx, self.hy = Nx, Ny, hx, hy
        solid = problem.solid if problem.solid is not None \
            else np.zeros((Nx, Ny), bool)
        self.solid = solid
        has_out = {s: problem.side(s).kind == "outflow"
                   for s in ("left", "right", "bottom", "top")}
        self.has_out = has_out

        def lid(i, j):
            return i + j * Nx

        rows, cols, vals = [], [], []
        for j in range(Ny):
            for i in range(Nx):
                c = lid(i, j)
                if solid[i, j]:
                    rows.append(c); cols.append(c); vals.append(1.0); continue
                diag = 0.0
                # vizinhos internos (fluido)
                for (di, dj, d, ln) in ((1, 0, hx, hy), (-1, 0, hx, hy),
                                        (0, 1, hy, hx), (0, -1, hy, hx)):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < Nx and 0 <= jj < Ny and not solid[ii, jj]:
                        w = dt * ln / d
                        rows.append(c); cols.append(lid(ii, jj))
                        vals.append(w); diag -= w
                # faces de SAÍDA (Dirichlet p=0): fantasma p=0 a meia célula
                for (side, at_edge, d, ln) in (
                        ("left", i == 0, hx, hy), ("right", i == Nx - 1, hx, hy),
                        ("bottom", j == 0, hy, hx), ("top", j == Ny - 1, hy, hx)):
                    if at_edge and has_out[side]:
                        diag -= dt * ln / (0.5 * d)     # p_ghost=0 (Dirichlet)
                rows.append(c); cols.append(c); vals.append(diag)
        A = sp.csr_matrix((vals, (rows, cols)), shape=(Nx * Ny, Nx * Ny)).tolil()
        self._pure_neumann = not any(has_out.values())
        if self._pure_neumann:
            pin = int(np.argmin(solid.flatten(order="F")))
            A[pin, :] = 0.0; A[pin, pin] = 1.0
            self._pin = pin
        self.lu = spla.splu(A.tocsc())

    def solve(self, w1, w2):
        dt, hx, hy = self.dt, self.hx, self.hy
        Nx, Ny, solid = self.Nx, self.Ny, self.solid
        divw = mac_divergence(w1, w2, hx, hy)
        divw = divw.copy(); divw[solid] = 0.0
        b = (divw * hx * hy).flatten(order="F")
        b[solid.flatten(order="F")] = 0.0
        if self._pure_neumann:
            fluid = ~solid.flatten(order="F")
            b[fluid] -= b[fluid].sum() / fluid.sum()
            b[self._pin] = 0.0
        p = self.lu.solve(b).reshape((Nx, Ny), order="F")
        p[solid] = 0.0
        u = w1.copy(); v = w2.copy()
        u[1:-1, :] -= dt * (p[1:, :] - p[:-1, :]) / hx
        v[:, 1:-1] -= dt * (p[:, 1:] - p[:, :-1]) / hy
        # saída: gradiente de p do lado da célula de borda; entrada/parede v·n
        _apply_external_normals_after_step(u, v, self.problem)
        fx, fy = solid_faces(solid)
        u[fx] = 0.0; v[fy] = 0.0
        return u, v, p

    def step(self, w1, w2):
        u, v, _ = self.solve(w1, w2)
        return u, v


class DiagnosticPressureSolver:
    """Pressão física via equação de Poisson: ∇²p = -∇·((u·∇)u).

    A matriz do sistema (geometria + BCs) é constante no tempo; montada e
    fatorada UMA vez no __init__. A cada passo, apenas a fonte (termo
    convectivo) é recomputada e o sistema resolvido com a fatoração cacheada.

    K=1 (Poisson padrão), Dirichlet p=0 na saída, Neumann homogêneo nas
    demais fronteiras.
    """

    def __init__(self, problem):
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        self.problem = problem
        hx, hy = problem.hx, problem.hy
        Nx, Ny = problem.Nx, problem.Ny
        self.hx, self.hy = hx, hy
        self.Nx, self.Ny = Nx, Ny
        solid = problem.solid if problem.solid is not None else np.zeros((Nx, Ny), bool)
        self.solid = solid

        has_out = {s: problem.side(s).kind == "outflow"
                   for s in ("left", "right", "bottom", "top")}

        def lid(i, j):
            return i + j * Nx

        rows, cols, vals = [], [], []
        for j in range(Ny):
            for i in range(Nx):
                c = lid(i, j)
                if solid[i, j]:
                    rows.append(c); cols.append(c); vals.append(1.0); continue
                diag = 0.0
                for (di, dj, d, ln) in ((1, 0, hx, hy), (-1, 0, hx, hy),
                                        (0, 1, hy, hx), (0, -1, hy, hx)):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < Nx and 0 <= jj < Ny and not solid[ii, jj]:
                        w = 1.0 * ln / d    # K=1
                        rows.append(c); cols.append(lid(ii, jj))
                        vals.append(w); diag -= w
                for (side, at_edge, d, ln) in (
                        ("left", i == 0, hx, hy), ("right", i == Nx-1, hx, hy),
                        ("bottom", j == 0, hy, hx), ("top", j == Ny-1, hy, hx)):
                    if at_edge and has_out[side]:
                        diag -= 1.0 * ln / (0.5 * d)
                rows.append(c); cols.append(c); vals.append(diag)

        A = sp.csr_matrix((vals, (rows, cols)), shape=(Nx * Ny, Nx * Ny)).tolil()
        pure_neumann = not any(has_out.values())
        if pure_neumann:
            pin = int(np.argmin(solid.flatten(order="F"))) if solid.any() else 0
            A[pin, :] = 0.0; A[pin, pin] = 1.0
            self._pin = pin
        self._pure_neumann = pure_neumann
        self.lu = spla.splu(A.tocsc())

    def solve(self, u, v):
        """Resolve ∇²p = -∇·((u·∇)u) com a fatoração cacheada."""
        hx, hy = self.hx, self.hy
        Nx, Ny = self.Nx, self.Ny
        solid = self.solid

        uc = 0.5 * (u[:-1, :] + u[1:, :])
        vc = 0.5 * (v[:, :-1] + v[:, 1:])

        dudx = np.zeros((Nx, Ny)); dudy = np.zeros((Nx, Ny))
        dvdx = np.zeros((Nx, Ny)); dvdy = np.zeros((Nx, Ny))
        dudx[1:-1, :] = (uc[2:, :] - uc[:-2, :]) / (2 * hx)
        dudx[0, :] = (uc[1, :] - uc[0, :]) / hx
        dudx[-1, :] = (uc[-1, :] - uc[-2, :]) / hx
        dudy[:, 1:-1] = (uc[:, 2:] - uc[:, :-2]) / (2 * hy)
        dudy[:, 0] = (uc[:, 1] - uc[:, 0]) / hy
        dudy[:, -1] = (uc[:, -1] - uc[:, -2]) / hy
        dvdx[1:-1, :] = (vc[2:, :] - vc[:-2, :]) / (2 * hx)
        dvdx[0, :] = (vc[1, :] - vc[0, :]) / hx
        dvdx[-1, :] = (vc[-1, :] - vc[-2, :]) / hx
        dvdy[:, 1:-1] = (vc[:, 2:] - vc[:, :-2]) / (2 * hy)
        dvdy[:, 0] = (vc[:, 1] - vc[:, 0]) / hy
        dvdy[:, -1] = (vc[:, -1] - vc[:, -2]) / hy

        conv_x = uc * dudx + vc * dudy
        conv_y = uc * dvdx + vc * dvdy

        source = np.zeros((Nx, Ny))
        source[1:-1, :] = -(conv_x[2:, :] - conv_x[:-2, :]) / (2 * hx)
        source[0, :] = -(conv_x[1, :] - conv_x[0, :]) / hx
        source[-1, :] = -(conv_x[-1, :] - conv_x[-2, :]) / hx
        source[:, 1:-1] += -(conv_y[:, 2:] - conv_y[:, :-2]) / (2 * hy)
        source[:, 0] += -(conv_y[:, 1] - conv_y[:, 0]) / hy
        source[:, -1] += -(conv_y[:, -1] - conv_y[:, -2]) / hy

        if solid is not None:
            source[solid] = 0.0

        b = (source * hx * hy).flatten(order="F")
        if solid is not None:
            b[solid.flatten(order="F")] = 0.0
        if self._pure_neumann:
            fluid = ~solid.flatten(order="F") if solid is not None else np.ones(Nx*Ny, bool)
            b[fluid] -= b[fluid].sum() / fluid.sum()
            b[self._pin] = 0.0

        p = self.lu.solve(b).reshape((Nx, Ny), order="F")
        if solid is not None:
            p[solid] = 0.0
        return p


def _diagnostic_pressure(u, v, problem, dt):
    """Wrapper retrocompatível — usa DiagnosticPressureSolver (refatora a
    cada chamada). Prefira instanciar DiagnosticPressureSolver e chamar
    .solve(u,v) para reutilizar a fatoração LU."""
    solver = DiagnosticPressureSolver(problem)
    return solver.solve(u, v)


def obstacle_forces(u, v, p, problem, U_ref, D, u_prev=None, v_prev=None, dt=None):
    """C_D, C_L por balanço de momento num volume de controle retangular que
    envolve o obstáculo (uma célula de folga). Inclui fluxo convectivo +
    pressão + VISCOSO (σ = -pI + ν(∇u+∇uᵀ)) nas 4 faces do VC e, se
    u_prev/dt fornecidos, o termo transiente d/dt ∫_VC u dV.
    ρ=1; C = 2F/(ρ U² D)."""
    solid = problem.solid
    if solid is None or not solid.any():
        return 0.0, 0.0
    hx, hy, nu = problem.hx, problem.hy, problem.nu
    Nx, Ny = problem.Nx, problem.Ny
    xs = np.where(solid.any(axis=1))[0]; ys = np.where(solid.any(axis=0))[0]
    i0, i1 = max(xs.min() - 1, 0), min(xs.max() + 1, Nx - 1)
    j0, j1 = max(ys.min() - 1, 0), min(ys.max() + 1, Ny - 1)

    uc = 0.5 * (u[:-1, :] + u[1:, :])
    vc = 0.5 * (v[:, :-1] + v[:, 1:])
    # gradientes centrados nos centros das células
    dudx = np.zeros((Nx, Ny)); dudy = np.zeros((Nx, Ny))
    dvdx = np.zeros((Nx, Ny)); dvdy = np.zeros((Nx, Ny))
    dudx[1:-1, :] = (uc[2:, :] - uc[:-2, :]) / (2 * hx)
    dudx[0, :] = (uc[1, :] - uc[0, :]) / hx
    dudx[-1, :] = (uc[-1, :] - uc[-2, :]) / hx
    dudy[:, 1:-1] = (uc[:, 2:] - uc[:, :-2]) / (2 * hy)
    dudy[:, 0] = (uc[:, 1] - uc[:, 0]) / hy
    dudy[:, -1] = (uc[:, -1] - uc[:, -2]) / hy
    dvdx[1:-1, :] = (vc[2:, :] - vc[:-2, :]) / (2 * hx)
    dvdx[0, :] = (vc[1, :] - vc[0, :]) / hx
    dvdx[-1, :] = (vc[-1, :] - vc[-2, :]) / hx
    dvdy[:, 1:-1] = (vc[:, 2:] - vc[:, :-2]) / (2 * hy)
    dvdy[:, 0] = (vc[:, 1] - vc[:, 0]) / hy
    dvdy[:, -1] = (vc[:, -1] - vc[:, -2]) / hy

    Fx = 0.0; Fy = 0.0
    # faces verticais — normal ±x
    for i, s in ((i0, -1.0), (i1 + 1, +1.0)):
        ii = min(max(i, 0), Nx)
        ic = min(max(i - (s > 0), 0), Nx - 1)
        un = u[ii, j0:j1 + 1]
        vc_f = 0.5 * (vc[min(ii, Nx-1), j0:j1+1] + vc[max(ii-1, 0), j0:j1+1])
        pf   = 0.5 * (p[min(ii, Nx-1), j0:j1+1] + p[max(ii-1, 0), j0:j1+1])
        visc_xx = 2.0 * nu * dudx[ic, j0:j1+1]
        visc_yx = nu * (dvdx[ic, j0:j1+1] + dudy[ic, j0:j1+1])
        Fx += -s * np.sum((un * un + pf - visc_xx) * hy)
        Fy += -s * np.sum((vc_f * un        - visc_yx) * hy)
    # faces horizontais — normal ±y
    for j, s in ((j0, -1.0), (j1 + 1, +1.0)):
        jj = min(max(j, 0), Ny)
        jc = min(max(j - (s > 0), 0), Ny - 1)
        vn = v[i0:i1 + 1, jj]
        uc_f = 0.5 * (uc[i0:i1+1, min(jj, Ny-1)] + uc[i0:i1+1, max(jj-1, 0)])
        pf   = 0.5 * (p[i0:i1+1, min(jj, Ny-1)] + p[i0:i1+1, max(jj-1, 0)])
        visc_xy = nu * (dudy[i0:i1+1, jc] + dvdx[i0:i1+1, jc])
        visc_yy = 2.0 * nu * dvdy[i0:i1+1, jc]
        Fx += -s * np.sum((uc_f * vn        - visc_xy) * hx)
        Fy += -s * np.sum((vn * vn + pf     - visc_yy) * hx)

    # termo transiente: d/dt ∫_VC u dV
    if u_prev is not None and dt is not None:
        upc = 0.5 * (u_prev[:-1, :] + u_prev[1:, :])
        vpc = 0.5 * (v_prev[:, :-1] + v_prev[:, 1:])
        dVol = hx * hy
        Fx -= (uc[i0:i1+1, j0:j1+1].sum() - upc[i0:i1+1, j0:j1+1].sum()) * dVol / dt
        Fy -= (vc[i0:i1+1, j0:j1+1].sum() - vpc[i0:i1+1, j0:j1+1].sum()) * dVol / dt

    q = 0.5 * U_ref * U_ref * D
    return Fx / q, Fy / q


def strouhal(signal, dt, U_ref, D, t_skip_frac=0.5, pad=8):
    """St = f·D/U da frequência dominante do sinal.

    Zero-padding (pad=8) refina os bins da FFT para resolução sub-bin.
    Interpolação parabólica de 3 pontos no pico para precisão adicional.
    Descartam-se os primeiros t_skip_frac do sinal (transiente).
    """
    sig = np.asarray(signal, float)
    n0 = int(len(sig) * t_skip_frac)
    s = sig[n0:]
    s = s - s.mean()
    if s.size < 16:
        return float("nan")
    s = s * np.hanning(s.size)
    n = int(2 ** np.ceil(np.log2(s.size * pad)))        # zero-padding
    amp = np.abs(np.fft.rfft(s, n=n))
    amp[0] = 0.0
    freqs = np.fft.rfftfreq(n, d=dt)
    k = int(np.argmax(amp))
    # Interpolação parabólica de 3 pontos para refinar o pico
    if 0 < k < len(amp) - 1:
        a, b, c = amp[k - 1], amp[k], amp[k + 1]
        denom = a - 2 * b + c
        if abs(denom) > 1e-30:
            k = k + 0.5 * (a - c) / denom
    f_dom = k * (freqs[1] - freqs[0]) if k > 0 else freqs[1]
    return float(f_dom * D / U_ref)


def make_frame_saver(output_dir="frames_mprm", dpi=100, Re=100.0, title_prefix="MPrM"):
    """Factory: retorna um callback que salva frames PNG da cavidade.

    Uso:
        saver = make_frame_saver("frames", Re=100)
        result = run_cavity(..., frame_callback=saver, frame_every=50)
        # Depois: make_gif("frames", "cavity.gif")

    Cada frame mostra magnitude |u| + quiver + linha do tempo.
    """
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    frame_idx = [0]  # mutable counter

    def _save(step, t, u, v, hx, hy):
        N = u.shape[0] - 1
        xc = (np.arange(N) + 0.5) * hx
        yc = (np.arange(N) + 0.5) * hy
        uc = 0.5 * (u[:-1, :] + u[1:, :])
        vc = 0.5 * (v[:, :-1] + v[:, 1:])
        mag = np.sqrt(uc**2 + vc**2)

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.pcolormesh(xc, yc, mag.T, shading='auto', cmap='turbo', vmin=0, vmax=1.0)
        skip = max(1, N // 25)
        ax.quiver(xc[::skip], yc[::skip],
                  uc[::skip, ::skip].T, vc[::skip, ::skip].T,
                  scale=4, width=0.002, color='white', alpha=0.7)
        ax.set_title(f"{title_prefix} — Cavidade Re={Re:.0f}  "
                     f"t = {t:.3f}")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='|u|', shrink=0.8)
        fname = os.path.join(output_dir, f"frame_{frame_idx[0]:05d}.png")
        fig.savefig(fname, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        frame_idx[0] += 1

    return _save


def make_gif(frame_dir="frames_mprm", output="cavity_mprm.gif",
             fps=15, duration=None):
    """Cria um GIF animado a partir dos frames PNG salvos.

    Requer: pip install imageio  (ou Pillow)

    Parâmetros:
      frame_dir: diretório com frame_*.png
      output:    arquivo de saída (.gif ou .mp4)
      fps:       quadros por segundo
      duration:  duração por frame em ms (alternativa a fps)
    """
    import os
    import glob

    files = sorted(glob.glob(os.path.join(frame_dir, "frame_*.png")))
    if not files:
        raise FileNotFoundError(f"Nenhum frame encontrado em {frame_dir}")

    if duration is None:
        duration = 1000.0 / fps

    try:
        import imageio.v2 as imageio
        frames = [imageio.imread(f) for f in files]
        imageio.mimsave(output, frames, duration=duration, loop=0)
        print(f"✓ GIF salvo: {output} ({len(frames)} frames)")
        return output
    except ImportError:
        # fallback: Pillow
        try:
            from PIL import Image
            frames = [Image.open(f) for f in files]
            frames[0].save(
                output, save_all=True, append_images=frames[1:],
                duration=duration, loop=0,
            )
            print(f"✓ GIF salvo (Pillow): {output} ({len(frames)} frames)")
            return output
        except ImportError:
            raise ImportError(
                "Instale imageio ou Pillow: pip install imageio"
            )


# ===========================================================================
# Benchmark analítico da decomposição de Helmholtz-Hodge (artigo §3.1)
# ===========================================================================

A_OLSH = 0.1  # parâmetro do campo de Olshanskii et al. (Eq. 40 do artigo)


def olshanskii_u_np(x, y, a=A_OLSH):
    """Campo solenoidal analítico (Eq. 40), tangente ao contorno (numpy)."""
    E = np.e
    th1 = 2 * np.pi * (np.exp(x) - 1.0) / (E - 1.0)
    th2 = 2 * np.pi * (np.exp(a * y) - 1.0) / (E ** a - 1.0)
    u1 = (a * np.exp(a * y) / (2 * np.pi * (E ** a - 1.0))) \
        * (1.0 - np.cos(th1)) * np.sin(th2)
    u2 = (-np.exp(x) / (2 * np.pi * (E - 1.0))) \
        * (1.0 - np.cos(th2)) * np.sin(th1)
    return u1, u2


def hh_p_np(x, y):
    """Potencial p̃ = cos(πx)cos(πy): ∇p̃·n = 0 no contorno (compatível com
    o caminho impermeável/no-flow forte do mrcm — ver notebook §2)."""
    return np.cos(np.pi * x) * np.cos(np.pi * y)


def hh_gradp_np(x, y):
    gx = -np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)
    gy = -np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
    return gx, gy


def hh_build_w(N):
    """Amostra o campo do benchmark HH nas faces MAC (malha N×N).

    Retorna (w1, w2, u1, u2): w = u + ∇p̃ e o campo solenoidal analítico u
    nas faces verticais/horizontais, com w·n = 0 imposto no contorno.
    """
    h = 1.0 / N
    xf = np.linspace(0.0, 1.0, N + 1)      # faces verticais
    yc = (np.arange(N) + 0.5) * h
    Xu, Yu = np.meshgrid(xf, yc, indexing="ij")
    xc = (np.arange(N) + 0.5) * h
    yf = np.linspace(0.0, 1.0, N + 1)
    Xv, Yv = np.meshgrid(xc, yf, indexing="ij")

    u1, _ = olshanskii_u_np(Xu, Yu)
    _, u2 = olshanskii_u_np(Xv, Yv)
    g1, _ = hh_gradp_np(Xu, Yu)
    _, g2 = hh_gradp_np(Xv, Yv)
    w1, w2 = u1 + g1, u2 + g2
    w1[0, :] = w1[-1, :] = 0.0             # w·n = 0 (exato para p̃; u é tangente)
    w2[:, 0] = w2[:, -1] = 0.0
    return w1, w2, u1, u2


def hh_benchmark_mac(Ndom=3, nloc=30, degreeU=1, degreeP=1, alpha=1.0,
                     comm=None):
    """Decomposição HH modificada via pipeline MPrM completo na malha MAC.

    Amostra w = u + ∇p̃ nas faces MAC, projeta com o MRCM (K=1, δt=1),
    aplica o downscaling e compara u_h = w − ṽ com o u analítico.
    Retorna dict(err_u, divmax, defect).
    """
    if comm is None:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    N = Ndom * nloc
    h = 1.0 / N
    dt = 1.0

    w1, w2, u1, u2 = hh_build_w(N)

    cfg = MRCMConfig(Ndomx=Ndom, Ndomy=Ndom, nlocx=nloc, nlocy=nloc,
                     degreeU=degreeU, degreeP=degreeP, alpha=alpha,
                     mixed_family="BDM", mixed_order=0,
                     executor_kind="sequential")
    projector = MPrMProjector(cfg, dt, comm=comm)
    downscaler = MACDownscaler(cfg, dt)
    uh1, uh2 = mprm_projection_step(w1, w2, projector, downscaler)

    num = np.sum((uh1 - u1) ** 2) + np.sum((uh2 - u2) ** 2)
    den = np.sum(u1 ** 2) + np.sum(u2 ** 2)
    return {
        "err_u": float(np.sqrt(num / den)),
        "divmax": float(np.abs(mac_divergence(uh1, uh2, h, h)).max()),
        "defect": downscaler.last_defect,
        "u": (uh1, uh2), "u_exact": (u1, u2), "w": (w1, w2), "N": N,
    }
