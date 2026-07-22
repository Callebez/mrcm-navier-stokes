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
                       inflow="uniform", walls="wall",
                       outflow="pressure") -> FlowProblem:
    """Escoamento em torno de um cilindro quadrado (von Kármán). Re = U_in*D/nu.

    inflow: "uniform" (u=U_in) ou "parabolic" (perfil de canal, média U_in).
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
        def profile(y):                       # média = U_in
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

    def __init__(self, cfg: MRCMConfig, dt: float, comm=None, verbose=False):
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

        self.skel = Skeleton.cartesian(cfg.Ndomx, cfg.Ndomy)
        self.dofm = DOFManager(self.skel.NGamma, cfg.ndofsU, cfg.ndofsP)
        self.my_subs = list(range(self.rank, cfg.Nsubd, self.size))

        # Problema de Darcy da projeção: K = δt, fonte preenchida online,
        # contorno impermeável (fluxo nulo ⇒ pino forte, ver apply_bc_patch).
        dt_c = self.dt

        def _K(mesh):
            from firedrake import Constant
            return Constant(dt_c)

        def _invK(mesh):
            from firedrake import Constant
            return Constant(1.0 / dt_c)

        def _zero(x, y):
            from firedrake import Constant
            return Constant(0.0)

        def _noflow(x, y):
            from firedrake import Constant
            return Constant(0.0)

        self.problem = DarcyProblem(
            Lx=cfg.Lx, Ly=cfg.Ly,
            permeability=_K, inv_permeability=_invK,
            source=_zero,
            boundary={s: BoundaryData(flux=_noflow)
                      for s in ("left", "right", "bottom", "top")},
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
        self.uh = {}          # isub -> Function(V) para extração de traço

        hx_f = cfg.Hx / cfg.nlocx
        hy_f = cfg.Hy / cfg.nlocy
        self.hx_f, self.hy_f = hx_f, hy_f

        for isub in self.my_subs:
            ix, iy = isub % cfg.Ndomx, isub // cfg.Ndomx
            flag_side, internal_sides = mark_sides(ix, iy, cfg.Ndomx, cfg.Ndomy)
            edge_l = {s: int(self.skel.ieGt[self.skel.el2edge[isub, s]])
                      for s in internal_sides}
            task = {"cfg_dict": cfg.as_dict(), "isub": isub,
                    "NunkU": NunkU, "edge_l": edge_l,
                    "problem": self.problem}
            # Offline: MsBF + caches B1/B2 (fonte nula — descartamos a
            # parte particular do workspace; a online é resolvida abaixo).
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
            bcs = _build_flux_dirichlet_bcs_allsides(W, flag_side,
                                                     problem=self.problem)
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
            # Q é DG_{mixed_order}; o mapa por ponto médio vale para DG0
            # (mixed_order=0, caso usado aqui).
            assert cfg.mixed_order == 0, "fonte por célula requer DG0"
            ic = np.floor(xcd[:, 0] / hx_f).astype(int)
            jc = np.floor(xcd[:, 1] / hy_f).astype(int)
            self.cell_map[isub] = (np.clip(ic, 0, cfg.Nglobx - 1),
                                   np.clip(jc, 0, cfg.Ngloby - 1))

            # Operadores de traço: fluxo médio por face fina de cada lado
            # interno (base da média da Eq. 21). Trace space HDivT grau 0:
            # 1 dof (momento ∫u·n) por faceta.
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
                # ordena as facetas ao longo do lado
                coord = ym if s in (1, 3) else xm     # lados verticais: y
                order = np.argsort(coord)
                sign = +1.0 if s in (1, 2) else -1.0  # right/top: n = +e
                self.trace[(isub, s)] = {
                    "form": dot(uh, n_facet) * tau * ds_m(marker),
                    "idx": idx, "len": ln[idx], "order": order, "sign": sign,
                }

        # Montador global (matriz de interface a partir de triplets — o
        # mesmo caminho do pipeline MPI do mrcm).
        self.assembler = None
        if self.rank == 0:
            self.assembler = GlobalAssembler(
                cfg, self.skel, self.dofm, ispace, robin,
                basis_per_sides=None, n_facets=None, dss=None, meshes=None,
                Vs=None, Qs=None, face_mass={}, cofunc_flux={}, cofunc_pres={})

        self.last_X = None
        self.last_pT = {}
        self.last_uT = {}

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

def mprm_projection_step(w1, w2, projector: MPrMProjector,
                         downscaler: MACDownscaler):
    """Passos 2-4 do MPrM: projeção MRCM + downscaling + uⁿ⁺¹ = w* − ṽ.

    Recebe w* na malha MAC global; devolve (u, v) com ∇·u = 0 célula a
    célula (até o defeito de compatibilidade reportado pelo downscaler).
    Coletivo sobre projector.comm (todos os ranks retornam o campo global).
    """
    cfg = projector.cfg
    comm = projector.comm
    n, m = cfg.nlocx, cfg.nlocy
    hx, hy = downscaler.hx, downscaler.hy

    divw = mac_divergence(w1, w2, hx, hy)
    fluxes = projector.project(divw)

    u = w1.copy()
    v = w2.copy()
    zeros_m, zeros_n = np.zeros(m), np.zeros(n)
    my_patch = {}
    for isub in projector.my_subs:
        ix, iy = isub % cfg.Ndomx, isub // cfg.Ndomx
        ox, oy = ix * n, iy * m
        fW = fluxes.get(("v", ix, iy), zeros_m)
        fE = fluxes.get(("v", ix + 1, iy), zeros_m)
        fS = fluxes.get(("h", ix, iy), zeros_n)
        fN = fluxes.get(("h", ix, iy + 1), zeros_n)
        vt1, vt2 = downscaler.solve_patch(
            divw[ox:ox + n, oy:oy + m], fW, fE, fS, fN)
        my_patch[isub] = (vt1, vt2)

    for d in comm.allgather(my_patch):
        for isub, (vt1, vt2) in d.items():
            ix, iy = isub % cfg.Ndomx, isub // cfg.Ndomx
            ox, oy = ix * n, iy * m
            u[ox:ox + n + 1, oy:oy + m] = w1[ox:ox + n + 1, oy:oy + m] - vt1
            v[ox:ox + n, oy:oy + m + 1] = w2[ox:ox + n, oy:oy + m + 1] - vt2
    # paredes impermeáveis (interfaces internas foram escritas 2×, com o
    # MESMO valor médio — consistência garantida pela Eq. 21)
    u[0, :] = u[-1, :] = 0.0
    v[:, 0] = v[:, -1] = 0.0
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


def run_cavity(Re=100.0, Ndom=3, nloc=16, T=8.0, dt=None, lid_velocity=1.0,
               degreeU=1, degreeP=1, alpha=1.0, method="mprm",
               basis_kind="nodal_Q",
               comm=None, verbose=True, log_every=200,
               frame_callback=None, frame_every=0):
    """Cavidade com tampa deslizante (artigo §3.3) integrada até t=T.

    method: "mprm" (projeção MRCM + downscaling) ou "fine" (referência de
    malha fina, sem decomposição). Malha global N = Ndom*nloc por direção.

    Parâmetros de animação:
      frame_callback: callable(step, t, u, v, hx, hy) chamado a cada
                      `frame_every` passos no rank 0. Use para salvar frames.
      frame_every:    intervalo (em passos) entre chamadas do callback.
                      0 = desabilitado.
    Retorna dict(u, v, dt, nsteps, t_wall, divmax, defect).
    """
    import time
    if comm is None:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD

    N = Ndom * nloc
    hx = hy = 1.0 / N
    nu = lid_velocity / Re
    if dt is None:
        dt = 0.4 * min(hx / lid_velocity, hx * hx / (4.0 * nu))
    nsteps = int(np.ceil(T / dt))

    if method == "mprm":
        # NOTA (D3/π₀): com graus assimétricos (degreeP < degreeU) use uma
        # família hierárquica ("legendre_P"/"monomial_P") — o nodal_Q trunca
        # a base de Lagrange do grau máximo e P_H perde a constante π₀,
        # quebrando o balanço de massa da média de fluxos (defeito ~1e-3).
        cfg = MRCMConfig(Ndomx=Ndom, Ndomy=Ndom, nlocx=nloc, nlocy=nloc,
                         degreeU=degreeU, degreeP=degreeP, alpha=alpha,
                         basis_kind=basis_kind,
                         mixed_family="BDM", mixed_order=0,
                         executor_kind="sequential")
        projector = MPrMProjector(cfg, dt, comm=comm)
        downscaler = MACDownscaler(cfg, dt)
    else:
        fine = FineProjector(N, N, hx, hy, dt)

    u = np.zeros((N + 1, N))
    v = np.zeros((N, N + 1))
    t0 = time.perf_counter()
    for step in range(nsteps):
        w1, w2 = mac_wstar(u, v, hx, hy, dt, nu, lid_velocity)
        if method == "mprm":
            u, v = mprm_projection_step(w1, w2, projector, downscaler)
        else:
            u, v = fine.step(w1, w2)

        # --- callback de frame ---
        if frame_every > 0 and frame_callback is not None and comm.rank == 0:
            if step == 0 or (step + 1) % frame_every == 0:
                frame_callback(step, dt * (step + 1), u, v, hx, hy)

        if verbose and comm.rank == 0 and (step + 1) % log_every == 0:
            dv = np.abs(mac_divergence(u, v, hx, hy)).max()
            print(f"  passo {step+1}/{nsteps}  t={dt*(step+1):.3f}  "
                  f"max|div u| = {dv:.2e}", flush=True)
    t_wall = time.perf_counter() - t0

    return {
        "u": u, "v": v, "dt": dt, "nsteps": nsteps, "N": N,
        "t_wall": t_wall,
        "divmax": float(np.abs(mac_divergence(u, v, hx, hy)).max()),
        "defect": (projector and downscaler.last_defect) if method == "mprm"
                  else 0.0,
    }


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
