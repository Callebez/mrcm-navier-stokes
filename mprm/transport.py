"""Transporte explícito na malha MAC — MPrM.

MAC staggered-grid transport: mac_transport, mac_wstar, mac_divergence.
Puro NumPy — não importa Firedrake.
"""
from __future__ import annotations
import numpy as np
from mprm.problem import SideBC, FlowProblem


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


def _minmod(a, b):
    """Minmod limiter: 0 se sinais opostos, senão menor em magnitude."""
    return np.where(a * b > 0, np.where(np.abs(a) < np.abs(b), a, b), 0.0)


def _superbee(a, b):
    """Superbee limiter: maximiza o gradiente dentro do regime TVD."""
    s = np.where(a * b > 0, np.where(np.abs(a) < np.abs(b), a, b), 0.0)
    return np.where(np.abs(a) < np.abs(b),
                    np.where(2*np.abs(a) < np.abs(b), 2*a, b),
                    np.where(2*np.abs(b) < np.abs(a), 2*b, a))


def _tvdupx(phi, vel_sign, limiter="minmod"):
    """Interpolação TVD 2ª ordem na direção x. phi shape (Nx+1, Ny)."""
    _lim = _superbee if limiter == "superbee" else _minmod
    result = np.where(vel_sign >= 0, phi[:-1, :], phi[1:, :])  # fallback 1ª ordem
    pos = vel_sign >= 0
    slope = _lim(phi[1:-1, :] - phi[:-2, :], phi[2:, :] - phi[1:-1, :])
    result[1:, :] = np.where(pos[1:, :], phi[1:-1, :] + 0.5 * slope, result[1:, :])
    neg = ~pos
    result[:-1, :] = np.where(neg[:-1, :], phi[1:-1, :] - 0.5 * slope, result[:-1, :])
    return result


def _tvdupy(phi, vel_sign, limiter="minmod"):
    """Interpolação TVD 2ª ordem na direção y. phi shape (Nx, Ny+1)."""
    _lim = _superbee if limiter == "superbee" else _minmod
    result = np.where(vel_sign >= 0, phi[:, :-1], phi[:, 1:])  # fallback 1ª ordem
    pos = vel_sign >= 0
    slope = _lim(phi[:, 1:-1] - phi[:, :-2], phi[:, 2:] - phi[:, 1:-1])
    result[:, 1:] = np.where(pos[:, 1:], phi[:, 1:-1] + 0.5 * slope, result[:, 1:])
    neg = ~pos
    result[:, :-1] = np.where(neg[:, :-1], phi[:, 1:-1] - 0.5 * slope, result[:, :-1])
    return result


def mac_transport(u, v, problem: "FlowProblem", dt, convective_order=2, limiter="minmod"):
    """Velocidade tentativa w* = uⁿ + δt(−∇·(uu) + ν∇²u + f) na malha MAC,
    para um `FlowProblem` arbitrário (contorno + obstáculo).

    Convecção upwind (donor-cell) + difusão central (§2.6 do artigo). A
    cavidade é o caso `cavity_problem(...)` e reproduz o estêncil antigo
    bit-a-bit. Entrada/paredes/obstáculo tratados via fantasmas + máscara
    de faces sólidas.

    Parameters
    ----------
    convective_order : int
        1 = upwind 1ª ordem; 2 = TVD 2ª ordem.
    limiter : str
        \"minmod\" (mais dissipativo) ou \"superbee\" (menos dissipativo).
        Só usado se convective_order >= 2.
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
    order = convective_order

    # ---- componente x (faces internas i=1..Nx-1) ----
    ue = 0.5 * (u[1:, :] + u[:-1, :])
    u_face = _tvdupx(u, ue, limiter) if order >= 2 else np.where(ue >= 0.0, u[:-1, :], u[1:, :])
    Fuu = u_face * ue
    dudx_conv = (Fuu[1:, :] - Fuu[:-1, :]) / hx
    v_at_corner = 0.5 * (vg[1:-2, :] + vg[2:-1, :])
    u_corner = _tvdupy(ug[1:-1, :], v_at_corner, limiter) if order >= 2 else np.where(v_at_corner >= 0.0, ug[1:-1, :-1], ug[1:-1, 1:])
    Fuv = u_corner * v_at_corner
    duvdy = (Fuv[:, 1:] - Fuv[:, :-1]) / hy
    lap_u = ((ug[2:, 1:-1] - 2.0 * ug[1:-1, 1:-1] + ug[:-2, 1:-1]) / hx**2
             + (ug[1:-1, 2:] - 2.0 * ug[1:-1, 1:-1] + ug[1:-1, :-2]) / hy**2)
    fbx = 0.0 if problem.force is None else problem.force[0]
    w1[1:-1, :] = u[1:-1, :] + dt * (-dudx_conv - duvdy + nu * lap_u + fbx)

    # ---- componente y (faces internas j=1..Ny-1) ----
    vc = 0.5 * (v[:, 1:] + v[:, :-1])
    v_face = _tvdupy(v, vc, limiter) if order >= 2 else np.where(vc >= 0.0, v[:, :-1], v[:, 1:])
    Fvv = v_face * vc
    dvdy_conv = (Fvv[:, 1:] - Fvv[:, :-1]) / hy
    u_at_corner = 0.5 * (ug[:, 1:-2] + ug[:, 2:-1])
    v_corner = _tvdupx(vg[:, 1:-1], u_at_corner, limiter) if order >= 2 else np.where(u_at_corner >= 0.0, vg[:-1, 1:-1], vg[1:, 1:-1])
    Fvu = v_corner * u_at_corner
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
    return mac_transport(u, v, prob, dt, convective_order=1)


def mac_divergence(w1, w2, hx, hy):
    """∇·w por célula (FV exato na malha MAC): (Nx, Ny)."""
    return (w1[1:, :] - w1[:-1, :]) / hx + (w2[:, 1:] - w2[:, :-1]) / hy


# ===========================================================================
