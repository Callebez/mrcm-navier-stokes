"""Transporte MAC generalizado: caracterização da cavidade (bit-idêntico ao
estêncil antigo) + comportamento de entrada/obstáculo. Puro NumPy."""
import numpy as np
import pytest

from mprm_core import mac_transport, mac_wstar, cavity_problem, \
    von_karman_problem, solid_faces


def _wstar_legacy(u, v, hx, hy, dt, nu, lid_velocity=1.0):
    """Cópia FIEL do estêncil da cavidade pré-refactor (referência dourada)."""
    Nx = u.shape[0] - 1
    Ny = v.shape[1] - 1
    ug = np.empty((Nx + 1, Ny + 2)); ug[:, 1:-1] = u
    ug[:, 0] = -u[:, 0]; ug[:, -1] = 2.0 * lid_velocity - u[:, -1]
    vg = np.empty((Nx + 2, Ny + 1)); vg[1:-1, :] = v
    vg[0, :] = -v[0, :]; vg[-1, :] = -v[-1, :]
    w1 = u.copy(); w2 = v.copy()
    ue = 0.5 * (u[1:, :] + u[:-1, :])
    Fuu = np.where(ue >= 0.0, u[:-1, :], u[1:, :]) * ue
    dudx_conv = (Fuu[1:, :] - Fuu[:-1, :]) / hx
    v_at_corner = 0.5 * (vg[1:-2, :] + vg[2:-1, :])
    u_upw_y = np.where(v_at_corner >= 0.0, ug[1:-1, :-1], ug[1:-1, 1:])
    Fuv = u_upw_y * v_at_corner
    duvdy = (Fuv[:, 1:] - Fuv[:, :-1]) / hy
    lap_u = ((ug[2:, 1:-1] - 2.0 * ug[1:-1, 1:-1] + ug[:-2, 1:-1]) / hx**2
             + (ug[1:-1, 2:] - 2.0 * ug[1:-1, 1:-1] + ug[1:-1, :-2]) / hy**2)
    w1[1:-1, :] = u[1:-1, :] + dt * (-dudx_conv - duvdy + nu * lap_u)
    vc = 0.5 * (v[:, 1:] + v[:, :-1])
    Fvv = np.where(vc >= 0.0, v[:, :-1], v[:, 1:]) * vc
    dvdy_conv = (Fvv[:, 1:] - Fvv[:, :-1]) / hy
    u_at_corner = 0.5 * (ug[:, 1:-2] + ug[:, 2:-1])
    v_upw_x = np.where(u_at_corner >= 0.0, vg[:-1, 1:-1], vg[1:, 1:-1])
    Fvu = v_upw_x * u_at_corner
    duvdx = (Fvu[1:, :] - Fvu[:-1, :]) / hx
    lap_v = ((vg[2:, 1:-1] - 2.0 * vg[1:-1, 1:-1] + vg[:-2, 1:-1]) / hx**2
             + (vg[1:-1, 2:] - 2.0 * vg[1:-1, 1:-1] + vg[1:-1, :-2]) / hy**2)
    w2[:, 1:-1] = v[:, 1:-1] + dt * (-duvdx - dvdy_conv + nu * lap_v)
    return w1, w2


def test_cavity_transporte_bit_identico_ao_legado():
    rng = np.random.default_rng(0)
    N = 16
    u = rng.standard_normal((N + 1, N)); u[0, :] = u[-1, :] = 0.0
    v = rng.standard_normal((N, N + 1)); v[:, 0] = v[:, -1] = 0.0
    hx = hy = 1.0 / N; dt = 1e-3; nu = 1e-2; lid = 1.3
    w1a, w2a = _wstar_legacy(u, v, hx, hy, dt, nu, lid)
    prob = cavity_problem(N, Re=lid / nu, lid_velocity=lid)
    w1b, w2b = mac_transport(u, v, prob, dt)
    assert np.allclose(w1a, w1b, atol=1e-14, rtol=0)
    assert np.allclose(w2a, w2b, atol=1e-14, rtol=0)


def test_wrapper_mac_wstar_igual_a_transport():
    rng = np.random.default_rng(1)
    N = 8
    u = rng.standard_normal((N + 1, N)); u[0, :] = u[-1, :] = 0.0
    v = rng.standard_normal((N, N + 1)); v[:, 0] = v[:, -1] = 0.0
    w1a, w2a = mac_wstar(u, v, 1.0 / N, 1.0 / N, 1e-3, 1e-2, 1.0)
    w1b, w2b = _wstar_legacy(u, v, 1.0 / N, 1.0 / N, 1e-3, 1e-2, 1.0)
    assert np.allclose(w1a, w1b, atol=1e-14)
    assert np.allclose(w2a, w2b, atol=1e-14)


def test_inflow_carrega_perfil_na_face_esquerda():
    prob = von_karman_problem(Lx=4.0, Ly=2.0, Nx=40, Ny=20, D=0.5,
                              xc=1.0, yc=1.0, U_in=1.0, Re=100.0)
    u = np.zeros((41, 20)); v = np.zeros((40, 21))
    w1, w2 = mac_transport(u, v, prob, dt=1e-3)
    # face de entrada (x=0) recebe U_in=1 (exceto onde há sólido — aqui não há)
    assert np.allclose(w1[0, :], 1.0)


def test_obstaculo_zera_faces_solidas():
    prob = von_karman_problem(Lx=4.0, Ly=2.0, Nx=40, Ny=20, D=0.5,
                              xc=1.0, yc=1.0, U_in=1.0, Re=100.0)
    rng = np.random.default_rng(2)
    u = rng.standard_normal((41, 20)); v = rng.standard_normal((40, 21))
    w1, w2 = mac_transport(u, v, prob, dt=1e-3)
    fx, fy = solid_faces(prob.solid)
    assert np.all(w1[fx] == 0.0)
    assert np.all(w2[fy] == 0.0)
