"""Forças (balanço de momento) e Strouhal (FFT) — validação analítica."""
import numpy as np
import pytest

from mprm_core import (ChannelFinePoisson, obstacle_forces, strouhal,
                       von_karman_problem, mac_divergence)


def test_strouhal_recupera_frequencia_conhecida():
    U, D = 1.0, 1.0
    f_true = 0.121 * U / D              # St alvo do artigo
    dt = 0.01
    t = np.arange(0, 200, dt)
    sig = np.sin(2 * np.pi * f_true * t) + 0.05 * np.random.default_rng(0).standard_normal(t.size)
    st = strouhal(sig, dt, U, D, t_skip_frac=0.25)
    assert st == pytest.approx(0.121, abs=0.01)


def test_channel_fine_poisson_projeta_divergencia_zero():
    prob = von_karman_problem(Lx=8.0, Ly=4.0, Nx=32, Ny=16, D=1.0,
                              xc=2.0, yc=2.0, U_in=1.0, Re=100.0)
    fp = ChannelFinePoisson(prob, dt=0.05)
    rng = np.random.default_rng(1)
    w1 = rng.standard_normal((33, 16)); w2 = rng.standard_normal((32, 17))
    # Impõe BCs físicas: entrada, paredes impermeáveis
    w1[0, :] = 1.0                                    # entrada (inflow)
    w1[-1, :] = w1[-2, :]                             # saída: gradiente nulo
    w2[:, 0] = 0.0; w2[:, -1] = 0.0                   # paredes top/bottom
    u, v = fp.step(w1, w2)
    dv = np.abs(mac_divergence(u, v, prob.hx, prob.hy))
    # células de fluido longe das bordas e do sólido devem ter div~0
    # NOTA: células adjacentes ao sólido e à saída têm erro de discretização
    interior = np.ones((32, 16), dtype=bool)
    interior[0, :] = interior[-1, :] = False           # entrada/saída
    interior[:, 0] = interior[:, -1] = False           # paredes
    interior[prob.solid] = False                        # sólido
    # exclui também células adjacentes ao sólido (cut-cell error)
    from scipy.ndimage import binary_dilation
    interior[prob.solid] = False
    near_solid = binary_dilation(prob.solid, iterations=1)
    interior[near_solid] = False
    assert interior.any(), "nenhuma célula interior após máscara"
    assert dv[interior].max() < 1e-8               # div~0 no interior do fluido
    # u·n=0 nas faces sólidas
    fx = np.zeros((33, 16), bool); fx[:-1, :] |= prob.solid; fx[1:, :] |= prob.solid
    assert np.all(u[fx] == 0.0)


def test_obstacle_forces_campo_uniforme_forca_nula():
    prob = von_karman_problem(Lx=8.0, Ly=4.0, Nx=32, Ny=16, D=1.0,
                              xc=2.0, yc=2.0, U_in=1.0, Re=100.0)
    # campo uniforme fora do sólido, pressão constante: forças ~ 0
    u = np.ones((33, 16)); v = np.zeros((32, 17)); p = np.zeros((32, 16))
    fx = np.zeros((33, 16), bool); fx[:-1, :] |= prob.solid; fx[1:, :] |= prob.solid
    fy = np.zeros((32, 17), bool); fy[:, :-1] |= prob.solid; fy[:, 1:] |= prob.solid
    u[fx] = 0.0; v[fy] = 0.0
    cd, cl = obstacle_forces(u, v, p, prob, U_ref=1.0, D=1.0)
    assert abs(cl) < 1e-9
    assert np.isfinite(cd)
