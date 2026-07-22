"""Downscaler com máscara de obstáculo (puro scipy/NumPy)."""
import numpy as np
import pytest

from mrcm.config import MRCMConfig
from mprm_core import MACDownscaler


def _cfg(n=8):
    return MRCMConfig(Ndomx=1, Ndomy=1, nlocx=n, nlocy=n,
                      degreeU=1, degreeP=1, alpha=1.0,
                      mixed_family="BDM", mixed_order=0,
                      executor_kind="sequential")


def test_sem_solido_reproduz_comportamento_atual():
    cfg = _cfg(8); dt = 0.5
    ds = MACDownscaler(cfg, dt)
    rng = np.random.default_rng(0)
    divw = rng.standard_normal((8, 8)); divw -= divw.mean()
    z = np.zeros(8)
    vt1, vt2 = ds.solve_patch(divw, z, z, z, z)          # sem solid_loc
    # conservação: div(vt) == divw (Neumann puro, RHS balanceado)
    dv = (vt1[1:, :] - vt1[:-1, :]) / ds.hx + (vt2[:, 1:] - vt2[:, :-1]) / ds.hy
    assert np.allclose(dv, divw, atol=1e-9)


def test_solido_zera_faces_e_conserva_no_fluido():
    cfg = _cfg(8); dt = 0.5
    ds = MACDownscaler(cfg, dt)
    solid = np.zeros((8, 8), dtype=bool)
    solid[3:5, 3:5] = True                                # bloco 2x2 interno
    rng = np.random.default_rng(1)
    divw = rng.standard_normal((8, 8))
    divw[solid] = 0.0                                     # fonte nula no sólido
    # Compatibiliza o RHS (Neumann puro): soma sobre fluido deve ser 0
    divw[~solid] -= divw[~solid].mean()
    z = np.zeros(8)
    vt1, vt2 = ds.solve_patch(divw, z, z, z, z, solid_loc=solid)
    # faces sólidas: vt·n = 0
    fx = np.zeros((9, 8), bool); fx[:-1, :] |= solid; fx[1:, :] |= solid
    fy = np.zeros((8, 9), bool); fy[:, :-1] |= solid; fy[:, 1:] |= solid
    assert np.all(vt1[fx] == 0.0)
    assert np.all(vt2[fy] == 0.0)
    # conservação nas células de FLUIDO: div(vt) == divw (fonte compatível)
    dv = (vt1[1:, :] - vt1[:-1, :]) / ds.hx + (vt2[:, 1:] - vt2[:, :-1]) / ds.hy
    assert np.allclose(dv[~solid], divw[~solid], atol=1e-9)
