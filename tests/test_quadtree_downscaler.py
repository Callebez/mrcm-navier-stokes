"""Downscaler com patches de tamanhos diferentes e folhas sólidas."""
import numpy as np
import pytest

from mrcm.config import MRCMConfig
from mprm.downscaler import MACDownscaler


def _cfg(nloc=8):
    return MRCMConfig(Ndomx=2, Ndomy=2, nlocx=nloc, nlocy=nloc,
                      degreeU=1, degreeP=1, alpha=1.0,
                      basis_kind="legendre_P", mixed_family="BDM",
                      mixed_order=0, executor_kind="sequential")


def _neumann_compativel(n, m, h, seed=0):
    """Fonte com integral nula e fluxos de contorno nulos."""
    rng = np.random.default_rng(seed)
    divw = rng.standard_normal((n, m))
    return divw - divw.mean()


def test_patches_de_tamanhos_diferentes():
    d = MACDownscaler(_cfg(), dt=0.01, h=0.5 / 8)
    for (n, m) in ((8, 8), (4, 4), (16, 16)):
        divw = _neumann_compativel(n, m, d.h)
        vt1, vt2, defect = d.solve_patch(divw, np.zeros(m), np.zeros(m),
                                         np.zeros(n), np.zeros(n))
        assert vt1.shape == (n + 1, m)
        assert vt2.shape == (n, m + 1)
        assert abs(defect) < 1e-12, f"patch {n}x{m}: defeito {defect:.3e}"


def test_cache_reutiliza_fatoracao_por_geometria():
    d = MACDownscaler(_cfg(), dt=0.01, h=0.5 / 8)
    divw8 = _neumann_compativel(8, 8, d.h)
    d.solve_patch(divw8, np.zeros(8), np.zeros(8), np.zeros(8), np.zeros(8))
    n_lu = len(d._lu_cache)
    d.solve_patch(divw8, np.zeros(8), np.zeros(8), np.zeros(8), np.zeros(8))
    assert len(d._lu_cache) == n_lu, "refatorou uma geometria já vista"
    divw4 = _neumann_compativel(4, 4, d.h)
    d.solve_patch(divw4, np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4))
    assert len(d._lu_cache) == n_lu + 1, "não fatorou a nova geometria"


def test_folha_totalmente_solida_devolve_zero_sem_nan():
    """spec §4.5 item 1: nfluid = 0 dividia por zero no defeito."""
    d = MACDownscaler(_cfg(), dt=0.01, h=0.5 / 8)
    n = m = 8
    solid = np.ones((n, m), dtype=bool)
    vt1, vt2, defect = d.solve_patch(np.zeros((n, m)), np.zeros(m), np.zeros(m),
                                     np.zeros(n), np.zeros(n), solid_loc=solid)
    assert np.all(vt1 == 0.0) and np.all(vt2 == 0.0)
    assert np.isfinite(defect) and defect == 0.0


def test_solucao_de_patch_uniforme_inalterada():
    """Regressão numérica: o caminho sem sólido continua dando o mesmo ṽ."""
    d = MACDownscaler(_cfg(), dt=0.01, h=0.5 / 8)
    n = m = 8
    divw = _neumann_compativel(n, m, d.h, seed=5)
    fW = np.linspace(-1.0, 1.0, m) * 1e-3
    fE = np.linspace(1.0, -1.0, m) * 1e-3
    fS = np.zeros(n)
    fN = np.zeros(n)
    vt1, vt2, defect = d.solve_patch(divw, fW, fE, fS, fN)
    # Condições de contorno impostas exatamente nas faces externas do patch.
    assert np.allclose(vt1[0, :], fW, atol=1e-14, rtol=0)
    assert np.allclose(vt1[-1, :], fE, atol=1e-14, rtol=0)
    assert np.allclose(vt2[:, 0], fS, atol=1e-14, rtol=0)
    assert np.allclose(vt2[:, -1], fN, atol=1e-14, rtol=0)


def test_defeito_e_por_chamada_nao_acumulado():
    d = MACDownscaler(_cfg(), dt=0.01, h=0.5 / 8)
    n = m = 8
    divw = np.ones((n, m))          # fonte incompatível: defeito não nulo
    _, _, defect_ruim = d.solve_patch(divw, np.zeros(m), np.zeros(m),
                                      np.zeros(n), np.zeros(n))
    assert abs(defect_ruim) > 0.0
    divw_ok = _neumann_compativel(n, m, d.h)
    _, _, defect_bom = d.solve_patch(divw_ok, np.zeros(m), np.zeros(m),
                                     np.zeros(n), np.zeros(n))
    assert abs(defect_bom) < 1e-12, (
        f"o defeito da chamada anterior vazou: {defect_bom:.3e}")
