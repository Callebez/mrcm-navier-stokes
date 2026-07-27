"""Projetor sobre a camada Decomposition.

Enquanto a decomposição for uniforme, nenhum número pode mudar: é isso que
torna a generalização verificável.
"""
import numpy as np
import pytest

from mrcm.config import MRCMConfig
from mprm.decomposition import Decomposition
from mprm.projector import MPrMProjector


def _cfg(Ndom=2, nloc=8, dU=1, dP=1):
    return MRCMConfig(Ndomx=Ndom, Ndomy=Ndom, nlocx=nloc, nlocy=nloc,
                      degreeU=dU, degreeP=dP, alpha=1.0,
                      basis_kind="legendre_P", mixed_family="BDM",
                      mixed_order=0, executor_kind="sequential")


def _fonte(cfg, seed=7):
    rng = np.random.default_rng(seed)
    divw = rng.standard_normal((cfg.Nglobx, cfg.Ngloby))
    return divw - divw.mean()


def test_decomposicao_explicita_da_o_mesmo_que_a_implicita():
    """Passar Decomposition.uniform == deixar o projetor construí-la."""
    cfg = _cfg()
    divw = _fonte(cfg)

    p_impl = MPrMProjector(cfg, dt=0.01)
    f_impl = p_impl.project(divw)

    d = Decomposition.uniform(cfg.Ndomx, cfg.Ndomy, cfg.nlocx, cfg.Lx, cfg.Ly)
    p_expl = MPrMProjector(cfg, dt=0.01, decomposition=d)
    f_expl = p_expl.project(divw)

    assert np.allclose(p_impl.last_X, p_expl.last_X, atol=1e-12, rtol=0)
    assert set(f_impl) == set(f_expl)
    for k in f_impl:
        assert np.allclose(f_impl[k], f_expl[k], atol=1e-12, rtol=0)


def test_projetor_expoe_a_decomposicao():
    cfg = _cfg()
    p = MPrMProjector(cfg, dt=0.01)
    assert p.decomp is not None
    assert len(p.decomp.leaves) == cfg.Ndomx * cfg.Ndomy
    assert p.decomp.h == pytest.approx(cfg.Lx / cfg.Nglobx)


def test_beta_do_solver_local_usa_o_L_da_folha():
    """Achado 6: o Robin do solver local tem de usar o mesmo L das MsBF."""
    cfg = _cfg()
    p = MPrMProjector(cfg, dt=0.01)
    lf = p.decomp.leaf(0)
    esperado = cfg.robin_L_xy_for(lf.Hx, lf.Hy, lf.nlocx, lf.nlocy)
    assert p.robin_L[0] == pytest.approx(esperado)


def test_cavidade_inalterada():
    from mprm.flow import run_cavity
    from mprm.transport import mac_divergence
    r = run_cavity(Re=100.0, Ndom=2, nloc=8, T=0.05, verbose=False)
    dv = np.abs(mac_divergence(r["u"], r["v"], 1.0 / 16, 1.0 / 16)).max()
    assert dv < 1e-11, f"max|div u| = {dv:.3e}"
