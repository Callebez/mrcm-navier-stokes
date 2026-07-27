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


# ═══════════════════════════════════════════════════════════════════════
# Traços por segmento e média por aresta de Γ (spec §5 Fase 3)
# ═══════════════════════════════════════════════════════════════════════
def _arvore_com_no_pendurado(L=1.0):
    """2x2 folhas com a folha 0 subdividida: 4 arestas com nó pendurado."""
    from mrcm.quadtree import QuadTree, QuadTreeNode
    H = L / 2
    root = QuadTreeNode(0.0, 0.0, L, L, 0)
    root.children = [QuadTreeNode(i * H, j * H, (i + 1) * H, (j + 1) * H, 0)
                     for j in range(2) for i in range(2)]
    qt = QuadTree(root=root, max_level=1, Lx=L, Ly=L)
    qt._build_neighbours()
    qt.leaves[0].subdivide()
    qt._invalidate_cache()
    qt.balance_2to1()
    return qt


def test_project_devolve_chaves_por_segmento():
    cfg = _cfg()
    p = MPrMProjector(cfg, dt=0.01)
    f = p.project(_fonte(cfg))
    for k, v in f.items():
        assert isinstance(k, tuple) and len(k) == 3, f"chave inesperada: {k}"
        isub, side, seg = k
        assert 0 <= side <= 3 and seg in (0, 1)
        assert isinstance(v, np.ndarray)
    # Os dois lados de cada aresta recebem o MESMO array (Eq. 21).
    for e in p.decomp.gamma_edges():
        assert np.array_equal(f[e.a], f[e.b]), f"aresta {e.ell} assimétrica"


def test_leaf_boundary_fluxes_posiciona_segmentos():
    from mprm.decomposition import leaf_boundary_fluxes
    cfg = _cfg()
    p = MPrMProjector(cfg, dt=0.01)
    f = p.project(_fonte(cfg))
    for lf in p.decomp.leaves:
        fS, fE, fN, fW = leaf_boundary_fluxes(p.decomp, f, lf.isub)
        assert fS.shape == (lf.nlocx,) and fN.shape == (lf.nlocx,)
        assert fW.shape == (lf.nlocy,) and fE.shape == (lf.nlocy,)
        # Lados externos ficam zerados (parede/entrada: ṽ·n = 0).
        if lf.x0 == 0.0:
            assert np.allclose(fW, 0.0)


def test_quadtree_com_no_pendurado_continuidade_fraca():
    """Salto de fluxo desprezível através das arestas penduradas.

    Roda com degreeU=2/degreeP=2: o modo constante (k=0) é idêntico nas duas
    parametrizações de base e não detectaria erro de segmento — foi essa a
    lição da RevisaoQuadtree.md do mrcm.

    O MRCM garante continuidade FRACA dos fluxos (iguais quando testados
    contra as cofunções), não pontual. Por isso verificamos três condições:
    1. Os fluxos médios (Eq. 21) são idênticos nos dois lados de cada aresta.
    2. O salto pontual dos traços crus não excede a escala do sinal.
    3. A integral do salto sobre cada aresta é zero (conservação).
    """
    cfg = MRCMConfig(Ndomx=2, Ndomy=2, nlocx=8, nlocy=8,
                     degreeU=2, degreeP=2, alpha=1.0,
                     basis_kind="legendre_P", mixed_family="BDM",
                     mixed_order=0, executor_kind="sequential",
                     use_quadtree=True)
    qt = _arvore_com_no_pendurado()
    d = Decomposition.from_quadtree(qt, h=0.25 / 4, Lx=1.0, Ly=1.0)
    p = MPrMProjector(cfg, dt=0.01, decomposition=d)

    rng = np.random.default_rng(11)
    divw = rng.standard_normal((d.Nx, d.Ny))
    divw -= divw.mean()
    f = p.project(divw)

    escala = max(np.abs(np.concatenate([v for v in f.values()])).max(), 1e-30)
    for e in p.decomp.gamma_edges():
        # 1. Fluxos médios são bit-idênticos nos dois lados (Eq. 21).
        assert np.array_equal(f[e.a], f[e.b]), \
            f"aresta ell={e.ell}: fluxos médios assimétricos"
        # 2. Salto pontual dos traços crus é limitado pela escala do sinal.
        salto = np.abs(p.raw_traces[e.a] - p.raw_traces[e.b]).max()
        assert salto <= 0.6 * escala + 1e-12, (
            f"aresta ell={e.ell} ({e.a} | {e.b}): salto {salto:.3e} "
            f"contra escala de fluxo {escala:.3e}")
        # 3. Integral do salto é zero (conservação de fluxo).
        dh = d.hx if e.a[1] in (0, 2) else d.hy
        salto_int = abs(np.sum(p.raw_traces[e.a] - p.raw_traces[e.b]) * dh)
        h_local = dh
        assert salto_int <= 1e-8 * escala * h_local + 1e-14, \
            f"aresta ell={e.ell}: integral do salto {salto_int:.3e}"
