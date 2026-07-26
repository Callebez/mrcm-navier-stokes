"""Camada de decomposição: geometria, arestas de Γ e tarefas de Fase 1.

Nesta tarefa a cobertura é do caminho cartesiano — o quadtree entra depois.
O valor do teste está em amarrar a `Decomposition` à aritmética que o
projetor faz hoje, para que a troca (Tarefa 6) seja verificável.
"""
import numpy as np
import pytest

from mrcm.config import MRCMConfig
from mrcm.topology import mark_sides
from mprm.decomposition import Decomposition, Leaf, GammaEdge


def test_uniform_geometria_das_folhas():
    d = Decomposition.uniform(Ndomx=2, Ndomy=2, nloc=8, Lx=1.0, Ly=1.0)
    assert len(d.leaves) == 4
    assert d.h == pytest.approx(0.5 / 8)
    assert (d.Nx, d.Ny) == (16, 16)

    # Folha 0 = canto inferior esquerdo; folha 3 = superior direito.
    lf0, lf3 = d.leaf(0), d.leaf(3)
    assert (lf0.x0, lf0.y0, lf0.x1, lf0.y1) == pytest.approx((0.0, 0.0, 0.5, 0.5))
    assert (lf0.i0, lf0.j0) == (0, 0)
    assert (lf3.x0, lf3.y0) == pytest.approx((0.5, 0.5))
    assert (lf3.i0, lf3.j0) == (8, 8)
    for lf in d.leaves:
        assert (lf.nlocx, lf.nlocy) == (8, 8)


def test_uniform_offsets_batem_com_a_aritmetica_do_projetor():
    """i0/j0 devem reproduzir exatamente `ox = ix*n, oy = iy*m`."""
    Ndomx, Ndomy, nloc = 3, 2, 6
    d = Decomposition.uniform(Ndomx, Ndomy, nloc, Lx=3.0, Ly=2.0)
    for isub in range(Ndomx * Ndomy):
        ix, iy = isub % Ndomx, isub // Ndomx
        lf = d.leaf(isub)
        assert (lf.i0, lf.j0) == (ix * nloc, iy * nloc)


def test_gamma_edges_contagem_e_pares():
    """Grade 3x2: 2*2 arestas verticais internas + 3*1 horizontais = 7."""
    d = Decomposition.uniform(3, 2, nloc=6, Lx=3.0, Ly=2.0)
    edges = d.gamma_edges()
    assert len(edges) == 7
    for e in edges:
        assert e.a[0] != e.b[0], "uma aresta de Γ liga dois subdomínios distintos"
        assert e.nfaces == 6, "com h uniforme, os dois lados têm o mesmo nº de faces"
    # Cada subdomínio interno aparece em tantas arestas quantos lados internos tem.
    for isub in range(6):
        ix, iy = isub % 3, isub // 3
        _, internal = mark_sides(ix, iy, 3, 2)
        n = sum(1 for e in edges if e.a[0] == isub or e.b[0] == isub)
        assert n == len(internal), f"isub={isub}"


def test_task_uniforme_tem_as_chaves_que_o_mrcm_espera():
    cfg = MRCMConfig(Ndomx=2, Ndomy=2, nlocx=8, nlocy=8, degreeU=1, degreeP=1,
                     alpha=1.0, basis_kind="legendre_P", mixed_family="BDM",
                     mixed_order=0, executor_kind="sequential")
    d = Decomposition.uniform(2, 2, 8, 1.0, 1.0)
    t = d.task(cfg, isub=0, NunkU=10, problem="DUMMY")
    assert set(t) == {"cfg_dict", "isub", "NunkU", "edge_l", "problem"}
    assert t["isub"] == 0 and t["NunkU"] == 10 and t["problem"] == "DUMMY"
    # edge_l mapeia lado interno -> índice de Γ, como em projector.py:137-138
    _, internal = mark_sides(0, 0, 2, 2)
    assert set(t["edge_l"]) == set(internal)


def test_segments_of_uniforme_tem_um_segmento_por_lado_interno():
    d = Decomposition.uniform(2, 2, 8, 1.0, 1.0)
    segs = d.segments_of(0)
    _, internal = mark_sides(0, 0, 2, 2)
    assert len(segs) == len(internal)
    for side, seg, ell, bbox, nfaces in segs:
        assert seg == 0
        assert nfaces == 8
        assert bbox is not None, "mesmo no uniforme, a bbox do lado é preenchida"
