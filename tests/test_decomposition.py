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


# ═══════════════════════════════════════════════════════════════════════
# Geometria do quadtree (spec §4.3, §4.3.1, §4.4)
# ═══════════════════════════════════════════════════════════════════════
from mprm.decomposition import plan_quadtree_geometry, build_obstacle_tree


class TestPlanGeometry:
    def test_familia_a_configuracao_de_referencia(self):
        """Ly=8D, N_by=3 (ímpar) -> H0=8D/3, níveis 0-2, nloc=(32,16,8)."""
        g = plan_quadtree_geometry(D=1.0, Ly=8.0, N_bx=10, N_by=3,
                                   max_level=2, nloc_min=8, family="a")
        assert g["H0"] == pytest.approx(8.0 / 3.0)
        assert g["Lx"] == pytest.approx(10 * 8.0 / 3.0)
        assert g["nloc"] == [32, 16, 8]
        assert g["h"] == pytest.approx(1.0 / 12.0)
        assert (g["Nx"], g["Ny"]) == (320, 96)
        assert g["yc"] == pytest.approx(4.0), "obstáculo centrado no canal"

    def test_familia_b_configuracao_de_referencia(self):
        """Ly=8D, N_by=2 (par) -> H0=4D, corpo = 2x2 folhas de D/2."""
        g = plan_quadtree_geometry(D=1.0, Ly=8.0, N_bx=6, N_by=2,
                                   max_level=4, nloc_min=4, family="b")
        assert g["H0"] == pytest.approx(4.0)
        assert g["nloc"] == [64, 32, 16, 8, 4]
        assert g["h"] == pytest.approx(1.0 / 16.0)
        assert (g["Nx"], g["Ny"]) == (384, 128)
        assert g["yc"] == pytest.approx(4.0)
        assert g["body_level"] == 3, "corpo = folhas de nível 3 (D/2)"
        assert g["H"][g["body_level"]] == pytest.approx(0.5)

    def test_familia_a_rejeita_N_by_par(self):
        with pytest.raises(ValueError, match="ímpar"):
            plan_quadtree_geometry(D=1.0, Ly=8.0, N_bx=10, N_by=4,
                                   max_level=2, nloc_min=8, family="a")

    def test_familia_b_rejeita_N_by_impar(self):
        with pytest.raises(ValueError, match="par"):
            plan_quadtree_geometry(D=1.0, Ly=8.0, N_bx=6, N_by=3,
                                   max_level=4, nloc_min=4, family="b")

    def test_rejeita_H0_menor_que_D_na_familia_a(self):
        with pytest.raises(ValueError, match="H0"):
            plan_quadtree_geometry(D=4.0, Ly=8.0, N_bx=10, N_by=3,
                                   max_level=2, nloc_min=8, family="a")

    def test_rejeita_nloc_impar(self):
        """nloc ímpar contra nó pendurado quebra o alinhamento de segmento."""
        with pytest.raises(ValueError, match="par"):
            plan_quadtree_geometry(D=1.0, Ly=8.0, N_bx=10, N_by=3,
                                   max_level=2, nloc_min=7, family="a")

    def test_rejeita_nloc_min_menor_que_grau(self):
        with pytest.raises(ValueError, match="degree"):
            plan_quadtree_geometry(D=1.0, Ly=8.0, N_bx=10, N_by=3,
                                   max_level=2, nloc_min=2, family="a",
                                   degreeU=1, degreeP=3)


class TestObstacleTree:
    def test_familia_a_obstaculo_interior_a_uma_folha(self):
        g = plan_quadtree_geometry(D=1.0, Ly=8.0, N_bx=10, N_by=3,
                                   max_level=2, nloc_min=8, family="a")
        qt = build_obstacle_tree(g, radii=(6.0, 3.0))
        xc, yc, D = g["xc"], g["yc"], 1.0
        conteiners = [lf for lf in qt.leaves
                      if lf.x0 <= xc - D / 2 and lf.x1 >= xc + D / 2
                      and lf.y0 <= yc - D / 2 and lf.y1 >= yc + D / 2]
        assert len(conteiners) == 1, "o corpo deve caber numa única folha"

    def test_familia_b_obstaculo_e_uniao_exata_de_4_folhas(self):
        g = plan_quadtree_geometry(D=1.0, Ly=8.0, N_bx=6, N_by=2,
                                   max_level=4, nloc_min=4, family="b")
        qt = build_obstacle_tree(g, radii=(8.0, 4.0, 2.0, 1.0))
        xc, yc, D = g["xc"], g["yc"], 1.0
        dentro = [lf for lf in qt.leaves
                  if lf.x0 >= xc - D / 2 - 1e-12 and lf.x1 <= xc + D / 2 + 1e-12
                  and lf.y0 >= yc - D / 2 - 1e-12 and lf.y1 <= yc + D / 2 + 1e-12]
        assert len(dentro) == 4, f"esperado 4 folhas no corpo, obtido {len(dentro)}"
        area = sum((lf.x1 - lf.x0) * (lf.y1 - lf.y0) for lf in dentro)
        assert area == pytest.approx(D * D), "as 4 folhas cobrem o corpo exatamente"

    def test_arvore_balanceada_2_para_1(self):
        g = plan_quadtree_geometry(D=1.0, Ly=8.0, N_bx=10, N_by=3,
                                   max_level=2, nloc_min=8, family="a")
        qt = build_obstacle_tree(g, radii=(6.0, 3.0))
        for lf in qt.leaves:
            for s in range(4):
                assert len(lf.neighbours[s]) <= 2, \
                    f"folha {lf} tem {len(lf.neighbours[s])} vizinhos no lado {s}"
                for nb in lf.neighbours[s]:
                    assert abs(lf.level - nb.level) <= 1

    def test_refino_decresce_com_a_distancia_ao_corpo(self):
        g = plan_quadtree_geometry(D=1.0, Ly=8.0, N_bx=10, N_by=3,
                                   max_level=2, nloc_min=8, family="a")
        qt = build_obstacle_tree(g, radii=(6.0, 3.0))
        xc, yc = g["xc"], g["yc"]
        # Janela de um H0 (=8/3): a de 2.0 do plano cai exatamente sobre o
        # centro do anel de nível 1, que o cap de nível deixa não-subdividido
        # (ver pre-flight-resolutions.md, item 1).
        perto = [lf.level for lf in qt.leaves
                 if abs(lf.cx - xc) < 3.0 and abs(lf.cy - yc) < 3.0]
        longe = [lf.level for lf in qt.leaves if abs(lf.cx - xc) > 12.0]
        assert max(perto) > max(longe), \
            f"níveis perto={max(perto)} não excedem longe={max(longe)}"

    def test_folha_travada_nao_e_subdividida_pelo_balanco(self):
        """Cap de nível: o anel imediato fica em `locked_level + 1`.

        Sem o cap, as folhas de nível 1 vizinhas do corpo refinam a nível 2 e
        `balance_2to1` subdivide a própria folha travada (pre-flight item 1).
        """
        g = plan_quadtree_geometry(D=1.0, Ly=8.0, N_bx=10, N_by=3,
                                   max_level=2, nloc_min=8, family="a")
        qt = build_obstacle_tree(g, radii=(6.0, 3.0))
        niveis = {}
        for lf in qt.leaves:
            niveis[lf.level] = niveis.get(lf.level, 0) + 1
        assert qt.Nsubd == 120, f"Nsubd={qt.Nsubd}"
        assert niveis == {0: 16, 1: 40, 2: 64}, f"níveis={niveis}"

    def test_cap_gradado_resiste_a_mais_de_um_anel_refinado(self):
        """Com max_level=3 e raios largos, o 2º anel quer chegar a nível 3.

        Um cap restrito ao anel que *encosta* na travada não bastaria: o
        balanço 2:1 subdividiria o anel de nível 1 e, em cascata, a própria
        travada (medido: 0 folhas contendo o corpo). O cap gradado por
        distância mantém exatamente 1.
        """
        g = plan_quadtree_geometry(D=1.0, Ly=8.0, N_bx=10, N_by=3,
                                   max_level=3, nloc_min=8, family="a")
        qt = build_obstacle_tree(g, radii=(8.0, 6.0, 4.0))
        xc, yc, D = g["xc"], g["yc"], 1.0
        conteiners = [lf for lf in qt.leaves
                      if lf.x0 <= xc - D / 2 and lf.x1 >= xc + D / 2
                      and lf.y0 <= yc - D / 2 and lf.y1 >= yc + D / 2]
        assert len(conteiners) == 1 and conteiners[0].level == 0
        for lf in qt.leaves:
            for s in range(4):
                for nb in lf.neighbours[s]:
                    assert abs(lf.level - nb.level) <= 1
