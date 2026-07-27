"""Camada de decomposição do MPrM — o único ponto que conhece a topologia.

O projetor, o downscaler e o passo de projeção param de fazer aritmética
`(ix, iy)` e passam a iterar sobre folhas e arestas de Γ. A decomposição
cartesiana vira um caso particular da mesma classe, o que torna o teste
"árvore uniforme ≡ cartesiano" a rede de segurança da generalização.

Convenção de lados (herdada do mrcm): 0=bottom, 1=right, 2=top, 3=left.
Convenção de malha MAC: hx = hy = h uniforme global (spec, decisão D1).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

from mrcm.topology import Skeleton, mark_sides

SIDE_BOTTOM, SIDE_RIGHT, SIDE_TOP, SIDE_LEFT = 0, 1, 2, 3


@dataclass(frozen=True)
class Leaf:
    """Um subdomínio: geometria física, resolução local e posição na malha MAC."""
    isub: int
    x0: float
    y0: float
    x1: float
    y1: float
    nlocx: int
    nlocy: int
    i0: int          # índice da primeira célula MAC em x
    j0: int          # índice da primeira célula MAC em y
    level: int

    @property
    def Hx(self) -> float:
        return self.x1 - self.x0

    @property
    def Hy(self) -> float:
        return self.y1 - self.y0


@dataclass(frozen=True)
class GammaEdge:
    """Uma aresta de Γ: o segmento de interface entre dois subdomínios.

    `a` e `b` são `(isub, side, seg)`. `nfaces` é o número de faces finas do
    segmento — igual dos dois lados porque h é uniforme (spec §4.2), o que
    torna a média da Eq. 21 uma operação face a face.
    """
    ell: int
    a: tuple
    b: tuple
    nfaces: int
    bbox: tuple      # (x0, y0, x1, y1) do segmento


class Decomposition:
    """Folhas + esqueleto + arestas de Γ, para um `h` uniforme global."""

    def __init__(self, skeleton, leaves, h, Lx, Ly, *, hy=None):
        self.skeleton = skeleton
        self.leaves = list(leaves)
        self.hx = float(h)
        self.hy = float(hy) if hy is not None else float(h)
        self.Lx, self.Ly = float(Lx), float(Ly)
        self.Nx = int(round(Lx / self.hx))
        self.Ny = int(round(Ly / self.hy))
        self._by_id = {lf.isub: lf for lf in self.leaves}
        self._gamma = None          # construído sob demanda

    @property
    def h(self) -> float:
        """Espaçamento de malha (atalho para hx; no quadtree hx == hy)."""
        return self.hx

    # ── fábricas ────────────────────────────────────────────────────────
    @classmethod
    def uniform(cls, Ndomx: int, Ndomy: int, nloc: int,
                Lx: float, Ly: float, *, nlocy: int = None) -> "Decomposition":
        """Decomposição cartesiana — o caminho retrocompatível."""
        if nlocy is None:
            nlocy = nloc
        Hx, Hy = Lx / Ndomx, Ly / Ndomy
        hx = Hx / nloc
        hy = Hy / nlocy
        leaves = []
        for isub in range(Ndomx * Ndomy):
            ix, iy = isub % Ndomx, isub // Ndomx
            leaves.append(Leaf(isub=isub,
                               x0=ix * Hx, y0=iy * Hy,
                               x1=(ix + 1) * Hx, y1=(iy + 1) * Hy,
                               nlocx=nloc, nlocy=nlocy,
                               i0=ix * nloc, j0=iy * nlocy, level=0))
        kw = {} if abs(hx - hy) <= 1e-12 * max(hx, hy) else {"hy": hy}
        return cls(Skeleton.cartesian(Ndomx, Ndomy), leaves, hx, Lx, Ly, **kw)

    @classmethod
    def from_quadtree(cls, qt, h: float, Lx: float, Ly: float,
                      solid=None) -> "Decomposition":
        """Decomposição a partir de uma árvore balanceada, com h uniforme.

        Cada folha recebe ``nloc_i = H_i/h`` (decisão D1), gravado em
        ``skeleton.sub_nloc``, que é o canal pelo qual o mrcm leva a resolução
        até a malha local (`mrcm/solver.py:120-136`).

        ``solid``, quando dado, é a máscara MAC global; a conexidade do fluido
        em cada folha é verificada na construção (spec §4.5).
        """
        leaves_qt = qt.leaves            # já ordenadas row-major, com id
        skel = Skeleton.from_quadtree(qt)   # valida o balanço 2:1 internamente

        leaves = []
        sub_nloc = np.zeros((len(leaves_qt), 2), dtype=int)
        for nd in leaves_qt:
            isub = nd.subdomain_id
            nlocx = int(round((nd.x1 - nd.x0) / h))
            nlocy = int(round((nd.y1 - nd.y0) / h))
            for nome, nl, H in (("nlocx", nlocx, nd.x1 - nd.x0),
                                ("nlocy", nlocy, nd.y1 - nd.y0)):
                if abs(H / h - nl) > 1e-9 * max(nl, 1):
                    raise ValueError(
                        f"folha {isub}: {nome} = {H:.6g}/{h:.6g} não é inteiro. "
                        f"h precisa dividir todo H_ℓ (spec §4.3).")
            sub_nloc[isub] = (nlocx, nlocy)
            leaves.append(Leaf(isub=isub, x0=nd.x0, y0=nd.y0,
                               x1=nd.x1, y1=nd.y1,
                               nlocx=nlocx, nlocy=nlocy,
                               i0=int(round(nd.x0 / h)),
                               j0=int(round(nd.y0 / h)),
                               level=nd.level))
        skel.sub_nloc = sub_nloc
        d = cls(skel, leaves, h, Lx, Ly)
        if solid is not None:
            d.check_fluid_connectivity(solid)
        return d

    # ── invariantes ─────────────────────────────────────────────────────
    def check_fluid_connectivity(self, solid) -> None:
        """Nenhuma folha pode ter região fluida desconexa (spec §4.5).

        O problema local do downscaler é Neumann puro: com fluido em duas
        componentes a nulidade é 2, e o pino numa única célula remove apenas um
        modo — a projeção sairia errada sem erro nenhum. Folhas 100% sólidas
        são válidas (a família (b) as produz por construção).
        """
        from scipy.ndimage import label
        for lf in self.leaves:
            blk = solid[lf.i0:lf.i0 + lf.nlocx, lf.j0:lf.j0 + lf.nlocy]
            fluid = ~np.asarray(blk, dtype=bool)
            if not fluid.any():
                continue                       # folha 100% sólida: ok
            ncomp = int(label(fluid)[1])
            if ncomp > 1:
                raise ValueError(
                    f"folha {lf.isub} (células x[{lf.i0}:{lf.i0+lf.nlocx}], "
                    f"y[{lf.j0}:{lf.j0+lf.nlocy}]) tem região fluida desconexa "
                    f"({ncomp} componentes). O problema de Neumann local teria "
                    f"nulidade {ncomp} e o pino remove só um modo.")

    def assert_consistent_across_ranks(self, comm) -> None:
        """Todos os ranks precisam ter construído o MESMO esqueleto.

        Cada rank constrói a árvore por conta própria. É determinístico, mas se
        um rank divergir, os triplets são montados contra numerações de Γ
        diferentes e o sistema global sai numericamente lixo, sem erro.

        Usa zlib.crc32 em vez de hash() porque PYTHONHASHSEED randomiza o hash
        de str/bytes por processo — hash() difere entre ranks MPI mesmo para
        esqueletos idênticos (pre-flight item 2; ver mrcm/solver.py:339).
        """
        if getattr(comm, "size", 1) == 1:
            return
        import zlib
        from mpi4py import MPI as _MPI
        skel = self.skeleton
        raw = (skel.NGamma.to_bytes(4, "little") +
               len(self.leaves).to_bytes(4, "little") +
               np.ascontiguousarray(skel.el2edge).tobytes() +
               np.ascontiguousarray(skel.edge2el).tobytes())
        fingerprint = zlib.crc32(raw)
        lo = comm.allreduce(fingerprint, op=_MPI.MIN)
        hi = comm.allreduce(fingerprint, op=_MPI.MAX)
        if lo != hi:
            raise RuntimeError(
                f"esqueletos divergem entre ranks (rank {comm.rank}: "
                f"NGamma={skel.NGamma}, {len(self.leaves)} folhas). "
                f"A montagem global seria feita contra numerações diferentes.")

    # ── acesso ──────────────────────────────────────────────────────────
    @property
    def is_uniform(self) -> bool:
        return self.skeleton.is_uniform

    def leaf(self, isub: int) -> Leaf:
        return self._by_id[isub]

    def segments_of(self, isub: int):
        """`(side, seg, ell, bbox, nfaces)` de cada incidência de Γ da folha.

        No caminho uniforme o mrcm devolve `bbox=None` (o segmento é o lado
        inteiro); aqui a bbox é sempre preenchida, para que os consumidores
        tenham um único formato.
        """
        lf = self.leaf(isub)
        out = []
        for inc in self.skeleton.gamma_incidences(isub):
            bbox = inc.bbox if inc.bbox is not None else _side_bbox(lf, inc.side)
            horiz = inc.side in (SIDE_BOTTOM, SIDE_TOP)
            span = (bbox[2] - bbox[0]) if horiz else (bbox[3] - bbox[1])
            dh = self.hx if horiz else self.hy
            nfaces = int(round(span / dh))
            out.append((inc.side, inc.seg, inc.ell, bbox, nfaces))
        return out

    def gamma_edges(self) -> list:
        """Uma `GammaEdge` por aresta de Γ, com os dois lados emparelhados."""
        if self._gamma is not None:
            return self._gamma
        by_ell: dict = {}
        for lf in self.leaves:
            for side, seg, ell, bbox, nfaces in self.segments_of(lf.isub):
                by_ell.setdefault(ell, []).append(
                    ((lf.isub, side, seg), bbox, nfaces))
        edges = []
        for ell in sorted(by_ell):
            entries = by_ell[ell]
            if len(entries) != 2:
                raise ValueError(
                    f"aresta de Γ ell={ell} tem {len(entries)} incidências "
                    f"(esperado 2). Decomposição inconsistente.")
            (ka, bba, na), (kb, bbb, nb) = entries
            if na != nb:
                raise ValueError(
                    f"aresta de Γ ell={ell}: {na} faces de um lado e {nb} do "
                    f"outro. Com h uniforme os dois lados devem coincidir.")
            edges.append(GammaEdge(ell=ell, a=ka, b=kb, nfaces=na, bbox=bba))
        self._gamma = edges
        return edges

    # ── tarefa de Fase 1 do mrcm ────────────────────────────────────────
    def task(self, cfg, isub: int, NunkU: int, problem) -> dict:
        """Tarefa consumida por `mrcm.local_solver.build_subdomain_workspace`.

        Reproduz `mrcm/solver.py:104-141`: no caminho uniforme a geometria é
        derivada de `(ix, iy)` pelo próprio mrcm e basta `edge_l`; no caminho
        quadtree é preciso mandar `geom`, `incidences` e `is_external`.
        """
        skel = self.skeleton
        if self.is_uniform:
            ix, iy = isub % cfg.Ndomx, isub // cfg.Ndomx
            _, internal = mark_sides(ix, iy, cfg.Ndomx, cfg.Ndomy)
            edge_l = {s: int(skel.ieGt[skel.el2edge[isub, s]]) for s in internal}
            return {"cfg_dict": cfg.as_dict(), "isub": isub, "NunkU": NunkU,
                    "edge_l": edge_l, "problem": problem}
        lf = self.leaf(isub)
        return {"cfg_dict": cfg.as_dict(), "isub": isub, "NunkU": NunkU,
                "geom": {"x0": lf.x0, "y0": lf.y0,
                         "Hx": lf.Hx, "Hy": lf.Hy,
                         "nlocx": lf.nlocx, "nlocy": lf.nlocy},
                "incidences": skel.gamma_incidences(isub),
                "is_external": skel.external_side_flags(isub),
                "cacheable": False,
                "problem": problem}


# ═══════════════════════════════════════════════════════════════════════════
# Geometria do quadtree (spec §4.3 e §4.3.1)
# ═══════════════════════════════════════════════════════════════════════════

def plan_quadtree_geometry(*, D, Ly, N_bx, N_by, max_level, nloc_min,
                           family, degreeU=1, degreeP=3):
    """Resolve as restrições geométricas e devolve a configuração completa.

    O quadtree é diádico: toda fronteira de folha cai em múltiplo de H0/2^ℓ.
    Duas famílias de alinhamento entre corpo e árvore (spec, decisão D7):

    * ``family="a"`` — o corpo fica **contido no interior** de uma folha, que
      por isso não pode ser subdividida. Exige ``N_by`` ímpar (o centro do
      canal precisa ser o centro de uma célula base) e ``H0 > D``.
    * ``family="b"`` — o corpo é a **união exata** de um bloco 2×2 de folhas de
      nível ``body_level``, com ``D = 2·H[body_level]``. Exige ``N_by`` par (o
      centro do canal é um vértice da grade). É a única forma de o refino
      alcançar a superfície: com o corpo em uma única folha de nível ℓ≥1 e
      centrado, Ly/D teria de ser ímpar e par ao mesmo tempo.

    Raises
    ------
    ValueError
        Nomeando a restrição violada e o parâmetro a corrigir.
    """
    if family not in ("a", "b"):
        raise ValueError(f"family deve ser 'a' ou 'b', recebido {family!r}")
    if family == "a" and N_by % 2 == 0:
        raise ValueError(
            f"family='a' exige N_by ímpar (recebido {N_by}): com N_by par, "
            f"y=Ly/2 é fronteira de folha em todo nível e nenhuma folha "
            f"contém o corpo centrado.")
    if family == "b" and N_by % 2 != 0:
        raise ValueError(
            f"family='b' exige N_by par (recebido {N_by}): o centro do corpo "
            f"precisa cair num vértice da grade diádica.")

    H0 = Ly / N_by
    Lx = N_bx * H0
    H = [H0 / 2 ** l for l in range(max_level + 1)]
    h = H[max_level] / nloc_min
    nloc = [int(round(Hl / h)) for Hl in H]

    if nloc_min < max(degreeU, degreeP):
        raise ValueError(
            f"nloc_min={nloc_min} < max(degreeU={degreeU}, degreeP={degreeP}): "
            f"o espaço de interface ficaria mais rico que a malha fina do "
            f"segmento (Arbogast et al., condição de solvabilidade).")
    for l, (Hl, nl) in enumerate(zip(H, nloc)):
        if abs(Hl / h - nl) > 1e-9 * nl:
            raise ValueError(
                f"nível {l}: H={Hl:.6g} não é múltiplo inteiro de h={h:.6g}.")
        if nl % 2 != 0:
            raise ValueError(
                f"nível {l}: nloc={nl} é ímpar. Um segmento de nó pendurado "
                f"terminaria no meio de uma célula e a base de interface, que "
                f"é mascarada por célula, integraria intervalos diferentes nos "
                f"dois lados (mrcm/topology.py:109-160). Use nloc_min par.")

    Nx, Ny = int(round(Lx / h)), int(round(Ly / h))
    for name, val, tot in (("Nx", Nx, Lx), ("Ny", Ny, Ly)):
        if abs(tot / h - val) > 1e-9 * val:
            raise ValueError(f"{name}: {tot:.6g}/h não é inteiro.")

    yc = 0.5 * Ly
    if family == "a":
        if H0 <= D:
            raise ValueError(
                f"family='a' exige H0 > D: H0={H0:.6g}, D={D:.6g}. A folha que "
                f"contém o corpo não pode ser subdividida, então ela precisa "
                f"caber o corpo inteiro.")
        body_level = 0
        # xc no centro de uma célula base, o mais próximo de 1/4 do canal.
        k = max(0, int(round(0.25 * N_bx - 0.5)))
        xc = (k + 0.5) * H0
    else:
        # D = 2·H[body_level] -> o corpo é um bloco 2x2 desse nível.
        cand = [l for l, Hl in enumerate(H) if abs(D - 2 * Hl) <= 1e-9 * D]
        if not cand:
            raise ValueError(
                f"family='b' exige D = 2·H[ℓ] para algum nível: "
                f"D={D:.6g}, H={[f'{x:.4g}' for x in H]}. Ajuste N_by, Ly ou "
                f"max_level.")
        body_level = cand[0]
        if body_level >= max_level:
            raise ValueError(
                f"body_level={body_level} == max_level={max_level}: não sobra "
                f"nível para refinar as vizinhas do corpo, que é o objetivo "
                f"da family='b'. Aumente max_level.")
        Hb = H[body_level]
        k = max(1, int(round(0.25 * Lx / Hb)))
        xc = k * Hb                       # vértice da grade do nível do corpo
    return {"family": family, "D": D, "Lx": Lx, "Ly": Ly, "H0": H0,
            "H": H, "nloc": nloc, "h": h, "Nx": Nx, "Ny": Ny,
            "xc": xc, "yc": yc, "N_bx": N_bx, "N_by": N_by,
            "max_level": max_level, "body_level": body_level}


def build_obstacle_tree(geom: dict, radii, wake_length: float = 0.0):
    """Constrói a árvore refinada em torno do obstáculo (spec §4.4).

    Uma folha de nível ``ℓ`` é subdividida sse ``ℓ < max_level`` e a distância
    da sua bbox à bbox do corpo é ``<= radii[ℓ]``. ``radii`` deve ser
    estritamente decrescente — é o que garante que o anel mais fino envolve o
    corpo em vez de aparecer solto no domínio.

    ``wake_length > 0`` estende a região refinada a jusante (x > xc); o default
    é isotrópico.

    Na família (a) a folha que contém o corpo é travada e nunca subdividida, e
    o refino em volta dela é **gradado** (ver ``_cap_de_nivel``).
    Na família (b) o corpo é refinado até ``body_level`` para virar um bloco
    2×2 exato, e não além.
    """
    from mrcm.quadtree import QuadTree, QuadTreeNode

    D, xc, yc = geom["D"], geom["xc"], geom["yc"]
    Lx, Ly = geom["Lx"], geom["Ly"]
    N_bx, N_by = geom["N_bx"], geom["N_by"]
    max_level, family = geom["max_level"], geom["family"]
    body_level = geom["body_level"]
    radii = list(radii)
    if any(radii[i] <= radii[i + 1] for i in range(len(radii) - 1)):
        raise ValueError(f"radii deve ser estritamente decrescente: {radii}")
    if len(radii) < max_level:
        raise ValueError(
            f"radii tem {len(radii)} entradas para max_level={max_level}; "
            f"são necessárias {max_level} (uma por nível subdividível).")

    bx0, bx1 = xc - D / 2, xc + D / 2
    by0, by1 = yc - D / 2, yc + D / 2

    # Grade base de células quadradas (H0 x H0). `QuadTree.from_uniform` exige
    # potências de dois, o que N_by ímpar viola; montamos a raiz à mão.
    H0 = geom["H0"]
    root = QuadTreeNode(0.0, 0.0, Lx, Ly, 0)
    root.children = [
        QuadTreeNode(i * H0, j * H0, (i + 1) * H0, (j + 1) * H0, 0)
        for j in range(N_by) for i in range(N_bx)
    ]
    qt = QuadTree(root=root, max_level=max_level, Lx=Lx, Ly=Ly)

    def _dist_ao_corpo(nd):
        dx = max(bx0 - nd.x1, 0.0, nd.x0 - bx1)
        dy = max(by0 - nd.y1, 0.0, nd.y0 - by1)
        return float(np.hypot(dx, dy))

    def _contem_o_corpo(nd):
        return (nd.x0 <= bx0 + 1e-12 and nd.x1 >= bx1 - 1e-12 and
                nd.y0 <= by0 + 1e-12 and nd.y1 >= by1 - 1e-12)

    def _dentro_do_corpo(nd):
        return (nd.x0 >= bx0 - 1e-12 and nd.x1 <= bx1 + 1e-12 and
                nd.y0 >= by0 - 1e-12 and nd.y1 <= by1 + 1e-12)

    # A folha travada da família (a): a célula base que contém o corpo. Ela
    # nunca é subdividida, mas `balance_2to1` (mrcm/quadtree.py:415-441) não
    # sabe disso — se uma vizinha imediata chegar a nível `travada+2`, o
    # balanço subdivide a travada e o corpo deixa de caber numa única folha.
    travada = next((nd for nd in root.children if _contem_o_corpo(nd)), None)
    if family == "a" and travada is None:
        raise ValueError(
            f"family='a': nenhuma célula base contém o corpo "
            f"[{bx0:.6g},{bx1:.6g}]x[{by0:.6g},{by1:.6g}]. Verifique xc/yc.")

    def _cap_de_nivel(nd):
        """Nível máximo desta folha por gradação em volta da travada.

        Implementa spec §4.3 item 3 ("as vizinhas imediatas ficam em nível 1,
        o anel seguinte em nível 2"): o anel que encosta na travada fica em
        `travada.level + 1`, e cada nível extra só é liberado quando cabe,
        entre a folha e a travada, um anel de cada nível intermediário — o
        buffer de gradação de de Berg et al., §14.3. Com esse cap o balanço
        2:1 nunca precisa tocar a folha travada.
        """
        gap = max(travada.x0 - nd.x1, nd.x0 - travada.x1,
                  travada.y0 - nd.y1, nd.y0 - travada.y1, 0.0)   # distância L∞
        Ht = travada.x1 - travada.x0
        m = 1
        while (travada.level + m < max_level
               and gap >= Ht * (1.0 - 2.0 ** -m) - 1e-12 * Ht):
            m += 1
        return travada.level + m

    while True:
        mudou = False
        for lf in qt._collect_leaves(use_cache=False):
            if lf.level >= max_level:
                continue
            if family == "a" and lf is travada:
                continue                      # folha travada (decisão D4)
            if family == "a" and lf.level >= _cap_de_nivel(lf):
                continue                      # anel gradado junto à travada
            if family == "b" and _dentro_do_corpo(lf) and lf.level >= body_level:
                continue                      # o corpo já é o bloco exato
            if family == "b" and _contem_o_corpo(lf) and lf.level < body_level:
                lf.subdivide(); mudou = True   # desce até o corpo virar 2x2
                continue
            d = _dist_ao_corpo(lf)
            if wake_length > 0.0 and lf.cx > xc and abs(lf.cy - yc) <= wake_length:
                d = min(d, max(0.0, lf.cx - bx1) * 0.5)
            if d <= radii[lf.level]:
                lf.subdivide(); mudou = True
        if not mudou:
            break
        qt._invalidate_cache()
        qt.balance_2to1()

    qt._invalidate_cache()
    qt.balance_2to1()

    # Validação pós-balanço: o balanceamento pode ter subdividido a folha
    # travada. Com o cap de nível acima isso não deve ocorrer, mas silenciar
    # essa verificação transformaria uma violação de D4 em resultado plausível.
    if family == "a":
        n = sum(1 for lf in qt.leaves if _contem_o_corpo(lf))
        if n != 1:
            raise ValueError(
                f"family='a': após o balanço 2:1 há {n} folhas contendo o "
                f"corpo (esperado 1). Reduza radii[0] ou aumente H0.")
    else:
        dentro = [lf for lf in qt.leaves if _dentro_do_corpo(lf)]
        area = sum((lf.x1 - lf.x0) * (lf.y1 - lf.y0) for lf in dentro)
        if abs(area - D * D) > 1e-9 * D * D:
            raise ValueError(
                f"family='b': as folhas internas ao corpo cobrem área "
                f"{area:.6g}, esperado {D*D:.6g}. O corpo não é união exata "
                f"de folhas.")
    return qt


def _side_bbox(lf: Leaf, side: int) -> tuple:
    """Extensão física do lado inteiro de uma folha."""
    if side == SIDE_BOTTOM:
        return (lf.x0, lf.y0, lf.x1, lf.y0)
    if side == SIDE_TOP:
        return (lf.x0, lf.y1, lf.x1, lf.y1)
    if side == SIDE_LEFT:
        return (lf.x0, lf.y0, lf.x0, lf.y1)
    return (lf.x1, lf.y0, lf.x1, lf.y1)         # SIDE_RIGHT


def leaf_boundary_fluxes(decomp: "Decomposition", flux_avg: dict, isub: int):
    """Fluxos dos 4 lados da folha, montados a partir dos segmentos de Γ.

    Sob quadtree um lado pode ter dois segmentos, cada um confrontando uma
    folha vizinha diferente. Como h é uniforme, cada segmento cobre um bloco
    contíguo e exato de faces finas do lado, e a montagem é concatenação — sem
    interpolação nem restrição.

    Returns
    -------
    (fS, fE, fN, fW) : arrays de tamanho (nlocx, nlocy, nlocx, nlocy).
        Orientação: fS/fN em +y, fW/fE em +x — a mesma que
        `MACDownscaler.solve_patch` espera.
    """
    lf = decomp.leaf(isub)
    f = {0: np.zeros(lf.nlocx), 1: np.zeros(lf.nlocy),
         2: np.zeros(lf.nlocx), 3: np.zeros(lf.nlocy)}
    for side, seg, ell, bbox, nfaces in decomp.segments_of(isub):
        val = flux_avg.get((isub, side, seg))
        if val is None:                       # lado externo: ṽ·n = 0
            continue
        horiz = side in (0, 2)
        origin = lf.x0 if horiz else lf.y0
        start = bbox[0] if horiz else bbox[1]
        dh = decomp.hx if horiz else decomp.hy
        off = int(round((start - origin) / dh))
        if off < 0 or off + val.size > f[side].size:
            raise ValueError(
                f"folha {isub} lado {side} seg {seg}: offset {off} + "
                f"{val.size} faces excede o lado ({f[side].size}).")
        f[side][off:off + val.size] = val
    return f[0], f[1], f[2], f[3]
