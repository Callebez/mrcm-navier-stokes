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

    def __init__(self, skeleton, leaves, h, Lx, Ly):
        self.skeleton = skeleton
        self.leaves = list(leaves)
        self.h = float(h)
        self.Lx, self.Ly = float(Lx), float(Ly)
        self.Nx = int(round(Lx / h))
        self.Ny = int(round(Ly / h))
        self._by_id = {lf.isub: lf for lf in self.leaves}
        self._gamma = None          # construído sob demanda

    # ── fábricas ────────────────────────────────────────────────────────
    @classmethod
    def uniform(cls, Ndomx: int, Ndomy: int, nloc: int,
                Lx: float, Ly: float) -> "Decomposition":
        """Decomposição cartesiana — o caminho retrocompatível."""
        Hx, Hy = Lx / Ndomx, Ly / Ndomy
        h = Hx / nloc
        if abs(Hy / nloc - h) > 1e-12 * h:
            raise ValueError(
                f"células não quadradas: Hx/nloc={h:.6g} != Hy/nloc="
                f"{Hy/nloc:.6g}. Ajuste Ndomx/Ndomy para hx == hy.")
        leaves = []
        for isub in range(Ndomx * Ndomy):
            ix, iy = isub % Ndomx, isub // Ndomx
            leaves.append(Leaf(isub=isub,
                               x0=ix * Hx, y0=iy * Hy,
                               x1=(ix + 1) * Hx, y1=(iy + 1) * Hy,
                               nlocx=nloc, nlocy=nloc,
                               i0=ix * nloc, j0=iy * nloc, level=0))
        return cls(Skeleton.cartesian(Ndomx, Ndomy), leaves, h, Lx, Ly)

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
            nfaces = int(round(span / self.h))
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


def _side_bbox(lf: Leaf, side: int) -> tuple:
    """Extensão física do lado inteiro de uma folha."""
    if side == SIDE_BOTTOM:
        return (lf.x0, lf.y0, lf.x1, lf.y0)
    if side == SIDE_TOP:
        return (lf.x0, lf.y1, lf.x1, lf.y1)
    if side == SIDE_LEFT:
        return (lf.x0, lf.y0, lf.x0, lf.y1)
    return (lf.x1, lf.y0, lf.x1, lf.y1)         # SIDE_RIGHT
