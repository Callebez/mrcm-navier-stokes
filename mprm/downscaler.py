"""Downscaling MAC — MPrM.

MACDownscaler: resolve problemas de Neumann locais (Eq. 22) na malha MAC.
Usa SciPy sparse LU — não importa Firedrake.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from mrcm.config import MRCMConfig

# Downscaling MAC (Eqs. 21-23 do artigo) — conservação na escala fina
# ===========================================================================

class MACDownscaler:
    """Problemas locais de Neumann (Eq. 22) na malha MAC, por folha.

    Sob decomposição quadtree as folhas têm tamanhos diferentes, então a
    matriz FV de 5 pontos deixa de ser única: é cacheada por
    ``(n, m, padrão de sólido)``. Como os níveis de refinamento são poucos, o
    número de fatorações distintas é pequeno.

    O defeito de compatibilidade é **retornado** por chamada. O antigo atributo
    acumulado nunca era reiniciado entre passos e, com folhas de tamanhos
    diferentes, misturava escalas de normalização.
    """

    def __init__(self, cfg: MRCMConfig, dt: float, h: float = None,
                 *, hy: float = None):
        self.cfg = cfg
        self.dt = float(dt)
        if h is not None:
            self.hx = float(h)
            self.hy = float(hy) if hy is not None else float(h)
        else:
            self.hx = cfg.Hx / cfg.nlocx
            self.hy = cfg.Hy / cfg.nlocy
        self._lu_cache = {}       # (n, m, solid_key) -> (splu, pin)
        self.last_defect = 0.0

    @property
    def h(self) -> float:
        """Espaçamento de malha (atalho para hx; células quadradas)."""
        return self.hx

    def reset_defect(self):
        """Zera o acumulador de defeito (chamar a cada passo de tempo)."""
        self.last_defect = 0.0

    # ── matrizes ────────────────────────────────────────────────────────
    def _build_lu(self, n, m, solid):
        """Fatoração FV de 5 pontos para o patch (n, m) e um padrão de sólido.

        Células sólidas viram p = 0 (linha identidade, desacopladas) e as faces
        sólidas somem do estêncil (no-flow). O pino da pressão vai na primeira
        célula de FLUIDO — o sistema é Neumann puro.
        """
        dt, hx, hy = self.dt, self.hx, self.hy

        def lid(i, j):
            return i + j * n

        rows, cols, vals = [], [], []
        for j in range(m):
            for i in range(n):
                c = lid(i, j)
                if solid is not None and solid[i, j]:
                    rows.append(c); cols.append(c); vals.append(1.0)
                    continue
                diag = 0.0
                for (di, dj, d, ln) in ((1, 0, hx, hy), (-1, 0, hx, hy),
                                        (0, 1, hy, hx), (0, -1, hy, hx)):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < n and 0 <= jj < m and \
                            not (solid is not None and solid[ii, jj]):
                        w = dt * ln / d
                        rows.append(c); cols.append(lid(ii, jj))
                        vals.append(w); diag -= w
                rows.append(c); cols.append(c); vals.append(diag)
        A = sp.csr_matrix((vals, (rows, cols)), shape=(n * m, n * m)).tolil()
        if solid is None:
            pin = 0
        else:
            pin = int(np.argmin(solid.flatten(order="F")))
        A[pin, :] = 0.0
        A[pin, pin] = 1.0
        return spla.splu(A.tocsc()), pin

    def _get_lu(self, n, m, solid):
        key = (n, m, None if solid is None else solid.tobytes())
        if key not in self._lu_cache:
            self._lu_cache[key] = self._build_lu(n, m, solid)
        return self._lu_cache[key]

    # ── solve ───────────────────────────────────────────────────────────
    def solve_patch(self, divw_loc, fW, fE, fS, fN, solid_loc=None):
        """Resolve o Neumann local (Eq. 22) e devolve ``(ṽ₁, ṽ₂, defeito)``.

        O tamanho do patch vem de ``divw_loc.shape`` — não do cfg —, o que é o
        que permite folhas de tamanhos diferentes.
        """
        dt, hx, hy = self.dt, self.hx, self.hy
        n, m = divw_loc.shape
        solid = None if solid_loc is None else \
            np.asarray(solid_loc, dtype=bool).reshape(n, m)

        # Folha 100% sólida: nada a resolver. Sem esta guarda, nfluid = 0 e a
        # normalização do defeito divide por zero (spec §4.5, item 1).
        if solid is not None and solid.all():
            self.last_defect = max(self.last_defect, 0.0)
            return np.zeros((n + 1, m)), np.zeros((n, m + 1)), 0.0

        rhs = (divw_loc * hx * hy).copy()
        rhs[0, :] += fW * hy
        rhs[-1, :] -= fE * hy
        rhs[:, 0] += fS * hx
        rhs[:, -1] -= fN * hx

        lu, pin = self._get_lu(n, m, solid)
        if solid is None:
            b = rhs.flatten(order="F")
            defect = float(b.sum())
            norm = hx * hy * n * m
            b -= defect / b.size
            b[pin] = 0.0
            p = lu.solve(b).reshape((n, m), order="F")
        else:
            rhs[solid] = 0.0
            b = rhs.flatten(order="F")
            fluid = ~solid.flatten(order="F")
            nfluid = int(fluid.sum())
            defect = float(b[fluid].sum())
            norm = hx * hy * nfluid
            b[fluid] -= defect / nfluid
            b[~fluid] = 0.0
            b[pin] = 0.0
            p = lu.solve(b).reshape((n, m), order="F")
            p[solid] = 0.0

        defect_rel = abs(defect) / norm
        self.last_defect = max(self.last_defect, defect_rel)

        vt1 = np.empty((n + 1, m))
        vt1[1:-1, :] = dt * (p[1:, :] - p[:-1, :]) / hx
        vt1[0, :], vt1[-1, :] = fW, fE
        vt2 = np.empty((n, m + 1))
        vt2[:, 1:-1] = dt * (p[:, 1:] - p[:, :-1]) / hy
        vt2[:, 0], vt2[:, -1] = fS, fN

        if solid is not None:
            fx = np.zeros((n + 1, m), bool); fx[:-1, :] |= solid; fx[1:, :] |= solid
            fy = np.zeros((n, m + 1), bool); fy[:, :-1] |= solid; fy[:, 1:] |= solid
            vt1[fx] = 0.0
            vt2[fy] = 0.0
        return vt1, vt2, defect_rel
