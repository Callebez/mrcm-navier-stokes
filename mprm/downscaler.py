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
    """Problemas locais de Neumann (Eq. 22) na malha MAC, por subdomínio.

    Matriz FV (5 pontos, K=δt) idêntica para todos os subdomínios: montada e
    fatorada UMA vez (offline). O sistema é singular (Neumann puro); o RHS é
    balanceado (compatibilizado) e a pressão pinada na célula 0. O defeito de
    compatibilidade por subdomínio (violação da continuidade média de fluxo
    após a média da Eq. 21) é reportado em `last_defect`.
    """

    def __init__(self, cfg: MRCMConfig, dt: float):
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        self.cfg = cfg
        self.dt = float(dt)
        n, m = cfg.nlocx, cfg.nlocy
        hx, hy = cfg.Hx / n, cfg.Hy / m
        self.hx, self.hy = hx, hy

        def lid(i, j):
            return i + j * n

        rows, cols, vals = [], [], []
        for j in range(m):
            for i in range(n):
                c = lid(i, j)
                diag = 0.0
                for (di, dj, d, ln) in ((1, 0, hx, hy), (-1, 0, hx, hy),
                                        (0, 1, hy, hx), (0, -1, hy, hx)):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < n and 0 <= jj < m:
                        w = dt * ln / d
                        rows.append(c); cols.append(lid(ii, jj)); vals.append(w)
                        diag -= w
                rows.append(c); cols.append(c); vals.append(diag)
        A = sp.csr_matrix((vals, (rows, cols)), shape=(n * m, n * m))
        # pino da pressão na célula 0 (remoção do modo constante)
        A = A.tolil()
        A[0, :] = 0.0
        A[0, 0] = 1.0
        self.lu = spla.splu(A.tocsc())
        self._A_base = A.tocsc()          # sem sólido (referência p/ máscara)
        self._masked_lu = {}              # solid_loc.tobytes() -> splu
        self.last_defect = 0.0

    def _build_masked_lu(self, solid_loc):
        """Fatoração FV para um padrão de sólido: células sólidas viram
        p=0 (linha identidade, desacopladas); faces sólidas somem do estêncil
        (no-flow). Cacheada por padrão."""
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        n, m = self.cfg.nlocx, self.cfg.nlocy
        dt, hx, hy = self.dt, self.hx, self.hy
        solid = solid_loc.reshape(n, m)

        def lid(i, j):
            return i + j * n

        rows, cols, vals = [], [], []
        for j in range(m):
            for i in range(n):
                c = lid(i, j)
                if solid[i, j]:
                    rows.append(c); cols.append(c); vals.append(1.0)
                    continue
                diag = 0.0
                for (di, dj, d, ln) in ((1, 0, hx, hy), (-1, 0, hx, hy),
                                        (0, 1, hy, hx), (0, -1, hy, hx)):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < n and 0 <= jj < m and not solid[ii, jj]:
                        w = dt * ln / d
                        rows.append(c); cols.append(lid(ii, jj))
                        vals.append(w); diag -= w
                rows.append(c); cols.append(c); vals.append(diag)
        A = sp.csr_matrix((vals, (rows, cols)), shape=(n * m, n * m)).tolil()
        # pino: primeira célula de FLUIDO
        pin = int(np.argmin(solid.flatten(order="F")))
        A[pin, :] = 0.0; A[pin, pin] = 1.0
        return spla.splu(A.tocsc()), pin

    def _get_masked(self, solid_loc):
        key = solid_loc.tobytes()
        if key not in self._masked_lu:
            self._masked_lu[key] = self._build_masked_lu(solid_loc)
        return self._masked_lu[key]

    def solve_patch(self, divw_loc, fW, fE, fS, fN, solid_loc=None):
        """Resolve o Neumann local (Eq. 22) e devolve ṽ (MAC).

        Se `solid_loc` (n,m bool) for dado, células sólidas são excluídas
        (p=0) e faces sólidas ficam com ṽ·n=0.
        """
        cfg, dt = self.cfg, self.dt
        n, m = cfg.nlocx, cfg.nlocy
        hx, hy = self.hx, self.hy

        rhs = (divw_loc * hx * hy).copy()
        rhs[0, :] += fW * hy
        rhs[-1, :] -= fE * hy
        rhs[:, 0] += fS * hx
        rhs[:, -1] -= fN * hx

        if solid_loc is None:
            b = rhs.flatten(order="F")
            defect = b.sum()
            self.last_defect = max(self.last_defect,
                                   abs(defect) / (hx * hy * n * m))
            b -= defect / b.size
            b[0] = 0.0
            p = self.lu.solve(b).reshape((n, m), order="F")
            solid = np.zeros((n, m), dtype=bool)
        else:
            solid = solid_loc.reshape(n, m)
            lu, pin = self._get_masked(solid)
            rhs[solid] = 0.0                       # fonte nula no sólido
            b = rhs.flatten(order="F")
            fluid = ~solid.flatten(order="F")
            nfluid = int(fluid.sum())
            defect = b[fluid].sum()
            self.last_defect = max(self.last_defect,
                                   abs(defect) / (hx * hy * nfluid))
            b[fluid] -= defect / nfluid            # compatibiliza só no fluido
            b[~fluid] = 0.0
            b[pin] = 0.0
            p = lu.solve(b).reshape((n, m), order="F")
            p[solid] = 0.0

        vt1 = np.empty((n + 1, m))
        vt1[1:-1, :] = dt * (p[1:, :] - p[:-1, :]) / hx
        vt1[0, :], vt1[-1, :] = fW, fE
        vt2 = np.empty((n, m + 1))
        vt2[:, 1:-1] = dt * (p[:, 1:] - p[:, :-1]) / hy
        vt2[:, 0], vt2[:, -1] = fS, fN

        if solid_loc is not None:
            fx = np.zeros((n + 1, m), bool); fx[:-1, :] |= solid; fx[1:, :] |= solid
            fy = np.zeros((n, m + 1), bool); fy[:, :-1] |= solid; fy[:, 1:] |= solid
            vt1[fx] = 0.0
            vt2[fy] = 0.0
        return vt1, vt2


# ===========================================================================
