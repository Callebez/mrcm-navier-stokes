"""
tests/common.py — Utilitários compartilhados pelos testes MPrM.

Centraliza o que, de outra forma, seria duplicado em cada `test_*.py`:
setup de `sys.path` (para `import mprm_core` funcionar independente do cwd),
comunicador MPI (com fallback se `mpi4py` não estiver disponível), estilo de
plotagem (paleta categórica CVD-safe, validada com o skill `dataviz`), erro
relativo, banner de seção e serialização JSON dos resultados.

Importado por todo `test_*.py` — mantém os arquivos de teste focados na
lógica do experimento, não em infraestrutura.
"""

from __future__ import annotations

import os
import sys
import json
from datetime import datetime
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# sys.path: garante que `import mprm_core` e `import mrcm.*` funcionem
# rodando de qualquer cwd. Precisa acontecer aqui (não só dentro de
# mprm_core.py) porque um `test_*.py` pode importar `mrcm.config` antes de
# `mprm_core` — a ordem de import não deve importar.
# ---------------------------------------------------------------------------
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(TESTS_DIR)              # ~/mrcm/NavierStokes
MRCM_REPO = os.path.expanduser("~/mrcm/mrcm-firedrake")
for _p in (ROOT_DIR, MRCM_REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_OUTDIR = os.path.join(ROOT_DIR, "results_alpha")


# ---------------------------------------------------------------------------
# Comunicador MPI — com fallback para ambientes sem mpi4py.
# ---------------------------------------------------------------------------

class _DummyComm:
    """Substituto sequencial de um comunicador MPI (rank 0, size 1)."""
    rank = 0
    size = 1

    def allgather(self, x):
        return [x]

    def gather(self, x, root=0):
        return [x]

    def bcast(self, x, root=0):
        return x

    def reduce(self, x, op=None, root=0):
        return x

    def Barrier(self):
        pass


def get_comm():
    """Retorna `MPI.COMM_WORLD`, ou um `_DummyComm` se `mpi4py` não existir."""
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD
    except ImportError:
        return _DummyComm()


# ---------------------------------------------------------------------------
# Estilo de plotagem — paleta categórica CVD-safe (Okabe-Ito), validada com
# scripts/validate_palette.js do skill `dataviz` (ΔE ≥ 8 em todos os pares).
# Ordem fixa por k (1..4); nunca cicla a paleta além disso.
# ---------------------------------------------------------------------------
COLORS = {1: "#0072B2", 2: "#E69F00", 3: "#009E73", 4: "#CC79A7"}
MARKERS = {1: "o", 2: "s", 3: "D", 4: "^"}
INK = "#333333"


def apply_plot_style():
    """Aplica o `rcParams` compartilhado. Chamar uma vez, antes de plotar."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "axes.edgecolor": "#BBBBBB", "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "font.size": 11, "axes.titlesize": 12,
    })
    return plt


# ---------------------------------------------------------------------------
# Métricas e I/O
# ---------------------------------------------------------------------------

def rel_err(a1, a2, b1, b2) -> float:
    """‖(a1,a2) − (b1,b2)‖₂ / ‖(b1,b2)‖₂ sobre as faces MAC (arrays NumPy)."""
    num = np.sum((a1 - b1) ** 2) + np.sum((a2 - b2) ** 2)
    den = np.sum(b1 ** 2) + np.sum(b2 ** 2)
    return float(np.sqrt(num / den))


def banner(msg: str, comm=None) -> None:
    """Cabeçalho de seção, impresso só no rank 0 (se `comm` for dado)."""
    if comm is not None and comm.rank != 0:
        return
    print("=" * 64)
    print(msg)
    print("=" * 64, flush=True)


def make_meta(study: str, comm, **fields: Any) -> dict:
    """Bloco `meta` padrão para o JSON de resultados de um teste."""
    meta = {
        "study": study,
        "date": datetime.now().isoformat(timespec="seconds"),
        "n_ranks": getattr(comm, "size", 1),
    }
    meta.update(fields)
    return meta


def save_json(results: dict, outdir: str, filename: str, comm=None) -> str:
    """Salva `results` como JSON em `outdir/filename` (só no rank 0)."""
    if comm is not None and comm.rank != 0:
        return os.path.join(outdir, filename)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, filename)
    with open(path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"  ✓ JSON: {path}")
    return path


def save_figure(fig, outdir: str, filename: str) -> str:
    """Salva uma figura matplotlib em `outdir/filename` e fecha a figura."""
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Figura: {path}")
    return path
