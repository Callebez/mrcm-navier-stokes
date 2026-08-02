#!/usr/bin/env python3
"""Figura dos perfis da cavidade contra Ghia et al. (1982) — Secao 5.7.4.

    python -m figures.plot_ghia

Le `results_alpha/cavity_ghia/ghia_validation.json`. Nao roda simulacao:
a campanha de Re=100 sozinha custou 21 699 s de parede.

Os valores tabelados de Ghia sao importados de `tests/test_cavity_ghia.py`,
que e a fonte unica e versionada no repositorio.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import numpy as np

from mprm import plotting

RAIZ = Path(__file__).resolve().parents[1]
JSON = RAIZ / "results_alpha" / "cavity_ghia" / "ghia_validation.json"

# Valores tabelados de Ghia et al. (1982), J. Comput. Phys. 48:387-411.
# Fonte unica no repositorio: tests/test_cavity_ghia.py, GHIA_U e GHIA_V.
_ghia_fonte = (RAIZ / "tests" / "test_cavity_ghia.py").read_text(
    encoding="utf-8")
_ns = {}
_exec_ate = _ghia_fonte.index("def ", _ghia_fonte.index("GHIA_V ="))
exec(_ghia_fonte[_ghia_fonte.index("GHIA_U ="):_exec_ate], _ns)
GHIA_U = _ns["GHIA_U"]
GHIA_V = _ns["GHIA_V"]


def main() -> int:
    if not JSON.exists():
        print(f"{JSON} ausente")
        return 1
    doc = json.loads(JSON.read_text())
    casos = sorted(doc["cases"], key=lambda c: c["Re"])

    fig, axes = plotting.new_figure(2, len(casos), height_per_row=0.70)

    for col, caso in enumerate(casos):
        re = caso["Re"]
        # Painel de cima: u(y) na linha vertical central x = 0,5
        ax = axes[0][col]
        yu = np.array(caso["y_u"])
        up = np.array(caso["u_profile"])
        ax.plot(up, yu, color=plotting.C["mrcm"], lw=1.4,
                label="método multiescala")
        ax.plot(GHIA_U[re]["u"], GHIA_U[re]["y"], "o",
                color=plotting.C["ref_fem"], ms=4, mfc="none",
                label="Ghia et al. (1982)")
        ax.set_xlabel("$u$")
        ax.set_ylabel("$y$")
        ax.legend(loc="best", fontsize=7.0, title=f"$Re = {re}$")
        ax.grid(True, alpha=0.25, linestyle=":")
        plotting.panel_tag(ax, "abc"[col])

        # Painel de baixo: v(x) na linha horizontal central y = 0,5
        ax = axes[1][col]
        xv = np.array(caso["x_v"])
        vp = np.array(caso["v_profile"])
        ax.plot(xv, vp, color=plotting.C["mrcm"], lw=1.4)
        ax.plot(GHIA_V[re]["x"], GHIA_V[re]["v"], "o",
                color=plotting.C["ref_fem"], ms=4, mfc="none")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$v$")
        ax.grid(True, alpha=0.25, linestyle=":")
        plotting.panel_tag(ax, "def"[col])

    fig.tight_layout()
    plotting.save_figure(fig, "ns", "fig_ns-ghia")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
