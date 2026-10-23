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

    # `height_per_row` é a altura de cada LINHA de painéis, EM POLEGADAS.
    # `0.70` aqui foi lido como fração da largura e produziu uma figura de
    # 1,4 in para seis painéis, que saíram sobrepostos e ilegíveis. Com três
    # colunas em 6,30 in cada painel tem 2,1 in de largura, e 2,45 in de
    # altura por linha lhe dá proporção próxima da quadrada, que é a certa
    # para um perfil sobre a linha de centro de uma cavidade quadrada.
    fig, axes = plotting.new_figure(2, len(casos), height_per_row=2.45)

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
        # O número de Reynolds é CONFIGURAÇÃO: vai na figura como anotação, e
        # não como título de legenda. Como título ele arrastava consigo uma
        # caixa de legenda por painel, que cobria o próprio perfil.
        ax.annotate(f"$Re = {re}$", xy=(0.5, 1.03), xycoords="axes fraction",
                    ha="center", va="bottom", fontsize=9)
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
        ax.grid(True, alpha=0.25, linestyle=":")
        plotting.panel_tag(ax, "def"[col])

    # Rótulo do eixo y só na coluna da esquerda: as três colunas partilham a
    # grandeza, e repeti-lo três vezes rouba largura dos painéis.
    axes[0][0].set_ylabel("$y$")
    axes[1][0].set_ylabel("$v$")
    for linha in axes:
        for ax in linha[1:]:
            ax.set_yticklabels([])
    for linha in axes:
        lo = min(a.get_ylim()[0] for a in linha)
        hi = max(a.get_ylim()[1] for a in linha)
        for ax in linha:
            ax.set_ylim(lo, hi)

    # Uma legenda para os seis painéis: as séries são as mesmas em todos.
    plotting.shared_legend(fig, axes[0][0], ncol=2, fontsize=8.0)
    plotting.save_figure(fig, "ns", "fig_ns-ghia")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
