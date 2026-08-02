#!/usr/bin/env python3
"""Figura do par misto e do grau de interface na etapa de projecao (Secao 5.7.2).

    python -m figures.plot_par_misto

Le `alpha_hh.json` (BDM1) e `alpha_hh_bdm2.json` (BDM2). O caminho de RT0
(`alpha_hh_rt0.json`) NAO entra na figura: roda em quadrilateros enquanto
os demais rodam em triangulos, e os valores de alpha a partir de 1e2 sao
falha da fatoracao local. A ressalva vai ao texto.

CINCO curvas, nao oito: as quatro ordens com BDM1 (que exibem a escada do
enriquecimento), mais a ordem mais alta com BDM2 (que exibe a degradacao
suprimida). Oito curvas sobrepostas em regime moderado, onde elas sao
indistinguiveis por construcao, esconderiam o argumento.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")

from mprm import plotting

RAIZ = Path(__file__).resolve().parents[1]
EIXO_Y = "erro relativo contra a solução fina sem decomposição"


def _agrupar(caminho, por="k"):
    doc = json.loads(caminho.read_text())
    registros = doc.get("cases", doc.get("records", []))
    grupos = defaultdict(list)
    for r in registros:
        grupos[r[por]].append(r)
    for chave in grupos:
        grupos[chave].sort(key=lambda c: c["alpha"])
    return dict(sorted(grupos.items()))


def main() -> int:
    bdm1 = RAIZ / "results_alpha" / "alpha_hh.json"
    bdm2 = RAIZ / "results_alpha" / "alpha_hh_bdm2.json"
    for p in (bdm1, bdm2):
        if not p.exists():
            print(f"{p} ausente")
            return 1

    fig, ax = plotting.new_figure(1, 1)
    series = []

    curvas_bdm1 = _agrupar(bdm1)
    curvas_bdm2 = _agrupar(bdm2)

    # Quatro ordens com BDM1: a escada do enriquecimento
    for grau, cor in zip(sorted(curvas_bdm1), plotting.PALETTE):
        casos = curvas_bdm1[grau]
        series.append(dict(
            x=[c["alpha"] for c in casos],
            y=[c["err_vs_fine"] for c in casos],
            color=cor, marker="o",
            label=f"$BDM_1$, $\\ell = {grau}$"))

    # A ordem mais alta com BDM2, mesma cor, tracejada
    if sorted(curvas_bdm1) and sorted(curvas_bdm1)[-1] in curvas_bdm2:
        grau_max = sorted(curvas_bdm1)[-1]
        casos = curvas_bdm2[grau_max]
        # A cor e a mesma do BDM1 l=4, para que a comparacao seja direta
        cor_bdm2 = plotting.PALETTE[3]  # quarta cor da rampa
        series.append(dict(
            x=[c["alpha"] for c in casos],
            y=[c["err_vs_fine"] for c in casos],
            color=cor_bdm2, marker="s", ls="--",
            label=f"$BDM_2$, $\\ell = {grau_max}$"))

    plotting.curve_panel(
        ax, series, r"$\alpha$", EIXO_Y,
        legend_kw={"loc": "best", "ncol": 2, "fontsize": 7.0})
    fig.tight_layout()
    plotting.save_figure(fig, "ns", "fig_ns-par-misto")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
