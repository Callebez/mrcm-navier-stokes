#!/usr/bin/env python3
"""Figura do par misto e do grau de interface na etapa de projecao (Secao 5.7.2).

    python -m figures.plot_par_misto

Le `alpha_hh.json` (BDM1) e `alpha_hh_bdm2.json` (BDM2). O caminho de RT0
(`alpha_hh_rt0.json`) NAO entra na figura: roda em quadrilateros enquanto
os demais rodam em triangulos, e os valores de alpha a partir de 1e2 sao
falha da fatoracao local. A ressalva vai ao texto.

CINCO curvas, nao oito. As quatro ordens de interface sao tracadas com
BDM2, e nao com BDM1, e a escolha e deliberada: BDM1 DEGRADA em alpha grande
quando o espaco de interface e rico, de modo que uma escada tracada com ele
misturaria o efeito do grau de interface com um artefato do par misto. Com
BDM2 a escada e limpa em toda a varredura, e o efeito do grau de interface
aparece isolado. A quinta curva e BDM1 na ordem mais alta, tracejada: ela
mede exatamente o que o par de ordem mais baixa custa, e so se separa da
sua gemea em alpha grande.

Oito curvas sobrepostas em regime moderado, onde elas sao indistinguiveis
por construcao, esconderiam o argumento.
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
EIXO_Y = plotting.err_label("u", "fina_sem_decomposicao")


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

    # Quatro ordens de interface com BDM2: a escada do enriquecimento, limpa.
    # Com BDM1 a escada seria contaminada pela degradacao do par em alpha
    # grande, que e' efeito do PAR MISTO e nao do grau de interface.
    for grau, cor in zip(sorted(curvas_bdm2), plotting.PALETTE):
        casos = curvas_bdm2[grau]
        series.append(dict(
            x=[c["alpha"] for c in casos],
            y=[c["err_vs_fine"] for c in casos],
            color=cor, marker="o",
            label=f"$BDM_2$, $\\ell = {grau}$"))

    # BDM1 na ordem mais alta, tracejado e na mesma cor da sua gemea: e' a
    # medida do que o par de ordem mais baixa custa.
    if sorted(curvas_bdm2) and sorted(curvas_bdm2)[-1] in curvas_bdm1:
        grau_max = sorted(curvas_bdm2)[-1]
        casos = curvas_bdm1[grau_max]
        cor_par = plotting.PALETTE[len(sorted(curvas_bdm2)) - 1]
        series.append(dict(
            x=[c["alpha"] for c in casos],
            y=[c["err_vs_fine"] for c in casos],
            color=cor_par, marker="s", ls="--",
            label=f"$BDM_1$, $\\ell = {grau_max}$"))

    plotting.curve_panel(
        ax, series, r"$\alpha$", EIXO_Y,
        legend_kw={"loc": "best", "ncol": 2, "fontsize": 8.5})
    plotting.save_figure(fig, "ns", "fig_ns-par-misto")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
