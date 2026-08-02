#!/usr/bin/env python3
"""Figura da verificacao contra o campo analitico de Olshanskii (Secao 5.7.1).

    python -m figures.plot_olshanskii

Le `results_alpha/convergence_olshanskii/e1_hh_alpha.json`. O erro e medido
contra a solucao fina sem decomposicao (`err_vs_fine`), que e a unica
referencia em que a taxa multiescala aparece isolada: o piso do par misto
local cancela-se entre as duas solucoes.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")

from mprm import plotting

RAIZ = Path(__file__).resolve().parents[1]
JSON = RAIZ / "results_alpha" / "convergence_olshanskii" / "e1_hh_alpha.json"


def _taxa(xs, ys):
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    n = len(lx)
    mx, my = sum(lx) / n, sum(ly) / n
    num = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    den = sum((a - mx) ** 2 for a in lx)
    return num / den if den else float("nan")


def main() -> int:
    if not JSON.exists():
        print(f"{JSON} ausente")
        return 1
    doc = json.loads(JSON.read_text())

    fig, ax = plotting.new_figure(1, 1)
    series = []
    for curva, cor in zip(doc["curves"], plotting.PALETTE):
        casos = sorted(curva["cases"], key=lambda c: c["H"])
        xs = [c["H"] for c in casos]
        ys = [c["err_vs_fine"] for c in casos]
        taxa = _taxa(xs, ys)
        series.append(dict(x=xs, y=ys, color=cor, marker="o",
                          label=f"$\\alpha = {curva['alpha']:g}$: {taxa:.2f}"))

    # Reta de referencia O(H^2) ancorada no ponto mais fino da primeira curva.
    base = sorted(doc["curves"][0]["cases"], key=lambda c: c["H"])[0]
    ref_x = [base["H"] * 0.5, base["H"]]
    series.append(dict(
        x=ref_x,
        y=[base["err_vs_fine"] * (x / base["H"]) ** 2 for x in ref_x],
        color=plotting.C["ref_line"], marker="",
        ls=plotting.LS["reference"], label=r"$\mathcal{O}(H^2)$"))

    plotting.curve_panel(
        ax, series, "$H$",
        "erro relativo contra a solução fina sem decomposição")
    fig.tight_layout()
    plotting.save_figure(fig, "ns", "fig_ns-olshanskii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
