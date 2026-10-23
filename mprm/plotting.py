"""Estilo de figuras dos estudos — cópia versionada de `plot_style.py`.

`convergence_study/` saiu do versionamento no commit `f45c6cf` e está no
`.gitignore`, de modo que `convergence_study/plot_style.py` NÃO pode ser
importado por um pacote versionado. Esta é a cópia que o repositório
carrega; a paleta e os `rcParams` são os validados de lá, e as únicas
mudanças são de caminho (`save_figure` grava em `studies/<id>/plots/`) e
dois atalhos de eixo log-log.

Regra dura herdada do guia de estilo (§5): eixo y único. `twinx`/`twiny`
são PROIBIDOS — duas grandezas de escalas diferentes viram dois painéis
alinhados com x compartilhado (`stacked_panels`).
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from . import paths

# ═══════════════════════════════════════════════════════════════════
# Paleta (colorblind-safe, print-safe, ColorBrewer Set1 adaptada)
# ═══════════════════════════════════════════════════════════════════

C = {
    # ── Categórico: famílias de elementos ──
    "family_bdm":       "#2166ac",   # BDM — azul profundo
    "family_rt":        "#b2182b",   # RT  — vermelho tijolo

    # ── Categórico: referências / métodos ──
    "ref_fem":          "#4daf4a",   # referência FEM — verde
    "mrcm":             "#ff7f00",   # solução MRCM — laranja

    # ── Categórico: regimes ──
    "regime_mmmfem":    "#377eb8",   # α→0
    "regime_mhm":       "#e41a1c",   # α→∞

    # ── Categórico: solvers ──
    "direct_solver":    "#984ea3",
    "iterative_solver": "#a65628",

    # ── Categórico: camadas do SPE10 ──
    "layer_0":          "#2166ac",
    "layer_35":         "#b2182b",

    # ── ORDINAL: fases do pipeline (barras empilhadas do 1e) ──
    # Fases 1→5 são ORDENADAS ⇒ rampa de UM matiz (azul), claro→escuro,
    # com luminosidade monótona. Uma paleta categórica aqui falharia os
    # checks de ordinalidade.
    "phase0":           "#c6dbef",
    "phase1":           "#79abdf",
    "phase2":           "#5493cb",
    "phase3":           "#3474ae",
    "phase4_5":         "#1d4e7f",

    # ── Anotações ──
    "ideal":            "#000000",
    "ref_line":         "#888888",
    "annotation":       "#555555",
    "grid":             "#cccccc",
}

# Rampa categórica de uso geral, para curvas indexadas por um parâmetro
# (pares (ℓ_U, ℓ_P), contrastes, valores de α). Ordem estável.
#
# Base Okabe-Ito, desenhada para deficiência de visão cromática e validada
# em impressão. A rampa ColorBrewer anterior não sustentava sete categorias:
# `#2166ac` e `#377eb8` distavam ΔE = 12,9 em visão normal e ΔE = 2,9 sob
# protanopia, e caíam lado a lado justamente nas figuras de sete pares de
# graus de interface, onde as duas curvas colididas eram a pior e a melhor.
# Esta rampa mantém ΔE ≥ 26 em visão normal e ΔE ≥ 19 sob deuteranopia e
# protanopia até a nona cor (`studies/tests/test_plotting_style.py`).
# Os dois azuis diferem em luminância (46 e 70), de modo que separam
# também em escala de cinza.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7",
           "#8B4A0F", "#56B4E9", "#332288", "#3A3A3A"]

# ═══════════════════════════════════════════════════════════════════
# Cores por solução de referência
#
# Não existe "régua R1/R2/R3": cada cor nomeia a SOLUÇÃO contra a qual o
# erro foi medido, e as etiquetas de eixo repetem esse nome por extenso.
# ═══════════════════════════════════════════════════════════════════

REF = {
    "analitica":             "#b2182b",
    "fina_sem_decomposicao": "#2166ac",
    "monolitica_mais_fina":  "#4daf4a",
}

REF_LABEL = {
    "analitica": "contra a solução analítica",
    "fina_sem_decomposicao": "contra a solução fina sem decomposição",
    "monolitica_mais_fina":
        "contra a solução monolítica em malha oito vezes mais fina",
}

# ═══════════════════════════════════════════════════════════════════
# Rótulo do eixo de erro
#
# `REF_LABEL` nomeia a referência por extenso e serve à LEGENDA LaTeX e ao
# arquivo de proveniência. No EIXO ele não cabe: "erro relativo $L^2$ de
# pressão contra a solução monolítica em malha oito vezes mais fina" é mais
# alto que os eixos e saía truncado no meio da palavra em oito das figuras.
#
# No eixo vai a notação matemática, que nomeia a solução de referência pelo
# símbolo e ocupa um terço do espaço. Quem é cada símbolo:
#
#   p,  u          solução analítica do problema manufaturado
#   p_h, u_h       elementos finitos mistos na malha fina, SEM decomposição
#   p_{h/8}        monolítica no grid oito vezes mais fino que o dado
#   p_H, u_H       solução multiescala — a que está sendo medida
#
# A legenda LaTeX continua declarando por extenso qual é a referência, que
# é onde a regra do plano a exige.
# ═══════════════════════════════════════════════════════════════════

REF_SYMBOL = {
    "analitica":             "{campo}",
    "fina_sem_decomposicao": "{campo}_h",
    "monolitica_mais_fina":  "{campo}_{{h/8}}",
}


def err_label(campo: str, reference: str) -> str:
    """Rótulo curto do eixo de erro, em notação matemática.

    ``campo`` é o símbolo da grandeza (``"p"``, ``"u"``); ``reference`` é
    uma chave de ``REF``. Devolve, por exemplo, para
    ``err_label("p", "monolitica_mais_fina")``::

        $\\|p_H - p_{h/8}\\|_{L^2} / \\|p_{h/8}\\|_{L^2}$
    """
    if reference not in REF_SYMBOL:
        raise KeyError(f"referência desconhecida: {reference!r}")
    ref = REF_SYMBOL[reference].format(campo=campo)
    return (rf"$\|{campo}_H - {ref}\|_{{L^2}}"
            rf"\,/\,\|{ref}\|_{{L^2}}$")

# \the\textwidth da classe USPSC = 455,0 pt = 6,2960 in.
FIG_W = {"full": 6.30, "half": 3.05}

M = {
    "bdm": "o", "rt": "s", "fem": "^", "mrcm": "D",
    "direct": "o", "iterative": "s",
    "p": "o", "u": "s", "hdiv": "^",
}

LS = {
    "pre_asymptotic": "--",
    "asymptotic":     "-",
    "reference":      ":",
    "ideal":          "--",
    "dashed":         "--",
    "dotted":         ":",
    "dashdot":        "-.",
}

# Ordem FIXA da rampa de fases (empilhamento de baixo para cima).
PHASES = [C["phase0"], C["phase1"], C["phase2"], C["phase3"],
          C["phase4_5"]]

# Barras: borda branca = "respiro" entre segmentos que se tocam.
BAR = dict(edgecolor="white", linewidth=0.8)

CMAP_SEQUENTIAL = "viridis"
CMAP_DIVERGING = "cividis"
CMAP_MAGNITUDE = "plasma"


def get_cmap(name=CMAP_SEQUENTIAL):
    return plt.get_cmap(name)


# ═══════════════════════════════════════════════════════════════════
# rcParams
# ═══════════════════════════════════════════════════════════════════

def apply_style() -> None:
    """Aplica o estilo das figuras da dissertação globalmente."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "font.size": 11,
        "mathtext.fontset": "dejavuserif",

        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.titlesize": 13,

        "figure.dpi": 150,
        "savefig.dpi": 300,
        # `bbox="tight"` RECORTA a página para a tinta desenhada e assim
        # DESCARTA o figsize: as páginas saíam de 410 a 454 pt em vez de 455,
        # e `\includegraphics[width=\textwidth]` reescalava cada figura por um
        # fator diferente, fazendo o corpo de letra variar até 11% entre
        # figuras vizinhas. A largura só é travada se a página for a figura.
        # O ajuste do conteúdo passa a ser do layout restrito (`new_figure`).
        "savefig.bbox": "standard",
        "savefig.pad_inches": 0.0,

        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": ":",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
        "lines.markersize": 6,
        "legend.framealpha": 0.85,
        "legend.edgecolor": "#cccccc",
    })


def new_figure(nrows: int = 1, ncols: int = 1, width: str = "full", **kwargs):
    """Figura com a largura travada na ``\\textwidth`` da dissertação.

    ``width`` é ``"full"`` ou ``"half"``. Passar ``figsize`` explícito é erro:
    a largura não é negociável, porque as figuras precisam formar um conjunto
    quando postas lado a lado no documento.

    ``height_per_row`` é a altura de CADA LINHA de painéis, **em polegadas**.
    Por omissão vale ``0.74 * largura``, que é onde mora a armadilha: o valor
    default é calculado a partir da largura e por isso PARECE uma fração dela.
    Quem passou ``height_per_row=0.70`` supondo fração obteve 0,70 in por
    linha e uma figura de 126 pt de altura com seis painéis sobrepostos. Daí
    a validação abaixo: nenhuma linha de painéis útil mede menos de uma
    polegada, de modo que um valor menor só pode ser fração passada por
    engano.

    O layout é restrito (`constrained`), e não `tight_layout`: o layout
    restrito acomoda o conteúdo DENTRO do figsize dado, ao passo que
    `bbox_inches="tight"` recortava a página e destruía a trava de largura.
    Por isso `fig.tight_layout()` não deve ser chamado sobre estas figuras.
    """
    if "figsize" in kwargs:
        raise ValueError(
            "figsize explícito é proibido; use width='full' ou 'half'")
    w = FIG_W[width]
    hpr = kwargs.pop("height_per_row", 0.74 * w)
    if hpr < 1.0:
        raise ValueError(
            f"height_per_row={hpr} é altura POR LINHA em polegadas, não "
            f"fração da largura; use algo como {hpr * w:.2f}")
    kwargs.setdefault("layout", "constrained")
    return plt.subplots(nrows, ncols, figsize=(w, hpr * nrows), **kwargs)


def stacked_panels(nrows: int = 2, height_ratios=None, **kwargs):
    """Painéis empilhados com x compartilhado — o substituto OBRIGATÓRIO do
    eixo duplo. Ex.: erro em cima e κ₁(A_Γ) embaixo, na varredura de α."""
    defaults = dict(
        figsize=(FIG_W["full"], 2.55 * nrows),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": height_ratios or [1] * nrows},
    )
    defaults.update(kwargs)
    return plt.subplots(nrows, 1, **defaults)


def panel_tag(ax, letter: str) -> None:
    """Marcador ``(a)``, ``(b)``, … no canto superior esquerdo do painel.

    Substitui ``set_title``: o conteúdo do painel é explicado na legenda
    LaTeX, e não dentro da figura.
    """
    ax.text(-0.02, 1.04, f"({letter})", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="right")


def loglog_axes(ax, xlabel: str, ylabel: str) -> None:
    """Eixos log-log com grade menor — o par padrão dos estudos."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", alpha=0.25, linestyle=":")
    ax.grid(True, which="minor", alpha=0.12, linestyle=":")


def add_slope_triangle(ax, x, y, slope, label, color=None, alpha=0.5,
                       lw=0.8, fontsize=8) -> None:
    """Triângulo de inclinação de referência num gráfico log-log."""
    if color is None:
        color = C["ref_line"]
    x_end = x * 0.5
    y_end = y * (0.5 ** slope)
    ax.loglog([x, x_end], [y, y_end], color=color, lw=lw, alpha=alpha,
              ls=LS["reference"])
    ax.text(x_end * 1.05, y_end * 0.95, label, fontsize=fontsize,
            color=color, va="top")


def slope_reference(ax, xs, ys, slope, label, color=None) -> None:
    """Reta de inclinação de referência ancorada DENTRO da faixa de dados.

    `add_slope_triangle` ancora onde o chamador mandar, e ancorar no ponto
    extremo joga a reta para fora do eixo, onde ela some ou colide com a
    legenda. Esta versão ancora no ponto mais grosso e desce meia década,
    de modo que a reta acompanha a curva sem cobri-la.
    """
    xs, ys = list(xs), list(ys)
    if len(xs) < 2:
        return
    i = max(range(len(xs)), key=lambda k: xs[k])       # ponto mais grosso
    x0, y0 = xs[i], ys[i] * 0.35
    add_slope_triangle(ax, x0, y0, slope, label, color=color)


def annotate_rate(ax, xs, ys, rate: float, label: str, color=None) -> None:
    """Ancora o triângulo de inclinação no ponto mais fino da curva."""
    xs = list(xs)
    ys = list(ys)
    if not xs:
        return
    i = min(range(len(xs)), key=lambda k: xs[k])
    add_slope_triangle(ax, xs[i] * 2.2, ys[i] * 2.2, rate, label, color=color)


def shade_regime(ax, x_start, x_end, color, alpha=0.06, label=None) -> None:
    ax.axvspan(x_start, x_end, alpha=alpha, color=color, label=label)


def regime_annotations(ax, alpha_lo: float = 1e-2, alpha_hi: float = 1e2,
                       fontsize: float = 7.5) -> None:
    """Faixas sombreadas e rótulos dos regimes-limite num eixo de α.

    O argumento da figura de varredura de α é que a curva SATURA nas duas
    pontas, isto é, que a varredura cobre os regimes-limite inteiros e não
    um trecho do meio. As faixas mostram onde cada limite vale: abaixo de
    ``alpha_lo`` o MRCM tende ao MMMFEM, acima de ``alpha_hi`` tende ao MHM.

    Chamar DEPOIS de traçar as curvas: os limites do eixo são lidos daqui, e
    as faixas são fixadas com `axvspan`, que acompanha o eixo y.
    """
    x0, x1 = ax.get_xlim()
    ax.axvspan(x0, alpha_lo, color=C["regime_mmmfem"], alpha=0.07, zorder=0)
    ax.axvspan(alpha_hi, x1, color=C["regime_mhm"], alpha=0.07, zorder=0)
    # Os rótulos vão ACIMA do eixo, e não dentro: dentro eles caíam sobre a
    # curva de velocidade, que satura justamente nas duas pontas — isto é,
    # exatamente onde as faixas estão.
    for x, texto, cor in (
            (math.sqrt(max(x0, 1e-300) * alpha_lo),
             r"MMMFEM ($\alpha\!\to\!0$)", C["regime_mmmfem"]),
            (math.sqrt(alpha_hi * x1),
             r"MHM ($\alpha\!\to\!\infty$)", C["regime_mhm"])):
        ax.text(x, 1.01, texto, transform=ax.get_xaxis_transform(),
                fontsize=fontsize, color=cor, ha="center", va="bottom",
                alpha=0.9)
    ax.set_xlim(x0, x1)


def save_figure(fig, study_id: str, name: str) -> Path:
    """Grava PDF (300 dpi) e PNG (200 dpi) em `studies/<study_id>/plots/`.

    Grava com a página IGUAL ao ``figsize``. Sem ``bbox_inches="tight"``:
    recortar a página para a tinta desenhada descarta a trava de largura e
    faz cada figura chegar ao documento com um fator de escala diferente.
    """
    plot_dir = paths.plots_dir(study_id)
    out = plot_dir / f"{name}.pdf"
    largura = fig.get_figwidth()
    if not any(abs(largura - v) < 1e-6 for v in FIG_W.values()):
        raise ValueError(
            f"{name}: largura {largura:.3f} in não é nenhuma de "
            f"{sorted(FIG_W.values())}; a figura não foi criada por "
            f"new_figure/stacked_panels")
    for ext, dpi in (("pdf", 300), ("png", 200)):
        path = plot_dir / f"{name}.{ext}"
        fig.savefig(path, dpi=dpi)
        print(f"  gravado: {path}")
    plt.close(fig)
    return out


def curve_panel(ax, series, xlabel: str, ylabel: str,
                logx: bool = True, logy: bool = True,
                legend_kw=None) -> None:
    """Desenha uma família de curvas num painel, no estilo da dissertação.

    ``series`` é uma lista de dicionários com as chaves:
      ``x``, ``y``   — sequências de mesmo comprimento
      ``label``      — texto da legenda (já com a taxa, quando houver)
      ``color``      — cor (de ``C``, ``REF`` ou ``PALETTE``)
      ``marker``     — chave de ``M`` ou símbolo do matplotlib
      ``ls``         — estilo de linha (opcional; sólido por omissão)
      ``slope``      — inclinação do triângulo de referência (opcional)
      ``slope_label``— rótulo do triângulo (obrigatório se ``slope`` for dado)
    """
    for s in series:
        ax.plot(s["x"], s["y"],
                marker=M.get(s.get("marker", ""), s.get("marker", "o")),
                color=s["color"], ls=s.get("ls", "-"), label=s["label"])
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    # As retas de referência entram depois das curvas, para que a âncora
    # possa ser calculada com os eixos já em escala logarítmica.
    for s in series:
        if s.get("slope") is not None:
            slope_reference(ax, s["x"], s["y"], s["slope"], s["slope_label"],
                            color=s["color"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", alpha=0.25, linestyle=":")
    ax.grid(True, which="minor", alpha=0.12, linestyle=":")
    # Rótulos de escala menor colidem quando a faixa cobre várias décadas, e
    # são a ÚNICA marcação quando ela cobre menos de uma. Decidir pelo dado.
    if logx:
        _ajustar_rotulos_log(ax.xaxis, [v for s in series for v in s["x"]])
    if logy:
        _ajustar_rotulos_log(ax.yaxis, [v for s in series for v in s["y"]])
    ax.legend(**(legend_kw or {"loc": "best"}))


def _ajustar_rotulos_log(eixo, valores) -> None:
    """Suprime os rótulos de escala menor só quando há décadas suficientes."""
    positivos = [v for v in valores if v is not None and v > 0]
    if not positivos:
        return
    decadas = math.log10(max(positivos) / min(positivos))
    if decadas >= 1.0:
        eixo.set_minor_formatter(mpl.ticker.NullFormatter())
    else:
        # Menos de uma década: sem rótulo menor o eixo fica sem marcação.
        eixo.set_minor_formatter(mpl.ticker.LogFormatterSciNotation(
            labelOnlyBase=False, minor_thresholds=(2, 0.5)))


def shared_legend(fig, ax, ncol: int = 4, title: str | None = None,
                  fontsize: float = 8.0,
                  loc: str = "outside lower center") -> None:
    """Legenda única abaixo dos painéis, retirada de `ax`.

    Quando há muitas curvas (os sete pares de graus de interface, por
    exemplo), a legenda dentro do painel cobre justamente as curvas que o
    leitor precisa ver. Aqui ela sai para fora e serve os dois painéis, que
    partilham as mesmas séries.

    ``loc`` só deve sair do valor padrão quando o pé da figura já estiver
    ocupado: o layout restrito reserva UMA faixa inferior, e um `supxlabel`
    dividiria essa faixa com a legenda, imprimindo um sobre o outro. Nesse
    caso use ``"outside upper center"``.
    """
    handles, labels = ax.get_legend_handles_labels()
    # Remove a legenda de TODOS os painéis, e não só a de `ax`: as séries são
    # as mesmas em todos, e a que sobrasse num painel vizinho seria uma cópia
    # da legenda compartilhada, cobrindo justamente as curvas.
    for outro in fig.axes:
        leg = outro.get_legend()
        if leg is not None:
            leg.remove()
    # `loc="outside lower center"` é reservado ao layout restrito, que abre
    # espaço para a legenda dentro do figsize. `subplots_adjust`, que estava
    # aqui antes, DESLIGA o layout restrito e devolve a colisão de rótulos.
    fig.legend(handles, labels, loc=loc, ncol=ncol,
               fontsize=fontsize, title=title, frameon=True)


def stacked_bars(ax, categories, parts, labels, colors=None) -> None:
    """Barras empilhadas — o perfil por fase do estudo 1e.

    ``categories`` são os rótulos do eixo x; ``parts`` é a lista de séries
    (uma por fase), cada uma com um valor por categoria; ``labels`` nomeia
    cada fase. As cores por omissão são a rampa ordinal ``PHASES``.
    """
    import numpy as _np
    colors = colors or PHASES
    base = _np.zeros(len(categories), dtype=float)
    xs = _np.arange(len(categories))
    for part, label, color in zip(parts, labels, colors):
        vals = _np.asarray(part, dtype=float)
        ax.bar(xs, vals, bottom=base, label=label, color=color, **BAR)
        base += vals
    ax.set_xticks(xs)
    ax.set_xticklabels(categories)
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    ax.legend(loc="best")


apply_style()
