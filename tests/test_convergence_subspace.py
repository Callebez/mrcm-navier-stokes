#!/usr/bin/env python3
"""
tests/test_convergence_subspace.py — Experimento E12: convergência por subespaço.

Roda a curva Hh separadamente para cada subespaço da entrada:

  gradient   : w = grad p̃   -> resposta exata u = 0. Todo valor devolvido é
               erro puro. Mede o VAZAMENTO SOLENOIDAL ESPÚRIO.
  solenoidal : w = u        -> resposta exata u_h = w. Mede o vazamento
               IRROTACIONAL espúrio.
  mixed      : w = u + grad p̃ -> erro agregado, o benchmark histórico.

A hipótese (Cap. 3 da dissertação, sec:erro-decomposicao) é que o erro da
decomposição multiescala se aloja na parcela irrotacional. Se for verdade,
o modo `gradient` domina e o `mixed` reproduz o `gradient`.

A métrica comparável entre modos é `leak_ratio` = ‖u_h − u_ex‖ / ‖w‖: a
fração da energia da ENTRADA que aparece como erro. `err_vs_exact` não serve,
porque no modo gradient seu denominador é zero.

Uso:
  python tests/test_convergence_subspace.py --quick
  mpiexec -n 4 python tests/test_convergence_subspace.py \
      --modes gradient,solenoidal,mixed --Ndoms 2,4,8,16
"""

from __future__ import annotations
import os
import time
import subprocess

import numpy as np

from common import (get_comm, apply_plot_style, banner, save_json,
                    INK, COLORS, DEFAULT_OUTDIR)
from params import (CommonParams, parse_args, parse_int_list,
                    parse_float_list)
from mprm.benchmarks.convergence import (convergence_curve_hh,
                                         convergence_curve_h,
                                         convergence_curve_p)

OUTDIR = os.path.join(DEFAULT_OUTDIR, "convergence_olshanskii")
DEFAULT_NDOMS = "2,4,8,16"
DEFAULT_MODES = "gradient,solenoidal,mixed"
MODE_STYLE = {"gradient": ("o-", COLORS[1]),
              "solenoidal": ("s--", COLORS[2]),
              "mixed": ("^:", COLORS[3])}


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            text=True).strip()
    except Exception:
        return "unknown"


# ═══════════════════════════════════════════════════════════════════════════
# Estudo (função pura de orquestração — o cálculo vive em convergence.py)
# ═══════════════════════════════════════════════════════════════════════════

def run_subspace_study(comm, modes, Ndoms, nloc, degreeU, degreeP, alpha,
                       verbose=True) -> dict:
    """Uma curva Hh por modo, com envelope autodescritivo."""
    curves = {}
    for mode in modes:
        if verbose and comm.rank == 0:
            banner(f"modo {mode}", comm)
        curves[mode] = convergence_curve_hh(
            comm, Ndoms, nloc, degreeU, degreeP, alpha,
            mode=mode, verbose=verbose)

    return {
        "hypothesis": {
            "E12a": "o erro da decomposição multiescala aloja-se na parcela "
                    "irrotacional (sec:erro-decomposicao)",
            "E12b": "o modo mixed reproduz o modo gradient, porque a parcela "
                    "solenoidal contribui pouco (pipeline é linear)",
        },
        "criterion": {
            "E12a": "leak_ratio(gradient) >> leak_ratio(solenoidal) no mesmo "
                    "nível de malha e grau",
            "E12b": "|leak_ratio(mixed) − leak_ratio(gradient)| / "
                    "leak_ratio(gradient) < 0.1",
        },
        "verdict": _verdict(curves),
        "caveats": [
            "o resíduo do modo solenoidal é piso de amostragem, não defeito: "
            "o campo de Olshanskii amostrado nas faces MAC tem divergência "
            "discreta O(h), que o projetor legitimamente remove",
            "err_vs_exact é None no modo gradient (denominador zero); as "
            "taxas desse modo vêm de leak_ratio",
            "err_vs_fine infla as taxas em graus altos — use err_vs_exact ou "
            "leak_ratio para comparar",
        ],
        "git_commit": _git_commit(),
        "params": {"modes": list(modes), "Ndoms": list(Ndoms), "nloc": nloc,
                   "degreeU": degreeU, "degreeP": degreeP, "alpha": alpha},
        "curves": curves,
    }


def run_saturation_study(comm, modes, Ndom, nlocs, degreeU, degreeP, alpha,
                         verbose=True) -> dict:
    """H1 — o defeito de ortogonalidade é da escala H, não da escala h.

    Com H fixo (Ndom fixo) e h→0, a previsão é que `ortho_cos` SATURE num
    platô não nulo. Se ele cair com h, o erro é de discretização e não da
    decomposição, e isso contradiz sec:erro-decomposicao.
    """
    curves = {}
    for mode in modes:
        if verbose and comm.rank == 0:
            banner(f"saturação em h — modo {mode}", comm)
        curves[mode] = convergence_curve_h(comm, Ndom, nlocs, degreeU,
                                           degreeP, alpha, mode=mode,
                                           verbose=verbose)
    return {
        "hypothesis": {
            "H1": "ortho_cos satura em platô não nulo sob refino de h com "
                  "H fixo — o defeito é da escala H, não da escala h",
        },
        "criterion": {
            "H1": "|rate_ortho vs h| < 0.2 e ortho_cos > 1e-6 no h mais fino",
        },
        "verdict": _verdict_saturacao(curves),
        "caveats": [
            "ortho_cos no modo solenoidal é razão entre duas quantidades "
            "pequenas (u_h ≈ w, logo ṽ ≈ 0) e pode se comportar de forma "
            "errática; o modo de referência para H1 é gradient",
        ],
        "git_commit": _git_commit(),
        "params": {"modes": list(modes), "Ndom": Ndom, "nlocs": list(nlocs),
                   "degreeU": degreeU, "degreeP": degreeP, "alpha": alpha},
        "curves": curves,
    }


def _verdict_saturacao(curves) -> dict:
    out = {}
    for mode, c in curves.items():
        r = c.get("rate_ortho")
        cases = sorted(c["cases"], key=lambda k: k["h"])
        fino = cases[0]["ortho_cos"] if cases else None
        taxa = r["rate"] if r else None
        out[mode] = {
            "rate_ortho_vs_h": taxa,
            "ortho_cos_no_h_mais_fino": fino,
            "satura": bool(taxa is not None and abs(taxa) < 0.2
                           and fino is not None and fino > 1e-6),
        }
    ref = out.get("gradient", {})
    return {"H1": {"por_modo": out,
                   "confirmada": bool(ref.get("satura"))}}


def run_degree_study(comm, modes, Ndom, nloc, ks, alpha,
                     verbose=True) -> dict:
    """H3 — enriquecer a interface reduz o defeito de ortogonalidade."""
    curves = {}
    for mode in modes:
        if verbose and comm.rank == 0:
            banner(f"graus de interface — modo {mode}", comm)
        curves[mode] = convergence_curve_p(comm, Ndom, nloc, ks, alpha,
                                           mode=mode, verbose=verbose)
    return {
        "hypothesis": {
            "H3": "enriquecer a interface (degreeU=degreeP) reduz ortho_cos",
        },
        "criterion": {
            "H3": "ortho_cos no maior grau < ortho_cos no menor grau",
        },
        "verdict": _verdict_graus(curves),
        "caveats": [
            "convergence_curve_p não ajusta taxa (grau não é escala de "
            "comprimento); o veredito compara valores extremos",
        ],
        "git_commit": _git_commit(),
        "params": {"modes": list(modes), "Ndom": Ndom, "nloc": nloc,
                   "ks": list(ks), "alpha": alpha},
        "curves": curves,
    }


def _verdict_graus(curves) -> dict:
    out = {}
    for mode, c in curves.items():
        cases = sorted(c["cases"], key=lambda k: k.get("degreeU", 0))
        vals = [k["ortho_cos"] for k in cases]
        out[mode] = {
            "graus": [k.get("degreeU") for k in cases],
            "ortho_cos": vals,
            "fator_de_reducao": (vals[0] / vals[-1])
            if vals and vals[-1] else None,
            "reduz": bool(len(vals) >= 2 and vals[-1] < vals[0]),
        }
    ref = out.get("gradient", {})
    return {"H3": {"por_modo": out, "confirmada": bool(ref.get("reduz"))}}


def run_alpha_study(comm, modes, alphas, Ndom, nloc, degreeU, degreeP,
                    verbose=True) -> dict:
    """Varredura de α por subespaço, com malha e graus fixos.

    A pergunta: o parâmetro de Robin controla o balanço entre continuidade
    de pressão (α→0) e de fluxo (α→∞) — precisamente as duas grandezas cujo
    salto aparece no resíduo de ortogonalidade. A previsão a testar é que o
    α que minimiza o vazamento no modo `gradient` NÃO precisa coincidir com
    o que o minimiza no modo `solenoidal`, porque cada modo penaliza um dos
    dois saltos.
    """
    curves = {}
    for mode in modes:
        if verbose and comm.rank == 0:
            banner(f"varredura de alpha — modo {mode}", comm)
        cases = []
        for alpha in alphas:
            r = convergence_curve_hh(comm, [Ndom], nloc, degreeU, degreeP,
                                     alpha, mode=mode, verbose=False)
            case = dict(r["cases"][0])
            case["alpha"] = float(alpha)
            cases.append(case)
            if verbose and comm.rank == 0:
                print(f"  α={alpha:<9g} leak={case['leak_ratio']:.4e}  "
                      f"err_exato={_fmt_opt(case['err_vs_exact'])}  "
                      f"div={case['divmax']:.1e}", flush=True)
        curves[mode] = {"cases": cases}

    return {
        "hypothesis": {
            "E12c": "o alpha que minimiza o vazamento no modo gradient não "
                    "coincide com o que o minimiza no modo solenoidal, "
                    "porque cada modo penaliza um dos dois saltos",
        },
        "criterion": {
            "E12c": "argmin_alpha leak_ratio(gradient) != "
                    "argmin_alpha leak_ratio(solenoidal)",
        },
        "verdict": _verdict_alpha(curves),
        "caveats": [
            "alpha=1 é a escolha do capítulo do MRCM, herdada de estudos em "
            "meios HETEROGÊNEOS; a etapa de projeção é homogênea e não herda "
            "essa justificativa automaticamente",
            "a varredura é de alpha, não de amplitude: o pipeline é linear em "
            "w, logo varrer m_pr seria vazio por construção",
        ],
        "git_commit": _git_commit(),
        "params": {"modes": list(modes), "alphas": [float(a) for a in alphas],
                   "Ndom": Ndom, "nloc": nloc,
                   "degreeU": degreeU, "degreeP": degreeP},
        "curves": curves,
    }


def _fmt_opt(x, spec=".4e") -> str:
    return "None      " if x is None else format(x, spec)


def _argmin_alpha(cases, key):
    """α do menor |key|, ignorando casos em que a métrica não existe."""
    valid = [c for c in cases if c.get(key) is not None]
    if not valid:
        return None
    return min(valid, key=lambda c: abs(c[key]))["alpha"]


def _verdict_alpha(curves) -> dict:
    """Decide E12c registrando também a SENSIBILIDADE a α.

    O argmin sozinho engana: se a curva de um modo for plana, seu mínimo é
    ruído e "os ótimos divergem" não significa nada. `spread` = (máx−mín)/mín
    diz se o mínimo é real. Um modo só conta como sensível a α se variar
    mais de 5% ao longo da varredura.
    """
    mins, spreads = {}, {}
    for m, c in curves.items():
        vals = [k["leak_ratio"] for k in c["cases"]
                if k.get("leak_ratio") is not None]
        mins[m] = _argmin_alpha(c["cases"], "leak_ratio")
        spreads[m] = ((max(vals) - min(vals)) / min(vals)) if vals else None

    sensiveis = [m for m, s in spreads.items() if s is not None and s > 0.05]
    divergem = ("gradient" in mins and "solenoidal" in mins
                and mins["gradient"] != mins["solenoidal"])
    # `mixed` NÃO conta como testemunha independente: E12b estabelece que ele
    # reproduz `gradient`, e contá-lo seria dupla contagem da mesma medição.
    # Os dois subespaços independentes são gradient e solenoidal.
    ambos_sensiveis = all(
        (spreads.get(m) or 0.0) > 0.05 for m in ("gradient", "solenoidal")
        if m in spreads)
    return {
        "E12c": {
            "alpha_otimo_por_modo": mins,
            "spread_relativo_por_modo": spreads,
            "modos_sensiveis_a_alpha": sensiveis,
            "argmins_divergem": divergem,
            # só é achado se os argmins divergirem E os DOIS subespaços
            # independentes forem sensíveis a α; se um for plano, seu
            # mínimo é ruído e a divergência não significa nada
            "confirmada": bool(divergem and ambos_sensiveis),
        }
    }


def _verdict(curves) -> dict:
    """Decide E12a e E12b a partir das curvas medidas."""
    out = {}
    if "gradient" in curves and "solenoidal" in curves:
        g = curves["gradient"]["cases"][0]["leak_ratio"]
        s = curves["solenoidal"]["cases"][0]["leak_ratio"]
        out["E12a"] = {
            "leak_gradient": g, "leak_solenoidal": s,
            "razao": (g / s) if s else None,
            "confirmada": bool(s and g > 10.0 * s),
        }
    if "gradient" in curves and "mixed" in curves:
        g = curves["gradient"]["cases"][0]["leak_ratio"]
        m = curves["mixed"]["cases"][0]["leak_ratio"]
        rel = abs(m - g) / g if g else None
        out["E12b"] = {
            "leak_mixed": m, "leak_gradient": g, "desvio_relativo": rel,
            "confirmada": bool(rel is not None and rel < 0.1),
        }
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Testes de regressão (pytest) — este arquivo NÃO é só um script
# ═══════════════════════════════════════════════════════════════════════════

def test_modo_gradient_nao_quebra_o_ajuste_de_taxa():
    """err_vs_exact é None no modo gradient; _fit_rate deve ignorá-lo."""
    comm = get_comm()
    r = convergence_curve_hh(comm, [2, 3], 8, 1, 1, 1.0,
                             mode="gradient", verbose=False)
    assert r["cases"][0]["err_vs_exact"] is None
    assert r["rate_exact"] is None, "curva ausente não pode virar taxa"
    assert r["rate_leak"] is not None, "leak_ratio existe em todo modo"


def test_default_mixed_preserva_o_comportamento_anterior():
    """Constraint 7: quem não passar mode vê exatamente o que via antes."""
    comm = get_comm()
    a = convergence_curve_hh(comm, [2, 3], 8, 1, 1, 1.0, verbose=False)
    b = convergence_curve_hh(comm, [2, 3], 8, 1, 1, 1.0,
                             mode="mixed", verbose=False)
    for ca, cb in zip(a["cases"], b["cases"]):
        assert ca["err_vs_exact"] == cb["err_vs_exact"]
        assert ca["err_vs_fine"] == cb["err_vs_fine"]


def test_curva_plana_nao_vira_veredito():
    """Um argmin sobre curva plana é ruído, e não pode confirmar E12c.

    `mixed` reproduz `gradient` (E12b), logo não é testemunha independente:
    contá-lo levaria a confirmar E12c com uma única medição real.
    """
    # plana, com mínimo em α=1 — como o modo solenoidal medido, cujo
    # spread real é 0.28% ao longo de seis décadas de α
    plana = [{"alpha": a, "leak_ratio": r}
             for a, r in [(0.1, 1.0003e-4), (1.0, 1.0000e-4),
                          (10.0, 1.0002e-4)]]
    fundo = [{"alpha": a, "leak_ratio": r}
             for a, r in [(0.1, 9.3e-3), (1.0, 1.1e-2), (10.0, 1.3e-2)]]
    v = _verdict_alpha({"gradient": {"cases": fundo},
                        "solenoidal": {"cases": plana},
                        "mixed": {"cases": fundo}})["E12c"]
    assert v["argmins_divergem"] is True
    assert v["confirmada"] is False, "curva plana não sustenta o veredito"
    assert v["spread_relativo_por_modo"]["solenoidal"] < 0.05


def test_erro_do_misto_e_o_erro_do_gradiente():
    """E12b — a previsão de que a parcela solenoidal quase não contribui."""
    comm = get_comm()
    kw = dict(comm=comm, Ndoms=[3], nloc=8, degreeU=1, degreeP=1,
              alpha=1.0, verbose=False)
    g = convergence_curve_hh(**{**kw, "mode": "gradient"})["cases"][0]
    m = convergence_curve_hh(**{**kw, "mode": "mixed"})["cases"][0]
    s = convergence_curve_hh(**{**kw, "mode": "solenoidal"})["cases"][0]
    rel = abs(m["leak_ratio"] - g["leak_ratio"]) / g["leak_ratio"]
    assert rel < 0.1, (f"mixed={m['leak_ratio']:.4e} "
                       f"gradient={g['leak_ratio']:.4e} rel={rel:.3f}")
    assert g["leak_ratio"] > 10.0 * s["leak_ratio"], (
        f"E12a falhou: gradient={g['leak_ratio']:.4e} "
        f"solenoidal={s['leak_ratio']:.4e}")


# ═══════════════════════════════════════════════════════════════════════════
# Figura e CLI
# ═══════════════════════════════════════════════════════════════════════════

def plot_results(results, outdir):
    plt = apply_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for mode, curve in results["curves"].items():
        cases = sorted(curve["cases"], key=lambda c: -c["H"])
        Hs = np.array([c["H"] for c in cases])
        leak = np.array([c["leak_ratio"] for c in cases])
        style, color = MODE_STYLE.get(mode, ("o-", INK))
        rate = curve.get("rate_leak")
        lbl = f"{mode}" + (f"  O(H^{rate['rate']:.2f})" if rate else "")
        ax.loglog(Hs, leak, style, color=color, markersize=7,
                  linewidth=2, label=lbl)

    ax.set_xlabel("$H$ (tamanho do subdomínio)")
    ax.set_ylabel(r"leak_ratio $= \|u_h - u_{ex}\|_2 / \|w\|_2$")
    ax.set_title("E12 — vazamento por subespaço da decomposição")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "e12_subspace.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Figura: {path}")
    return path


def plot_saturacao(results, outdir):
    """H1 — o defeito de ortogonalidade não vê h.

    Duas curvas no mesmo eixo: ortho_cos, que deve ficar PLANO, e leak_ratio,
    para contraste. O achado é a horizontalidade, não a inclinação.
    """
    plt = apply_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for mode, curve in results["curves"].items():
        cases = sorted(curve["cases"], key=lambda c: -c["h"])
        hs = np.array([c["h"] for c in cases])
        _, color = MODE_STYLE.get(mode, ("o-", INK))
        ortho = np.array([c["ortho_cos"] for c in cases])
        r = curve.get("rate_ortho")
        taxa = f"  taxa={r['rate']:+.3f}" if r else ""
        ax.loglog(hs, ortho, "o-", color=color, markersize=8, linewidth=3.0,
                  label=f"$|\\cos\\theta|$  {mode}{taxa}", zorder=3)
        leak = [c["leak_ratio"] for c in cases]
        if all(v is not None for v in leak):
            # tracejado branco POR CIMA: no modo gradient as duas curvas
            # coincidem por álgebra exata (cos = −leak/√(1+leak²), pois
            # <u_h, w> = 0), e sem isto a de baixo fica invisível
            ax.loglog(hs, leak, "--", color="white", linewidth=1.4,
                      dashes=(4, 4), zorder=4)
            ax.loglog(hs, leak, "s", color=color, markersize=4,
                      markerfacecolor="white", markeredgewidth=1.2,
                      zorder=5, label=f"leak_ratio  {mode}")

    # referência O(h): o que uma quantidade governada por h faria
    href = np.array([max(c["h"] for c in results["curves"][
        next(iter(results["curves"]))]["cases"]),
        min(c["h"] for c in results["curves"][
            next(iter(results["curves"]))]["cases"])])
    anchor = max(c["ortho_cos"] for c in
                 results["curves"][next(iter(results["curves"]))]["cases"])
    ax.loglog(href, anchor * (href / href[0]), "--", color=INK,
              linewidth=1, alpha=0.45, label="referência $O(h)$")

    Ndom = results["params"]["Ndom"]
    ax.set_xlabel("$h$ (malha fina)")
    ax.set_ylabel("métrica")
    ax.set_title(f"H1 — o defeito de ortogonalidade não vê $h$ "
                 f"($N_{{dom}}={Ndom}$, $H$ fixo)")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "e13_saturacao.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Figura: {path}")
    return path


def plot_graus(results, outdir):
    """H3 — enriquecer a interface derruba o defeito."""
    plt = apply_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for mode, curve in results["curves"].items():
        cases = sorted(curve["cases"], key=lambda c: c.get("degreeU", 0))
        ks = np.array([c.get("degreeU") for c in cases])
        ortho = np.array([c["ortho_cos"] for c in cases])
        style, color = MODE_STYLE.get(mode, ("o-", INK))
        fator = ortho[0] / ortho[-1] if ortho[-1] else float("nan")
        ax.semilogy(ks, ortho, style, color=color, markersize=9,
                    linewidth=2.2, label=f"{mode}  ({fator:.0f}× do grau "
                                         f"{ks[0]} ao {ks[-1]})")
        for k, o in zip(ks, ortho):
            ax.annotate(f"{o:.2e}", (k, o), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=7, color=INK)

    Ndom = results["params"]["Ndom"]
    nloc = results["params"]["nloc"]
    ax.set_xlabel("grau do espaço de interface ($k_U = k_P$)")
    ax.set_ylabel(r"$|\cos\theta|$ — defeito de ortogonalidade")
    ax.set_title(f"H3 — enriquecer a interface reduz o defeito "
                 f"($N_{{dom}}={Ndom}$, nloc={nloc})")
    ax.set_xticks(list(ks))
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "e13_graus.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Figura: {path}")
    return path


def plot_alpha(results, outdir):
    plt = apply_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for mode, curve in results["curves"].items():
        cases = sorted(curve["cases"], key=lambda c: c["alpha"])
        al = np.array([c["alpha"] for c in cases])
        leak = np.array([c["leak_ratio"] for c in cases])
        style, color = MODE_STYLE.get(mode, ("o-", INK))
        ax.loglog(al, leak, style, color=color, markersize=7,
                  linewidth=2, label=mode)
        i = int(np.argmin(leak))
        ax.plot(al[i], leak[i], "*", color=color, markersize=16,
                markeredgecolor=INK, markeredgewidth=0.6)

    ax.set_xlabel(r"$\alpha$ (parâmetro de Robin, $\beta = \alpha H/K$)")
    ax.set_ylabel(r"leak_ratio $= \|u_h - u_{ex}\|_2 / \|w\|_2$")
    ax.set_title(r"E12c — $\alpha$ ótimo por subespaço (★ = mínimo)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "e12_subspace_alpha.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Figura: {path}")
    return path


def main(argv=None):
    defaults = CommonParams(nloc=16, degreeU=1, degreeP=1, alpha=1.0)
    args = parse_args(
        "E12: convergência por subespaço da decomposição de Helmholtz-Hodge",
        defaults=defaults,
        extra_args=lambda p: (
            p.add_argument("--Ndoms", default=DEFAULT_NDOMS,
                           help="lista de Ndom (nloc fixo)"),
            p.add_argument("--modes", default=DEFAULT_MODES,
                           help="subespaços da entrada, separados por vírgula"),
            p.add_argument("--alphas", default=None,
                           help="se dado, varre alpha em vez de Ndom "
                                "(malha e graus fixos)"),
            p.add_argument("--nlocs", default=None,
                           help="se dado, varre h com H fixo (saturação, H1)"),
            p.add_argument("--degrees", default=None,
                           help="se dado, varre graus de interface (H3)"),
        ),
        argv=argv,
    )

    if args.quick:
        Ndoms, modes, nloc = [2, 3], ["gradient", "solenoidal", "mixed"], 8
        alphas = parse_float_list(args.alphas) if args.alphas else None
    else:
        Ndoms = parse_int_list(args.Ndoms)
        modes = [m.strip() for m in args.modes.split(",") if m.strip()]
        nloc = args.nloc
        alphas = parse_float_list(args.alphas) if args.alphas else None

    comm = get_comm()
    t0 = time.perf_counter()
    if args.nlocs:
        results = run_saturation_study(comm, modes, args.Ndom,
                                       parse_int_list(args.nlocs),
                                       args.degreeU, args.degreeP, args.alpha,
                                       verbose=args.verbose)
        stem, plotter = "e13_saturacao", plot_saturacao
    elif args.degrees:
        results = run_degree_study(comm, modes, args.Ndom, nloc,
                                   parse_int_list(args.degrees), args.alpha,
                                   verbose=args.verbose)
        stem, plotter = "e13_graus", plot_graus
    elif alphas:
        results = run_alpha_study(comm, modes, alphas, args.Ndom, nloc,
                                  args.degreeU, args.degreeP,
                                  verbose=args.verbose)
        stem, plotter = "e12_subspace_alpha", plot_alpha
    else:
        results = run_subspace_study(comm, modes, Ndoms, nloc,
                                     args.degreeU, args.degreeP, args.alpha,
                                     verbose=args.verbose)
        stem, plotter = "e12_subspace", plot_results
    elapsed = time.perf_counter() - t0

    if comm.rank == 0:
        os.makedirs(OUTDIR, exist_ok=True)
        save_json(results, OUTDIR, f"{stem}.json", comm)
        if plotter is not None:
            try:
                plotter(results, OUTDIR)
            except Exception as e:
                print(f"  ⚠ Plot failed: {e}")

        banner("vereditos", comm)
        for key, v in results["verdict"].items():
            mark = "✓" if v["confirmada"] else "✗"
            print(f"  {mark} {key}: " + "  ".join(
                f"{k}={v[k]:.4e}" if isinstance(v[k], float) else f"{k}={v[k]}"
                for k in v if k != "confirmada"))
        print(f"\nTotal: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
