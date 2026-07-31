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
from params import CommonParams, parse_args, parse_int_list
from mprm.benchmarks.convergence import convergence_curve_hh

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
        ),
        argv=argv,
    )

    if args.quick:
        Ndoms, modes, nloc = [2, 3], ["gradient", "solenoidal", "mixed"], 8
    else:
        Ndoms = parse_int_list(args.Ndoms)
        modes = [m.strip() for m in args.modes.split(",") if m.strip()]
        nloc = args.nloc

    comm = get_comm()
    t0 = time.perf_counter()
    results = run_subspace_study(comm, modes, Ndoms, nloc,
                                 args.degreeU, args.degreeP, args.alpha,
                                 verbose=args.verbose)
    elapsed = time.perf_counter() - t0

    if comm.rank == 0:
        os.makedirs(OUTDIR, exist_ok=True)
        save_json(results, OUTDIR, "e12_subspace.json", comm)
        try:
            plot_results(results, OUTDIR)
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
