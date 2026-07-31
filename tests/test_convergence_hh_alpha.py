#!/usr/bin/env python3
"""
tests/test_convergence_hh_alpha.py — Experimento E1: Modo Hh × α.

Varre α ∈ {0.01, 0.1, 1, 10, 100, 1000} com (P₁,U₁), nloc=16,
Ndom ∈ {2,4,8,16}. Para cada α, ajusta a taxa de convergência.

Uso:
  python tests/test_convergence_hh_alpha.py
  python tests/test_convergence_hh_alpha.py --quick
  mpiexec -n 4 python tests/test_convergence_hh_alpha.py
"""

from __future__ import annotations
import time, os, json
import numpy as np

from common import (get_comm, apply_plot_style, banner, make_meta,
                    save_json, COLORS, MARKERS, INK, DEFAULT_OUTDIR)
from params import CommonParams, parse_args, parse_float_list, parse_int_list
from mprm.benchmarks.convergence import convergence_curve_hh

OUTDIR = os.path.join(DEFAULT_OUTDIR, "convergence_olshanskii")
DEFAULT_ALPHAS = "0.01,0.1,1.0,10.0,100.0,1000.0"
DEFAULT_NDOMS = "2,4,8,16"
DEFAULT_NLOC = 16


def run_study(comm, alphas, Ndoms, nloc, degreeU, degreeP, verbose=True):
    results = {"meta": make_meta("e1_hh_alpha", comm,
                                 nloc=nloc, degreeU=degreeU, degreeP=degreeP,
                                 Ndoms=list(Ndoms), alphas=[float(a) for a in alphas]),
               "curves": []}
    for a in alphas:
        if verbose:
            banner(f"α = {a:.4f}", comm)
        curve = convergence_curve_hh(comm, Ndoms, nloc, degreeU, degreeP, a,
                                     verbose=verbose)
        results["curves"].append({"alpha": float(a), **curve})
    return results


def plot_results(results, outdir):
    plt = apply_plot_style()
    curves = results["curves"]
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, c in enumerate(curves):
        a = c["alpha"]
        cases = sorted(c["cases"], key=lambda x: -x["H"])
        Hs = np.array([cs["H"] for cs in cases])
        errs = np.array([cs["err_vs_fine"] for cs in cases])
        color = plt.cm.viridis(i / max(1, len(curves) - 1))
        ax.loglog(Hs, errs, "o-", color=color, markersize=6, linewidth=1.5,
                  label=f"α={a:g}")
        rate = c.get("rate_fine")
        if rate:
            fit = np.exp(rate["intercept"]) * Hs ** rate["rate"]
            ax.loglog(Hs, fit, ":", color=color, alpha=0.5, linewidth=1)

    ref = errs[0] * (Hs / Hs[0]) ** 1.0
    ax.loglog(Hs, ref, "--", color=INK, alpha=0.4, label="O(H)")
    ref2 = errs[0] * (Hs / Hs[0]) ** 2.0
    ax.loglog(Hs, ref2, "--", color=INK, alpha=0.25, label="O(H²)")

    ax.set_xlabel("H (tamanho do subdomínio)")
    ax.set_ylabel("erro relativo vs referência fina")
    ax.set_title(f"E1: Modo Hh × α — (P₁,U₁), nloc={results['meta']['nloc']}")
    ax.invert_xaxis()
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    path = os.path.join(outdir, "e1_hh_alpha.png")
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Figura: {path}")
    return path


def main(argv=None):
    defaults = CommonParams(nloc=DEFAULT_NLOC, degreeU=1, degreeP=1, alpha=1.0)
    args = parse_args(
        "E1: Modo Hh × α — convergência do MPrM vs parâmetro de Robin",
        defaults=defaults,
        extra_args=lambda p: (
            p.add_argument("--alphas", default=DEFAULT_ALPHAS,
                          help="lista de α ou 'logspace:ini,fim,n'"),
            p.add_argument("--Ndoms", default=DEFAULT_NDOMS,
                          help="lista de Ndom"),
        ),
        argv=argv,
    )

    if args.quick:
        alphas = [0.1, 1.0, 10.0]
        Ndoms = [2, 4]
    else:
        alphas = parse_float_list(args.alphas)
        Ndoms = parse_int_list(args.Ndoms)

    comm = get_comm()
    t0 = time.perf_counter()
    results = run_study(comm, alphas, Ndoms, args.nloc,
                        args.degreeU, args.degreeP, verbose=args.verbose)
    elapsed = time.perf_counter() - t0

    if comm.rank == 0:
        os.makedirs(OUTDIR, exist_ok=True)
        save_json(results, OUTDIR, "e1_hh_alpha.json", comm)
        try:
            plot_results(results, OUTDIR)
        except Exception as e:
            print(f"  ⚠ Plot failed: {e}")
        print(f"\nTotal: {elapsed:.0f}s")
        # Tabela resumo
        print("\nResumo de taxas (err_vs_fine vs H):")
        for c in results["curves"]:
            rf = c.get("rate_fine")
            if rf:
                print(f"  α={c['alpha']:9.1e}:  taxa={rf['rate']:.2f}  "
                      f"R²={rf['r_squared']:.4f}")
            else:
                print(f"  α={c['alpha']:9.1e}:  (insuficiente para taxa)")


if __name__ == "__main__":
    main()
