#!/usr/bin/env python3
"""
tests/test_convergence_p_mode.py — Experimento E6: Modo p (grau).

Malha fixa (Ndom=4, nloc=16), graus simétricos crescentes k ∈ {1,2,3,4}.
Mede convergência espectral para campo C^∞ de Olshanskii.

Uso:
  python tests/test_convergence_p_mode.py
  python tests/test_convergence_p_mode.py --quick
  mpiexec -n 4 python tests/test_convergence_p_mode.py
"""

from __future__ import annotations
import time, os
import numpy as np

from common import (get_comm, apply_plot_style, banner, make_meta,
                    save_json, COLORS, INK, DEFAULT_OUTDIR)
from params import CommonParams, parse_args, parse_float_list, parse_int_list
from mprm.benchmarks.convergence import convergence_curve_p

OUTDIR = os.path.join(DEFAULT_OUTDIR, "convergence_olshanskii")
DEFAULT_KS = "1,2,3,4"
DEFAULT_NDOM = 4
DEFAULT_NLOC = 16


def plot_results(results, outdir):
    plt = apply_plot_style()
    cases = results["cases"]
    # Extrair ks únicos na ordem
    ks = sorted(set(c["degreeU"] for c in cases))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for metric, ax, ylabel in [
        ("err_vs_fine", ax1, "erro relativo vs referência fina"),
        ("err_vs_exact", ax2, "erro relativo vs solução exata"),
    ]:
        errs = np.array([c[metric] for c in cases])
        ax.semilogy(ks, errs, "o-", color="#0072B2", markersize=8,
                    linewidth=2)
        for k, err in zip(ks, errs):
            ax.annotate(f"{err:.2e}", xy=(k, err), xytext=(5, 5),
                        textcoords="offset points", fontsize=7, color=INK)
        ax.set_xlabel("k (grau polinomial, (P_k, U_k))")
        ax.set_ylabel(ylabel)
        ax.set_xticks(ks)
        ax.grid(True, alpha=0.3)

    ax1.set_title(f"E6: Convergência-p (vs fino) — "
                  f"Ndom={results['meta']['Ndom']}, nloc={results['meta']['nloc']}")
    ax2.set_title(f"E6: Convergência-p (vs exato)")
    fig.suptitle(f"α={results['meta']['alpha']}", fontsize=10, color=INK)
    fig.tight_layout()
    path = os.path.join(outdir, "e6_p_degree.png")
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Figura: {path}")
    return path


def main(argv=None):
    defaults = CommonParams(Ndom=DEFAULT_NDOM, nloc=DEFAULT_NLOC,
                            degreeU=1, degreeP=1, alpha=1.0)
    args = parse_args(
        "E6: Convergência-p (grau) — campo C^∞ de Olshanskii",
        defaults=defaults,
        extra_args=lambda p: (
            p.add_argument("--ks", default=DEFAULT_KS,
                          help="lista de k para graus simétricos"),
        ),
        argv=argv,
    )

    if args.quick:
        ks = [1, 2]
    else:
        ks = parse_int_list(args.ks)

    comm = get_comm()
    t0 = time.perf_counter()
    results = convergence_curve_p(comm, args.Ndom, args.nloc, ks, args.alpha,
                                  verbose=args.verbose)
    elapsed = time.perf_counter() - t0

    if comm.rank == 0:
        os.makedirs(OUTDIR, exist_ok=True)
        save_json(results, OUTDIR, "e6_p_degree.json", comm)
        try:
            plot_results(results, OUTDIR)
        except Exception as e:
            print(f"  ⚠ Plot failed: {e}")
        print(f"\nTotal: {elapsed:.0f}s")
        print("\nResumo (err_vs_fine):")
        for c in results["cases"]:
            print(f"  (U{c['degreeU']},P{c['degreeP']}):  "
                  f"err={c['err_vs_fine']:.4e}  "
                  f"div={c['divmax']:.1e}  ({c['wall_s']:.1f}s)")


if __name__ == "__main__":
    main()
