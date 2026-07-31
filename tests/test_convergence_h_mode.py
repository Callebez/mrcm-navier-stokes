#!/usr/bin/env python3
"""
tests/test_convergence_h_mode.py — Experimento E5: Modo h (saturação).

H fixo (Ndom=4), nloc varre → h→0. Mede a saturação do erro.

Uso:
  python tests/test_convergence_h_mode.py
  python tests/test_convergence_h_mode.py --quick
  mpiexec -n 4 python tests/test_convergence_h_mode.py
"""

from __future__ import annotations
import time, os
import numpy as np

from common import (get_comm, apply_plot_style, banner, make_meta,
                    save_json, INK, DEFAULT_OUTDIR)
from params import CommonParams, parse_args, parse_int_list
from mprm.benchmarks.convergence import convergence_curve_h

OUTDIR = os.path.join(DEFAULT_OUTDIR, "convergence_olshanskii")
DEFAULT_NLOCS = "2,4,8,16,32,64"
DEFAULT_NDOM = 4


def plot_results(results, outdir):
    plt = apply_plot_style()
    cases = sorted(results["cases"], key=lambda x: -x["h"])
    hs = np.array([c["h"] for c in cases])
    err_fine = np.array([c["err_vs_fine"] for c in cases])
    err_exact = np.array([c["err_vs_exact"] for c in cases])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.loglog(hs, err_fine, "o-", color="#0072B2", markersize=7,
              linewidth=2, label="err_vs_fine")
    ax.loglog(hs, err_exact, "s--", color="#E69F00", markersize=7,
              linewidth=1.5, label="err_vs_exact")

    for rf_key, errs, rate_key in [("rate_fine", err_fine, "rate_fine"),
                                    ("rate_exact", err_exact, "rate_exact")]:
        rate = results.get(rate_key)
        if rate:
            fit = np.exp(rate["intercept"]) * hs ** rate["rate"]
            ax.loglog(hs, fit, ":", alpha=0.5, linewidth=1, color="gray",
                      label=f"  ajuste {rate_key}: O(h^{rate['rate']:.2f}), "
                            f"R²={rate['r_squared']:.4f}")

    ref2 = err_fine[0] * (hs / hs[0]) ** 2.0
    ax.loglog(hs, ref2, "--", color=INK, alpha=0.4, label="O(h²)")

    for nloc, h, err in zip((c["N"] // results["meta"]["Ndom"] for c in cases),
                             hs, err_fine):
        ax.annotate(f"nloc={nloc}", xy=(h, err), xytext=(6, 6),
                    textcoords="offset points", fontsize=8, color=INK)

    ax.set_xlabel("h (tamanho da célula MAC)")
    ax.set_ylabel("erro relativo")
    ax.set_title(f"E5: Modo h (saturação) — Ndom={results['meta']['Ndom']} "
                 f"(H={results['meta']['H']:.4f}), "
                 f"(U{results['meta']['degreeU']},P{results['meta']['degreeP']}), "
                 f"α={results['meta']['alpha']}")
    ax.invert_xaxis()
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = os.path.join(outdir, "e5_h_saturation.png")
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Figura: {path}")
    return path


def main(argv=None):
    defaults = CommonParams(Ndom=DEFAULT_NDOM, nloc=16, degreeU=1, degreeP=1,
                            alpha=1.0)
    args = parse_args(
        "E5: Modo h (saturação) — H fixo, h→0",
        defaults=defaults,
        extra_args=lambda p: (
            p.add_argument("--nlocs", default=DEFAULT_NLOCS,
                          help="lista de nloc"),
        ),
        argv=argv,
    )

    if args.quick:
        nlocs = [4, 8, 16]
    else:
        nlocs = parse_int_list(args.nlocs)

    comm = get_comm()
    t0 = time.perf_counter()
    results = convergence_curve_h(comm, args.Ndom, nlocs,
                                  args.degreeU, args.degreeP, args.alpha,
                                  verbose=args.verbose)
    elapsed = time.perf_counter() - t0

    if comm.rank == 0:
        os.makedirs(OUTDIR, exist_ok=True)
        save_json(results, OUTDIR, "e5_h_saturation.json", comm)
        try:
            plot_results(results, OUTDIR)
        except Exception as e:
            print(f"  ⚠ Plot failed: {e}")
        print(f"\nTotal: {elapsed:.0f}s")
        for rk in ["rate_exact", "rate_fine"]:
            r = results.get(rk)
            if r:
                print(f"  {rk}: taxa={r['rate']:.2f}  R²={r['r_squared']:.4f}")


if __name__ == "__main__":
    main()
