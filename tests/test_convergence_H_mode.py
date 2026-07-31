#!/usr/bin/env python3
"""
tests/test_convergence_H_mode.py — Experimento E4: Modo H (multiescala).

h ~ fixo (target ≈ 1/256), Ndom varia → H→0. Mede a taxa multiescala
O(H^{degree+2}).

Uso:
  python tests/test_convergence_H_mode.py
  python tests/test_convergence_H_mode.py --quick
  mpiexec -n 4 python tests/test_convergence_H_mode.py
"""

from __future__ import annotations
import time, os
import numpy as np

from common import (get_comm, apply_plot_style, banner, make_meta,
                    save_json, INK, DEFAULT_OUTDIR)
from params import CommonParams, parse_args, parse_int_list
from mprm.benchmarks.convergence import convergence_curve_H

OUTDIR = os.path.join(DEFAULT_OUTDIR, "convergence_olshanskii")
DEFAULT_NDOMS = "3,4,6,8,12"
DEFAULT_H_TARGET = 1.0 / 256  # ≈ 0.00390625


def plot_results(results, outdir):
    plt = apply_plot_style()
    cases = sorted(results["cases"], key=lambda x: -x["H"])
    Hs = np.array([c["H"] for c in cases])
    err_fine = np.array([c["err_vs_fine"] for c in cases])
    err_exact = np.array([c["err_vs_exact"] for c in cases])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.loglog(Hs, err_fine, "o-", color="#0072B2", markersize=7,
              linewidth=2, label="err_vs_fine")
    ax.loglog(Hs, err_exact, "s--", color="#E69F00", markersize=7,
              linewidth=1.5, label="err_vs_exact")

    for rf_key, errs, rate_key in [("rate_fine", err_fine, "rate_fine"),
                                    ("rate_exact", err_exact, "rate_exact")]:
        rate = results.get(rate_key)
        if rate:
            fit = np.exp(rate["intercept"]) * Hs ** rate["rate"]
            ax.loglog(Hs, fit, ":", alpha=0.5, linewidth=1, color="gray",
                      label=f"  ajuste {rate_key}: O(H^{rate['rate']:.2f}), "
                            f"R²={rate['r_squared']:.4f}")

    for Ndom, H, err in zip((c["Ndom"] for c in cases), Hs, err_fine):
        ax.annotate(f"Ndom={Ndom}", xy=(H, err), xytext=(6, 6),
                    textcoords="offset points", fontsize=8, color=INK)

    ax.set_xlabel("H (tamanho do subdomínio)")
    ax.set_ylabel("erro relativo")
    ax.set_title(f"E4: Modo H (multiescala) — h≈{results['meta']['h_target']:.5f}, "
                 f"(U{results['meta']['degreeU']},P{results['meta']['degreeP']}), "
                 f"α={results['meta']['alpha']}")
    ax.invert_xaxis()
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = os.path.join(outdir, "e4_H_multiscale.png")
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Figura: {path}")
    return path


def main(argv=None):
    defaults = CommonParams(nloc=16, degreeU=1, degreeP=1, alpha=1.0)
    args = parse_args(
        "E4: Modo H (multiescala) — taxa O(H^{degree+2})",
        defaults=defaults,
        extra_args=lambda p: (
            p.add_argument("--Ndoms", default=DEFAULT_NDOMS,
                          help="lista de Ndom (h varia para manter h~target)"),
            p.add_argument("--h-target", type=float, default=DEFAULT_H_TARGET,
                          help="tamanho alvo da malha fina h"),
        ),
        argv=argv,
    )

    if args.quick:
        Ndoms = [2, 4, 8]
        h_target = 1.0 / 64
    else:
        Ndoms = parse_int_list(args.Ndoms)
        h_target = args.h_target

    comm = get_comm()
    t0 = time.perf_counter()
    results = convergence_curve_H(comm, Ndoms, h_target,
                                  args.degreeU, args.degreeP, args.alpha,
                                  verbose=args.verbose)
    elapsed = time.perf_counter() - t0

    if comm.rank == 0:
        os.makedirs(OUTDIR, exist_ok=True)
        save_json(results, OUTDIR, "e4_H_multiscale.json", comm)
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
