#!/usr/bin/env python3
"""
tests/test_convergence_downscaling.py — Experimento E10: Com vs sem downscaling.

Compara a taxa de convergência Hh COM e SEM o passo de downscaling.
Hipótese central: com downscaling → O(H²); sem → O(H¹).

Uso:
  python tests/test_convergence_downscaling.py
  python tests/test_convergence_downscaling.py --quick
  mpiexec -n 4 python tests/test_convergence_downscaling.py
"""

from __future__ import annotations
import time, os
import numpy as np

from common import (get_comm, apply_plot_style, banner, make_meta,
                    save_json, COLORS, INK, DEFAULT_OUTDIR)
from params import CommonParams, parse_args, parse_int_list
from mprm.benchmarks.convergence import convergence_curve_hh

OUTDIR = os.path.join(DEFAULT_OUTDIR, "convergence_olshanskii")
DEFAULT_NDOMS = "2,4,8,16"
DEFAULT_NLOC = 16


def plot_results(results, outdir):
    plt = apply_plot_style()
    cases = sorted(results["cases"], key=lambda x: -x["H"])
    Hs = np.array([c["H"] for c in cases])

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Com downscaling
    err_with = np.array([c["err_vs_fine"] for c in cases])
    ax.loglog(Hs, err_with, "o-", color=COLORS[1], markersize=7,
              linewidth=2, label="com downscaling (MPrM completo)")
    rw = results.get("rate_fine")
    if rw:
        fit = np.exp(rw["intercept"]) * Hs ** rw["rate"]
        ax.loglog(Hs, fit, ":", color=COLORS[1], alpha=0.5, linewidth=1,
                  label=f"  O(H^{rw['rate']:.2f}), R²={rw['r_squared']:.4f}")

    # Sem downscaling (raw MRCM)
    raw_cases = [c for c in cases if c.get("err_raw_mrcm") is not None]
    if raw_cases:
        raw_Hs = np.array([c["H"] for c in raw_cases])
        err_raw = np.array([c["err_raw_mrcm"] for c in raw_cases])
        ax.loglog(raw_Hs, err_raw, "s--", color=COLORS[2], markersize=7,
                  linewidth=2, label="sem downscaling (BDM₁-DG₀ cru)")
        rr = results.get("rate_raw")
        if rr:
            fit = np.exp(rr["intercept"]) * raw_Hs ** rr["rate"]
            ax.loglog(raw_Hs, fit, ":", color=COLORS[2], alpha=0.5, linewidth=1,
                      label=f"  O(H^{rr['rate']:.2f}), R²={rr['r_squared']:.4f}")

    ref1 = err_with[0] * (Hs / Hs[0]) ** 1.0
    ax.loglog(Hs, ref1, "--", color=INK, alpha=0.4, label="O(H)")
    ref2 = err_with[0] * (Hs / Hs[0]) ** 2.0
    ax.loglog(Hs, ref2, "--", color=INK, alpha=0.25, label="O(H²)")

    ax.set_xlabel("H (tamanho do subdomínio)")
    ax.set_ylabel("erro relativo vs referência fina")
    ax.set_title(f"E10: Efeito do downscaling — (P₁,U₁), α=1, nloc={results['meta']['nloc']}")
    ax.invert_xaxis()
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = os.path.join(outdir, "e10_downscaling_on_off.png")
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Figura: {path}")
    return path


def main(argv=None):
    defaults = CommonParams(nloc=DEFAULT_NLOC, degreeU=1, degreeP=1, alpha=1.0)
    args = parse_args(
        "E10: Com vs sem downscaling — teste da hipótese de superconvergência",
        defaults=defaults,
        extra_args=lambda p: (
            p.add_argument("--Ndoms", default=DEFAULT_NDOMS,
                          help="lista de Ndom"),
        ),
        argv=argv,
    )

    if args.quick:
        Ndoms = [2, 4]
    else:
        Ndoms = parse_int_list(args.Ndoms)

    comm = get_comm()
    t0 = time.perf_counter()
    banner("E10: Com vs sem downscaling", comm)
    results = convergence_curve_hh(comm, Ndoms, args.nloc,
                                   args.degreeU, args.degreeP, args.alpha,
                                   include_raw_mrcm=True,
                                   verbose=args.verbose)
    elapsed = time.perf_counter() - t0

    if comm.rank == 0:
        os.makedirs(OUTDIR, exist_ok=True)
        save_json(results, OUTDIR, "e10_downscaling_on_off.json", comm)
        try:
            plot_results(results, OUTDIR)
        except Exception as e:
            print(f"  ⚠ Plot failed: {e}")
        print(f"\nTotal: {elapsed:.0f}s")
        for rk in ["rate_fine", "rate_raw"]:
            r = results.get(rk)
            if r:
                print(f"  {rk}: taxa={r['rate']:.2f}  R²={r['r_squared']:.4f}")


if __name__ == "__main__":
    main()
