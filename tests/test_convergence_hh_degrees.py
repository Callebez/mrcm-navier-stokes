#!/usr/bin/env python3
"""
tests/test_convergence_hh_degrees.py — Experimentos E2+E3: Modo Hh × graus.

E2 (simétrico): varre (P_k, U_k), k ∈ {1,2,3}, com α ∈ {0.1, 1.0}.
E3 (assimétrico): varre pares (P_k, U_{k-1}) e (P_{k-1}, U_k), k ∈ {2,3},
                   com α = 1.0.

Uso:
  python tests/test_convergence_hh_degrees.py                     # E2
  python tests/test_convergence_hh_degrees.py --asymmetric        # E3
  python tests/test_convergence_hh_degrees.py --quick
  mpiexec -n 4 python tests/test_convergence_hh_degrees.py
"""

from __future__ import annotations
import time, os, json
import numpy as np

from common import (get_comm, apply_plot_style, banner, make_meta,
                    save_json, COLORS, INK, DEFAULT_OUTDIR)
from params import CommonParams, parse_args, parse_float_list, parse_int_list
from mprm.benchmarks.convergence import convergence_curve_hh

OUTDIR = os.path.join(DEFAULT_OUTDIR, "convergence_olshanskii")
DEFAULT_NDOMS = "2,4,8,16"
DEFAULT_NLOC = 16


def run_study_symmetric(comm, ks, alphas, Ndoms, nloc, verbose=True):
    results = {"meta": make_meta("e2_hh_degrees_sym", comm,
                                 nloc=nloc, ks=list(ks),
                                 alphas=[float(a) for a in alphas],
                                 Ndoms=list(Ndoms)),
               "curves": []}
    for a in alphas:
        for k in ks:
            if verbose:
                banner(f"(P{k},U{k}), α={a}", comm)
            curve = convergence_curve_hh(comm, Ndoms, nloc, k, k, a,
                                         verbose=verbose)
            results["curves"].append({"alpha": float(a), "k": k, **curve})
    return results


def run_study_asymmetric(comm, pairs, alpha, Ndoms, nloc, verbose=True):
    results = {"meta": make_meta("e3_hh_degrees_asym", comm,
                                 nloc=nloc, alpha=alpha,
                                 pairs=[list(p) for p in pairs],
                                 Ndoms=list(Ndoms)),
               "curves": []}
    for degU, degP in pairs:
        if verbose:
            banner(f"(U{degU},P{degP}), α={alpha}", comm)
        curve = convergence_curve_hh(comm, Ndoms, nloc, degU, degP, alpha,
                                     verbose=verbose)
        results["curves"].append({"alpha": float(alpha),
                                 "degreeU": degU, "degreeP": degP,
                                 **curve})
    return results


def plot_results(results, outdir, filename, title):
    plt = apply_plot_style()
    curves = results["curves"]
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, c in enumerate(curves):
        cases = sorted(c["cases"], key=lambda x: -x["H"])
        Hs = np.array([cs["H"] for cs in cases])
        errs = np.array([cs["err_vs_fine"] for cs in cases])

        if "k" in c:
            label = f"(P{c['k']},U{c['k']}), α={c['alpha']:g}"
            color = COLORS.get(c["k"], INK)
            marker = "osD^"[min(c["k"] - 1, 3)]
        else:
            label = f"(U{c['degreeU']},P{c['degreeP']})"
            color = plt.cm.tab10(i % 10)
            marker = "o"

        ax.loglog(Hs, errs, f"{marker}-", color=color, markersize=6,
                  linewidth=1.5, label=label)
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
    ax.set_title(title)
    ax.invert_xaxis()
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = os.path.join(outdir, filename)
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Figura: {path}")
    return path


def main(argv=None):
    defaults = CommonParams(nloc=DEFAULT_NLOC, degreeU=1, degreeP=1, alpha=1.0)
    args = parse_args(
        "E2/E3: Modo Hh × graus (simétrico e assimétrico)",
        defaults=defaults,
        extra_args=lambda p: (
            p.add_argument("--asymmetric", action="store_true",
                          help="Modo assimétrico (E3): pares não-simétricos"),
            p.add_argument("--ks", default="1,2,3",
                          help="lista de k para modo simétrico (default: 1,2,3)"),
            p.add_argument("--alphas", default="1.0,0.1",
                          help="lista de α (modo simétrico)"),
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

    if args.asymmetric:
        # E3: pares assimétricos
        pairs = [(2, 1), (3, 2), (1, 2), (2, 3)]
        results = run_study_asymmetric(comm, pairs, args.alpha, Ndoms,
                                       args.nloc, verbose=args.verbose)
        fname = "e3_hh_degrees_asym"
        title = f"E3: Modo Hh × graus assimétricos — α={args.alpha}, nloc={args.nloc}"
    else:
        # E2: simétrico
        ks = parse_int_list(args.ks)
        alphas = parse_float_list(args.alphas)
        results = run_study_symmetric(comm, ks, alphas, Ndoms, args.nloc,
                                      verbose=args.verbose)
        fname = "e2_hh_degrees_sym"
        title = f"E2: Modo Hh × graus simétricos — nloc={args.nloc}"

    elapsed = time.perf_counter() - t0

    if comm.rank == 0:
        os.makedirs(OUTDIR, exist_ok=True)
        save_json(results, OUTDIR, f"{fname}.json", comm)
        try:
            plot_results(results, OUTDIR, f"{fname}.png", title)
        except Exception as e:
            print(f"  ⚠ Plot failed: {e}")
        print(f"\nTotal: {elapsed:.0f}s")
        print("\nResumo de taxas (err_vs_fine vs H):")
        for c in results["curves"]:
            rf = c.get("rate_fine")
            label = (f"(P{c['k']},U{c['k']})" if "k" in c
                     else f"(U{c['degreeU']},P{c['degreeP']})")
            alpha_str = f" α={c.get('alpha', args.alpha):g}"
            if rf:
                print(f"  {label}{alpha_str}:  taxa={rf['rate']:.2f}  "
                      f"R²={rf['r_squared']:.4f}")
            else:
                print(f"  {label}{alpha_str}:  (insuficiente para taxa)")


if __name__ == "__main__":
    main()
