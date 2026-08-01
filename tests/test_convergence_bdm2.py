#!/usr/bin/env python3
"""
tests/test_convergence_bdm2.py — Estudo de convergência Olshanskii E1–E13 com BDM₂.

Reexecuta os experimentos E1–E13 do estudo de convergência com BDM₂
(mixed_order=1) em vez de BDM₁.  Resultados salvos em
results_alpha/convergence_olshanskii_bdm2/.

Uso:
  python tests/test_convergence_bdm2.py                          # tudo (30+ min)
  python tests/test_convergence_bdm2.py --experiments E1,E4,E6  # subset
  mpiexec -n 4 python tests/test_convergence_bdm2.py             # paralelo
"""

from __future__ import annotations
import os, sys, time, json, argparse
import numpy as np

# ── Config ──
OUTDIR = "results_alpha/convergence_olshanskii_bdm2"
MIXED_ORDER = 1
BASIS_KIND = "legendre_P"

def get_comm():
    from mpi4py import MPI
    return MPI.COMM_WORLD

def rel_err(a1, a2, b1, b2):
    d1, d2 = a1 - b1, a2 - b2
    n = np.sqrt(np.sum(d1**2) + np.sum(d2**2))
    d = np.sqrt(np.sum(b1**2) + np.sum(b2**2))
    return float(n / d) if d > 0 else 0.0

def save_json(data, name, comm=None):
    if comm is not None and comm.rank != 0:
        return
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, name)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  ✓ {path}")

def banner(text, comm):
    if comm.rank == 0:
        print(f"\n{'='*70}\n{text}\n{'='*70}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# E1 — Hh × α
# ══════════════════════════════════════════════════════════════════════════════

def run_E1(comm):
    from mprm.benchmarks.convergence import convergence_curve_hh
    banner("E1 — Hh × α (BDM₂)", comm)
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    Ndoms = [2, 4, 8, 16]
    results = {"meta": {"experiment": "E1", "mixed_order": MIXED_ORDER}, "curves": []}
    for a in alphas:
        if comm.rank == 0:
            print(f"  α={a:.2e}...", flush=True)
        curve = convergence_curve_hh(comm, Ndoms, nloc=16, degreeU=1, degreeP=1,
                                     alpha=a, basis_kind=BASIS_KIND,
                                     mixed_order=MIXED_ORDER, verbose=False)
        results["curves"].append({"alpha": a, **curve})
    save_json(results, "e1_hh_alpha.json", comm)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# E2 — Hh × graus simétricos
# ══════════════════════════════════════════════════════════════════════════════

def run_E2(comm):
    from mprm.benchmarks.convergence import convergence_curve_hh
    banner("E2 — Hh × graus simétricos (BDM₂)", comm)
    Ndoms = [2, 4, 8, 16]
    results = {"meta": {"experiment": "E2", "mixed_order": MIXED_ORDER}, "curves": []}
    for a in [1.0, 0.1]:
        for k in [1, 2, 3]:
            if comm.rank == 0:
                print(f"  α={a}, k={k}...", flush=True)
            curve = convergence_curve_hh(comm, Ndoms, nloc=16, degreeU=k, degreeP=k,
                                         alpha=a, basis_kind=BASIS_KIND,
                                         mixed_order=MIXED_ORDER, verbose=False)
            results["curves"].append({"alpha": a, "k": k, **curve})
    save_json(results, "e2_hh_degrees_sym.json", comm)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# E3 — Hh × graus assimétricos
# ══════════════════════════════════════════════════════════════════════════════

def run_E3(comm):
    from mprm.benchmarks.convergence import convergence_curve_hh
    banner("E3 — Hh × graus assimétricos (BDM₂)", comm)
    Ndoms = [2, 4, 8, 16]
    pairs = [(2, 1), (2, 2), (1, 2), (2, 3), (3, 2), (3, 3)]
    results = {"meta": {"experiment": "E3", "mixed_order": MIXED_ORDER}, "curves": []}
    for degU, degP in pairs:
        if comm.rank == 0:
            print(f"  (U_{degU}, P_{degP})...", flush=True)
        curve = convergence_curve_hh(comm, Ndoms, nloc=16, degreeU=degU, degreeP=degP,
                                     alpha=1.0, basis_kind=BASIS_KIND,
                                     mixed_order=MIXED_ORDER, verbose=False)
        results["curves"].append({"degU": degU, "degP": degP, **curve})
    save_json(results, "e3_hh_degrees_asym.json", comm)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# E4 — Modo H multiescala
# ══════════════════════════════════════════════════════════════════════════════

def run_E4(comm):
    from mprm.benchmarks.convergence import convergence_curve_H
    banner("E4 — Modo H multiescala (BDM₂)", comm)
    Ndoms = [3, 4, 6, 8, 12]
    curve = convergence_curve_H(comm, Ndoms, h_fine_target=1/256,
                                degreeU=1, degreeP=1, alpha=1.0,
                                basis_kind=BASIS_KIND, mixed_order=MIXED_ORDER,
                                verbose=False)
    results = {"meta": {"experiment": "E4", "mixed_order": MIXED_ORDER}, **curve}
    save_json(results, "e4_H_multiscale.json", comm)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# E5 — Modo h saturação
# ══════════════════════════════════════════════════════════════════════════════

def run_E5(comm):
    from mprm.benchmarks.convergence import convergence_curve_h
    banner("E5 — Modo h saturação (BDM₂)", comm)
    nlocs = [2, 4, 8, 16, 32, 64]
    curve = convergence_curve_h(comm, Ndom=4, nlocs=nlocs,
                                degreeU=1, degreeP=1, alpha=1.0,
                                basis_kind=BASIS_KIND, mixed_order=MIXED_ORDER,
                                verbose=False)
    results = {"meta": {"experiment": "E5", "mixed_order": MIXED_ORDER}, **curve}
    save_json(results, "e5_h_saturation.json", comm)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# E6 — Modo p (grau)
# ══════════════════════════════════════════════════════════════════════════════

def run_E6(comm):
    from mprm.benchmarks.convergence import convergence_curve_p
    banner("E6 — Modo p grau (BDM₂)", comm)
    ks = [1, 2, 3, 4]
    curve = convergence_curve_p(comm, Ndom=4, nloc=16, ks=ks, alpha=1.0,
                                basis_kind=BASIS_KIND, mode="mixed",
                                mixed_order=MIXED_ORDER, verbose=False)
    results = {"meta": {"experiment": "E6", "mixed_order": MIXED_ORDER}, **curve}
    save_json(results, "e6_p_degree.json", comm)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# E10 — Downscaling on/off
# ══════════════════════════════════════════════════════════════════════════════

def run_E10(comm):
    from mprm.benchmarks.convergence import convergence_curve_hh
    banner("E10 — Downscaling on/off (BDM₂)", comm)
    Ndoms = [2, 4, 8, 16]
    curve = convergence_curve_hh(comm, Ndoms, nloc=16, degreeU=1, degreeP=1,
                                 alpha=1.0, basis_kind=BASIS_KIND,
                                 include_raw_mrcm=True,
                                 mixed_order=MIXED_ORDER, verbose=False)
    results = {"meta": {"experiment": "E10", "mixed_order": MIXED_ORDER}, **curve}
    save_json(results, "e10_downscaling_on_off.json", comm)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# E12 — Hh × subespaço
# ══════════════════════════════════════════════════════════════════════════════

def run_E12(comm):
    from mprm.benchmarks.convergence import convergence_curve_hh
    banner("E12 — Hh × subespaço (BDM₂)", comm)
    Ndoms = [2, 4, 8, 16]
    modes = ["gradient", "solenoidal", "mixed"]
    results = {"meta": {"experiment": "E12", "mixed_order": MIXED_ORDER}, "curves": []}
    for mode in modes:
        if comm.rank == 0:
            print(f"  mode={mode}...", flush=True)
        curve = convergence_curve_hh(comm, Ndoms, nloc=16, degreeU=1, degreeP=1,
                                     alpha=1.0, basis_kind=BASIS_KIND,
                                     mode=mode, mixed_order=MIXED_ORDER,
                                     verbose=False)
        results["curves"].append({"mode": mode, **curve})
    save_json(results, "e12_subspace.json", comm)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# E12c — α × subespaço
# ══════════════════════════════════════════════════════════════════════════════

def run_E12c(comm):
    from mprm.benchmarks.convergence import convergence_curve_hh
    banner("E12c — α × subespaço (BDM₂)", comm)
    alphas = [1e-3, 1e-2, 0.1, 1, 10, 100, 1e3]
    modes = ["gradient", "solenoidal", "mixed"]
    results = {"meta": {"experiment": "E12c", "mixed_order": MIXED_ORDER}, "curves": []}
    for mode in modes:
        cur = []
        for a in alphas:
            if comm.rank == 0:
                print(f"  mode={mode}, α={a:.1e}...", flush=True)
            case = convergence_curve_hh(comm, [4], nloc=16, degreeU=1, degreeP=1,
                                        alpha=a, basis_kind=BASIS_KIND,
                                        mode=mode, mixed_order=MIXED_ORDER,
                                        verbose=False)
            cur.append({"alpha": a, **case})
        results["curves"].append({"mode": mode, "cases": cur})
    save_json(results, "e12_subspace_alpha.json", comm)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# E12d — Hh × H/h
# ══════════════════════════════════════════════════════════════════════════════

def run_E12d(comm):
    from mprm.benchmarks.convergence import convergence_curve_hh
    banner("E12d — Hh × H/h (BDM₂)", comm)
    results = {"meta": {"experiment": "E12d", "mixed_order": MIXED_ORDER}, "curves": []}
    for hh_ratio in [8, 16, 32]:
        for mode in ["gradient", "solenoidal", "mixed"]:
            if comm.rank == 0:
                print(f"  H/h={hh_ratio}, mode={mode}...", flush=True)
            curve = convergence_curve_hh(comm, [2, 4, 8], nloc=hh_ratio,
                                         degreeU=1, degreeP=1, alpha=1.0,
                                         basis_kind=BASIS_KIND, mode=mode,
                                         mixed_order=MIXED_ORDER, verbose=False)
            results["curves"].append({"Hh_ratio": hh_ratio, "mode": mode, **curve})
    save_json(results, "e12d_hh_ratio.json", comm)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# E13-H1 — h × ortogonalidade (saturação)
# ══════════════════════════════════════════════════════════════════════════════

def run_E13_H1(comm):
    from mprm.benchmarks.convergence import convergence_curve_hh
    banner("E13-H1 — h × ortogonalidade (BDM₂)", comm)
    nlocs = [8, 16, 24, 32]
    modes = ["gradient", "mixed"]
    results = {"meta": {"experiment": "E13-H1", "mixed_order": MIXED_ORDER}, "curves": []}
    for mode in modes:
        cur = []
        for nloc in nlocs:
            if comm.rank == 0:
                print(f"  mode={mode}, nloc={nloc}...", flush=True)
            case = convergence_curve_hh(comm, [3], nloc=nloc, degreeU=1, degreeP=1,
                                        alpha=1.0, basis_kind=BASIS_KIND,
                                        mode=mode, mixed_order=MIXED_ORDER,
                                        verbose=False)
            cur.append({"nloc": nloc, **case})
        results["curves"].append({"mode": mode, "cases": cur})
    save_json(results, "e13_saturacao.json", comm)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# E13-H3 — grau × ortogonalidade
# ══════════════════════════════════════════════════════════════════════════════

def run_E13_H3(comm):
    from mprm.benchmarks.convergence import convergence_curve_hh
    banner("E13-H3 — grau × ortogonalidade (BDM₂)", comm)
    degrees = [1, 2, 3]
    modes = ["gradient", "mixed"]
    results = {"meta": {"experiment": "E13-H3", "mixed_order": MIXED_ORDER}, "curves": []}
    for mode in modes:
        cur = []
        for deg in degrees:
            if comm.rank == 0:
                print(f"  mode={mode}, degree={deg}...", flush=True)
            case = convergence_curve_hh(comm, [3], nloc=16, degreeU=deg, degreeP=deg,
                                        alpha=1.0, basis_kind=BASIS_KIND,
                                        mode=mode, mixed_order=MIXED_ORDER,
                                        verbose=False)
            cur.append({"degree": deg, **case})
        results["curves"].append({"mode": mode, "cases": cur})
    save_json(results, "e13_graus.json", comm)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

EXPERIMENTS = {
    "E1": run_E1, "E2": run_E2, "E3": run_E3, "E4": run_E4,
    "E5": run_E5, "E6": run_E6, "E10": run_E10,
    "E12": run_E12, "E12c": run_E12c, "E12d": run_E12d,
    "E13-H1": run_E13_H1, "E13-H3": run_E13_H3,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convergence study E1–E13 with BDM₂")
    parser.add_argument("--experiments", default="all",
                       help="Comma-separated list: E1,E2,... or 'all'")
    args = parser.parse_args()

    comm = get_comm()
    os.makedirs(OUTDIR, exist_ok=True)

    if args.experiments == "all":
        to_run = list(EXPERIMENTS.keys())
    else:
        to_run = [e.strip() for e in args.experiments.split(",")]

    t0 = time.perf_counter()
    for name in to_run:
        if name not in EXPERIMENTS:
            if comm.rank == 0:
                print(f"  ⚠  Experimento desconhecido: {name}")
            continue
        t1 = time.perf_counter()
        EXPERIMENTS[name](comm)
        if comm.rank == 0:
            print(f"  ⏱  {name}: {time.perf_counter()-t1:.0f}s", flush=True)

    if comm.rank == 0:
        print(f"\nBDM₂ convergence study done in {time.perf_counter()-t0:.0f}s")
        print(f"Results: {OUTDIR}/")
