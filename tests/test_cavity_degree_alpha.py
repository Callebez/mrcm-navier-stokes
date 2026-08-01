#!/usr/bin/env python3
"""
tests/test_cavity_degree_alpha.py — Cavidade: grau de interface × α.

Estuda como a solução da cavidade (Re=100) é afetada pela variação
simultânea do grau dos espaços de interface (k=1,2,3) e do parâmetro
de Robin (α ∈ [1e-2, 1e4]).

Para cada (k, α), mede:
  - erro vs referência fina (mesma malha, mesmo δt, mesmo T)
  - perfis u(y) em x=0.5 e v(x) em y=0.5
  - divmax, defect, wall time

Uso:
  python tests/test_cavity_degree_alpha.py                        # tudo
  python tests/test_cavity_degree_alpha.py --ks 2,3 --alphas 1,100
  mpiexec -n 4 python tests/test_cavity_degree_alpha.py           # MPI

Saída: results_alpha/cavity_degree_alpha/
"""

from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# Ghia reference (for profile comparison plots)
# ══════════════════════════════════════════════════════════════════════════════
GHIA_U_100 = {
    "y": [0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813,
          0.4531, 0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609,
          0.9688, 0.9766, 1.0000],
    "u": [0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150,
          -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151,
          0.68717, 0.73722, 0.78871, 0.84123, 1.00000],
}
GHIA_V_100 = {
    "x": [0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266,
          0.2344, 0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531,
          0.9609, 0.9688, 1.0000],
    "v": [0.00000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077,
          0.17507, 0.17527, 0.05454, -0.24533, -0.22445, -0.16914,
          -0.10313, -0.08864, -0.07391, -0.05906, 0.00000],
}

# ══════════════════════════════════════════════════════════════════════════════
# Core
# ══════════════════════════════════════════════════════════════════════════════

def run_one(Re, Ndom, nloc, degreeU, degreeP, alpha, T, basis_kind, comm):
    """Roda a cavidade e extrai métricas + perfis."""
    from mprm_core import run_cavity

    N = Ndom * nloc
    h = 1.0 / N
    nu = 1.0 / Re
    dt = 0.4 * min(h, h * h / (4.0 * nu))

    t0 = time.perf_counter()
    r = run_cavity(Re=Re, Ndom=Ndom, nloc=nloc, T=T, dt=dt,
                   degreeU=degreeU, degreeP=degreeP, alpha=alpha,
                   basis_kind=basis_kind, method="mprm", comm=comm,
                   verbose=False)
    wall = time.perf_counter() - t0

    u, v = r["u"], r["v"]
    Nx, Ny = u.shape[0] - 1, v.shape[1] - 1
    hx, hy = 1.0 / Nx, 1.0 / Ny

    # u along vertical centerline
    ix = Nx // 2
    y_u = (np.arange(Ny) + 0.5) * hy
    u_profile = 0.5 * (u[ix, :] + u[ix + 1, :])

    # v along horizontal centerline
    jy = Ny // 2
    x_v = (np.arange(Nx) + 0.5) * hx
    v_profile = 0.5 * (v[:, jy] + v[:, jy + 1])

    return {
        "Re": Re, "N": Nx, "Ndom": Ndom, "nloc": nloc,
        "degreeU": degreeU, "degreeP": degreeP, "alpha": alpha,
        "T": T, "dt": dt, "nsteps": r["nsteps"],
        "divmax": r["divmax"], "defect": r["defect"],
        "wall_s": wall,
        "u": u, "v": v,                               # campos completos
        "y_u": y_u.tolist(), "u_profile": u_profile.tolist(),
        "x_v": x_v.tolist(), "v_profile": v_profile.tolist(),
    }


def rel_err(u1_a, v1_a, u1_b, v1_b):
    d1, d2 = u1_a - u1_b, v1_a - v1_b
    n = np.sqrt(np.sum(d1**2) + np.sum(d2**2))
    d = np.sqrt(np.sum(u1_b**2) + np.sum(v1_b**2))
    return float(n / d) if d > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(results: dict, outdir: str):
    plt = None
    try:
        from common import apply_plot_style, save_figure, COLORS, MARKERS, INK
        plt = apply_plot_style()
    except ImportError:
        import matplotlib.pyplot as plt
        plt.style.use("default")

    cases = results["cases"]
    alphas = sorted(set(c["alpha"] for c in cases))
    ks = sorted(set(c["degreeU"] for c in cases))
    Re = cases[0]["Re"]

    # ── Figure 1: error vs α, one curve per k ──
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for k in ks:
        pts = [(c["alpha"], c["err_vs_fine"]) for c in cases
               if c["degreeU"] == k and np.isfinite(c["err_vs_fine"])]
        if not pts: continue
        pts.sort()
        aa, ee = zip(*pts)
        ax.loglog(aa, ee, "o-", markersize=7, linewidth=1.5,
                  color=COLORS.get(k, f"C{k}"),
                  label=f"$(P_{{H,{k}}},U_{{H,{k}}})$")
        i = np.argmin(ee)
        ax.plot(aa[i], ee[i], "*", color=COLORS.get(k, f"C{k}"),
                markersize=16, markeredgecolor="black", markeredgewidth=0.5)
    ax.set_xlabel("α"); ax.set_ylabel(r"$\|u_{MPrM}-u_{fino}\|_2/\|u_{fino}\|_2$")
    ax.set_title(f"Cavidade Re={Re}: erro vs α e grau de interface  · ★ = mínimo")
    ax.grid(True, alpha=0.25); ax.legend(fontsize=9)
    fig.tight_layout()
    save_figure(fig, outdir, "error_vs_alpha.png")

    # ── Figure 2: u(y) profiles vs Ghia, one panel per k at best α ──
    nk = len(ks)
    fig, axes = plt.subplots(1, nk, figsize=(5 * nk, 5.5))
    if nk == 1: axes = [axes]
    for ax, k in zip(axes, ks):
        # Best α for this k
        k_cases = [c for c in cases if c["degreeU"] == k]
        if not k_cases: continue
        best = min(k_cases, key=lambda c: c["err_vs_fine"])
        prof = best.get("profiles", best)
        ax.plot(prof["u"], prof["y_u"], "-", linewidth=2,
                color=COLORS.get(k, f"C{k}"),
                label=f"MPrM k={k}, α={best['alpha']:.1e}")
        ax.plot(GHIA_U_100["u"], GHIA_U_100["y"], "ko", markersize=5,
                label="Ghia")
        ax.set_xlabel("u"); ax.set_ylabel("y")
        ax.set_title(f"k={k} (α={best['alpha']:.0e})", fontweight="bold")
        ax.grid(True, alpha=0.25); ax.legend(fontsize=7)
    fig.suptitle(f"Cavidade Re={Re}: u(y) em x=0.5 — melhor α por grau", fontsize=12)
    fig.tight_layout()
    save_figure(fig, outdir, "profiles_u_vs_ghia.png")

    # ── Figure 3: u(y) for all α at fixed k=2 ──
    fig, ax = plt.subplots(figsize=(7, 5.5))
    k = 2
    for a in alphas:
        c = [c for c in cases if c["degreeU"] == k and abs(c["alpha"] - a) < 1e-9]
        if not c: continue
        c = c[0]
        prof = c.get("profiles", c)
        ax.plot(prof["u"], prof["y_u"], "-", linewidth=1, alpha=0.7,
                label=f"α={a:.0e}")
    ax.plot(GHIA_U_100["u"], GHIA_U_100["y"], "ko", markersize=5, label="Ghia")
    ax.set_xlabel("u"); ax.set_ylabel("y")
    ax.set_title(f"Cavidade Re={Re}: u(y) em x=0.5 — k=2, todos α")
    ax.grid(True, alpha=0.25); ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    save_figure(fig, outdir, "profiles_u_k2_all_alpha.png")

    print(f"  ✓ {len([f for f in os.listdir(outdir) if f.endswith('.png')])} figuras")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Cavidade: grau de interface × α")
    parser.add_argument("--Re", type=float, default=100.0)
    parser.add_argument("--Ndom", type=int, default=4)
    parser.add_argument("--nloc", type=int, default=16)
    parser.add_argument("--ks", default="1,2,3")
    parser.add_argument("--alphas", default="1e-2,1e-1,1,1e1,1e2,1e4")
    parser.add_argument("--T", type=float, default=20.0)
    parser.add_argument("--outdir", default="results_alpha/cavity_degree_alpha")
    parser.add_argument("--basis-kind", default="legendre_P")
    args = parser.parse_args()

    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    ks = [int(x) for x in args.ks.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]
    Re = int(args.Re)
    T = args.T
    N = args.Ndom * args.nloc
    os.makedirs(args.outdir, exist_ok=True)

    if comm.rank == 0:
        print(f"Cavidade grau×α: Re={Re}, N={N}², {args.Ndom}×{args.Ndom}, "
              f"T={T}, ks={ks}, alphas={alphas}")
        print(f"MPI ranks: {comm.size}, total cases: {len(ks)*len(alphas)}")
        print()

    # ── Fine reference ──
    from mprm_core import run_cavity
    if comm.rank == 0: print("Computando referência fina...", flush=True)
    t0 = time.perf_counter()
    ref = run_cavity(Re=Re, Ndom=args.Ndom, nloc=args.nloc, T=T,
                     method="fine", comm=comm, verbose=False)
    if comm.rank == 0:
        print(f"  referência: {time.perf_counter()-t0:.0f}s, "
              f"div={ref['divmax']:.1e}\n", flush=True)

    # ── Sweep (k, α) ──
    cases = []
    for k in ks:
        for a in alphas:
            if comm.rank == 0:
                print(f"  k={k}, α={a:.1e} ...", flush=True)
            rec = run_one(Re, args.Ndom, args.nloc, k, k, a, T,
                         args.basis_kind, comm)
            rec["err_vs_fine"] = rel_err(rec["u"], rec["v"],
                                         ref["u"], ref["v"])
            # Remove full fields (huge for JSON), keep 1D profiles
            del rec["u"], rec["v"]
            prof_u = rec.pop("u_profile")
            prof_v = rec.pop("v_profile")
            y_u = rec.pop("y_u")
            x_v = rec.pop("x_v")
            rec["profiles"] = {"y_u": y_u, "u": prof_u, "x_v": x_v, "v": prof_v}
            cases.append(rec)
            if comm.rank == 0:
                print(f"    err={rec['err_vs_fine']:.4e}, "
                      f"div={rec['divmax']:.1e}, {rec['wall_s']:.0f}s",
                      flush=True)

    results = {
        "meta": {
            "test": "cavity_degree_alpha",
            "Re": Re, "N": N, "Ndom": args.Ndom, "nloc": args.nloc,
            "T": T, "ks": ks, "alphas": alphas,
            "basis_kind": args.basis_kind,
        },
        "cases": cases,
    }

    if comm.rank == 0:
        json_path = os.path.join(args.outdir, "results.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  ✓ {json_path}")
        plot_results(results, args.outdir)
        print(f"Concluído. Resultados em {args.outdir}/")


if __name__ == "__main__":
    main()
