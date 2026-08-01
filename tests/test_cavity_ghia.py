#!/usr/bin/env python3
"""
tests/test_cavity_ghia.py — Validação da cavidade contra Ghia et al. (1982).

Roda a cavidade com tampa deslizante para Re ∈ {100, 400, 1000}, extrai os
perfis de velocidade nas linhas centrais (x=0.5 e y=0.5) e compara contra
os dados tabulados de Ghia et al., J. Comput. Phys. 48:387–411, 1982.

Uso:
  python tests/test_cavity_ghia.py                           # Re=100,400,1000
  python tests/test_cavity_ghia.py --Re 100                  # só Re=100
  python tests/test_cavity_ghia.py --Re 100 --Ndom 4 --nloc 32  # mais refinado
  mpiexec -n 4 python tests/test_cavity_ghia.py --Re 100,400,1000

Saída: results_alpha/cavity_ghia/
"""

from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# Ghia et al. (1982) — dados tabulados
# ══════════════════════════════════════════════════════════════════════════════

# u-velocity along vertical line through geometric center (x=0.5, y ∈ [0,1])
GHIA_U = {
    100: {
        "y":    [0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813,
                 0.4531, 0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609,
                 0.9688, 0.9766, 1.0000],
        "u":    [0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150,
                 -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151,
                 0.68717, 0.73722, 0.78871, 0.84123, 1.00000],
    },
    400: {
        "y":    [0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813,
                 0.4531, 0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609,
                 0.9688, 0.9766, 1.0000],
        "u":    [0.00000, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299,
                 -0.32726, -0.17119, -0.11477, 0.02135, 0.16256, 0.29093,
                 0.55892, 0.61756, 0.68439, 0.75837, 1.00000],
    },
    1000: {
        "y":    [0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813,
                 0.4531, 0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609,
                 0.9688, 0.9766, 1.0000],
        "u":    [0.00000, -0.18109, -0.20196, -0.22220, -0.29730, -0.38289,
                 -0.27805, -0.10648, -0.06080, 0.05702, 0.18719, 0.33304,
                 0.46604, 0.51117, 0.57492, 0.65928, 1.00000],
    },
}

# v-velocity along horizontal line through geometric center (y=0.5, x ∈ [0,1])
GHIA_V = {
    100: {
        "x":    [0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266,
                 0.2344, 0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531,
                 0.9609, 0.9688, 1.0000],
        "v":    [0.00000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077,
                 0.17507, 0.17527, 0.05454, -0.24533, -0.22445, -0.16914,
                 -0.10313, -0.08864, -0.07391, -0.05906, 0.00000],
    },
    400: {
        "x":    [0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266,
                 0.2344, 0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531,
                 0.9609, 0.9688, 1.0000],
        "v":    [0.00000, 0.18360, 0.19713, 0.20920, 0.22965, 0.28124,
                 0.30203, 0.30174, 0.05186, -0.38598, -0.44993, -0.23827,
                 -0.22847, -0.19254, -0.15663, -0.12146, 0.00000],
    },
    1000: {
        "x":    [0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266,
                 0.2344, 0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531,
                 0.9609, 0.9688, 1.0000],
        "v":    [0.00000, 0.27485, 0.29012, 0.30353, 0.32627, 0.37095,
                 0.33075, 0.32235, 0.02526, -0.31966, -0.42665, -0.51550,
                 -0.39188, -0.33714, -0.27669, -0.21394, 0.00000],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Simulação
# ══════════════════════════════════════════════════════════════════════════════

def run_validation(Re, Ndom=8, nloc=16, degreeU=2, degreeP=2, alpha=1.0,
                   T=None, basis_kind="legendre_P", comm=None, verbose=True):
    """Roda a cavidade e extrai perfis de velocidade nas linhas centrais."""
    from mprm_core import run_cavity

    if T is None:
        T = {100: 30.0, 400: 40.0, 1000: 50.0}.get(Re, 30.0)

    if verbose and (comm is None or comm.rank == 0):
        print(f"  Re={Re}, N={Ndom*nloc}², T={T}, α={alpha}, "
              f"U_H,{degreeU}/P_H,{degreeP} ...", flush=True)

    t0 = time.perf_counter()
    r = run_cavity(Re=Re, Ndom=Ndom, nloc=nloc, T=T,
                   degreeU=degreeU, degreeP=degreeP, alpha=alpha,
                   basis_kind=basis_kind, method="mprm", comm=comm,
                   verbose=False)
    wall = time.perf_counter() - t0

    u, v = r["u"], r["v"]
    Nx, Ny = u.shape[0] - 1, v.shape[1] - 1
    hx, hy = 1.0 / Nx, 1.0 / Ny

    # u-velocity along vertical centerline (i = Nx//2)
    ix = Nx // 2
    y_u = (np.arange(Ny) + 0.5) * hy
    u_profile = 0.5 * (u[ix, :] + u[ix + 1, :])  # média das faces adjacentes

    # v-velocity along horizontal centerline (j = Ny//2)
    jy = Ny // 2
    x_v = (np.arange(Nx) + 0.5) * hx
    v_profile = 0.5 * (v[:, jy] + v[:, jy + 1])

    return {
        "Re": Re, "N": Nx, "Ndom": Ndom, "nloc": nloc,
        "degreeU": degreeU, "degreeP": degreeP, "alpha": alpha, "T": T,
        "dt": r["dt"], "nsteps": r["nsteps"],
        "divmax": r["divmax"], "defect": r["defect"],
        "wall_s": wall,
        "y_u": y_u.tolist(), "u_profile": u_profile.tolist(),
        "x_v": x_v.tolist(), "v_profile": v_profile.tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Plotagem
# ══════════════════════════════════════════════════════════════════════════════

def plot_validation(results: dict, outdir: str):
    """Plota perfis de velocidade contra Ghia para cada Re."""
    plt = None
    try:
        from common import apply_plot_style, save_figure, COLORS, INK
        plt = apply_plot_style()
    except ImportError:
        import matplotlib.pyplot as plt
        plt.style.use("default")

    for rec in results["cases"]:
        Re = rec["Re"]
        if Re not in GHIA_U:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

        # ── u-velocity along vertical centerline ──
        ax = axes[0]
        ghia = GHIA_U[Re]
        ax.plot(ghia["u"], ghia["y"], "ko", markersize=6, label="Ghia et al. (1982)")
        ax.plot(rec["u_profile"], rec["y_u"], "-", color=COLORS.get(1, "#1f77b4"),
                linewidth=2, label=f'MPrM (N={rec["N"]}$^2$, {rec["Ndom"]}$\\times${rec["Ndom"]})')
        ax.set_xlabel("u"); ax.set_ylabel("y")
        ax.set_title(f"u-velocity at x=0.5, Re={Re}")
        ax.grid(True, alpha=0.25); ax.legend(fontsize=8)
        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")

        # ── v-velocity along horizontal centerline ──
        ax = axes[1]
        ghia_v = GHIA_V[Re]
        ax.plot(ghia_v["x"], ghia_v["v"], "ko", markersize=6, label="Ghia et al. (1982)")
        ax.plot(rec["x_v"], rec["v_profile"], "-", color=COLORS.get(2, "#ff7f0e"),
                linewidth=2, label=f'MPrM (N={rec["N"]}$^2$, {rec["Ndom"]}$\\times${rec["Ndom"]})')
        ax.set_xlabel("x"); ax.set_ylabel("v")
        ax.set_title(f"v-velocity at y=0.5, Re={Re}")
        ax.grid(True, alpha=0.25); ax.legend(fontsize=8)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

        fig.suptitle(f"Cavidade com tampa deslizante — Re={Re}  ·  "
                     f"α={rec['alpha']}, U_H,{rec['degreeU']}/P_H,{rec['degreeP']}",
                     fontsize=12)
        fig.tight_layout()

        path = os.path.join(outdir, f"ghia_Re{Re}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  ✓ {path}")
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Validação da cavidade contra Ghia et al. (1982)")
    parser.add_argument("--Re", default="100,400,1000",
                       help="Reynolds numbers (comma-separated)")
    parser.add_argument("--Ndom", type=int, default=8,
                       help="Subdomínios por direção")
    parser.add_argument("--nloc", type=int, default=16,
                       help="Células por subdomínio")
    parser.add_argument("--degreeU", type=int, default=2)
    parser.add_argument("--degreeP", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--T", type=float, default=None,
                       help="Tempo final (default: Re-dependente)")
    parser.add_argument("--outdir", default="results_alpha/cavity_ghia")
    parser.add_argument("--basis-kind", default="legendre_P")
    args = parser.parse_args()

    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    Res = [float(r) for r in args.Re.split(",")]
    os.makedirs(args.outdir, exist_ok=True)

    if comm.rank == 0:
        print(f"Cavidade × Ghia: Re ∈ {Res}, "
              f"N={args.Ndom*args.nloc}², "
              f"{args.Ndom}×{args.Ndom}, nloc={args.nloc}, "
              f"U_H,{args.degreeU}/P_H,{args.degreeP}, α={args.alpha}")
        print(f"MPI ranks: {comm.size}")
        print()

    cases = []
    for Re in Res:
        rec = run_validation(int(Re), Ndom=args.Ndom, nloc=args.nloc,
                            degreeU=args.degreeU, degreeP=args.degreeP,
                            alpha=args.alpha, T=args.T,
                            basis_kind=args.basis_kind, comm=comm,
                            verbose=(comm.rank == 0))
        cases.append(rec)
        if comm.rank == 0:
            print(f"    divmax={rec['divmax']:.1e}  "
                  f"nsteps={rec['nsteps']}  dt={rec['dt']:.4e}  "
                  f"{rec['wall_s']:.0f}s")

    results = {
        "meta": {
            "test": "cavity_ghia_validation",
            "reference": "Ghia et al. (1982), J. Comput. Phys. 48:387-411",
            "N": args.Ndom * args.nloc,
            "Ndom": args.Ndom, "nloc": args.nloc,
            "degreeU": args.degreeU, "degreeP": args.degreeP,
            "alpha": args.alpha, "basis_kind": args.basis_kind,
        },
        "cases": cases,
    }

    if comm.rank == 0:
        json_path = os.path.join(args.outdir, "ghia_validation.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  ✓ {json_path}")

        plot_validation(results, args.outdir)
        print(f"\nValidação concluída. Resultados em {args.outdir}/")


if __name__ == "__main__":
    main()
