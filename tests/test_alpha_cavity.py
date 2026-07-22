#!/usr/bin/env python3
"""
tests/test_alpha_cavity.py — Estudo 2: cavidade com tampa deslizante vs α.

Adaptação da **Fig. 11** do artigo do MPrM (Lages, §3.4): varredura de α com
espaços U_{H,1} (fluxo linear) e P_{H,0} (pressão constante) — os mesmos da
Fig. 11. No artigo o problema é o degrau (backward-facing step); aqui
usamos a **cavidade com tampa deslizante** como análogo de contorno
impermeável, pois BCs de entrada/saída (fluxo imposto ≠ 0) não são
suportadas corretamente pelo mrcm
(Vault/bugs/2026-07-19-noflow-strong-bc-left-right-missing.md).

Referência: solução de malha fina (mesma malha MAC, mesmo δt e T).

O estudo da Tabela 3 (von Kármán, α ∈ {1e-6, 1, 1e6}) NÃO é replicado —
exige as mesmas BCs de entrada/saída bloqueadas acima.

Uso:
  python tests/test_alpha_cavity.py                      # config completa
  python tests/test_alpha_cavity.py --quick               # smoke test
  python tests/test_alpha_cavity.py --alphas 1e-2,1,1e2,1e4
  python tests/test_alpha_cavity.py --Re 400 --T 4
  mpiexec -n 3 python tests/test_alpha_cavity.py           # paralelo (recomendado)

Saída: results_alpha/alpha_cavity.json, results_alpha/alpha_cavity_fig11.png.
"""

from __future__ import annotations
import time

import numpy as np

from common import (
    get_comm, apply_plot_style, rel_err, banner, make_meta,
    save_json, save_figure, COLORS, INK, DEFAULT_OUTDIR,
)
from params import CommonParams, parse_args, parse_float_list

from mprm_core import run_cavity

# Espaços da Fig. 11 do artigo: fluxo linear / pressão constante.
DEFAULT_DEGREE_U = 1
DEFAULT_DEGREE_P = 0

# α da varredura completa: cobre os limites MMMFEM (α→0) e MHM (α→∞),
# incluindo os pontos que confirmaram a saturação do platô MHM (1e4, 1e6).
DEFAULT_ALPHAS = "1e-4,1e-2,1e-1,3e-1,1,3,10,100,1e4,1e6"


# ===========================================================================
# Computação
# ===========================================================================

def run_study(comm, alphas: list[float], Ndom: int, nloc: int,
              Re: float, T: float, degreeU: int, degreeP: int,
              basis_kind: str = "legendre_P", dt: float | None = None,
              verbose: bool = True) -> dict:
    """Varre α e mede o erro da cavidade MPrM vs a cavidade de malha fina.

    Retorna um dict `{"meta": {...}, "cases": [...]}` — um `case` por α,
    com `err_vs_fine`, `divmax`, `defect`, `wall_s`.
    """
    N = Ndom * nloc
    h = 1.0 / N
    nu = 1.0 / Re
    if dt is None:
        dt = 0.4 * min(h, h * h / (4.0 * nu))
    nsteps = int(np.ceil(T / dt))

    if verbose:
        banner(f"Estudo 2 (Fig. 11 adaptada): cavidade Re={Re} vs α — "
              f"{N}², {Ndom}×{Ndom} subdomínios, "
              f"U_H,{degreeU}/P_H,{degreeP}", comm)
        if comm.rank == 0:
            print(f"  T={T}  δt={dt:.5f}  ({nsteps} passos/α)")
            print(f"  α ∈ {np.array(alphas)}", flush=True)

    t0 = time.perf_counter()
    ref = run_cavity(Re=Re, Ndom=Ndom, nloc=nloc, T=T, dt=dt,
                     method="fine", comm=comm, verbose=False)
    if verbose and comm.rank == 0:
        print(f"  Referência fina: {time.perf_counter()-t0:.1f}s  "
              f"div={ref['divmax']:.1e}", flush=True)

    results = {"meta": make_meta(
        "cavity_alpha (Fig. 11 adaptada: BFS→cavidade)", comm,
        N=N, Ndom=Ndom, nloc=nloc, Re=Re, T=T, dt=dt, nsteps=nsteps,
        degreeU=degreeU, degreeP=degreeP, basis_kind=basis_kind,
        alphas=[float(a) for a in alphas],
        reference="fine MAC cavity, mesmo δt e T",
    ), "cases": []}

    for a in alphas:
        case = _run_one_case(comm, ref, Ndom, nloc, Re, T, dt,
                             degreeU, degreeP, a, basis_kind)
        results["cases"].append(case)
        if verbose and comm.rank == 0:
            print(f"  α={a:9.1e}:  err_fino={case['err_vs_fine']:.4e}  "
                  f"div={case['divmax']:.1e}  ({case['wall_s']:.0f}s)",
                  flush=True)

    return results


def _run_one_case(comm, ref, Ndom, nloc, Re, T, dt,
                  degreeU, degreeP, alpha, basis_kind) -> dict:
    """Um único ponto α da varredura: roda a cavidade MPrM e mede o erro."""
    t0 = time.perf_counter()
    try:
        r = run_cavity(Re=Re, Ndom=Ndom, nloc=nloc, T=T, dt=dt,
                       degreeU=degreeU, degreeP=degreeP, alpha=float(alpha),
                       basis_kind=basis_kind,
                       method="mprm", comm=comm, verbose=False)
        err = rel_err(r["u"], r["v"], ref["u"], ref["v"])
        return {"alpha": float(alpha), "err_vs_fine": err,
                "divmax": r["divmax"], "defect": r["defect"],
                "wall_s": time.perf_counter() - t0}
    except Exception as e:  # instabilidade em α extremo → registrar, não abortar
        return {"alpha": float(alpha), "err_vs_fine": float("nan"),
                "divmax": float("nan"), "defect": float("nan"),
                "wall_s": time.perf_counter() - t0,
                "error": f"{type(e).__name__}: {e}"}


# ===========================================================================
# Plotagem
# ===========================================================================

def plot_results(results: dict, outdir: str) -> str:
    """Erro vs α — análogo da Fig. 11, com o mínimo anotado."""
    plt = apply_plot_style()
    m = results["meta"]
    alphas = np.array([c["alpha"] for c in results["cases"]])
    errs = np.array([c["err_vs_fine"] for c in results["cases"]])
    ok = np.isfinite(errs)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.loglog(alphas[ok], errs[ok], "-", marker="o", markersize=7,
              linewidth=2, color=COLORS[1])
    if ok.any():
        i_min = int(np.nanargmin(errs))
        exp = int(np.round(np.log10(alphas[i_min])))
        ax.annotate(rf"mínimo: $\alpha=10^{{{exp}}}$"
                    f"\nerro = {errs[i_min]:.2e}",
                    xy=(alphas[i_min], errs[i_min]),
                    xytext=(-14, 40), textcoords="offset points",
                    ha="right", fontsize=10, color=INK,
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1))
    ax.set_xlabel(r"$\alpha$ (parâmetro de Robin)")
    ax.set_ylabel(r"$\|u_{\rm MPrM} - u_{\rm fino}\|_2\,/\,\|u_{\rm fino}\|_2$")
    ax.set_title(f"Cavidade Re={m['Re']:.0f}: erro vs α — análogo da Fig. 11 "
                 f"($U_{{H,{m['degreeU']}}}$, $P_{{H,{m['degreeP']}}}$, "
                 f"T={m['T']:g})")
    fig.tight_layout()
    return save_figure(fig, outdir, "alpha_cavity_fig11.png")


# ===========================================================================
# CLI
# ===========================================================================

def main(argv=None):
    defaults = CommonParams(Ndom=3, nloc=16, degreeU=DEFAULT_DEGREE_U,
                            degreeP=DEFAULT_DEGREE_P, Re=100.0, T=8.0,
                            outdir=DEFAULT_OUTDIR)
    args = parse_args(
        "Estudo 2: cavidade com tampa deslizante vs α (análogo da Fig. 11, "
        "Lages §3.4)",
        defaults=defaults,
        extra_args=lambda p: p.add_argument(
            "--alphas", default=DEFAULT_ALPHAS,
            help="lista 'a,b,c' ou 'logspace:ini,fim,n'"),
        argv=argv,
    )

    if args.quick:
        alphas = [1.0]
        Ndom, nloc, T = 3, 16, 0.3
    else:
        alphas = parse_float_list(args.alphas)
        Ndom, nloc, T = args.Ndom, args.nloc, args.T

    comm = get_comm()
    t0 = time.perf_counter()
    results = run_study(comm, alphas, Ndom, nloc, args.Re, T,
                        args.degreeU, args.degreeP,
                        basis_kind=args.basis_kind, dt=args.dt,
                        verbose=args.verbose)

    if comm.rank == 0:
        save_json(results, args.outdir, "alpha_cavity.json")
        plot_results(results, args.outdir)
        print(f"\nEstudo concluído em {time.perf_counter()-t0:.0f}s "
              f"({comm.size} rank(s)).")
    return results


if __name__ == "__main__":
    main()
