#!/usr/bin/env python3
"""
tests/test_alpha_hh.py — Estudo 1: decomposição HH modificada vs α.

Réplica da **Fig. 6** do artigo do MPrM (Lages, §3.1): erro da decomposição
de Helmholtz-Hodge modificada em função de α, para os espaços de interface
(P_{H,k}, U_{H,k}), k = 1..4, na configuração do artigo — malha fina 90×90,
decomposição 3×3 (30×30 células/subdomínio por padrão).

Como no artigo, a referência é a **solução não decomposta** (projeção MAC
fina sobre o mesmo w); o erro vs o campo analítico de Olshanskii também é
registrado no JSON.

Tendências esperadas pelo artigo: erro melhora com o enriquecimento dos
espaços; varia suavemente com α; pode haver um mínimo perto de α ≈ 1.
(Ver docs/05-estudo-alpha.md para o confronto detalhado com os resultados
medidos aqui — nem todas as tendências do artigo se confirmam.)

Uso:
  python tests/test_alpha_hh.py                          # config completa
  python tests/test_alpha_hh.py --quick                  # smoke test
  python tests/test_alpha_hh.py --alphas 1e-2,1,1e2 --ks 1,2
  mpiexec -n 3 python tests/test_alpha_hh.py              # paralelo (recomendado)

Saída: results_alpha/alpha_hh.json, results_alpha/alpha_hh_fig6.png.
"""

from __future__ import annotations
import time

import numpy as np

from common import (
    get_comm, apply_plot_style, rel_err, banner, make_meta,
    save_json, save_figure, COLORS, MARKERS, INK, DEFAULT_OUTDIR,
)
from params import CommonParams, parse_args, parse_float_list, parse_int_list

from mrcm.config import MRCMConfig
from mprm_core import (
    MPrMProjector, MACDownscaler, FineProjector,
    mprm_projection_step, mac_divergence, hh_build_w,
)


# ===========================================================================
# Computação (sem side effects de I/O — retorna um dict de resultados)
# ===========================================================================

def run_study(comm, alphas: list[float], ks: list[int],
              Ndom: int, nloc: int, basis_kind: str = "legendre_P",
              verbose: bool = True) -> dict:
    """Varre (k, α) e mede o erro da projeção MPrM vs a projeção fina.

    Retorna um dict `{"meta": {...}, "cases": [...]}` — um `case` por par
    (k, α), com `err_vs_fine`, `err_vs_exact`, `divmax`, `defect`, `wall_s`.
    """
    N = Ndom * nloc
    h = 1.0 / N
    dt = 1.0

    if verbose:
        banner(f"Estudo 1 (Fig. 6): HH vs α — {N}×{N}, "
              f"{Ndom}×{Ndom} subdomínios", comm)
        if comm.rank == 0:
            print(f"  α ∈ {np.array(alphas)}")
            print(f"  k ∈ {list(ks)}  (P_H,k = U_H,k; degreeU = degreeP = k)",
                  flush=True)

    w1, w2, u1, u2 = hh_build_w(N)

    # Referência do artigo: solução NÃO decomposta (projeção fina no mesmo w).
    fine = FineProjector(N, N, h, h, dt)
    uf1, uf2 = fine.step(w1, w2)
    err_fine_vs_exact = rel_err(uf1, uf2, u1, u2)
    if verbose and comm.rank == 0:
        print(f"  Referência fina: ‖u_fino − u_exato‖/‖u_exato‖ = "
              f"{err_fine_vs_exact:.4e}", flush=True)

    results = {"meta": make_meta(
        "hh_alpha (Fig. 6, Lages §3.1)", comm,
        N=N, Ndom=Ndom, nloc=nloc, dt=dt,
        alphas=[float(a) for a in alphas], ks=list(ks),
        basis_kind=basis_kind,
        reference="undecomposed fine MAC projection (artigo Fig. 6)",
        err_fine_vs_exact=err_fine_vs_exact,
    ), "cases": []}

    for k in ks:
        for a in alphas:
            case = _run_one_case(comm, w1, w2, uf1, uf2, u1, u2,
                                 Ndom, nloc, k, a, dt, h, basis_kind)
            results["cases"].append(case)
            if verbose and comm.rank == 0:
                print(f"  k={k}  α={a:9.1e}:  "
                      f"err_fino={case['err_vs_fine']:.4e}"
                      f"  err_exato={case['err_vs_exact']:.4e}"
                      f"  div={case['divmax']:.1e}  ({case['wall_s']:.1f}s)",
                      flush=True)

    return results


def _run_one_case(comm, w1, w2, uf1, uf2, u1, u2,
                  Ndom, nloc, k, alpha, dt, h, basis_kind) -> dict:
    """Um único ponto (k, α) da varredura: monta o offline, projeta, mede."""
    cfg = MRCMConfig(Ndomx=Ndom, Ndomy=Ndom, nlocx=nloc, nlocy=nloc,
                     degreeU=k, degreeP=k, alpha=float(alpha),
                     basis_kind=basis_kind,
                     mixed_family="BDM", mixed_order=0,
                     executor_kind="sequential")
    t0 = time.perf_counter()
    projector = MPrMProjector(cfg, dt, comm=comm)
    downscaler = MACDownscaler(cfg, dt)
    uh1, uh2 = mprm_projection_step(w1, w2, projector, downscaler)
    wall = time.perf_counter() - t0

    return {
        "k": k, "alpha": float(alpha),
        "err_vs_fine": rel_err(uh1, uh2, uf1, uf2),
        "err_vs_exact": rel_err(uh1, uh2, u1, u2),
        "divmax": float(np.abs(mac_divergence(uh1, uh2, h, h)).max()),
        "defect": float(downscaler.last_defect),
        "wall_s": wall,
    }


# ===========================================================================
# Plotagem
# ===========================================================================

def plot_results(results: dict, outdir: str) -> str:
    """Erro vs α, uma curva por k — réplica da Fig. 6. Retorna o path salvo."""
    plt = apply_plot_style()
    alphas = results["meta"]["alphas"]
    ks = results["meta"]["ks"]

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    # deslocamentos verticais dos rótulos diretos (curvas k=2..4 convergem
    # no limite MHM — sem o dodge os rótulos colidem)
    dodge = {1: 0, 2: 10, 3: 0, 4: -11}
    for k in ks:
        errs = [c["err_vs_fine"] for c in results["cases"] if c["k"] == k]
        ax.loglog(alphas, errs, "-", marker=MARKERS[k], markersize=7,
                  linewidth=2, color=COLORS[k],
                  label=f"$(P_{{H,{k}}},\\,U_{{H,{k}}})$")
        ax.annotate(f"k={k}", xy=(alphas[-1], errs[-1]),
                    xytext=(7, dodge.get(k, 0)), textcoords="offset points",
                    va="center", fontsize=10, color=INK)
    ax.set_xlabel(r"$\alpha$ (parâmetro de Robin)")
    ax.set_ylabel(r"$\|u_h - u_{\rm fino}\|_2\,/\,\|u_{\rm fino}\|_2$")
    ax.set_title("Decomposição HH modificada: erro vs α — réplica da Fig. 6 "
                 f"(malha {results['meta']['N']}², "
                 f"{results['meta']['Ndom']}×{results['meta']['Ndom']} subdomínios)")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    return save_figure(fig, outdir, "alpha_hh_fig6.png")


# ===========================================================================
# CLI
# ===========================================================================

def main(argv=None):
    defaults = CommonParams(Ndom=3, nloc=30, outdir=DEFAULT_OUTDIR)
    args = parse_args(
        "Estudo 1: decomposição HH modificada vs α (réplica da Fig. 6, "
        "Lages §3.1)",
        defaults=defaults,
        extra_args=lambda p: (
            p.add_argument("--alphas", default="logspace:-4,4,9",
                          help="lista 'a,b,c' ou 'logspace:ini,fim,n'"),
            p.add_argument("--ks", default="1,2,3,4",
                          help="graus de interface P_H,k=U_H,k; lista ou 'range:a,b'"),
        ),
        argv=argv,
    )

    if args.quick:
        alphas = [1e-2, 1.0, 1e2]
        ks = [1]
        Ndom, nloc = 3, 16
    else:
        alphas = parse_float_list(args.alphas)
        ks = parse_int_list(args.ks)
        Ndom, nloc = args.Ndom, args.nloc

    comm = get_comm()
    t0 = time.perf_counter()
    results = run_study(comm, alphas, ks, Ndom, nloc,
                        basis_kind=args.basis_kind, verbose=args.verbose)

    if comm.rank == 0:
        save_json(results, args.outdir, "alpha_hh.json")
        plot_results(results, args.outdir)
        print(f"\nEstudo concluído em {time.perf_counter()-t0:.0f}s "
              f"({comm.size} rank(s)).")
    return results


if __name__ == "__main__":
    main()
