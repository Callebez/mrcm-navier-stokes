#!/usr/bin/env python3
"""
tests/test_alpha_hh_bdm2.py — Estudo 1 (BDM₂): decomposição HH modificada vs α.

Versão com BDM₂ (mixed_order=1): velocidade quadrática (BDM₂) e pressão
linear (DG₁) nos subdomínios. Malhas locais ~3× maiores que BDM₁ —
o offline é significativamente mais caro. Para nloc=30, usar
mpiexec -n 4.

Réplica da **Fig. 6** do artigo do MPrM (Lages, §3.1) com BDM₂ em vez de
BDM₁. Resultados de nloc=8 mostram que BDM₂ reduz o erro no limite MHM
em até 22× para k=3 comparado ao BDM₁.

Uso:
  python tests/test_alpha_hh_bdm2.py --nloc 16           # rápido (~5 min)
  python tests/test_alpha_hh_bdm2.py --nloc 30           # completo (~30+ min)
  python tests/test_alpha_hh_bdm2.py --quick              # smoke test
  mpiexec -n 4 python tests/test_alpha_hh_bdm2.py --nloc 30  # paralelo

Saída: results_alpha/alpha_hh_bdm2.json, results_alpha/alpha_hh_bdm2_fig6.png.
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
        "hh_alpha_BDM2 (Fig. 6, Lages §3.1, BDM₂)", comm,
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
                     mixed_family="BDM", mixed_order=1,
                     local_solver_mode="direct",
                     executor_kind="mpi")
    from mprm.benchmarks.hh import (hh_ortho_defect, hh_assemble_psi,
                                    hh_ortho_skeleton, _l2_mac)

    t0 = time.perf_counter()
    projector = MPrMProjector(cfg, dt, comm=comm)
    downscaler = MACDownscaler(cfg, dt)
    downscaler.keep_potential = True
    uh1, uh2 = mprm_projection_step(w1, w2, projector, downscaler)
    wall = time.perf_counter() - t0

    # eq:ErroOrtogonalidade — a métrica que não depende de solução de
    # referência. O artigo compara contra a discretização não-decomposta
    # (err_vs_fine); esta mede a inconsistência interna da solução.
    ortho_abs, ortho_cos = hh_ortho_defect(uh1, uh2, w1 - uh1, w2 - uh2, h)
    psi = hh_assemble_psi(downscaler, projector, Ndom, nloc)
    ortho_skel = hh_ortho_skeleton(uh1, uh2, psi, Ndom, nloc, h, dt)
    scale = max(abs(ortho_abs), abs(ortho_skel))

    return {
        "k": k, "alpha": float(alpha),
        "err_vs_fine": rel_err(uh1, uh2, uf1, uf2),
        "err_vs_exact": rel_err(uh1, uh2, u1, u2),
        "leak_ratio": _l2_mac(uh1 - u1, uh2 - u2, h) / _l2_mac(w1, w2, h),
        "ortho_cos": None if ortho_cos is None else abs(ortho_cos),
        "ortho_skel": ortho_skel,
        "ortho_identity_rel": (None if scale == 0.0
                               else abs(ortho_abs - ortho_skel) / scale),
        "divmax": float(np.abs(mac_divergence(uh1, uh2, h, h)).max()),
        "defect": float(downscaler.last_defect),
        "wall_s": wall,
    }


# ===========================================================================
# As afirmações do artigo sobre a Fig. 6, como testes
# ===========================================================================
# Lages, §3.1, texto da Fig. 6 — quatro afirmações verificáveis:
#   (a) "solution improves when interface spaces are enriched"
#   (b) "a minimum error can occur for some choice of α around unity"
#   (c) "Errors are better for large values of α"
#   (d) "(P_H,4,U_H,4) ... the choice of α will have less impact"
# Medidas em malha 90², 3×3 subdomínios — o arranjo da figura.

def _fig6_grid(alphas=(1e-2, 1e-1, 1.0, 1e2, 1e4), ks=(1, 4),
               Ndom=3, nloc=30):
    """Subconjunto do grid da Fig. 6, barato o bastante para pytest."""
    from mprm.benchmarks.convergence import _run_one_hh_level
    comm = get_comm()
    out = []
    for k in ks:
        for a in alphas:
            c = _run_one_hh_level(comm, Ndom, nloc, k, k, a,
                                  "legendre_P", 1.0, mode="mixed")
            c["k"], c["alpha"] = k, a
            out.append(c)
    return out


def test_fig6a_enriquecer_a_interface_melhora():
    """(a) ✓ — a afirmação do artigo que se confirma sem ressalva."""
    g = _fig6_grid(alphas=(1e-1,), ks=(1, 2, 3, 4))
    errs = [c["err_vs_fine"] for c in sorted(g, key=lambda c: c["k"])]
    assert all(b < a for a, b in zip(errs, errs[1:])), errs


def test_fig6c_erro_NAO_melhora_com_alpha_grande_em_k_alto():
    """(c) ✗ — refuta 'errors are better for large values of α'.

    Vale para k=1, inverte-se para k=4: em α=1e4 o erro é ordens de
    grandeza PIOR que em α pequeno. Achado contra o artigo, registrado.
    """
    g = _fig6_grid(alphas=(1e-2, 1e4), ks=(4,))
    pequeno = [c for c in g if c["alpha"] == 1e-2][0]["err_vs_fine"]
    grande = [c for c in g if c["alpha"] == 1e4][0]["err_vs_fine"]
    assert grande > 10.0 * pequeno, (
        f"α grande deveria ser MUITO pior em k=4: "
        f"α=1e-2 → {pequeno:.3e}, α=1e4 → {grande:.3e}")


def test_fig6d_k4_e_a_mais_sensivel_a_alpha_nao_a_menos():
    """(d) ✗ — inverte 'α will have less impact' para (P_H,4,U_H,4).

    Mede por err_vs_exact, e NÃO por err_vs_fine como o artigo: a referência
    fina tem erro de discretização 9.34e-05 em N=90, e 3 dos 9 pontos da
    curva k=4 caem abaixo disso — ali err_vs_fine mede a diferença entre dois
    erros e cava um mínimo artificialmente fundo (569× em vez de 197×).
    """
    g = _fig6_grid(alphas=(1e-2, 1e-1, 1.0, 1e2, 1e4), ks=(1, 4))
    spread = {}
    for k in (1, 4):
        v = [c["err_vs_exact"] for c in g if c["k"] == k]
        spread[k] = max(v) / min(v)
    assert spread[4] > 10.0 * spread[1], (
        f"k=4 deveria ser a MAIS sensível: k=1 varia {spread[1]:.1f}×, "
        f"k=4 varia {spread[4]:.1f}×")


def test_referencia_fina_e_resolvida_por_LU_direta():
    """A precisão algébrica da referência é de máquina — não é o limitante.

    FineProjector usa splu (LU esparsa direta), sem tolerância. O limitante
    é o erro de DISCRETIZAÇÃO da malha fina, medido aqui: é ele que põe piso
    em err_vs_fine, e é por isso que o teste acima usa err_vs_exact.
    """
    from mprm.benchmarks.hh import hh_build_w_mode, _l2_mac
    from mprm.flow import FineProjector

    N = 90
    h = 1.0 / N
    w1, w2, u1, u2 = hh_build_w_mode(N, mode="mixed")
    uf1, uf2 = FineProjector(N, N, h, h, 1.0).step(w1, w2)

    piso = _l2_mac(uf1 - u1, uf2 - u2, h) / _l2_mac(u1, u2, h)
    assert 5e-5 < piso < 2e-4, f"piso da referência fina mudou: {piso:.3e}"
    assert np.abs(mac_divergence(uf1, uf2, h, h)).max() < 1e-8


# ===========================================================================
# Plotagem
# ===========================================================================

def plot_results(results: dict, outdir: str) -> str:
    """Erro vs α, uma curva por k — réplica da Fig. 6. Retorna o path salvo."""
    plt = apply_plot_style()
    alphas = results["meta"]["alphas"]
    ks = results["meta"]["ks"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    # deslocamentos verticais dos rótulos diretos (curvas k=2..4 convergem
    # no limite MHM — sem o dodge os rótulos colidem)
    dodge = {1: 0, 2: 10, 3: 0, 4: -11}

    for ax, key, ylbl, titulo in (
            (axes[0], "err_vs_fine",
             r"$\|u_h - u_{\rm fino}\|_2\,/\,\|u_{\rm fino}\|_2$",
             "réplica da Fig. 6 — erro vs a discretização não-decomposta"),
            (axes[1], "ortho_cos",
             r"$|\cos\theta|$ — defeito de ortogonalidade",
             "eq:ErroOrtogonalidade — sem solução de referência")):
        for k in ks:
            errs = [c[key] for c in results["cases"] if c["k"] == k]
            if any(e is None for e in errs):
                continue
            ax.loglog(alphas, errs, "-", marker=MARKERS[k], markersize=7,
                      linewidth=2, color=COLORS[k],
                      label=f"$(P_{{H,{k}}},\\,U_{{H,{k}}})$")
            ax.annotate(f"k={k}", xy=(alphas[-1], errs[-1]),
                        xytext=(7, dodge.get(k, 0)),
                        textcoords="offset points",
                        va="center", fontsize=10, color=INK)
            i = int(np.argmin(errs))
            ax.plot(alphas[i], errs[i], "*", color=COLORS[k], markersize=15,
                    markeredgecolor=INK, markeredgewidth=0.6, zorder=6)
        ax.set_xlabel(r"$\alpha$ (parâmetro de Robin)")
        ax.set_ylabel(ylbl)
        ax.set_title(titulo, fontsize=10)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False, loc="best", fontsize=8)

    fig.suptitle("Decomposição HH modificada vs α — Lages Fig. 6 (BDM₂) "
                 f"(malha {results['meta']['N']}², "
                 f"{results['meta']['Ndom']}×{results['meta']['Ndom']} "
                 "subdomínios) · ★ = mínimo")
    fig.tight_layout()
    return save_figure(fig, outdir, "alpha_hh_bdm2_fig6.png")


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
        save_json(results, args.outdir, "alpha_hh_bdm2.json")
        plot_results(results, args.outdir)
        print(f"\nEstudo concluído em {time.perf_counter()-t0:.0f}s "
              f"({comm.size} rank(s)).")
    return results


if __name__ == "__main__":
    main()
