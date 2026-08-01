#!/usr/bin/env python3
"""
tests/test_alpha_hh_rt0.py — Réplica da Fig. 6 com RT₀ em malha Cartesiana.

Versão modificada de test_alpha_hh.py que substitui BDM₁ em triângulos por
RTCF(1) (RT₀) em quadriláteros — equivalente a volumes finitos (FV), que é a
discretização do artigo de Lages.

Diferenças vs test_alpha_hh.py:
  - mixed_family="RT", mixed_order=0 → RTCF em quads (1 fluxo normal/face = FV)
  - make_local_mesh com quadrilateral=True (malha Cartesiana, sem triângulos)
  - DG vira DQ automaticamente em quads (Firedrake warning cosmético)

ADVERTÊNCIAS:
  1. RTCF em quads apresenta problemas numéricos em α extremo (α ≥ 1e2) e
     k ≥ 2 — erros de 2.5e+15 indicam falha na fatoração LU local.
  2. O número de condicionamento das matrizes locais é ~10× pior que BDM.
  3. Este caminho NÃO é testado pela suíte de regressão do MRCM — é
     exploratório. Os resultados para α ∈ [1e-2, 1e1], k=1 são confiáveis;
     fora dessa faixa, investigar antes de usar.

Uso:
  python tests/test_alpha_hh_rt0.py --quick
  python tests/test_alpha_hh_rt0.py --alphas 1e-2,1e-1,1,1e1 --ks 1
"""

from __future__ import annotations
import time, numpy as np, sys, os

# ══════════════════════════════════════════════════════════════════════════════
# Monkey-patch: malha quadrilateral + RTCF no MRCM
# ══════════════════════════════════════════════════════════════════════════════

def _patch_mrcm_for_quads():
    """Substitui make_local_mesh e make_mixed_space no MRCM para usar
    quadriláteros + RTCF (FV-equivalente)."""
    import mrcm.local_form as lf
    import mrcm.local_solver as ls

    def _quad_make_local_mesh(x0, y0, Hx, Hy, nx, ny, comm=None):
        from firedrake import RectangleMesh, Mesh, Function, COMM_SELF
        if comm is None:
            comm = COMM_SELF
        key = (nx, ny, round(Hx, 12), round(Hy, 12), id(comm), 'quad')
        entry = lf._TOPO_CACHE.get(key)
        if entry is None:
            base = RectangleMesh(nx, ny, Hx, Hy, quadrilateral=True, comm=comm)
            Vc = base.coordinates.function_space()
            ref_coords = base.coordinates.dat.data_ro.copy()
            lf._TOPO_CACHE[key] = entry = (Vc, ref_coords)
        Vc, ref_coords = entry
        coords = Function(Vc)
        coords.dat.data[:, 0] = ref_coords[:, 0] + x0
        coords.dat.data[:, 1] = ref_coords[:, 1] + y0
        return Mesh(coords)

    def _quad_make_mixed_space(mesh, family="BDM", order=0):
        from firedrake import FunctionSpace
        is_quad = mesh.ufl_cell().cellname == 'quadrilateral'
        if family == "BDM":
            fam = "BDMCF" if is_quad else "BDM"
        elif family == "RT":
            # RT em quads → RTCF = 1 fluxo normal por face = FV exato
            fam = "RTCF" if is_quad else "RT"
        else:
            raise ValueError(f"Unknown mixed family={family!r}")
        V = FunctionSpace(mesh, fam, order + 1)
        Q = FunctionSpace(mesh, "DG", order)  # DG em quads → DQ (warning cosmético)
        return V, Q, V * Q

    lf.make_local_mesh = _quad_make_local_mesh
    lf.make_mixed_space = _quad_make_mixed_space
    ls.make_local_mesh = _quad_make_local_mesh
    ls.make_mixed_space = _quad_make_mixed_space


_patch_mrcm_for_quads()

# ══════════════════════════════════════════════════════════════════════════════
# Dependências (pós-patch)
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# Computação
# ══════════════════════════════════════════════════════════════════════════════

def run_study(comm, alphas: list[float], ks: list[int],
              Ndom: int, nloc: int, basis_kind: str = "legendre_P",
              verbose: bool = True) -> dict:
    """Varre (k, α) com RT₀ em quads e mede o erro vs solução analítica."""
    N = Ndom * nloc
    h = 1.0 / N
    dt = 1.0

    if verbose:
        banner(f"Estudo Fig. 6 — RT₀ (RTCF) em quads — {N}×{N}, "
               f"{Ndom}×{Ndom} subdomínios", comm)
        if comm.rank == 0:
            print(f"  α ∈ {np.array(alphas)}")
            print(f"  k ∈ {list(ks)}  (P_H,k = U_H,k)")
            print(f"  mixed_family=RT, mixed_order=0 → RTCF(1) = FV", flush=True)

    w1, w2, u1, u2 = hh_build_w(N)

    # Referência fina (Poisson MAC global — igual ao artigo)
    fine = FineProjector(N, N, h, h, dt)
    uf1, uf2 = fine.step(w1, w2)
    err_fine_vs_exact = rel_err(uf1, uf2, u1, u2)
    if verbose and comm.rank == 0:
        print(f"  Referência fina: ‖u_fino − u_exato‖/‖u_exato‖ = "
              f"{err_fine_vs_exact:.4e}", flush=True)

    results = {"meta": make_meta(
        "hh_alpha_RT0_quad (Fig. 6, RTCF=RT₀=FV)", comm,
        N=N, Ndom=Ndom, nloc=nloc, dt=dt,
        alphas=[float(a) for a in alphas], ks=list(ks),
        basis_kind=basis_kind,
        reference="solucao analitica (erro absoluto)",
        err_fine_vs_exact=err_fine_vs_exact,
        mixed_family="RT (→RTCF em quads)", mixed_order=0,
        mesh_type="quadrilateral",
    ), "cases": []}

    for k in ks:
        for a in alphas:
            case = _run_one_case(comm, w1, w2, uf1, uf2, u1, u2,
                                 Ndom, nloc, k, a, dt, h, basis_kind)
            results["cases"].append(case)
            if verbose and comm.rank == 0:
                print(f"  k={k}  α={a:9.1e}:  "
                      f"err_exato={case['err_vs_exact']:.4e}"
                      f"  err_fino={case['err_vs_fine']:.4e}"
                      f"  div={case['divmax']:.1e}  ({case['wall_s']:.1f}s)",
                      flush=True)

    return results


def _run_one_case(comm, w1, w2, uf1, uf2, u1, u2,
                  Ndom, nloc, k, alpha, dt, h, basis_kind) -> dict:
    cfg = MRCMConfig(Ndomx=Ndom, Ndomy=Ndom, nlocx=nloc, nlocy=nloc,
                     degreeU=k, degreeP=k, alpha=float(alpha),
                     basis_kind=basis_kind,
                     mixed_family="RT", mixed_order=0,   # RT → RTCF em quads
                     executor_kind="sequential")
    from mprm.benchmarks.hh import (hh_ortho_defect, hh_assemble_psi,
                                    hh_ortho_skeleton, _l2_mac)

    t0 = time.perf_counter()
    projector = MPrMProjector(cfg, dt, comm=comm)
    downscaler = MACDownscaler(cfg, dt)
    downscaler.keep_potential = True
    uh1, uh2 = mprm_projection_step(w1, w2, projector, downscaler)
    wall = time.perf_counter() - t0

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


# ══════════════════════════════════════════════════════════════════════════════
# Plotagem
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(results: dict, outdir: str) -> str:
    plt = apply_plot_style()
    alphas = results["meta"]["alphas"]
    ks = results["meta"]["ks"]
    N = results["meta"]["N"]
    Ndom = results["meta"]["Ndom"]

    # Computa ||u_ex|| para erro absoluto
    w1, w2, u1_ex, u2_ex = hh_build_w(N)
    NORM_U_EX = np.sqrt(np.sum(u1_ex**2) + np.sum(u2_ex**2))

    fig, ax = plt.subplots(figsize=(10, 7))

    dodge = {1: 14, 2: -8, 3: 12, 4: 14}

    for k in ks:
        errs = np.array([
            [c for c in results["cases"]
             if c["k"] == k and abs(c["alpha"] - a) < 1e-9][0]["err_vs_exact"]
            * NORM_U_EX
            for a in alphas
        ])
        # Filtra valores NaN/Inf (problemas numéricos em α extremo)
        valid = np.isfinite(errs) & (errs > 0)
        if not valid.any():
            continue
        ax.loglog(np.array(alphas)[valid], errs[valid], '-',
                  marker=MARKERS[k], markersize=8,
                  linewidth=2, color=COLORS[k],
                  label=f'$(P_{{H,{k}}},\\,U_{{H,{k}}})$')
        ax.annotate(f'$k={k}$',
                    xy=(np.array(alphas)[valid][-1], errs[valid][-1]),
                    xytext=(7, dodge.get(k, 0)),
                    textcoords='offset points', va='center', fontsize=10,
                    color=INK)
        i = np.argmin(errs[valid])
        ax.plot(np.array(alphas)[valid][i], errs[valid][i], '*',
                color=COLORS[k], markersize=16,
                markeredgecolor=INK, markeredgewidth=0.6, zorder=6)

    ax.set_xlabel(r'$\alpha$ (parâmetro de Robin)')
    ax.set_ylabel(r'$\|u_h - u_{\rm exato}\|_2$  — erro absoluto')
    ax.set_title('Decomposição HH modificada vs α — RT₀ (RTCF) em quads = FV\n'
                 f'(malha {N}$^2$, {Ndom}$\\times${Ndom} subdomínios, '
                 f'$n_{{\\rm loc}}$={results["meta"]["nloc"]})  · ★ = mínimo',
                 fontsize=11)
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(frameon=False, loc='lower left', fontsize=9)
    fig.tight_layout()

    return save_figure(fig, outdir, 'fig6_rt0_quad.png')


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main(argv=None):
    defaults = CommonParams(Ndom=3, nloc=30, outdir=DEFAULT_OUTDIR)
    args = parse_args(
        "Estudo Fig. 6 com RT₀ (RTCF) em quads — FV-equivalente",
        defaults=defaults,
        extra_args=lambda p: (
            p.add_argument("--alphas", default="logspace:-4,4,9",
                          help="lista ou 'logspace:ini,fim,n'"),
            p.add_argument("--ks", default="1,2,3,4",
                          help="graus de interface"),
        ),
        argv=argv,
    )

    if args.quick:
        alphas = [1e-2, 1e-1, 1.0, 1e1]
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
        save_json(results, args.outdir, "alpha_hh_rt0.json")
        plot_results(results, args.outdir)
        print(f"\nEstudo concluído em {time.perf_counter()-t0:.0f}s "
              f"({comm.size} rank(s)).")
    return results


if __name__ == "__main__":
    main()
