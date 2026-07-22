#!/usr/bin/env python3
"""
tests/test_convergence_hh.py — Taxa de convergência Hh do MPrM.

Mede a taxa de convergência **clássica de FEM interior** do MPrM: com o
grau dos espaços de interface (`degreeU`, `degreeP`) e α **fixos**,
refina H e h **proporcionalmente** — `nloc` (a razão H/h) fica fixo,
`Ndom` cresce — e ajusta a ordem observada por regressão log-log de
erro vs H (equivalente a vs h em modo Hh, já que h = H/nloc).

Isso replica o modo "Hh" do estudo de convergência nativo do `mrcm`
(`python -m mrcm.convergence --mode Hh`; ver
`Vault/concepts/Convergence-Rates.md`): "Hh-mode mede a taxa do FEM
interior (≈1.0 com RT₀/BDM₁), não a taxa multiescala O(H^{ℓ+2})" — essa é
a taxa de refinamento clássico, distinta da taxa multiescala medida pelo
modo H (subdomínios menores, h fixo) ou da saturação do modo h (só a
malha fina refina). Essa é a taxa esperada para a reconstrução BDM₁-DG₀
CRUA do MRCM (sem o passo de downscaling).

**Duas referências, dois erros relativos** (mesmo par usado em
`tests/test_alpha_hh.py`, seguindo a convenção do próprio `mrcm` —
memória de projeto "Fine-reference error convention": medir contra a
referência fina de MESMA malha, não só contra a analítica, para separar
as fontes de erro):

* `err_vs_exact` — erro vs a solução **analítica** de Olshanskii (campo
  fechado; convenção do artigo do MPrM, Lages §3.1, Tabela 1). Inclui
  TANTO o erro de decomposição multiescala QUANTO o erro intrínseco do
  esquema MAC/downscaling naquela resolução N.
* `err_vs_fine` — erro vs a projeção MAC **não decomposta** na MESMA
  resolução N (`FineProjector`, o "fine mesh solver" do artigo, §2.7).
  Isola o erro de **decomposição** (MSBF, acoplamento Robin, interface):
  o esquema MAC de referência tem o mesmo N, logo o mesmo erro de
  discretização "de fundo", que se cancela na diferença.

Cada um tem sua própria taxa ajustada (`rate_exact`, `rate_fine`); o
`--metric` da CLI escolhe qual dirige a validação PASS/FAIL padrão
(`--expected-rate`/`--tolerance`).

**ACHADO (medido, não teórico a priori)**: com (P_{H,1}, U_{H,1}), α=1,
n_loc=16 fixo, Ndom ∈ {2,4,8,16}, a taxa observada é **O(H^1.99)**, quase
exatamente **2**, não 1 — R²=1,0000 nos 4 níveis, taxa por par consecutivo
1,98/2,00/2,00; erro cai de 5,04e-1 para 8,00e-3 (fator ~63 ≈ 2⁶, batendo
com O(H²) para H caindo por 8×). Interpretação: diferente do modo Hh
nativo do `mrcm` (que mede a reconstrução BDM₁-DG₀ CRUA), aqui o campo
final `u_h` passa pelo **downscaling** (`MACDownscaler`, Eqs. 21-23 do
artigo — um solve de Poisson FV de 5 pontos na malha MAC usando o fluxo
médio de interface como condição de contorno). Esse passo extra de
pós-processamento local parece **recuperar superconvergência** — o solve
FV é O(h²) para campos suaves, e aparentemente domina sobre o piso O(H) do
MRCM cru. **Ressalva**: medido para um campo analítico suave (C^∞); não é
uma garantia teórica geral — um campo com fonte de baixa regularidade
poderia reexpor o piso O(H) do MRCM. Ver docs/06-convergencia-hh.md para
o experimento completo. A extração/validação da taxa reusa
`mrcm.convergence.compute_convergence_rate`/`validate_rate`, mantendo a
mesma metodologia (regressão log-log) do projeto mrcm.

Uso:
  python tests/test_convergence_hh.py                      # config padrão
  python tests/test_convergence_hh.py --quick                # smoke test
  python tests/test_convergence_hh.py --Ndoms 2,4,8,16 --nloc 12
  python tests/test_convergence_hh.py --expected-rate 1.0 --tolerance 0.25
  python tests/test_convergence_hh.py --metric fine           # valida a taxa isolada de decomposição
  mpiexec -n 3 python tests/test_convergence_hh.py           # paralelo

Saída: results_alpha/convergence_hh.json, results_alpha/convergence_hh.png.
"""

from __future__ import annotations
import time

import numpy as np

from common import (
    get_comm, apply_plot_style, rel_err, banner, make_meta,
    save_json, save_figure, COLORS, INK, DEFAULT_OUTDIR,
)
from params import CommonParams, parse_args, parse_int_list

from mrcm.config import MRCMConfig
from mrcm.convergence import compute_convergence_rate, validate_rate
from mprm_core import (
    MPrMProjector, MACDownscaler, FineProjector, mprm_projection_step,
    mac_divergence, hh_build_w,
)

DEFAULT_NDOMS = "2,4,8"
DEFAULT_NLOC = 16
# Taxa esperada da SAÍDA do MPrM (após downscaling), medida empiricamente
# em (P_H,1, U_H,1), α=1: O(H^1.99), R²=1.0000 em 4 níveis (ver docstring
# do módulo). NÃO é a taxa O(H) da reconstrução BDM1-DG0 crua do MRCM —
# essa é medida por `mrcm.convergence --mode Hh` diretamente, sem
# downscaling. Regressão/validação: `mrcm.convergence.compute_convergence_rate`.
DEFAULT_EXPECTED_RATE = 2.0
DEFAULT_TOLERANCE = 0.3


# ===========================================================================
# Computação (sem side effects de I/O — retorna um dict de resultados)
# ===========================================================================

def run_study(comm, Ndoms: list[int], nloc: int, degreeU: int, degreeP: int,
              alpha: float, basis_kind: str = "legendre_P",
              verbose: bool = True) -> dict:
    """Refina Hh (nloc fixo ⇒ H/h fixo; Ndom cresce) e mede, em cada nível,
    o erro relativo vs a solução analítica de Olshanskii E vs a projeção
    fina (não decomposta) na mesma resolução N.

    Retorna um dict `{"meta": {...}, "cases": [...], "rate_exact": {...},
    "rate_fine": {...}}` — um `case` por Ndom (`err_vs_exact`,
    `err_vs_fine`, `err_fine_vs_exact`, `H`, `h`, `N`, `divmax`, `defect`,
    `wall_s`) e os blocos de regressão log-log de cada métrica (ausentes
    se só houver um nível).
    """
    if verbose:
        banner(f"Convergência Hh: nloc={nloc} fixo (H/h={nloc}), "
              f"Ndom ∈ {list(Ndoms)}, "
              f"(degreeU,degreeP)=({degreeU},{degreeP}), α={alpha}", comm)

    dt = 1.0
    results = {"meta": make_meta(
        "convergence_hh (classical FEM interior rate, cf. mrcm.convergence "
        "mode='Hh')", comm,
        nloc=nloc, degreeU=degreeU, degreeP=degreeP, alpha=alpha,
        basis_kind=basis_kind, Ndoms=list(Ndoms), dt=dt,
        reference_exact="analytic Olshanskii field (artigo §3.1, Tabela 1)",
        reference_fine="undecomposed MAC projection, same N per level "
                        "(Fine-reference error convention)",
    ), "cases": []}

    for Ndom in Ndoms:
        case = _run_one_level(comm, Ndom, nloc, degreeU, degreeP, alpha,
                              basis_kind, dt)
        results["cases"].append(case)
        if verbose and comm.rank == 0:
            print(f"  Ndom={Ndom:3d}  H={case['H']:.4f}  h={case['h']:.5f}  "
                  f"N={case['N']:4d}:  err_relativo(exato)={case['err_vs_exact']:.4e}  "
                  f"err_relativo(fino)={case['err_vs_fine']:.4e}  "
                  f"div={case['divmax']:.1e}  ({case['wall_s']:.1f}s)",
                  flush=True)

    if comm.rank == 0 and len(results["cases"]) >= 2:
        if verbose:
            print()
        results["rate_exact"] = _fit_rate(
            results["cases"], "err_vs_exact",
            "vs solução exata (inclui erro do esquema MAC)", verbose)
        results["rate_fine"] = _fit_rate(
            results["cases"], "err_vs_fine",
            "vs referência fina (isola o erro de decomposição)", verbose)

    return results


def _run_one_level(comm, Ndom, nloc, degreeU, degreeP, alpha, basis_kind,
                   dt) -> dict:
    """Um único nível de refinamento Hh: monta o offline, projeta, mede
    contra as duas referências (analítica e fina, ambas na mesma N)."""
    N = Ndom * nloc
    h = 1.0 / N
    H = 1.0 / Ndom  # Lx = 1.0

    w1, w2, u1, u2 = hh_build_w(N)

    # Referência fina: mesma malha MAC N×N, sem decomposição de domínio —
    # isola o erro de decomposição multiescala (Fine-reference convention).
    fine = FineProjector(N, N, h, h, dt)
    uf1, uf2 = fine.step(w1, w2)

    cfg = MRCMConfig(Ndomx=Ndom, Ndomy=Ndom, nlocx=nloc, nlocy=nloc,
                     degreeU=degreeU, degreeP=degreeP, alpha=float(alpha),
                     basis_kind=basis_kind,
                     mixed_family="BDM", mixed_order=0,
                     executor_kind="sequential")
    t0 = time.perf_counter()
    projector = MPrMProjector(cfg, dt, comm=comm)
    downscaler = MACDownscaler(cfg, dt)
    uh1, uh2 = mprm_projection_step(w1, w2, projector, downscaler)
    wall = time.perf_counter() - t0

    return {
        "Ndom": Ndom, "H": float(H), "h": float(h), "N": N,
        "err_vs_exact": rel_err(uh1, uh2, u1, u2),
        "err_vs_fine": rel_err(uh1, uh2, uf1, uf2),
        "err_fine_vs_exact": rel_err(uf1, uf2, u1, u2),
        "divmax": float(np.abs(mac_divergence(uh1, uh2, h, h)).max()),
        "defect": float(downscaler.last_defect),
        "wall_s": wall,
    }


def _fit_rate(cases: list[dict], metric_key: str, label: str,
             verbose: bool = True) -> dict:
    """Regressão log-log `metric_key` vs H (ordenado por H decrescente —
    requisito de `compute_convergence_rate`). Reusa a extração de taxa do
    `mrcm` — mesma metodologia usada pelos estudos T1–T7."""
    ordered = sorted(cases, key=lambda c: -c["H"])
    Hs = [c["H"] for c in ordered]
    errs = [c[metric_key] for c in ordered]
    rate = compute_convergence_rate(Hs, errs)
    if verbose:
        print(f"  Taxa Hh {label}: {rate['rate']:.3f}  "
             f"(R² = {rate['r_squared']:.4f})  "
             f"por par: {[f'{s:.2f}' for s in rate['per_step']]}",
             flush=True)
    return rate


# ===========================================================================
# Plotagem
# ===========================================================================

def plot_results(results: dict, outdir: str) -> str:
    """Erro relativo vs H em log-log — duas séries (vs exato, vs fino),
    cada uma com sua reta ajustada, mais uma referência O(H)."""
    plt = apply_plot_style()
    m = results["meta"]
    cases = sorted(results["cases"], key=lambda c: -c["H"])
    Hs = np.array([c["H"] for c in cases])
    err_exact = np.array([c["err_vs_exact"] for c in cases])
    err_fine = np.array([c["err_vs_fine"] for c in cases])

    series = [
        ("err_vs_exact", err_exact, "vs solução exata", COLORS[1], "o"),
        ("err_vs_fine", err_fine, "vs referência fina (mesma N)", COLORS[2], "s"),
    ]

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    for key, errs, label, color, marker in series:
        ax.loglog(Hs, errs, "-", marker=marker, markersize=7, linewidth=2,
                  color=color, label=f"MPrM {label}")
        rate = results.get(f"rate_{'exact' if key.endswith('exact') else 'fine'}")
        if rate is not None:
            fit = np.exp(rate["intercept"]) * Hs ** rate["rate"]
            ax.loglog(Hs, fit, ":", color=color, alpha=0.6, linewidth=1.5,
                      label=f"  ajuste: $O(H^{{{rate['rate']:.2f}}})$, "
                            f"$R^2={rate['r_squared']:.4f}$")

    ref = min(err_exact[0], err_fine[0]) * (Hs / Hs[0]) ** 1.0
    ax.loglog(Hs, ref, "--", color=INK, alpha=0.5,
              label=r"$O(H)$ — piso MRCM cru (sem downscaling)")

    for Ndom, H, err in zip((c["Ndom"] for c in cases), Hs, err_exact):
        ax.annotate(f"Ndom={Ndom}", xy=(H, err), xytext=(6, 6),
                    textcoords="offset points", fontsize=9, color=INK)

    ax.set_xlabel(r"$H$ (tamanho do subdomínio; $h = H/n_{\rm loc}$, "
                 f"$n_{{\\rm loc}}={m['nloc']}$ fixo)")
    ax.set_ylabel(r"erro relativo $\|u_h - u_{\rm ref}\|_2\,/\,\|u_{\rm ref}\|_2$")
    ax.set_title("Convergência Hh do MPrM — "
                 f"$(P_{{H,{m['degreeP']}}}, U_{{H,{m['degreeU']}}})$, "
                 f"α={m['alpha']:g}")
    ax.invert_xaxis()
    ax.legend(frameon=False, loc="best", fontsize=9)
    fig.tight_layout()
    return save_figure(fig, outdir, "convergence_hh.png")


# ===========================================================================
# CLI
# ===========================================================================

def main(argv=None):
    defaults = CommonParams(nloc=DEFAULT_NLOC, degreeU=1, degreeP=1,
                            alpha=1.0, outdir=DEFAULT_OUTDIR)
    args = parse_args(
        "Taxa de convergência Hh do MPrM (taxa de FEM interior — cf. "
        "mrcm.convergence mode='Hh')",
        defaults=defaults,
        extra_args=lambda p: (
            p.add_argument("--Ndoms", default=DEFAULT_NDOMS,
                          help="lista de Ndom (subdomínios/direção); "
                               "nloc fica fixo ⇒ H/h fixo. "
                               "Aceita 'a,b,c' ou 'range:a,b'"),
            p.add_argument("--metric", choices=["exact", "fine"],
                          default="exact",
                          help="qual erro relativo dirige a validação "
                               "PASS/FAIL: vs solução exata (inclui o "
                               "esquema MAC) ou vs referência fina na "
                               "mesma N (isola a decomposição)"),
            p.add_argument("--expected-rate", type=float,
                          default=DEFAULT_EXPECTED_RATE,
                          help="taxa esperada para validação PASS/FAIL "
                               "(None desabilita)"),
            p.add_argument("--tolerance", type=float,
                          default=DEFAULT_TOLERANCE,
                          help="tolerância absoluta da validação de taxa"),
        ),
        argv=argv,
    )

    if args.quick:
        Ndoms = [2, 4]
        nloc = 8
    else:
        Ndoms = parse_int_list(args.Ndoms)
        nloc = args.nloc

    comm = get_comm()
    t0 = time.perf_counter()
    results = run_study(comm, Ndoms, nloc, args.degreeU, args.degreeP,
                        args.alpha, basis_kind=args.basis_kind,
                        verbose=args.verbose)

    if comm.rank == 0:
        rate = results.get(f"rate_{args.metric}")
        if rate is not None and args.expected_rate is not None:
            ok = validate_rate(rate["rate"], args.expected_rate, args.tolerance)
            flag = "PASS" if ok else "FAIL"
            print(f"  [{flag}] Taxa Hh ({args.metric}): {rate['rate']:.3f} vs "
                 f"esperada {args.expected_rate:g} (tol={args.tolerance:g})")
        save_json(results, args.outdir, "convergence_hh.json")
        if len(results["cases"]) >= 2:
            plot_results(results, args.outdir)
        print(f"\nEstudo concluído em {time.perf_counter()-t0:.0f}s "
             f"({comm.size} rank(s)).")
    return results


if __name__ == "__main__":
    main()
