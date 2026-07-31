"""Estudos de convergência para o benchmark de Helmholtz-Hodge de Olshanskii.

Fornece funções puras (sem I/O) que executam os 4 modos de convergência:
Hh (FEM interior), H (multiescala), h (saturação), p (grau).

Cada função recebe parâmetros, executa os jobs do MRCM e retorna um
dicionário de resultados com metadados, casos e taxas de convergência.
"""
from __future__ import annotations
import time
import numpy as np
from mpi4py import MPI

from mrcm.config import MRCMConfig
from mrcm.convergence import compute_convergence_rate


# ═══════════════════════════════════════════════════════════════════════════
# Helpers internos
# ═══════════════════════════════════════════════════════════════════════════

def _make_meta(study: str, comm, **fields) -> dict:
    from datetime import datetime
    meta = {
        "study": study,
        "date": datetime.now().isoformat(timespec="seconds"),
        "n_ranks": getattr(comm, "size", 1),
    }
    meta.update(fields)
    return meta


def _rel_err(a1, a2, b1, b2) -> float:
    """‖(a1,a2) − (b1,b2)‖₂ / ‖(b1,b2)‖₂."""
    num = np.sum((a1 - b1) ** 2) + np.sum((a2 - b2) ** 2)
    den = np.sum(b1 ** 2) + np.sum(b2 ** 2)
    return float(np.sqrt(num / den))


def _fit_rate(cases: list[dict], x_key: str, err_key: str,
              verbose: bool = True) -> dict | None:
    """Regressão log-log de err_key vs x_key."""
    if len(cases) < 2:
        return None
    ordered = sorted(cases, key=lambda c: -c[x_key])
    xs = [c[x_key] for c in ordered]
    errs = [c[err_key] for c in ordered]
    rate = compute_convergence_rate(xs, errs)
    if verbose:
        print(f"  Taxa ({err_key} vs {x_key}): {rate['rate']:.3f}  "
              f"(R² = {rate['r_squared']:.4f})  "
              f"por par: {[f'{s:.2f}' for s in rate['per_step']]}",
              flush=True)
    return rate


def _run_one_hh_level(comm, Ndom, nloc, degreeU, degreeP, alpha,
                      basis_kind, dt, include_raw_mrcm=False) -> dict:
    """Um nível Hh: monta offline, projeta, mede erro."""
    from mprm.benchmarks.hh import hh_build_w
    from mprm.projector import MPrMProjector
    from mprm.downscaler import MACDownscaler
    from mprm.flow import mprm_projection_step, FineProjector
    from mprm.transport import mac_divergence

    N = Ndom * nloc
    h = 1.0 / N
    H = 1.0 / Ndom

    w1, w2, u1, u2 = hh_build_w(N)

    fine = FineProjector(N, N, h, h, dt)
    uf1, uf2 = fine.step(w1, w2)

    t0 = time.perf_counter()
    cfg = MRCMConfig(Ndomx=Ndom, Ndomy=Ndom, nlocx=nloc, nlocy=nloc,
                     degreeU=degreeU, degreeP=degreeP, alpha=float(alpha),
                     basis_kind=basis_kind,
                     mixed_family="BDM", mixed_order=0,
                     executor_kind="sequential")
    projector = MPrMProjector(cfg, dt, comm=comm)
    downscaler = MACDownscaler(cfg, dt)
    uh1, uh2 = mprm_projection_step(w1, w2, projector, downscaler)
    wall = time.perf_counter() - t0

    case = {
        "Ndom": Ndom, "H": float(H), "h": float(h), "N": N,
        "err_vs_exact": _rel_err(uh1, uh2, u1, u2),
        "err_vs_fine": _rel_err(uh1, uh2, uf1, uf2),
        "err_fine_vs_exact": _rel_err(uf1, uf2, u1, u2),
        "divmax": float(np.abs(mac_divergence(uh1, uh2, h, h)).max()),
        "defect": float(downscaler.last_defect),
        "wall_s": wall,
    }

    if include_raw_mrcm:
        # Reconstrução BDM₁-DG₀ crua (sem downscaling): avalia o campo
        # BDM₁ de cada subdomínio nas faces da malha MAC correspondente.
        try:
            ur1, ur2 = _evaluate_bdm1_on_mac_faces_from_projector(
                projector, N, N)
            case["err_raw_mrcm"] = _rel_err(ur1, ur2, u1, u2)
        except Exception as e:
            if comm.rank == 0:
                print(f"  ⚠ raw MRCM eval failed: {e}")
            case["err_raw_mrcm"] = None

    return case


def _evaluate_bdm1_on_mac_faces_from_projector(projector, Nx, Ny,
                                                Lx=1.0, Ly=1.0):
    """Avalia os campos BDM₁ de cada subdomínio nas faces da malha MAC.

    Itera sobre os subdomínios, avalia `last_uT[isub]` nas faces MAC
    do patch correspondente, e monta o campo global.
    """
    cfg = projector.cfg
    nlocx, nlocy = cfg.nlocx, cfg.nlocy
    Ndomx, Ndomy = cfg.Ndomx, cfg.Ndomy
    hx, hy = Lx / Nx, Ly / Ny

    uh1 = np.zeros((Nx + 1, Ny))
    uh2 = np.zeros((Nx, Ny + 1))

    for isub in projector.my_subs:
        if isub not in projector.last_uT:
            continue
        uT = projector.last_uT[isub]  # firedrake.Function no espaço BDM₁
        ix, iy = isub % Ndomx, isub // Ndomx
        ox, oy = ix * nlocx, iy * nlocy

        # Faces verticais neste patch: componente x
        xf_local = np.linspace(ox * hx, (ox + nlocx) * hx, nlocx + 1)
        yc_local = (np.arange(oy, oy + nlocy) + 0.5) * hy
        for j in range(nlocy):
            for i in range(nlocx + 1):
                px, py = xf_local[i], yc_local[j]
                try:
                    val = uT.at(px, py)
                    uh1[ox + i, oy + j] = val[0] if hasattr(val, '__len__') else val
                except Exception:
                    uh1[ox + i, oy + j] = 0.0

        # Faces horizontais neste patch: componente y
        xc_local = (np.arange(ox, ox + nlocx) + 0.5) * hx
        yf_local = np.linspace(oy * hy, (oy + nlocy) * hy, nlocy + 1)
        for j in range(nlocy + 1):
            for i in range(nlocx):
                px, py = xc_local[i], yf_local[j]
                try:
                    val = uT.at(px, py)
                    uh2[ox + i, oy + j] = val[1] if hasattr(val, '__len__') else val
                except Exception:
                    uh2[ox + i, oy + j] = 0.0

    # Gather de todos os ranks (allreduce com SUM — cada rank contribui
    # apenas seus subdomínios, os demais têm zeros)
    comm = projector.comm
    uh1 = comm.allreduce(uh1, op=MPI.SUM)
    uh2 = comm.allreduce(uh2, op=MPI.SUM)

    # Forçar contorno impermeável
    uh1[0, :] = uh1[-1, :] = 0.0
    uh2[:, 0] = uh2[:, -1] = 0.0
    return uh1, uh2


# ═══════════════════════════════════════════════════════════════════════════
# API pública — Modos de convergência
# ═══════════════════════════════════════════════════════════════════════════

def convergence_curve_hh(comm, Ndoms: list[int], nloc: int,
                         degreeU: int, degreeP: int, alpha: float,
                         basis_kind: str = "legendre_P",
                         include_raw_mrcm: bool = False,
                         verbose: bool = True) -> dict:
    """Modo Hh: nloc fixo, Ndom varre → H e h proporcionais.

    Retorna dict com meta, cases, rate_exact, rate_fine (e rate_raw).
    """
    dt = 1.0
    if verbose and comm.rank == 0:
        print(f"Convergência Hh: nloc={nloc}, (U{degreeU},P{degreeP}), "
              f"α={alpha}, Ndom ∈ {Ndoms}", flush=True)

    results = {
        "meta": _make_meta("convergence_hh", comm,
                           nloc=nloc, degreeU=degreeU, degreeP=degreeP,
                           alpha=alpha, basis_kind=basis_kind,
                           Ndoms=list(Ndoms), dt=dt),
        "cases": [],
    }

    for Ndom in Ndoms:
        case = _run_one_hh_level(comm, Ndom, nloc, degreeU, degreeP,
                                 alpha, basis_kind, dt, include_raw_mrcm)
        results["cases"].append(case)
        if verbose and comm.rank == 0:
            extra = ""
            if include_raw_mrcm and case.get("err_raw_mrcm") is not None:
                extra = f"  raw={case['err_raw_mrcm']:.4e}"
            print(f"  Ndom={Ndom:3d}  H={case['H']:.4f}  h={case['h']:.5f}  "
                  f"N={case['N']:4d}:  err_exato={case['err_vs_exact']:.4e}  "
                  f"err_fino={case['err_vs_fine']:.4e}{extra}  "
                  f"div={case['divmax']:.1e}  ({case['wall_s']:.1f}s)",
                  flush=True)

    # Compute rates on ALL ranks (pure numpy, no MPI side effects)
    if len(results["cases"]) >= 2:
        if verbose and comm.rank == 0:
            print()
        results["rate_exact"] = _fit_rate(results["cases"], "H",
                                          "err_vs_exact",
                                          verbose and comm.rank == 0)
        results["rate_fine"] = _fit_rate(results["cases"], "H",
                                         "err_vs_fine",
                                         verbose and comm.rank == 0)
        if include_raw_mrcm:
            raw_cases = [c for c in results["cases"]
                        if c.get("err_raw_mrcm") is not None]
            if len(raw_cases) >= 2:
                results["rate_raw"] = _fit_rate(raw_cases, "H",
                                                "err_raw_mrcm",
                                                verbose and comm.rank == 0)

    return results


def convergence_curve_H(comm, Ndoms: list[int], h_fine_target: float,
                        degreeU: int, degreeP: int, alpha: float,
                        basis_kind: str = "legendre_P",
                        verbose: bool = True) -> dict:
    """Modo H (multiescala): h ~ fixo, Ndom varia → H→0.

    Para cada Ndom, calcula nloc = round(1/(h_target * Ndom)).
    """
    dt = 1.0
    if verbose and comm.rank == 0:
        print(f"Convergência H (multiescala): h≈{h_fine_target:.5f}, "
              f"(U{degreeU},P{degreeP}), α={alpha}, "
              f"Ndom ∈ {Ndoms}", flush=True)

    results = {
        "meta": _make_meta("convergence_H", comm,
                           h_target=h_fine_target, degreeU=degreeU,
                           degreeP=degreeP, alpha=alpha, basis_kind=basis_kind,
                           Ndoms=list(Ndoms), dt=dt),
        "cases": [],
    }

    for Ndom in Ndoms:
        nloc = max(2, round(1.0 / (h_fine_target * Ndom)))
        case = _run_one_hh_level(comm, Ndom, nloc, degreeU, degreeP,
                                 alpha, basis_kind, dt)
        results["cases"].append(case)
        if verbose and comm.rank == 0:
            print(f"  Ndom={Ndom:3d}  nloc={nloc:4d}  H={case['H']:.4f}  "
                  f"h={case['h']:.5f}  N={case['N']:4d}:  "
                  f"err_exato={case['err_vs_exact']:.4e}  "
                  f"err_fino={case['err_vs_fine']:.4e}  "
                  f"div={case['divmax']:.1e}  ({case['wall_s']:.1f}s)",
                  flush=True)

    if len(results["cases"]) >= 2:
        if verbose and comm.rank == 0:
            print()
        results["rate_exact"] = _fit_rate(results["cases"], "H",
                                          "err_vs_exact",
                                          verbose and comm.rank == 0)
        results["rate_fine"] = _fit_rate(results["cases"], "H",
                                         "err_vs_fine",
                                         verbose and comm.rank == 0)

    return results


def convergence_curve_h(comm, Ndom: int, nlocs: list[int],
                        degreeU: int, degreeP: int, alpha: float,
                        basis_kind: str = "legendre_P",
                        verbose: bool = True) -> dict:
    """Modo h (saturação): H fixo (Ndom fixo), nloc varre → h→0."""
    dt = 1.0
    if verbose and comm.rank == 0:
        print(f"Convergência h (saturação): Ndom={Ndom} (H={1.0/Ndom:.4f}), "
              f"(U{degreeU},P{degreeP}), α={alpha}, "
              f"nloc ∈ {nlocs}", flush=True)

    results = {
        "meta": _make_meta("convergence_h", comm,
                           Ndom=Ndom, H=1.0/Ndom, degreeU=degreeU,
                           degreeP=degreeP, alpha=alpha, basis_kind=basis_kind,
                           nlocs=list(nlocs), dt=dt),
        "cases": [],
    }

    for nloc in nlocs:
        case = _run_one_hh_level(comm, Ndom, nloc, degreeU, degreeP,
                                 alpha, basis_kind, dt)
        results["cases"].append(case)
        if verbose and comm.rank == 0:
            print(f"  nloc={nloc:3d}  h={case['h']:.5f}  N={case['N']:4d}:  "
                  f"err_exato={case['err_vs_exact']:.4e}  "
                  f"err_fino={case['err_vs_fine']:.4e}  "
                  f"div={case['divmax']:.1e}  ({case['wall_s']:.1f}s)",
                  flush=True)

    if len(results["cases"]) >= 2:
        if verbose and comm.rank == 0:
            print()
        results["rate_exact"] = _fit_rate(results["cases"], "h",
                                          "err_vs_exact",
                                          verbose and comm.rank == 0)
        results["rate_fine"] = _fit_rate(results["cases"], "h",
                                         "err_vs_fine",
                                         verbose and comm.rank == 0)

    return results


def convergence_curve_p(comm, Ndom: int, nloc: int,
                        ks: list[int], alpha: float,
                        basis_kind: str = "legendre_P",
                        asymmetric_pairs: list[tuple[int, int]] | None = None,
                        verbose: bool = True) -> dict:
    """Modo p (grau): malha fixa, graus crescentes.

    Se asymmetric_pairs=None, usa degreeU=degreeP=k para k em ks.
    Caso contrário, usa os pares (degreeU, degreeP) fornecidos.
    """
    dt = 1.0
    if verbose and comm.rank == 0:
        pairs_str = (f"pares {asymmetric_pairs}" if asymmetric_pairs
                     else f"ks={ks} (simétrico)")
        print(f"Convergência p (grau): Ndom={Ndom}, nloc={nloc}, "
              f"α={alpha}, {pairs_str}", flush=True)

    results = {
        "meta": _make_meta("convergence_p", comm,
                           Ndom=Ndom, nloc=nloc, alpha=alpha,
                           basis_kind=basis_kind, dt=dt),
        "cases": [],
    }

    if asymmetric_pairs:
        pairs = asymmetric_pairs
        results["meta"]["pairs"] = asymmetric_pairs
    else:
        pairs = [(k, k) for k in ks]
        results["meta"]["ks"] = list(ks)

    for degU, degP in pairs:
        case = _run_one_hh_level(comm, Ndom, nloc, degU, degP,
                                 alpha, basis_kind, dt)
        case["degreeU"] = degU
        case["degreeP"] = degP
        results["cases"].append(case)
        if verbose and comm.rank == 0:
            print(f"  (U{degU},P{degP}):  "
                  f"err_exato={case['err_vs_exact']:.4e}  "
                  f"err_fino={case['err_vs_fine']:.4e}  "
                  f"div={case['divmax']:.1e}  ({case['wall_s']:.1f}s)",
                  flush=True)

    return results
