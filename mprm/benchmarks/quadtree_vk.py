"""Estudo comparativo: decomposição quadtree vs cartesiana no von Kármán.

Funções puras — sem I/O, sem plot, sem escrita de arquivo. O script
`tests/test_quadtree_von_karman.py` cuida de CLI, JSON e figuras, seguindo o
mesmo contrato de `mprm/benchmarks/convergence.py`.

Métricas por configuração (spec §5, Fase 5): C_D médio, amplitude de C_L, St,
erro relativo contra a referência de malha fina, max|∇·u|, defeito de
compatibilidade, N_sub, n_Γ, min/max H, condicionamento de A_Γ, e tempos de
offline e online separados.
"""
from __future__ import annotations
import time
from datetime import datetime
import numpy as np

from mprm.decomposition import (Decomposition, plan_quadtree_geometry,
                                build_obstacle_tree)
from mprm.problem import von_karman_problem
from mprm.flow import run_flow
from mprm.forces import strouhal


def build_case(*, family, D, Ly, N_bx, N_by, max_level, nloc_min, radii,
               Re=100.0, U_in=1.0, wake_length=0.0, label=None) -> dict:
    """Geometria + problema + decomposição, sem resolver nada."""
    g = plan_quadtree_geometry(D=D, Ly=Ly, N_bx=N_bx, N_by=N_by,
                               max_level=max_level, nloc_min=nloc_min,
                               family=family)
    qt = build_obstacle_tree(g, radii=radii, wake_length=wake_length)
    problem = von_karman_problem(
        Lx=g["Lx"], Ly=g["Ly"], Nx=g["Nx"], Ny=g["Ny"], D=g["D"],
        xc=g["xc"], yc=g["yc"], U_in=U_in, Re=Re,
        inflow="parabolic", walls="slip", inflow_norm="max")
    decomp = Decomposition.from_quadtree(qt, h=g["h"], Lx=g["Lx"], Ly=g["Ly"],
                                         solid=problem.solid)
    return {"label": label or f"quadtree_{family}_L{max_level}",
            "geom": g, "problem": problem, "decomp": decomp,
            "Re": Re, "U_in": U_in}


def uniform_counterpart(case: dict, target: str = "n_gamma",
                        label: str = None) -> dict:
    """Decomposição cartesiana comparável, com o MESMO h e o MESMO problema.

    `target` é "N_sub" ou "n_gamma". Fixada a malha global, as decomposições
    cartesianas com células quadradas formam uma família discreta: `Nx/Ndomx`
    e `Ny/Ndomy` precisam ser inteiros iguais. Percorremos os divisores comuns
    e escolhemos o que mais se aproxima do alvo — a diferença residual é
    reportada em `target_gap`, porque não há como zerá-la em geral (spec §5).
    """
    g, decomp = case["geom"], case["decomp"]
    Nx, Ny = g["Nx"], g["Ny"]
    alvo = (len(decomp.leaves) if target == "N_sub"
            else len(decomp.gamma_edges()))

    melhor, melhor_erro = None, None
    for nloc in range(2, min(Nx, Ny) + 1, 2):
        if Nx % nloc or Ny % nloc:
            continue
        Ndomx, Ndomy = Nx // nloc, Ny // nloc
        d = Decomposition.uniform(Ndomx, Ndomy, nloc, g["Lx"], g["Ly"])
        valor = (Ndomx * Ndomy if target == "N_sub" else len(d.gamma_edges()))
        erro = abs(valor - alvo)
        if melhor_erro is None or erro < melhor_erro:
            melhor, melhor_erro = (d, Ndomx, Ndomy, nloc, valor), erro
    if melhor is None:
        raise ValueError(
            f"nenhuma decomposição cartesiana com células quadradas para "
            f"Nx={Nx}, Ny={Ny}.")
    d, Ndomx, Ndomy, nloc, valor = melhor
    return {"label": label or f"cartesian_{Ndomx}x{Ndomy}",
            "geom": g, "problem": case["problem"], "decomp": d,
            "Re": case["Re"], "U_in": case["U_in"],
            "Ndomx": Ndomx, "Ndomy": Ndomy, "nloc": nloc,
            "target": target, "target_value": valor, "target_gap": melhor_erro}


def _cond_A_gamma(projector) -> float:
    """Estimativa do número de condição de A_Γ (norma 1), ou nan."""
    A = getattr(projector, "A_gamma", None)
    if A is None or A.shape[0] > 4500:
        return float("nan")
    try:
        return float(np.linalg.cond(A.toarray(), p=1))
    except Exception:
        return float("nan")


def run_case(case: dict, T: float, comm=None, degreeU=1, degreeP=3,
             alpha=1.0, basis_kind="legendre_P", u_ref=None, v_ref=None,
             verbose=False) -> dict:
    """Roda uma configuração e devolve o dicionário do schema (spec §5)."""
    g, problem, decomp = case["geom"], case["problem"], case["decomp"]
    Ndomx = case.get("Ndomx", g["N_bx"])
    Ndomy = case.get("Ndomy", g["N_by"])
    nloc = case.get("nloc", g["nloc"][0])

    t0 = time.perf_counter()
    r = run_flow(problem, Ndomx=Ndomx, Ndomy=Ndomy, nloc=nloc, T=T,
                 decomposition=decomp, degreeU=degreeU, degreeP=degreeP,
                 alpha=alpha, basis_kind=basis_kind, comm=comm,
                 verbose=verbose, obstacle_metrics=True,
                 U_ref=case["U_in"], D=g["D"])
    t_total = time.perf_counter() - t0

    info = r["decomposition"]
    cl = r.get("cl_history")
    if cl is not None and len(cl) > 4:
        metade = len(cl) // 2
        Cd = cl[metade:, 1]
        Cl = cl[metade:, 2]
        Cd_mean, Cd_std = float(Cd.mean()), float(Cd.std())
        Cl_amp = float(0.5 * (Cl.max() - Cl.min()))
        St = float(strouhal(cl[:, 2], r["dt"], case["U_in"], g["D"]))
    else:
        Cd_mean = Cd_std = Cl_amp = St = float("nan")

    err_u = float("nan")
    if u_ref is not None:
        num = np.sum((r["u"] - u_ref) ** 2) + np.sum((r["v"] - v_ref) ** 2)
        den = np.sum(u_ref ** 2) + np.sum(v_ref ** 2)
        err_u = float(np.sqrt(num / den))

    t_online = t_total / max(r["nsteps"], 1)
    return {"label": case["label"],
            "N_sub": info["N_sub"],
            "n_gamma_edges": info["n_gamma_edges"],
            "n_gamma_dofs": info["n_gamma_dofs"],
            "H_min": info["H_min"], "H_max": info["H_max"],
            "levels": info["levels"],
            "cond_A_gamma": float(r.get("cond_A_gamma", float("nan"))),
            "t_offline_s": float(r.get("t_offline", 0.0)),
            "t_online_per_step_s": float(t_online),
            "Cd_mean": Cd_mean, "Cd_std": Cd_std,
            "Cl_amp": Cl_amp, "St": St,
            "div_max": float(r["divmax"]),
            "defect_max": float(r["defect"]),
            "err_u_rel": err_u,
            "target_gap": case.get("target_gap", 0),
            "u": r["u"], "v": r["v"]}


def run_study(cases, T, comm=None, degreeU=1, degreeP=3, alpha=1.0,
              basis_kind="legendre_P", with_uniform_baseline=True,
              with_fine_reference=True, verbose=False) -> dict:
    """Executa todas as configurações e monta o JSON do spec §5."""
    g = cases[0]["geom"]
    problem = cases[0]["problem"]

    u_ref = v_ref = None
    if with_fine_reference:
        r_fine = run_flow(problem, Ndomx=1, Ndomy=1, nloc=1, T=T,
                          method="fine", comm=comm, verbose=verbose,
                          obstacle_metrics=False, U_ref=cases[0]["U_in"],
                          D=g["D"])
        u_ref, v_ref = r_fine["u"], r_fine["v"]

    todos = list(cases)
    if with_uniform_baseline:
        for alvo in ("n_gamma", "N_sub"):
            todos.append(uniform_counterpart(cases[0], target=alvo))

    configs = []
    for c in todos:
        res = run_case(c, T=T, comm=comm, degreeU=degreeU, degreeP=degreeP,
                       alpha=alpha, basis_kind=basis_kind,
                       u_ref=u_ref, v_ref=v_ref, verbose=verbose)
        res.pop("u", None)
        res.pop("v", None)
        configs.append(res)

    return {"meta": {"date": datetime.now().isoformat(timespec="seconds"),
                     "n_ranks": getattr(comm, "size", 1)},
            "geometry": {k: g[k] for k in
                         ("family", "Lx", "Ly", "D", "h", "Nx", "Ny",
                          "xc", "yc", "H0", "nloc", "max_level")},
            "flow": {"Re": cases[0]["Re"], "U_in": cases[0]["U_in"],
                     "inflow": "parabolic", "inflow_norm": "max",
                     "walls": "slip", "T": T},
            "spaces": {"degreeU": degreeU, "degreeP": degreeP,
                       "alpha": alpha, "basis_kind": basis_kind},
            "configs": configs}
