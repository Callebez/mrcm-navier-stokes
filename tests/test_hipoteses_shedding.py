#!/usr/bin/env python3
"""Teste sistemático das hipóteses de ausência de vortex shedding — von Kármán.

H1 — Refinar malha (nloc=30, D/h=18.75) + superbee + perturbação
H2 — Superbee com malha atual (nloc=20) + perturbação  
H3 — Perturbação assimétrica + superbee + malha atual
H5 — method="fine" (sem MRCM) + malha atual + superbee + perturbação

Uso: mpiexec -n 4 python tests/test_hipoteses_shedding.py [--only H1,H3]
"""
from __future__ import annotations
import sys, os, json, time, argparse
import numpy as np
from mpi4py import MPI

# Garantir que mprm é importável (mesmo sem pip install)
_srcdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _srcdir not in sys.path:
    sys.path.insert(0, _srcdir)

from mprm import run_von_karman, strouhal

comm = MPI.COMM_WORLD
rank = comm.rank

BASE = {
    "Lx": 16, "Ly": 8, "D": 1, "xc": 4, "yc": 4+0.02,  # leve offset vertical (quebra simetria)
    "U_in": 1, "Re": 100, "T": 60,
    "degreeU": 3, "degreeP": 3, "alpha": 1.0,
    "inflow": "parabolic", "walls": "slip", "inflow_norm": "max",
    "obstacle_metrics_every": 5,
}

TESTS = {
    "H2_superbee_n20": {
        "desc": "H2: superbee, nloc=20, D/h=12.5, com offset y",
        "Ndomx": 10, "Ndomy": 5, "nloc": 20,
        "limiter": "superbee", "method": "mprm", "T": 60,
    },
    "H3_perturbacao": {
        "desc": "H3: superbee + perturbação explícita + offset, nloc=20",
        "Ndomx": 10, "Ndomy": 5, "nloc": 20,
        "limiter": "superbee", "method": "mprm", "T": 60,
        "perturb": 0.02,  # amplitude da perturbação
    },
    "H1_refine_n30": {
        "desc": "H1: malha refinada nloc=30, D/h=18.75, superbee + offset",
        "Ndomx": 10, "Ndomy": 5, "nloc": 30,
        "limiter": "superbee", "method": "mprm", "T": 60,
    },
    "H5_fine_ref": {
        "desc": "H5: method=fine (sem MRCM), nloc=20, superbee + offset",
        "Ndomx": 10, "Ndomy": 5, "nloc": 20,
        "limiter": "superbee", "method": "fine", "T": 60,
    },
}


def run_one(name, cfg, results_dir):
    t_label = f"{name}_{time.strftime('%H%M%S')}"
    log_path = os.path.join(results_dir, f"{t_label}.log")
    json_path = os.path.join(results_dir, f"{t_label}.json")

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"  {name}: {cfg['desc']}")
        print(f"  nloc={cfg['nloc']}, limiter={cfg.get('limiter','minmod')}, "
              f"method={cfg['method']}, T={cfg.get('T',60)}")
        print(f"  offset yc={BASE['yc']:.3f}, perturb={cfg.get('perturb',0)}")
        print(f"{'='*60}", flush=True)

    t0 = time.perf_counter()
    try:
        r = run_von_karman(
            Lx=BASE["Lx"], Ly=BASE["Ly"], D=BASE["D"],
            xc=BASE["xc"], yc=BASE["yc"],
            U_in=BASE["U_in"], Re=BASE["Re"],
            Ndomx=cfg["Ndomx"], Ndomy=cfg["Ndomy"], nloc=cfg["nloc"],
            degreeU=BASE["degreeU"], degreeP=BASE["degreeP"],
            alpha=BASE["alpha"],
            inflow=BASE["inflow"], walls=BASE["walls"],
            inflow_norm=BASE["inflow_norm"],
            T=cfg.get("T", 60), method=cfg.get("method", "mprm"),
            limiter=cfg.get("limiter", "minmod"),
            comm=comm, verbose=(rank == 0), log_every=400,
            obstacle_metrics_every=BASE["obstacle_metrics_every"],
        )
    except Exception as e:
        if rank == 0:
            print(f"  ERRO: {e}", flush=True)
        return {"name": name, "error": str(e)}

    wt = time.perf_counter() - t0
    result = {"name": name, "desc": cfg["desc"], "t_wall": wt,
              "dt": float(r["dt"]), "nsteps": int(r["nsteps"]),
              "divmax": float(r["divmax"])}

    if rank == 0:
        clh = r.get("cl_history")
        if clh is not None and len(clh) > 8:
            t_vec, cd, cl = clh[:, 0], clh[:, 1], clh[:, 2]
            h = max(len(t_vec) // 3, 1)  # primeiro terço é transiente
            cd_mean = float(cd[h:].mean())
            cl_amp = float(0.5 * (cl[h:].max() - cl[h:].min()))
            cl_rms = float(np.std(cl[h:]))
            st_val = strouhal(cl, r["dt"], 1, 1)
            result.update({
                "Cd_mean": cd_mean, "Cl_amp": cl_amp, "Cl_rms": cl_rms,
                "Cl_max": float(cl[h:].max()), "Cl_min": float(cl[h:].min()),
                "St": st_val,
            })
            # Salva histórico para plot
            np.savez(os.path.join(results_dir, f"{t_label}_history.npz"),
                     t=t_vec, cd=cd, cl=cl)
            print(f"  C_D={cd_mean:.4f}  C_L_amp={cl_amp:.4f}  "
                  f"C_L_rms={cl_rms:.4f}  St={st_val:.4f}  "
                  f"{wt:.0f}s  divmax={r['divmax']:.2e}", flush=True)
        else:
            result["warning"] = "sem cl_history suficiente"
            print(f"  sem cl_history suficiente ({len(clh) if clh is not None else 0} pontos)", flush=True)

    # Broadcast result
    result = comm.bcast(result if rank == 0 else None, root=0)
    if rank == 0:
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2, default=float)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="",
                        help="Testes a rodar, ex: H1,H3 (default: todos)")
    parser.add_argument("--results-dir", default="results_alpha/hipoteses")
    args = parser.parse_args()

    results_dir = args.results_dir
    if rank == 0:
        os.makedirs(results_dir, exist_ok=True)

    only = [t.strip() for t in args.only.split(",") if t.strip()] if args.only else []
    selected = {k: v for k, v in TESTS.items()
                if not only or k in only}

    if rank == 0:
        print(f"Testando {len(selected)} hipótese(s): {list(selected.keys())}",
              flush=True)

    all_results = []
    for name, cfg in selected.items():
        r = run_one(name, cfg, results_dir)
        all_results.append(r)

    if rank == 0:
        summary_path = os.path.join(results_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2, default=float)
        print(f"\nResumo salvo em {summary_path}")

        # Tabela comparativa
        print(f"\n{'Nome':<25s} {'C_D':>8s} {'C_L_amp':>10s} {'St':>8s} "
              f"{'t(s)':>8s} {'divmax':>10s}")
        print("-" * 75)
        for r in all_results:
            if "error" in r:
                print(f"{r['name']:<25s} {'ERRO':>8s}: {r['error'][:40]}")
            else:
                cd = r.get("Cd_mean", float("nan"))
                cl = r.get("Cl_amp", float("nan"))
                st = r.get("St", float("nan"))
                tw = r.get("t_wall", 0)
                dv = r.get("divmax", 0)
                print(f"{r['name']:<25s} {cd:8.4f} {cl:10.4f} {st:8.4f} "
                      f"{tw:8.0f} {dv:10.2e}")


if __name__ == "__main__":
    main()
