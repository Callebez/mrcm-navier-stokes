#!/usr/bin/env python3
"""Roda von Kármán sobre quadtree e imprime um resumo em JSON no rank 0.

Usado por `tests/test_quadtree_flow.py` para comparar sequencial e MPI. Cada
rank constrói a árvore por conta própria — é exatamente o que a verificação de
consistência do esqueleto precisa exercitar.
"""
import json
import sys
import numpy as np

from mpi4py import MPI
from mprm.decomposition import (Decomposition, plan_quadtree_geometry,
                                build_obstacle_tree)
from mprm.problem import von_karman_problem
from mprm.transport import mac_divergence
from mprm.flow import run_flow


def main():
    comm = MPI.COMM_WORLD
    g = plan_quadtree_geometry(D=1.0, Ly=6.0, N_bx=4, N_by=3,
                               max_level=1, nloc_min=4, family="a")
    qt = build_obstacle_tree(g, radii=(3.0,))
    problem = von_karman_problem(
        Lx=g["Lx"], Ly=g["Ly"], Nx=g["Nx"], Ny=g["Ny"], D=g["D"],
        xc=g["xc"], yc=g["yc"], U_in=1.0, Re=100.0,
        inflow="parabolic", walls="slip", inflow_norm="max")
    d = Decomposition.from_quadtree(qt, h=g["h"], Lx=g["Lx"], Ly=g["Ly"],
                                    solid=problem.solid)
    r = run_flow(problem, Ndomx=g["N_bx"], Ndomy=g["N_by"], nloc=g["nloc"][0],
                 T=0.2, decomposition=d, comm=comm, verbose=False,
                 obstacle_metrics=False, U_ref=1.0, D=g["D"])
    if comm.rank == 0:
        dv = np.abs(mac_divergence(r["u"], r["v"], g["h"], g["h"]))
        print("JSON " + json.dumps({
            "ranks": comm.size,
            "u_norm": float(np.linalg.norm(r["u"])),
            "v_norm": float(np.linalg.norm(r["v"])),
            "u_sum": float(r["u"].sum()),
            "div_max": float(dv[~problem.solid].max()),
            "N_sub": r["decomposition"]["N_sub"],
        }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
