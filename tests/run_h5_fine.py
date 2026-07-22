#!/usr/bin/env python3
"""H5: method='fine' (sem MRCM) — isola o efeito da decomposição multiescala."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, time
from mpi4py import MPI
from mprm import run_von_karman, strouhal

comm = MPI.COMM_WORLD; rank = comm.rank
name = "H5_fine_ref"

if rank == 0:
    print(f"{'='*60}\n  {name}: method=fine (Poisson global, sem MRCM), nloc=20, offset yc=4.02\n{'='*60}", flush=True)

t0 = time.perf_counter()
r = run_von_karman(Lx=16, Ly=8, D=1, xc=4, yc=4.02, U_in=1, Re=100,
                   Ndomx=10, Ndomy=5, nloc=20,
                   degreeU=3, degreeP=3,
                   inflow='parabolic', walls='slip', inflow_norm='max',
                   alpha=1.0, T=60, method='fine', limiter='superbee',
                   comm=comm, verbose=(rank==0), log_every=400,
                   obstacle_metrics_every=5)
wt = time.perf_counter() - t0

if rank == 0:
    clh = r.get('cl_history')
    print(f"  dt={r['dt']:.4f}  nsteps={r['nsteps']}  divmax={r['divmax']:.2e}  {wt:.0f}s", flush=True)
    if clh is not None and len(clh) > 8:
        t_vec, cd, cl = clh[:, 0], clh[:, 1], clh[:, 2]
        h = len(t_vec) // 3
        cd_m, cd_s = float(cd[h:].mean()), float(cd[h:].std())
        cl_amp = float(0.5 * (cl[h:].max() - cl[h:].min()))
        cl_rms = float(np.sqrt((cl[h:] ** 2).mean()))
        st = strouhal(cl, r['dt'], 1, 1)
        print(f"  C_D={cd_m:.4f}±{cd_s:.4f}  C_L_amp={cl_amp:.4f}  C_L_rms={cl_rms:.4f}  St={st:.4f}", flush=True)
        print(f"  C_L min={cl[h:].min():.6f}  max={cl[h:].max():.6f}", flush=True)
        np.savez(f"results_alpha/hipoteses/{name}_history.npz", t=t_vec, cd=cd, cl=cl)
    else:
        print(f"  AVISO: cl_history ausente ou curto ({len(clh) if clh is not None else 0} pontos)", flush=True)
