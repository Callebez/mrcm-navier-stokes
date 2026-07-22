#!/usr/bin/env python3
"""run_mprm_mpi.py — Cavidade MPrM com MPI."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/mrcm/mrcm-firedrake"))

from mpi4py import MPI
from mprm_core import run_cavity

comm = MPI.COMM_WORLD
rank, size = comm.rank, comm.size

result = run_cavity(
    Re=100.0, Ndom=9, nloc=16, T=16.0, dt=0.004340277777777778,
    lid_velocity=1.0,
    degreeU=2, degreeP=2, alpha=1.0,
    method="mprm", comm=comm, verbose=(rank == 0),
    log_every=max(1, 1844 // 10),
)

if rank == 0:
    print(f"\nMPI com {size} ranks:")
    print(f"  Tempo: {result['t_wall']:.2f} s")
    print(f"  max|div u|: {result['divmax']:.6e}")
    print(f"  Defeito: {result['defect']:.6e}")
