"""gif.py — Gera GIF do escoamento von Karman com MPrM.

Compativel com execucao sequencial e MPI:
  python gif.py                    # sequencial (1 rank)
  mpiexec -n 4 python gif.py       # paralelo (4 ranks)

Arquitetura MPI:
  - Rank 0: I/O (frames, GIF, prints) + coordenacao
  - Todos os ranks: transporte MAC (dados replicados), downscaler (replicado)
  - Projetor MRCM: subdominios distribuidos round-robin entre ranks
  - Frame callback: chamado apenas no rank 0 (run_flow ja faz a guarda)

Operacoes de I/O sao executadas APENAS no rank 0 para evitar race conditions.
"""
import numpy as np, os, glob, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpi4py import MPI
from mprm_core import run_von_karman, make_gif, strouhal

comm = MPI.COMM_WORLD
rank = comm.rank
n_ranks = comm.size

OUTDIR      = "frames_von_karman_hires"
GIF_OUT     = "results_alpha/von_karman_U4P4_40_3x3.gif"
FPS         = 10
FRAME_EVERY = 12
Ndomx, Ndomy, nloc =3, 3,60       # 160×80 cells, hx=hy=0.1, square cells
dU, dP      = 3, 3                   # U_H,2 / P_H,2
T           = 20.0
# Obstacle centered at (4,4) — fits in subdomain (interfaces at x=1.6,3.2,4.8,... y=1.6,3.2,4.8,...)
# bbox x=[3.5,4.5] ⊂ [3.2,4.8] ✓  bbox y=[3.5,4.5] ⊂ [3.2,4.8] ✓
XC, YC      = 4.0, 4.0

# I/O apenas no rank 0
if rank == 0:
    os.makedirs(OUTDIR, exist_ok=True)
    for f in glob.glob(f"{OUTDIR}/frame_*.png"):
        os.remove(f)

frame_idx = [0]
def frame_callback(step, t, u, v, hx, hy):
    """Chamado apenas no rank 0 (run_flow ja tem guarda comm.rank==0)."""
    Nx, Ny = u.shape[0]-1, v.shape[1]-1
    dvdx = (v[1:,:] - v[:-1,:]) / hx
    dudy = (u[:,1:] - u[:,:-1]) / hy
    vort = dvdx[:, 1:-1] - dudy[1:-1, :]
    fig, ax = plt.subplots(figsize=(10, 5))
    xc = (np.arange(1, Nx) + 0.5) * hx; yc = (np.arange(1, Ny) + 0.5) * hy
    ax.pcolormesh(xc, yc, vort.T, shading="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.add_patch(plt.Rectangle((XC - 0.5, YC - 0.5), 1.0, 1.0, color="k"))
    ax.set_aspect("equal")
    ax.set_title(f"von Karman  Re=100  U_H,{dU}/P_H,{dP}  {Ndomx*nloc}x{Ndomy*nloc}  t={t:.1f}")
    plt.colorbar(ax.collections[0], ax=ax, label=r"$\omega_z$", shrink=0.7)
    fname = os.path.join(OUTDIR, f"frame_{frame_idx[0]:05d}.png")
    fig.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close(fig)
    frame_idx[0] += 1

if rank == 0:
    print("=" * 60)
    print(f"von Karman MPI — Re=100, {Ndomx*nloc}x{Ndomy*nloc} cel, U_H,{dU}/P_H,{dP}")
    print(f"hx=hy={16/(Ndomx*nloc):.3f}, T={T}, {n_ranks} ranks MPI, {Ndomx}x{Ndomy} subdominios")
    print("=" * 60)

t0 = time.perf_counter()
r = run_von_karman(Lx=16.0, Ly=8.0, D=1.0, xc=XC, yc=YC, U_in=1.0, Re=100.0,
                   Ndomx=Ndomx, Ndomy=Ndomy, nloc=nloc, degreeU=dU, degreeP=dP,
                   inflow='uniform', walls='wall',
                   alpha=1.0, T=T, comm=comm, verbose=(rank == 0), log_every=500,
                   frame_callback=frame_callback, frame_every=FRAME_EVERY)
wall_t = time.perf_counter() - t0

# Metricas e GIF apenas no rank 0
if rank == 0:
    clh = r.get('cl_history')
    if clh is not None and len(clh) > 8:
        t_vec, cd, cl = clh[:, 0], clh[:, 1], clh[:, 2]
        half = len(t_vec) // 2
        st = strouhal(cl, r['dt'], 1.0, 1.0)
        print(f"\n===== RESULTADOS =====")
        print(f"Malha:  {r['N']}x{Ndomy*nloc} cel ({Ndomx}x{Ndomy} subd, nloc={nloc})")
        print(f"Passos: {r['nsteps']} (dt={r['dt']:.4f})")
        print(f"Tempo:  {wall_t:.0f}s  ({n_ranks} ranks MPI)")
        print(f"C_D:    {cd[half:].mean():.4f}")
        print(f"C_L_amp:{0.5*(cl[half:].max()-cl[half:].min()):.4f}")
        print(f"St:     {st:.4f}")
        print(f"divmax: {r['divmax']:.2e}")

    frames = sorted(glob.glob(f"{OUTDIR}/frame_*.png"))
    print(f"\n{len(frames)} frames. Gerando GIF...")
    gif_path = make_gif(OUTDIR, GIF_OUT, fps=FPS)
    print(f"✓ {gif_path}")