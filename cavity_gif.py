import numpy as np, os, glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpi4py import MPI
from mprm import run_cavity, make_gif

comm = MPI.COMM_WORLD
rank = comm.rank

outdir = "frames_cavity"
gif_out = "results_alpha/cavity_Re100.gif"

if rank == 0:
    os.makedirs(outdir, exist_ok=True)
    for f in glob.glob(f"{outdir}/frame_*.png"):
        os.remove(f)
    print("Cavidade Re=100, Ndom=4, nloc=16, T=10, 4 ranks MPI...")

def cb(step, t, u, v, hx, hy):
    N = u.shape[0] - 1
    xc = (np.arange(N) + 0.5) * hx
    mag = np.sqrt((0.5*(u[:-1,:]+u[1:,:]))**2 + (0.5*(v[:,:-1]+v[:,1:]))**2)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pcolormesh(xc, xc, mag.T, cmap="turbo", shading="auto", vmin=0, vmax=1)
    ax.set_aspect("equal")
    ax.set_title(f"Cavidade Re=100  t={t:.2f}")
    fig.savefig(f"{outdir}/frame_{step:05d}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

r = run_cavity(Re=100, Ndom=4, nloc=16, T=10, degreeU=3, degreeP=3,
               basis_kind="legendre_P", comm=comm, verbose=(rank==0),
               frame_callback=cb, frame_every=20)

if rank == 0:
    print(f"✓ {r['nsteps']} passos, divmax={r['divmax']:.1e}, {r['t_wall']:.0f}s")
    frames = sorted(glob.glob(f"{outdir}/frame_*.png"))
    print(f"{len(frames)} frames -> GIF...")
    make_gif(outdir, gif_out, fps=15)
    print(f"✓ {gif_out}")
