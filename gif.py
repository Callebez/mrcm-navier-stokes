import numpy as np, os, glob, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mprm_core import run_von_karman, make_gif, strouhal

OUTDIR      = "frames_von_karman_hires"
GIF_OUT     = "results_alpha/von_karman_U1P1.gif"
FPS         = 10
FRAME_EVERY = 12
Ndom, nloc  = 4, 30          # 64² células (vs 48² anterior)
dU, dP      = 1, 1           # U_H,3 / P_H,3
T           = 50.0           # 50 D/U∞

os.makedirs(OUTDIR, exist_ok=True)
for f in glob.glob(f"{OUTDIR}/frame_*.png"):
    os.remove(f)

frame_idx = [0]
def frame_callback(step, t, u, v, hx, hy):
    Nx, Ny = u.shape[0]-1, v.shape[1]-1
    dvdx = (v[1:,:] - v[:-1,:]) / hx
    dudy = (u[:,1:] - u[:,:-1]) / hy
    vort = dvdx[:, 1:-1] - dudy[1:-1, :]
    fig, ax = plt.subplots(figsize=(10, 5))
    xc = (np.arange(1, Nx) + 0.5) * hx; yc = (np.arange(1, Ny) + 0.5) * hy
    ax.pcolormesh(xc, yc, vort.T, shading="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.add_patch(plt.Rectangle((2.5, 2.5), 1.0, 1.0, color="k"))
    ax.set_aspect("equal")
    ax.set_title(f"von Karman  Re=100  U_H,3 / P_H,3  64² celulas  t={t:.1f}")
    plt.colorbar(ax.collections[0], ax=ax, label=r"$\omega_z$", shrink=0.7)
    fname = os.path.join(OUTDIR, f"frame_{frame_idx[0]:05d}.png")
    fig.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close(fig)
    frame_idx[0] += 1

print("=" * 60)
print(f"von Karman — Re=100, Ndom={Ndom}, nloc={nloc} ({Ndom*nloc}² celulas)")
print(f"U_H,{dU} / P_H,{dP}, T={T}")
print("=" * 60)

t0 = time.perf_counter()
r = run_von_karman(Lx=16.0, Ly=8.0, D=1.0, xc=3.0, yc=3.0, U_in=1.0, Re=100.0,
                   Ndom=Ndom, nloc=nloc, degreeU=dU, degreeP=dP,
                   alpha=1.0, T=T, comm=None, verbose=True, log_every=500,
                   frame_callback=frame_callback, frame_every=FRAME_EVERY)
wall_t = time.perf_counter() - t0

# Métricas
clh = r.get('cl_history')
if clh is not None and len(clh) > 8:
    t_vec, cd, cl = clh[:, 0], clh[:, 1], clh[:, 2]
    half = len(t_vec) // 2
    st = strouhal(cl, r['dt'], 1.0, 1.0)
    print(f"\n===== RESULTADOS =====")
    print(f"Malha:        {r['N']}² celulas ({Ndom}x{Ndom} subd, nloc={nloc})")
    print(f"Passos:       {r['nsteps']}  (dt={r['dt']:.4f})")
    print(f"Tempo:        {wall_t:.0f}s")
    print(f"C_D (medio):  {cd[half:].mean():.4f}")
    print(f"C_L (rms):    {np.sqrt((cl[half:]**2).mean()):.4f}")
    print(f"Strouhal:     {st:.4f}")
    print(f"divmax:       {r['divmax']:.2e}")

frames = sorted(glob.glob(f"{OUTDIR}/frame_*.png"))
print(f"\n{len(frames)} frames. Gerando GIF...")
gif_path = make_gif(OUTDIR, GIF_OUT, fps=FPS)
print(f"✓ GIF: {gif_path}")