"""quadtree_vk_gif.py — von Kármán com decomposição quadtree adaptativa + GIF.

Uso:
  python quadtree_vk_gif.py                  # sequencial (1 rank)
  mpiexec -n 4 python quadtree_vk_gif.py     # paralelo (4 ranks)

Arquitetura:
  - Cada rank constrói a árvore independentemente (determinística)
  - Projetor MRCM: subdomínios distribuídos round-robin entre ranks
  - Rank 0: I/O (frames, GIF, métricas)
  - assert_consistent_across_ranks impede divergência silenciosa

Família (b): corpo como bloco 2×2 de folhas — refino alcança a superfície.
"""
import numpy as np
import os
import glob
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpi4py import MPI

from mprm.decomposition import (Decomposition, plan_quadtree_geometry,
                                build_obstacle_tree)
from mprm.problem import von_karman_problem
from mprm.flow import run_von_karman
from mprm.forces import strouhal
from mprm.viz import make_gif

# ── Configuração ───────────────────────────────────────────────────────────

comm = MPI.COMM_WORLD
rank = comm.rank
n_ranks = comm.size

# Parâmetros físicos
Re = 100.0
U_in = 1.0
D = 1.0
Ly = 8.0                # altura do canal (Ly/D = 8)

# ── Escolha da família ────────────────────────────────────────────────────
# Família (a): corpo contido em 1 folha travada (N_by ímpar, H0 > D).
#   ~120 folhas com max_level=2. Mais rápido, refino em anéis concêntricos.
# Família (b): corpo como bloco 2×2 exato (N_by par). Refino até a superfície.
#   ~1500 folhas com max_level=3. Mais caro, geometricamente mais fiel.

# --- Descomente a configuração desejada ---

# [RÁPIDO] Família (a) — ~120 folhas, smoke test / iteração rápida
config = "a"
N_bx, N_by = 10, 3
max_level = 2
nloc_min = 8
radii = (6.0, 3.0)
wake_length_val = 0.0

# [COMPLETO] Família (b) — ~1500 folhas, refino na superfície (mais pesado)
# config = "b"
# N_bx, N_by = 10, 4
# max_level = 3
# nloc_min = 6
# radii = (12.0, 6.0, 3.0)
# wake_length_val = 6.0

# Parâmetros numéricos
degreeU = 3
degreeP = 3
alpha = 1.0
T = 20.0                # tempo final de simulação

# Saída
OUTDIR = "frames_quadtree_vk"
GIF_OUT = "results_alpha/quadtree_von_karman.gif"
FPS = 10
FRAME_EVERY = 15        # intervalo entre frames (passos de tempo)

# ── Construção da decomposição ─────────────────────────────────────────────

if rank == 0:
    print("=" * 60)
    print(f"von Kármán QUADTREE — Re={Re}, família ({config}), "
          f"max_level={max_level}")
    print(f"{n_ranks} rank(s) MPI")
    print("=" * 60)

# Geometria diádica
g = plan_quadtree_geometry(
    D=D, Ly=Ly, N_bx=N_bx, N_by=N_by,
    max_level=max_level, nloc_min=nloc_min,
    family=config, degreeU=degreeU, degreeP=degreeP,
)

# Árvore refinada em torno do obstáculo (com esteira)
qt = build_obstacle_tree(g, radii=radii, wake_length=wake_length_val)

# Problema físico
problem = von_karman_problem(
    Lx=g["Lx"], Ly=g["Ly"], Nx=g["Nx"], Ny=g["Ny"],
    D=g["D"], xc=g["xc"], yc=g["yc"],
    U_in=U_in, Re=Re,
    inflow="parabolic", walls="slip", inflow_norm="max",
)

# Decomposição MPrM a partir da árvore
d = Decomposition.from_quadtree(
    qt, h=g["h"], Lx=g["Lx"], Ly=g["Ly"], solid=problem.solid,
)

if rank == 0:
    niveis = sorted({lf.level for lf in d.leaves})
    print(f"Decomposição: {len(d.leaves)} folhas, níveis {niveis}")
    print(f"  Nx×Ny = {g['Nx']}×{g['Ny']}, h = {g['h']:.4f}")
    print(f"  H = [{', '.join(f'{x:.3g}' for x in g['H'])}]")
    print(f"  nloc = {g['nloc']}")
    print(f"  n_gamma_edges = {len(d.gamma_edges())}")
    print(f"  H_min = {min(lf.Hx for lf in d.leaves):.3g}, "
          f"H_max = {max(lf.Hx for lf in d.leaves):.3g}")

# ── I/O ────────────────────────────────────────────────────────────────────

if rank == 0:
    os.makedirs(OUTDIR, exist_ok=True)
    for f in glob.glob(f"{OUTDIR}/frame_*.png"):
        os.remove(f)

frame_idx = [0]


def frame_callback(step, t, u, v, hx, hy):
    """Vorticidade + obstáculo. Chamado apenas no rank 0."""
    Nx, Ny = u.shape[0] - 1, v.shape[1] - 1
    dvdx = (v[1:, :] - v[:-1, :]) / hx
    dudy = (u[:, 1:] - u[:, :-1]) / hy
    vort = dvdx[:, 1:-1] - dudy[1:-1, :]

    fig, ax = plt.subplots(figsize=(14, 5))
    xc = (np.arange(1, Nx) + 0.5) * hx
    yc = (np.arange(1, Ny) + 0.5) * hy
    ax.pcolormesh(xc, yc, vort.T, shading="auto", cmap="RdBu_r",
                  vmin=-2, vmax=2)

    # Obstáculo
    Dc, xc0, yc0 = g["D"], g["xc"], g["yc"]
    ax.add_patch(plt.Rectangle(
        (xc0 - Dc / 2, yc0 - Dc / 2), Dc, Dc, color="k"))

    # Contorno das folhas do quadtree (leve, para visualizar a decomposição)
    for lf in d.leaves:
        ax.plot([lf.x0, lf.x1, lf.x1, lf.x0, lf.x0],
                [lf.y0, lf.y0, lf.y1, lf.y1, lf.y0],
                "w-", linewidth=0.15, alpha=0.3)

    ax.set_aspect("equal")
    ax.set_title(
        f"von Kármán — Quadtree (família {config}, nível {max_level})  "
        f"Re={Re}  U_H,{degreeU}/P_H,{degreeP}  "
        f"{g['Nx']}×{g['Ny']}  "
        f"{len(d.leaves)} folhas  "
        f"t={t:.1f}"
    )
    plt.colorbar(ax.collections[0], ax=ax, label=r"$\omega_z$", shrink=0.5)
    fname = os.path.join(OUTDIR, f"frame_{frame_idx[0]:05d}.png")
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    frame_idx[0] += 1


# ── Execução ────────────────────────────────────────────────────────────────

if rank == 0:
    print(f"\nT={T}, frame_every={FRAME_EVERY}")
    print(f"Iniciando...")

# run_von_karman com Ndomx/Ndomy/nloc dummy (substituídos pela decomposição)
t0 = time.perf_counter()
r = run_von_karman(
    Lx=g["Lx"], Ly=g["Ly"], D=g["D"], xc=g["xc"], yc=g["yc"],
    U_in=U_in, Re=Re,
    Ndomx=g["N_bx"], Ndomy=g["N_by"], nloc=g["nloc"][0],
    degreeU=degreeU, degreeP=degreeP, alpha=alpha,
    inflow="parabolic", walls="slip", inflow_norm="max",
    T=T, comm=comm, verbose=(rank == 0), log_every=500,
    decomposition=d,
    frame_callback=frame_callback, frame_every=FRAME_EVERY,
)
wall_t = time.perf_counter() - t0

# ── Métricas e GIF (rank 0) ────────────────────────────────────────────────

if rank == 0:
    clh = r.get("cl_history")
    if clh is not None and len(clh) > 8:
        t_vec, cd, cl = clh[:, 0], clh[:, 1], clh[:, 2]
        half = len(t_vec) // 2
        st = strouhal(cl, r["dt"], U_in, D)
        print(f"\n===== RESULTADOS =====")
        print(f"Malha:       {g['Nx']}×{g['Ny']} células")
        print(f"Folhas:      {len(d.leaves)} ({len(d.gamma_edges())} "
              f"arestas de Γ)")
        print(f"H:           {min(lf.Hx for lf in d.leaves):.3g} – "
              f"{max(lf.Hx for lf in d.leaves):.3g}")
        print(f"Passos:      {r['nsteps']} (dt={r['dt']:.4f})")
        print(f"Tempo:       {wall_t:.0f}s ({n_ranks} ranks MPI)")
        print(f"C_D médio:   {cd[half:].mean():.4f}")
        print(f"C_L amp:     {0.5 * (cl[half:].max() - cl[half:].min()):.4f}")
        print(f"St:          {st:.4f}")
        print(f"max|div u|:  {r['divmax']:.2e}")
        print(f"defeito:     {r['defect']:.2e}")
        info = r["decomposition"]
        print(f"cond(A_Γ):   {r.get('cond_A_gamma', float('nan'))}")

    frames = sorted(glob.glob(f"{OUTDIR}/frame_*.png"))
    n_frames = len(frames)
    if n_frames > 0:
        print(f"\n{n_frames} frames. Gerando GIF...")
        gif_path = make_gif(OUTDIR, GIF_OUT, fps=FPS)
        print(f"✓ {gif_path}")
    else:
        print("\nNenhum frame gerado (aumente T ou reduza frame_every).")
