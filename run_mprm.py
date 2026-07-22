#!/usr/bin/env python3
"""
run_mprm.py — Método de Projeção Multiescala (MPrM) para Navier-Stokes.

Script standalone que implementa todos os benchmarks e visualizações do
notebook mprm_notebook.ipynb, com suporte a execução sequencial e MPI.

Uso:
  python run_mprm.py                          # todos os benchmarks (sequencial)
  python run_mprm.py --hh                     # apenas Helmholtz-Hodge
  python run_mprm.py --cavity                 # apenas cavidade (MPrM + fino)
  python run_mprm.py --anim                   # apenas animação (GIF)
  python run_mprm.py --cavity --no-fine       # cavidade MPrM sem referência fina
  python run_mprm.py --plot-dir resultados    # diretório de saída customizado

  mpiexec -n 4 python run_mprm.py --cavity    # execução paralela com MPI

Flags:
  --all        Executa todos os benchmarks (padrão)
  --hh         Benchmark de decomposição Helmholtz-Hodge
  --cavity     Benchmark de cavidade com tampa deslizante
  --anim       Gerar animação (GIF) da evolução temporal
  --fine       Incluir referência de malha fina (mais lento)
  --no-fine    Pular referência de malha fina
  --plot-dir   Diretório para salvar gráficos (padrão: .)
  --frames-dir Diretório para frames da animação (padrão: frames_mprm)

Baseado em:
  Lages, C. — The multiscale projection method for incompressible Navier-Stokes
  Guiraldello et al. (2018) — MRCM, J. Comp. Phys. 355:1–21
  Chorin (1968) — Numerical solution of the Navier-Stokes equations
  Ghia et al. (1982) — High-Re solutions... J. Comp. Phys. 48:387–411
"""

from __future__ import annotations
import sys, os, time, argparse, glob
import numpy as np

# ---- paths ----
HERE = os.path.dirname(os.path.abspath(__file__))
MRCM_REPO = os.path.expanduser("~/mrcm/mrcm-firedrake")
for p in (HERE, MRCM_REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

# ---- MPI (opcional) ----
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank, size = comm.rank, comm.size
except ImportError:
    class _DummyComm:
        rank = 0; size = 1
        def allgather(self, x): return [x]
        def bcast(self, x, root=0): return x
        def gather(self, x, root=0): return [x]
    comm = _DummyComm()
    rank, size = 0, 1

# ---- matplotlib (modo não-interativo) ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ---- mrcm / mprm_core ----
from mrcm.config import MRCMConfig
from mprm_core import (
    mac_wstar, mac_divergence,
    MPrMProjector, MACDownscaler,
    mprm_projection_step,
    FineProjector, run_cavity,
    hh_benchmark_mac, hh_p_np, hh_gradp_np,
    olshanskii_u_np, A_OLSH,
    make_frame_saver, make_gif,
)


# =============================================================================
# Parâmetros — ajuste aqui antes de executar
# =============================================================================

# --- Domínio ---
Lx, Ly = 1.0, 1.0

# --- Decomposição de domínio (MRCM) ---
Ndom = 3                     # N_H: subdomínios por direção
nloc = 16                    # n_loc: células finas por subdomínio

# --- Espaços de interface (MRCM) ---
degreeU = 1                  # k_U - 1
degreeP = 1                  # k_P - 1

# --- Parâmetro de Robin ---
alpha = 1.0

# --- Fluido ---
Re = 100.0
lid_velocity = 1.0

# --- Tempo ---
T = 16.0                      # Tempo final para benchmark

# --- Animação ---
ANIM_Ndom = 3
ANIM_nloc = 12
ANIM_Re = 100.0
ANIM_T = 6.0
ANIM_frame_every = 15
ANIM_fps = 10


# =============================================================================
# Helpers
# =============================================================================

def _banner(msg):
    if rank == 0:
        print("\n" + "=" * 60)
        print(msg)
        print("=" * 60)


def _compute_dt(N, Re, lid_vel=1.0):
    h = 1.0 / N
    nu = lid_vel / Re
    dt_c = h / lid_vel
    dt_d = h * h / (4.0 * nu)
    return 0.4 * min(dt_c, dt_d)


def _make_config():
    return MRCMConfig(
        Ndomx=Ndom, Ndomy=Ndom,
        nlocx=nloc, nlocy=nloc,
        degreeU=degreeU, degreeP=degreeP,
        alpha=alpha,
        mixed_family="BDM", mixed_order=0,
        executor_kind="sequential",
    )


def _print_config(N, dt):
    if rank != 0:
        return
    cfg = _make_config()
    h = 1.0 / N
    nu = lid_velocity / Re
    print(f"  Malha global:           {N}×{N} = {N*N} células")
    print(f"  Subdomínios:            {cfg.Nsubd} ({Ndom}×{Ndom})")
    print(f"  Células/subdomínio:     {nloc}×{nloc}")
    print(f"  Tamanho subdomínio H:   {cfg.Hx:.4f}×{cfg.Hy:.4f}")
    print(f"  Tamanho célula fina h:  {h:.6f}")
    print(f"  Interface DOFs:         k_U={cfg.ndofsU}, k_P={cfg.ndofsP}")
    print(f"  Viscosidade ν:          {nu:.6f}")
    print(f"  δt (CFL):               {dt:.6f}")
    print(f"  Reynolds:               {Re}")


# =============================================================================
# Benchmark 1: Decomposição de Helmholtz-Hodge
# =============================================================================

def run_hh_benchmark(plot_dir="."):
    _banner("Benchmark 1: Decomposição de Helmholtz-Hodge (Olshanskii)")

    result = hh_benchmark_mac(
        Ndom=Ndom, nloc=nloc,
        degreeU=degreeU, degreeP=degreeP,
        alpha=alpha, comm=comm,
    )

    if rank == 0:
        print(f"  N = {result['N']}")
        print(f"  Erro relativo u:        {result['err_u']:.6e}")
        print(f"  max|div u_h|:            {result['divmax']:.6e}")
        print(f"  Defeito compatibilidade: {result['defect']:.6e}")
        ok = result['err_u'] < 0.1 and result['divmax'] < 1e-10
        print(f"  {'✓ PASSOU' if ok else '✗ FALHOU'}")

        # ---- plot ----
        uh1, uh2 = result["u"]
        ue1, ue2 = result["u_exact"]
        N_hh = result["N"]
        xc = (np.arange(N_hh) + 0.5) / N_hh
        yc = (np.arange(N_hh) + 0.5) / N_hh

        uc = 0.5 * (uh1[:-1, :] + uh1[1:, :])
        vc = 0.5 * (uh2[:, :-1] + uh2[:, 1:])
        ue_c = 0.5 * (ue1[:-1, :] + ue1[1:, :])
        ve_c = 0.5 * (ue2[:, :-1] + ue2[:, 1:])

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        mag_ex = np.sqrt(ue_c**2 + ve_c**2)
        im0 = axes[0].pcolormesh(xc, yc, mag_ex.T, shading='auto', cmap='viridis')
        axes[0].quiver(xc[::3], yc[::3], ue_c[::3,::3].T, ve_c[::3,::3].T,
                       scale=2, width=0.003, color='white', alpha=0.7)
        axes[0].set_title(f"u exato (Olshanskii, a={A_OLSH})")
        axes[0].set_xlabel("x"); axes[0].set_ylabel("y"); axes[0].set_aspect('equal')
        plt.colorbar(im0, ax=axes[0], label='|u|')

        mag_rh = np.sqrt(uc**2 + vc**2)
        im1 = axes[1].pcolormesh(xc, yc, mag_rh.T, shading='auto', cmap='viridis')
        axes[1].quiver(xc[::3], yc[::3], uc[::3,::3].T, vc[::3,::3].T,
                       scale=2, width=0.003, color='white', alpha=0.7)
        axes[1].set_title(f"u_h (MPrM, {Ndom}×{Ndom} subdomínios)")
        axes[1].set_xlabel("x"); axes[1].set_ylabel("y"); axes[1].set_aspect('equal')
        plt.colorbar(im1, ax=axes[1], label='|u_h|')

        err = np.sqrt((uc - ue_c)**2 + (vc - ve_c)**2)
        im2 = axes[2].pcolormesh(xc, yc, err.T, shading='auto', cmap='Reds',
                                 norm=mcolors.LogNorm(vmin=1e-6, vmax=err.max()))
        axes[2].set_title(f"|u_h − u| (erro rel = {result['err_u']:.2e})")
        axes[2].set_xlabel("x"); axes[2].set_ylabel("y"); axes[2].set_aspect('equal')
        plt.colorbar(im2, ax=axes[2], label='|erro|')

        plt.tight_layout()
        path = os.path.join(plot_dir, "hh_benchmark.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Figura salva: {path}")

    return result


# =============================================================================
# Benchmark 2: Cavidade
# =============================================================================

def run_cavity_benchmark(run_fine=True, plot_dir="."):
    N = Ndom * nloc
    dt = _compute_dt(N, Re, lid_velocity)
    nsteps = int(np.ceil(T / dt))

    _banner(f"Benchmark 2: Cavidade (MPrM) — Re={Re}, N={N}², subdomínios={Ndom}²")
    _print_config(N, dt)

    t0 = time.perf_counter()
    result_mprm = run_cavity(
        Re=Re, Ndom=Ndom, nloc=nloc, T=T, dt=dt,
        lid_velocity=lid_velocity,
        degreeU=degreeU, degreeP=degreeP, alpha=alpha,
        method="mprm", comm=comm, verbose=(rank == 0),
        log_every=max(1, nsteps // 10),
    )
    t_wall = time.perf_counter() - t0

    if rank == 0:
        print(f"\n  --- Resultados MPrM ---")
        print(f"  Tempo de parede:        {t_wall:.2f} s")
        print(f"  Passos:                 {result_mprm['nsteps']}")
        print(f"  Tempo/passo:            {t_wall/result_mprm['nsteps']:.4f} s")
        print(f"  max|div u|:              {result_mprm['divmax']:.6e}")
        print(f"  Defeito downscaling:     {result_mprm['defect']:.6e}")

    result_fine = None
    if run_fine:
        _banner(f"Referência: Malha Fina — Re={Re}, N={N}²")
        t0 = time.perf_counter()
        result_fine = run_cavity(
            Re=Re, Ndom=Ndom, nloc=nloc, T=T, dt=dt,
            lid_velocity=lid_velocity,
            method="fine", comm=comm, verbose=(rank == 0),
            log_every=max(1, nsteps // 10),
        )
        t_wall_f = time.perf_counter() - t0

        if rank == 0:
            print(f"\n  --- Resultados Malha Fina ---")
            print(f"  Tempo de parede:    {t_wall_f:.2f} s")
            print(f"  max|div u|:          {result_fine['divmax']:.6e}")
            u_m, v_m = result_mprm["u"], result_mprm["v"]
            u_f, v_f = result_fine["u"], result_fine["v"]
            err = np.sqrt(np.sum((u_m-u_f)**2) + np.sum((v_m-v_f)**2))
            den = np.sqrt(np.sum(u_f**2) + np.sum(v_f**2))
            print(f"  Erro relativo MPrM × Fino: {err/den:.6e}")

    # ---- plots ----
    if rank == 0:
        _plot_cavity_fields(result_mprm, plot_dir)
        _plot_cavity_profiles(result_mprm, plot_dir)

    return result_mprm, result_fine


def _plot_cavity_fields(result, plot_dir):
    u, v = result["u"], result["v"]
    N_v = result["N"]
    xc = (np.arange(N_v) + 0.5) / N_v
    yc = (np.arange(N_v) + 0.5) / N_v
    uc = 0.5 * (u[:-1, :] + u[1:, :])
    vc = 0.5 * (v[:, :-1] + v[:, 1:])
    mag = np.sqrt(uc**2 + vc**2)

    # função corrente
    psi = np.zeros((N_v + 1, N_v + 1))
    for i in range(N_v):
        psi[i + 1, :] = psi[i, :] - 0.5 * (vc[i, :] + vc[i + 1, :]) / N_v

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    im0 = axes[0].pcolormesh(xc, yc, mag.T, shading='auto', cmap='turbo')
    axes[0].quiver(xc[::2], yc[::2], uc[::2,::2].T, vc[::2,::2].T,
                   scale=3, width=0.003, color='white', alpha=0.6)
    axes[0].set_title(f"|u| — MPrM, Re={Re}, t={T}")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y"); axes[0].set_aspect('equal')
    plt.colorbar(im0, ax=axes[0], label='|u|')

    im1 = axes[1].contourf(xc, yc, psi.T, levels=30, cmap='Blues')
    axes[1].contour(xc, yc, psi.T, levels=30, colors='k', linewidths=0.3)
    axes[1].set_title(f"ψ — MPrM, Re={Re}")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y"); axes[1].set_aspect('equal')
    plt.colorbar(im1, ax=axes[1], label='ψ')

    div = mac_divergence(u, v, 1.0/N_v, 1.0/N_v)
    im2 = axes[2].pcolormesh(xc, yc, div.T, shading='auto', cmap='RdBu_r',
                             norm=mcolors.SymLogNorm(linthresh=1e-14,
                                                     vmin=-1e-12, vmax=1e-12))
    axes[2].set_title(f"∇·u (max = {result['divmax']:.2e})")
    axes[2].set_xlabel("x"); axes[2].set_ylabel("y"); axes[2].set_aspect('equal')
    plt.colorbar(im2, ax=axes[2], label='∇·u')

    plt.tight_layout()
    path = os.path.join(plot_dir, "cavity_fields.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Figura salva: {path}")


def _plot_cavity_profiles(result, plot_dir):
    u, v = result["u"], result["v"]
    N_v = result["N"]
    xc = (np.arange(N_v) + 0.5) / N_v
    yc = (np.arange(N_v) + 0.5) / N_v
    ix_mid = N_v // 2
    iy_mid = N_v // 2
    u_mid = u[ix_mid, :]
    v_mid = v[:, iy_mid]

    # Ghia et al. (1982) — Re = 100
    ghia_y = np.array([0.0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813,
                       0.4531, 0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609,
                       0.9688, 0.9766, 1.0])
    ghia_u = np.array([0.0, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150,
                       -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151,
                       0.68717, 0.73722, 0.78871, 0.84123, 1.0])
    ghia_x = np.array([0.0, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266,
                       0.2344, 0.5, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531,
                       0.9609, 0.9688, 1.0])
    ghia_v = np.array([0.0, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077,
                       0.17507, 0.17527, 0.05454, -0.24533, -0.22445, -0.16914,
                       -0.10313, -0.08864, -0.07391, -0.05906, 0.0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(u_mid, yc, 'b-', linewidth=2, label='MPrM')
    ax1.plot(ghia_u, ghia_y, 'ro', markersize=4, label='Ghia et al. (1982)')
    ax1.set_xlabel('u'); ax1.set_ylabel('y')
    ax1.set_title(f'u(x=0.5, y) — Re={Re}')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(xc, v_mid, 'b-', linewidth=2, label='MPrM')
    ax2.plot(ghia_x, ghia_v, 'ro', markersize=4, label='Ghia et al. (1982)')
    ax2.set_xlabel('x'); ax2.set_ylabel('v')
    ax2.set_title(f'v(x, y=0.5) — Re={Re}')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plot_dir, "cavity_profiles.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Figura salva: {path}")


# =============================================================================
# Animação
# =============================================================================

def run_animation(plot_dir=".", frames_dir="frames_mprm"):
    _banner(f"Animação — Re={ANIM_Re}, "
            f"malha={(ANIM_Ndom*ANIM_nloc)}², T={ANIM_T}")

    if rank == 0:
        try:
            import imageio
        except ImportError:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install",
                                   "imageio", "-q"])
        print("  Salvando frames...")

    saver = make_frame_saver(
        output_dir=frames_dir, dpi=100,
        Re=ANIM_Re, title_prefix="MPrM",
    )

    result = run_cavity(
        Re=ANIM_Re, Ndom=ANIM_Ndom, nloc=ANIM_nloc, T=ANIM_T,
        lid_velocity=1.0,
        degreeU=degreeU, degreeP=degreeP, alpha=alpha,
        method="mprm", comm=comm, verbose=(rank == 0),
        log_every=50,
        frame_callback=saver,
        frame_every=ANIM_frame_every,
    )

    if rank == 0:
        nf = len(glob.glob(f"{frames_dir}/frame_*.png"))
        print(f"  ✓ {nf} frames salvos em '{frames_dir}/'")
        print(f"  Tempo: {result['t_wall']:.2f} s, "
              f"max|div u|: {result['divmax']:.2e}")

        gif_path = os.path.join(plot_dir, "cavity_mprm.gif")
        make_gif(frame_dir=frames_dir, output=gif_path, fps=ANIM_fps)
        print(f"  ✓ GIF: {gif_path}")

    return result


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="MPrM — Método de Projeção Multiescala para Navier-Stokes")
    g = p.add_argument_group("Benchmarks")
    g.add_argument("--all", action="store_true", default=True,
                   help="Executar todos os benchmarks (padrão)")
    g.add_argument("--hh", action="store_true",
                   help="Benchmark Helmholtz-Hodge")
    g.add_argument("--cavity", action="store_true",
                   help="Benchmark cavidade")
    g.add_argument("--anim", action="store_true",
                   help="Gerar animação (GIF)")
    g.add_argument("--fine", action="store_true", default=True,
                   help="Incluir referência de malha fina (padrão)")
    g.add_argument("--no-fine", action="store_false", dest="fine",
                   help="Pular referência de malha fina")
    g.add_argument("--plot-dir", default=".",
                   help="Diretório para gráficos")
    g.add_argument("--frames-dir", default="frames_mprm",
                   help="Diretório para frames da animação")
    return p.parse_args()


def main():
    args = parse_args()

    # Se flags específicas foram passadas, desativa --all
    if args.hh or args.cavity or args.anim:
        args.all = False

    if rank == 0:
        print("=" * 60)
        print("MPrM — Método de Projeção Multiescala")
        print(f"MPI: {size} rank(s)")
        print("=" * 60)

    if args.all or args.hh:
        run_hh_benchmark(args.plot_dir)

    if args.all or args.cavity:
        run_cavity_benchmark(run_fine=args.fine, plot_dir=args.plot_dir)

    if args.all or args.anim:
        run_animation(args.plot_dir, args.frames_dir)

    if rank == 0:
        print("\n" + "=" * 60)
        print("Execução concluída.")
        print("=" * 60)


if __name__ == "__main__":
    main()
