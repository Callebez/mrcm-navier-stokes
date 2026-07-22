#!/usr/bin/env python3
"""tests/test_von_karman.py — von Kármán (cilindro quadrado, Re=100).

Benchmark da Tabela 3 do artigo do MPrM (Lages, §3.5), agora possível com a
API de BCs generalizada do mrcm (no-flow forte nos 4 lados + saída Dirichlet
de pressão). Duas facetas:

  * `test_von_karman_quick()` — smoke pytest: malha pequena, poucos passos.
    Verifica div≈0 no fluido, u·n=0 no obstáculo e balanço de massa
    entrada≈saída.
  * `main()` — estudo completo (CLI): transiente longo, mede C_D/C_L/St e
    salva figura de vortex shedding.

Uso:
  pytest tests/test_von_karman.py::test_von_karman_quick -q     # smoke
  python tests/test_von_karman.py                               # estudo
  python tests/test_von_karman.py --quick
  mpiexec -n 3 python tests/test_von_karman.py                  # paralelo
"""
from __future__ import annotations
import time
import numpy as np

from common import (get_comm, apply_plot_style, banner, make_meta,
                    save_json, save_figure, COLORS, INK, DEFAULT_OUTDIR)
from params import CommonParams, parse_args

from mprm_core import (run_von_karman, strouhal, von_karman_problem,
                       mac_divergence, solid_faces)

# Geometria default do benchmark (canal com cilindro quadrado).
# NOTA: xc,yc devem ficar no interior de um subdomínio (sem cruzar interfaces).
# Para Ndom=4 com Lx=16: subdomínios em x=[0,4),[4,8),[8,12),[12,16].
# Obstáculo centrado em xc=3.0 cabe no 1º subdomínio ([2.5,3.5] ⊂ [0,4)).
DEFAULT_GEOM = dict(Lx=16.0, Ly=8.0, D=1.0, xc=3.0, yc=3.0, U_in=1.0)


def _mass_balance(u, v, problem):
    """∮ u·n no contorno externo (entrada − saída); ~0 em incompressível."""
    hy, hx = problem.hy, problem.hx
    inflow = np.sum(u[0, :]) * hy
    outflow = np.sum(u[-1, :]) * hy
    top = np.sum(v[:, -1]) * hx
    bot = np.sum(v[:, 0]) * hx
    return inflow - outflow + bot - top


def test_von_karman_quick():
    """Smoke: estrutura + conservação (rápido, sequencial)."""
    comm = get_comm()
    # Geometria ajustada para caber no subdomínio com Ndom=2, nloc=8
    r = run_von_karman(Lx=8.0, Ly=4.0, D=1.0, xc=2.0, yc=1.0, U_in=1.0,
                       Re=100.0, Ndom=2, nloc=8, degreeU=1, degreeP=3,
                       T=0.3, comm=comm, verbose=False)
    if comm.rank != 0:
        return
    u, v = r["u"], r["v"]
    prob = von_karman_problem(Lx=8.0, Ly=4.0, Nx=16, Ny=16, D=1.0,
                              xc=2.0, yc=1.0, U_in=1.0, Re=100.0)
    # 1. div ~ 0 no fluido (longe do sólido e bordas)
    dv = np.abs(mac_divergence(u, v, prob.hx, prob.hy))
    # exclui sólido e bordas
    interior = np.ones((16, 16), dtype=bool)
    interior[0, :] = interior[-1, :] = False
    interior[:, 0] = interior[:, -1] = False
    interior[prob.solid] = False
    assert dv[interior].max() < 1e-7
    # 2. u·n = 0 nas faces do obstáculo
    fx, fy = solid_faces(prob.solid)
    assert np.all(u[fx] == 0.0) and np.all(v[fy] == 0.0)
    # 3. balanço de massa entrada≈saída
    assert abs(_mass_balance(u, v, prob)) < 1e-6
    assert np.isfinite(u).all() and np.isfinite(v).all()


def run_study(comm, geom, Re, Ndom, nloc, degreeU, degreeP, alpha, T, dt,
              verbose=True):
    """Roda o transiente completo e extrai C_D médio, C_L rms e Strouhal."""
    if verbose:
        banner(f"von Kármán Re={Re} — cilindro quadrado D={geom['D']} "
               f"({Ndom*nloc}² células, {Ndom}×{Ndom} subd., "
               f"U_H,{degreeU}/P_H,{degreeP})", comm)
    t0 = time.perf_counter()
    r = run_von_karman(**geom, Re=Re, Ndom=Ndom, nloc=nloc, degreeU=degreeU,
                       degreeP=degreeP, alpha=alpha, T=T, dt=dt, comm=comm,
                       verbose=verbose)
    res = {"meta": make_meta("von_karman", comm, Re=Re, Ndom=Ndom, nloc=nloc,
                             degreeU=degreeU, degreeP=degreeP, alpha=alpha,
                             T=T, dt=r["dt"], nsteps=r["nsteps"], **geom)}
    if comm.rank == 0:
        clh = r.get("cl_history")
        if clh is not None and len(clh) > 8:
            t, cd, cl = clh[:, 0], clh[:, 1], clh[:, 2]
            half = len(t) // 2                 # descarta transiente
            st = strouhal(cl, r["dt"], geom["U_in"], geom["D"])
            res["metrics"] = {
                "Cd_mean": float(cd[half:].mean()),
                "Cl_rms": float(np.sqrt((cl[half:] ** 2).mean())),
                "St": float(st)}
            res["cl_history"] = clh.tolist()
        res["divmax"] = r["divmax"]; res["wall_s"] = time.perf_counter() - t0
        res["u"], res["v"] = r["u"], r["v"]      # p/ a figura (removido no JSON)
    return res


def plot_shedding(res, geom, outdir):
    plt = apply_plot_style()
    u, v = res.pop("u"), res.pop("v")
    Nx = u.shape[0] - 1; Ny = v.shape[1] - 1
    hx = geom["Lx"] / Nx; hy = geom["Ly"] / Ny
    xc = (np.arange(Nx) + 0.5) * hx; yc = (np.arange(Ny) + 0.5) * hy
    # Vorticidade nos centros das células interiores
    dvdx = (v[1:, :] - v[:-1, :]) / hx      # (Nx-1, Ny+1)
    dudy = (u[:, 1:] - u[:, :-1]) / hy       # (Nx+1, Ny-1)
    vort = dvdx[:, 1:-1] - dudy[1:-1, :]     # (Nx-1, Ny-1)
    # xc_vort, yc_vort são os centros das células interiores
    xc_v = (np.arange(1, Nx) + 0.5) * hx
    yc_v = (np.arange(1, Ny) + 0.5) * hy
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.pcolormesh(xc_v, yc_v, vort.T, shading="auto",
                       cmap="RdBu_r", vmin=-2, vmax=2)
    ax.add_patch(plt.Rectangle((geom["xc"] - geom["D"] / 2,
                                geom["yc"] - geom["D"] / 2),
                               geom["D"], geom["D"], color="k"))
    ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
    m = res.get("metrics", {})
    ax.set_title(f"von Kármán Re={res['meta']['Re']:.0f} — vorticidade  "
                 f"(St={m.get('St', float('nan')):.3f}, "
                 f"C_D={m.get('Cd_mean', float('nan')):.3f})")
    plt.colorbar(im, ax=ax, label=r"$\omega_z$", shrink=0.7)
    return save_figure(fig, outdir, "von_karman_shedding.png")


def main(argv=None):
    defaults = CommonParams(Ndom=4, nloc=12, degreeU=1, degreeP=3, Re=100.0,
                            T=60.0, outdir=DEFAULT_OUTDIR)
    args = parse_args("von Kármán (cilindro quadrado) — MPrM", defaults=defaults,
                      argv=argv)
    geom = dict(DEFAULT_GEOM)
    if args.quick:
        Ndom, nloc, T = 2, 8, 1.0
        # Ajusta geometria para caber no subdomínio com Ndom=2
        geom["yc"] = 1.0  # longe da interface y=4.0
        geom["xc"] = 2.0  # longe da interface x=8.0
    else:
        Ndom, nloc, T = args.Ndom, args.nloc, args.T
    comm = get_comm()
    res = run_study(comm, geom, args.Re, Ndom, nloc, args.degreeU,
                    args.degreeP, args.alpha, T, args.dt, verbose=args.verbose)
    if comm.rank == 0:
        outdir = args.outdir
        fig = plot_shedding(res, geom, outdir)
        save_json(res, outdir, "von_karman.json")
        m = res.get("metrics", {})
        print(f"\nC_D={m.get('Cd_mean', float('nan')):.4f}  "
              f"C_L,rms={m.get('Cl_rms', float('nan')):.4f}  "
              f"St={m.get('St', float('nan')):.4f}  (alvo St≈0.121)")
        print(f"figura: {fig}")
    return res


if __name__ == "__main__":
    main()
