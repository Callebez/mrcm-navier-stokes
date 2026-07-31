"""Laços temporais e passo completo do MPrM.

mprm_projection_step, run_flow, run_cavity, run_von_karman.
"""
from __future__ import annotations
import time
import numpy as np
from mpi4py import MPI

from mrcm.config import MRCMConfig
from mprm.problem import cavity_problem, von_karman_problem, FlowProblem
from mprm.transport import (mac_transport, mac_divergence, solid_faces,
                            _apply_external_normals_after_step)
from mprm.projector import MPrMProjector
from mprm.downscaler import MACDownscaler
from mprm.forces import (ChannelFinePoisson, DiagnosticPressureSolver,
                         obstacle_forces, strouhal)

# Import local para _diagnostic_pressure (retrocompatibilidade)
from mprm.forces import DiagnosticPressureSolver as _DiagSolver
def _diagnostic_pressure(u, v, problem, dt):
    return _DiagSolver(problem).solve(u, v)

# Passo completo do MPrM e laços temporais
# ===========================================================================


def mprm_projection_step(w1, w2, projector: "MPrMProjector",
                         downscaler: "MACDownscaler"):
    """Passos 2-4 do MPrM. Lê `projector.problem` (obstáculo + saída).
    Assinatura inalterada — retrocompatível."""
    cfg = projector.cfg
    comm = projector.comm
    problem = projector.problem
    n, m = cfg.nlocx, cfg.nlocy
    hx, hy = downscaler.hx, downscaler.hy
    solid = problem.solid

    divw = mac_divergence(w1, w2, hx, hy)
    if solid is not None:
        divw = divw.copy(); divw[solid] = 0.0
    fluxes = projector.project(divw)
    out_fluxes = getattr(projector, "last_boundary_fluxes", {})

    u = w1.copy(); v = w2.copy()
    zeros_m, zeros_n = np.zeros(m), np.zeros(n)
    my_patch = {}
    keep_pot = getattr(downscaler, "keep_potential", False)
    my_pot = {}
    if keep_pot:
        downscaler.last_potential = {}
    for isub in projector.my_subs:
        ix, iy = isub % cfg.Ndomx, isub // cfg.Ndomx
        ox, oy = ix * n, iy * m
        fW = fluxes.get(("v", ix, iy), zeros_m)
        fE = fluxes.get(("v", ix + 1, iy), zeros_m)
        fS = fluxes.get(("h", ix, iy), zeros_n)
        fN = fluxes.get(("h", ix, iy + 1), zeros_n)
        # fluxos externos de SAÍDA (v·n reconstruído nas faces outflow)
        if (isub, "right") in out_fluxes:
            fE = out_fluxes[(isub, "right")]
        if (isub, "left") in out_fluxes:
            fW = out_fluxes[(isub, "left")]
        if (isub, "top") in out_fluxes:
            fN = out_fluxes[(isub, "top")]
        if (isub, "bottom") in out_fluxes:
            fS = out_fluxes[(isub, "bottom")]
        solid_loc = None
        if solid is not None:
            sl = solid[ox:ox + n, oy:oy + m]
            solid_loc = sl if sl.any() else None
        vt1, vt2 = downscaler.solve_patch(
            divw[ox:ox + n, oy:oy + m], fW, fE, fS, fN, solid_loc=solid_loc)
        my_patch[isub] = (vt1, vt2)
        if keep_pot:
            my_pot[isub] = downscaler._last_p

    if keep_pot:
        for d in comm.allgather(my_pot):
            downscaler.last_potential.update(d)

    for d in comm.allgather(my_patch):
        for isub, (vt1, vt2) in d.items():
            ix, iy = isub % cfg.Ndomx, isub // cfg.Ndomx
            ox, oy = ix * n, iy * m
            u[ox:ox + n + 1, oy:oy + m] = w1[ox:ox + n + 1, oy:oy + m] - vt1
            v[ox:ox + n, oy:oy + m + 1] = w2[ox:ox + n, oy:oy + m + 1] - vt2

    # contorno externo por lado (impermeável => 0; saída/entrada preservam w*)
    _apply_external_normals_after_step(u, v, problem)
    if solid is not None:
        fx, fy = solid_faces(solid)
        u[fx] = 0.0; v[fy] = 0.0
    return u, v


class FineProjector:
    """Referência: projeção pela malha fina global (Poisson MAC 5 pontos,
    Neumann homogêneo, pino de pressão) — o 'fine mesh solver' do artigo."""

    def __init__(self, Nx, Ny, hx, hy, dt):
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        self.Nx, self.Ny = Nx, Ny
        self.hx, self.hy = hx, hy
        self.dt = float(dt)

        def lid(i, j):
            return i + j * Nx

        rows, cols, vals = [], [], []
        for j in range(Ny):
            for i in range(Nx):
                c = lid(i, j)
                diag = 0.0
                for (di, dj, d, ln) in ((1, 0, hx, hy), (-1, 0, hx, hy),
                                        (0, 1, hy, hx), (0, -1, hy, hx)):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < Nx and 0 <= jj < Ny:
                        w = dt * ln / d
                        rows.append(c); cols.append(lid(ii, jj)); vals.append(w)
                        diag -= w
                rows.append(c); cols.append(c); vals.append(diag)
        A = sp.csr_matrix((vals, (rows, cols)), shape=(Nx * Ny, Nx * Ny)).tolil()
        A[0, :] = 0.0
        A[0, 0] = 1.0
        self.lu = spla.splu(A.tocsc())

    def step(self, w1, w2):
        dt, hx, hy = self.dt, self.hx, self.hy
        divw = mac_divergence(w1, w2, hx, hy)
        b = (divw * hx * hy).flatten(order="F")
        b -= b.sum() / b.size
        b[0] = 0.0
        p = self.lu.solve(b).reshape((self.Nx, self.Ny), order="F")
        u = w1.copy()
        v = w2.copy()
        u[1:-1, :] -= dt * (p[1:, :] - p[:-1, :]) / hx
        v[:, 1:-1] -= dt * (p[:, 1:] - p[:, :-1]) / hy
        u[0, :] = u[-1, :] = 0.0
        v[:, 0] = v[:, -1] = 0.0
        return u, v


def _seed_inflow(u, w1_0, problem):
    """Semeia a face de entrada no campo inicial (evita transiente de partida
    com entrada nula)."""
    if problem.side("left").kind == "inflow":
        u = u.copy(); u[0, :] = w1_0[0, :]
    return u


def _sample_probe(u, v, problem, px, py):
    """v-velocidade interpolada no ponto (px,py) — sonda de esteira p/ Strouhal."""
    hx, hy = problem.hx, problem.hy
    j = min(int(py / hy), problem.Ny - 1)
    i = min(int(px / hx), problem.Nx - 1)
    return 0.5 * (v[i, j] + v[i, j + 1])


def run_flow(problem, Ndomx, Ndomy, nloc, T, dt=None, method="mprm",
             degreeU=1, degreeP=1, alpha=1.0, basis_kind="legendre_P",
             eps_solid=1e-6, comm=None, verbose=True, log_every=200,
             frame_callback=None, frame_every=0, probes=None,
             obstacle_metrics=False, obstacle_metrics_every=1,
             convective_order=2, limiter="minmod", U_ref=1.0, D=1.0):
    """Laço temporal genérico do MPrM para um FlowProblem. Cavidade e von
    Kármán compartilham este caminho. Devolve dict(u,v,dt,nsteps,divmax,
    defect,t_wall[,cl_history,probe_history])."""
    import time
    if comm is None:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD

    Nx, Ny = problem.Nx, problem.Ny
    hx, hy = problem.hx, problem.hy
    if dt is None:
        umax = max(U_ref, 1e-12)
        nu = max(problem.nu, 1e-12)
        h_min = min(hx, hy)
        dt = 0.4 * min(h_min / umax, h_min * h_min / (4.0 * nu))
    nsteps = int(np.ceil(T / dt))

    if method == "mprm":
        cfg = MRCMConfig(Ndomx=Ndomx, Ndomy=Ndomy, nlocx=nloc, nlocy=nloc,
                         degreeU=degreeU, degreeP=degreeP, alpha=alpha,
                         basis_kind=basis_kind, Lx=problem.Lx, Ly=problem.Ly,
                         mixed_family="BDM", mixed_order=0,
                         executor_kind="sequential")
        projector = MPrMProjector(cfg, dt, comm=comm,
                                  problem=problem, eps_solid=eps_solid)
        downscaler = MACDownscaler(cfg, dt)
    else:
        fine = ChannelFinePoisson(problem, dt)
        projector = None
        downscaler = None

    u = np.zeros((Nx + 1, Ny)); v = np.zeros((Nx, Ny + 1))
    # inicializa a entrada em w* (campo inicial = perfil de entrada nas faces)
    w1_0, w2_0 = mac_transport(u, v, problem, dt, convective_order=convective_order, limiter=limiter)
    u = _seed_inflow(u, w1_0, problem)

    cl_hist = []
    probe_hist = [] if probes else None

    # Cache da fatoração LU da pressão diagnóstica (T5)
    diag_solver = None
    if obstacle_metrics and comm.rank == 0:
        diag_solver = DiagnosticPressureSolver(problem)

    t0 = time.perf_counter()
    u_prev, v_prev = u.copy(), v.copy()  # para termo transiente nas forças
    for step in range(nsteps):
        w1, w2 = mac_transport(u, v, problem, dt, convective_order=convective_order, limiter=limiter)
        if method == "mprm":
            u, v = mprm_projection_step(w1, w2, projector, downscaler)
        else:
            u, v = fine.step(w1, w2)

        if comm.rank == 0:
            if obstacle_metrics and problem.solid is not None and step % obstacle_metrics_every == 0:
                p = diag_solver.solve(u, v)
                cd, cl = obstacle_forces(u, v, p, problem, U_ref, D,
                                         u_prev=u_prev, v_prev=v_prev, dt=dt)
                cl_hist.append((dt * (step + 1), cd, cl))
            if probes:
                probe_hist.append(
                    [dt * (step + 1)] + [_sample_probe(u, v, problem, px, py)
                                         for (px, py) in probes])
        if frame_every > 0 and frame_callback is not None and comm.rank == 0:
            if step == 0 or (step + 1) % frame_every == 0:
                frame_callback(step, dt * (step + 1), u, v, hx, hy)
        if verbose and comm.rank == 0 and (step + 1) % log_every == 0:
            dv = np.abs(mac_divergence(u, v, hx, hy))
            dv = dv[~problem.solid].max() if problem.solid is not None else dv.max()
            print(f"  passo {step+1}/{nsteps}  t={dt*(step+1):.3f}  "
                  f"max|div u| = {dv:.2e}", flush=True)
        u_prev, v_prev = u.copy(), v.copy()
    t_wall = time.perf_counter() - t0

    dvall = np.abs(mac_divergence(u, v, hx, hy))
    divmax = float(dvall[~problem.solid].max() if problem.solid is not None
                   else dvall.max())
    out = {"u": u, "v": v, "dt": dt, "nsteps": nsteps, "N": Nx,
           "t_wall": t_wall, "divmax": divmax,
           "defect": (downscaler.last_defect if method == "mprm" else 0.0)}
    if cl_hist:
        out["cl_history"] = np.array(cl_hist)     # colunas: t, Cd, Cl
    if probe_hist is not None:
        out["probe_history"] = np.array(probe_hist)
    return out


def run_cavity(Re=100.0, Ndom=3, nloc=16, T=8.0, dt=None, lid_velocity=1.0,
               degreeU=1, degreeP=1, alpha=1.0, method="mprm",
               basis_kind="nodal_Q",
               comm=None, verbose=True, log_every=200,
               frame_callback=None, frame_every=0):
    """Cavidade com tampa deslizante (wrapper sobre run_flow). Assinatura e
    retorno preservados para compatibilidade com testes/notebook."""
    N = Ndom * nloc
    problem = cavity_problem(N, Re, lid_velocity=lid_velocity)
    if dt is None:
        hx = 1.0 / N
        dt = 0.4 * min(hx / lid_velocity, hx * hx / (4.0 * (lid_velocity / Re)))
    return run_flow(problem, Ndom, Ndom, nloc, T, dt=dt, method=method,
                    degreeU=degreeU, degreeP=degreeP, alpha=alpha,
                    basis_kind=basis_kind, comm=comm, verbose=verbose,
                    log_every=log_every, frame_callback=frame_callback,
                    frame_every=frame_every, U_ref=lid_velocity)


def run_von_karman(Lx, Ly, D, xc, yc, U_in=1.0, Re=100.0,
                   Ndomx=None, Ndomy=None, nloc=None,
                   degreeU=1, degreeP=3, alpha=1.0,
                   basis_kind="legendre_P", T=None, dt=None, method="mprm",
                   inflow="parabolic", walls="slip", inflow_norm="max",
                   eps_solid=1e-6,
                   comm=None, verbose=True, log_every=200,
                   probes=None, frame_callback=None, frame_every=0,
                   obstacle_metrics_every=1, limiter="minmod"):
    """Escoamento em torno de um cilindro quadrado (von Kármán, artigo §3.5).

    Malha global Nx = Ndomx * nloc, Ny = Ndomy * nloc.
    Use Ndomx = 2*Ndomy para células quadradas (hx = hy) num domínio 2:1.
    Exemplo canônico: Ndomx=10, Ndomy=5, nloc=30 → 300×150, h=0.053,
    obstáculo centrado em (4,4) com D=1 (cabe no subdomínio).
    """
    assert Ndomx is not None and Ndomy is not None and nloc is not None, \
        "Ndomx, Ndomy e nloc são obrigatórios"
    Nx = Ndomx * nloc
    Ny = Ndomy * nloc
    problem = von_karman_problem(Lx, Ly, Nx, Ny, D, xc, yc, U_in, Re,
                                 inflow=inflow, walls=walls,
                                 inflow_norm=inflow_norm)
    if T is None:
        T = 40.0 * D / U_in
    if probes is None:
        probes = [(xc + 3.0 * D, yc)]
    return run_flow(problem, Ndomx, Ndomy, nloc, T, dt=dt, method=method,
                    degreeU=degreeU, degreeP=degreeP, alpha=alpha,
                    basis_kind=basis_kind, eps_solid=eps_solid, comm=comm,
                    verbose=verbose, log_every=log_every, probes=probes,
                    obstacle_metrics=True, obstacle_metrics_every=obstacle_metrics_every,
                    convective_order=2, limiter=limiter,
                    U_ref=U_in, D=D,
                    frame_callback=frame_callback, frame_every=frame_every)


# ===========================================================================
