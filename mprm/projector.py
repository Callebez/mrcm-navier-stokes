"""Projetor MRCM — MPrM.

MPrMProjector: passo de projeção resolvido pelo MRCM.
Importa Firedrake (apenas dentro dos métodos — não no import do módulo).
"""
from __future__ import annotations
import numpy as np
import os, sys
import scipy.sparse as sp
import scipy.sparse.linalg as spla

MRCM_REPO = os.path.expanduser("~/mrcm/mrcm-firedrake")
if MRCM_REPO not in sys.path:
    sys.path.insert(0, MRCM_REPO)

from mrcm.config import MRCMConfig
from mrcm.problem import DarcyProblem, BoundaryData
from mrcm.topology import Skeleton, DOFManager, mark_sides
from mrcm.interface_space import make_interface_space
from mrcm.robin import make_robin
from mrcm.local_form import build_local_form
from mrcm.local_solver import LocalSolver, build_subdomain_workspace
from mrcm.global_assembly import GlobalAssembler
from mrcm.reconstructor import reconstruct_subdomain

from mprm.problem import (SideBC, FlowProblem, _impermeable_problem,
                          _darcy_boundary_from_problem,
                          assert_solid_interior_to_subdomains)
from mprm.triplets import subdomain_matrix_triplets, subdomain_rhs

from mpi4py import MPI as _MPI
MPI_SUM = _MPI.SUM

# Projetor MRCM — estágio offline/online (Algoritmo 1 do artigo)
# ===========================================================================

class MPrMProjector:
    """Passo de projeção do MPrM resolvido pelo MRCM.

    Offline (construtor): malhas/espaços locais, MsBF + caches B1/B2
    (via `mrcm.local_solver.build_subdomain_workspace`), fatoração local
    persistente (`LocalSolver`), montador global e operadores de extração
    de traço de fluxo nas interfaces.

    Online (`project(divw)`): parte particular por subdomínio (fonte
    ∇·w* célula a célula), sistema global de interface, reconstrução e
    média dos fluxos finos nas interfaces (Eq. 21 do artigo).

    Paralelismo: subdomínios distribuídos round-robin entre os ranks de
    `comm` (isub % size), como no executor mpi do mrcm.
    """

    def __init__(self, cfg: MRCMConfig, dt: float, comm=None,
                 problem=None, verbose=False, eps_solid=1e-6):
        from firedrake import (Constant, FunctionSpace, TestFunction,
                               TestFunctions, Function, FacetNormal,
                               VectorFunctionSpace, SpatialCoordinate,
                               assemble, dot)
        from mrcm.topology import SIDE_TO_MARKER

        if comm is None:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
        self.comm = comm
        self.rank, self.size = comm.rank, comm.size
        self.cfg = cfg
        self.dt = float(dt)
        self.verbose = verbose
        self.eps_solid = float(eps_solid)

        # problem=None => cavidade impermeável (comportamento retrocompatível)
        if problem is None:
            problem = _impermeable_problem(cfg)
        self.problem = problem
        if problem.solid is not None:
            assert_solid_interior_to_subdomains(problem.solid, cfg, strict=False)

        self.skel = Skeleton.cartesian(cfg.Ndomx, cfg.Ndomy)
        self.dofm = DOFManager(self.skel.NGamma, cfg.ndofsU, cfg.ndofsP)
        self.my_subs = list(range(self.rank, cfg.Nsubd, self.size))

        dt_c = self.dt
        eps = self.eps_solid
        solid = problem.solid
        hx_f = cfg.Hx / cfg.nlocx
        hy_f = cfg.Hy / cfg.nlocy
        self.hx_f, self.hy_f = hx_f, hy_f

        def _K(mesh):
            from firedrake import (Constant, Function, FunctionSpace,
                                   VectorFunctionSpace, SpatialCoordinate)
            if solid is None:
                return Constant(dt_c)
            Kf = FunctionSpace(mesh, "DG", 0)
            xc = Function(VectorFunctionSpace(mesh, "DG", 0)).interpolate(
                SpatialCoordinate(mesh))
            xcd = xc.dat.data_ro
            ic = np.clip(np.floor(xcd[:, 0] / hx_f).astype(int),
                         0, cfg.Nglobx - 1)
            jc = np.clip(np.floor(xcd[:, 1] / hy_f).astype(int),
                         0, cfg.Ngloby - 1)
            k = Function(Kf)
            k.dat.data[:] = np.where(solid[ic, jc], dt_c * eps, dt_c)
            return k

        def _invK(mesh):
            from firedrake import Constant, Function
            if solid is None:
                return Constant(1.0 / dt_c)
            K = _K(mesh)
            ik = Function(K.function_space())
            ik.dat.data[:] = 1.0 / K.dat.data_ro
            return ik

        def _zero(x, y):
            from firedrake import Constant
            return Constant(0.0)

        self.problem_darcy = DarcyProblem(
            Lx=cfg.Lx, Ly=cfg.Ly,
            permeability=_K, inv_permeability=_invK,
            source=_zero,
            boundary=_darcy_boundary_from_problem(problem),
        )

        robin = make_robin(cfg.robin_variant, cfg.alpha, *cfg.robin_L_xy(),
                           cfg.robin_clamp_eps, cfg.robin_extreme_beta)
        ispace = make_interface_space(cfg.basis_kind, cfg.degreeU, cfg.degreeP)

        NunkU = self.skel.NGamma * cfg.ndofsU
        self.ws = {}          # isub -> workspace (mesh, V, Q, MsBF, caches)
        self.solvers = {}     # isub -> LocalSolver persistente (LU offline)
        self.src_fn = {}      # isub -> Function DG (fonte ∇·w*)
        self.L_forms = {}     # isub -> forma RHS  -f q dx
        self.cell_map = {}    # isub -> (ic, jc) índices globais de célula p/ dof DG
        self.trace = {}       # (isub, side) -> dict(form, idx, len, order, sign)
        self.out_trace = {}   # (isub, side_name) -> dict p/ fluxo de saída
        self.uh = {}          # isub -> Function(V) para extração de traço

        for isub in self.my_subs:
            ix, iy = isub % cfg.Ndomx, isub // cfg.Ndomx
            flag_side, internal_sides = mark_sides(ix, iy, cfg.Ndomx, cfg.Ndomy)
            edge_l = {s: int(self.skel.ieGt[self.skel.el2edge[isub, s]])
                      for s in internal_sides}
            task = {"cfg_dict": cfg.as_dict(), "isub": isub,
                    "NunkU": NunkU, "edge_l": edge_l,
                    "problem": self.problem_darcy}
            # Offline: MsBF + caches B1/B2
            ws = build_subdomain_workspace(task)
            self.ws[isub] = ws
            mesh, V, Q = ws["mesh"], ws["V"], ws["Q"]
            W = V * Q

            # Solver local persistente (fatoração LU offline; artigo §2.7).
            beta_expr = robin.beta_for_subdomain(isub, mesh)
            a_form, dx_m, ds_m, _, _, n_facet = build_local_form(
                W, mesh, _invK(mesh), beta_expr, internal_sides,
                quadrature_degree=cfg.final_quadrature_degree(),
                facet_quadrature_degree=cfg.final_facet_quadrature_degree(),
            )
            # BCs nativas do mrcm (no-flow forte nos lados externos com fluxo)
            from mrcm.local_form import build_flux_dirichlet_bcs
            bcs = build_flux_dirichlet_bcs(W, mesh, flag_side,
                                           problem=self.problem_darcy)
            self.solvers[isub] = LocalSolver(
                W, a_form, bcs,
                use_iterative=(cfg.local_solver_mode == "iterative"), cfg=cfg)

            # Fonte DG por subdomínio + mapa célula fina global -> dof DG.
            f_i = Function(Q)
            self.src_fn[isub] = f_i
            _, q_test = TestFunctions(W)
            self.L_forms[isub] = -f_i * q_test * dx_m
            Xc = VectorFunctionSpace(mesh, "DG", 0)
            xc = Function(Xc).interpolate(SpatialCoordinate(mesh))
            xcd = xc.dat.data_ro
            assert cfg.mixed_order == 0, "fonte por célula requer DG0"
            ic = np.floor(xcd[:, 0] / hx_f).astype(int)
            jc = np.floor(xcd[:, 1] / hy_f).astype(int)
            self.cell_map[isub] = (np.clip(ic, 0, cfg.Nglobx - 1),
                                   np.clip(jc, 0, cfg.Ngloby - 1))

            # Operadores de traço: fluxo médio por face fina de cada lado interno.
            T = FunctionSpace(mesh, "HDiv Trace", 0)
            tau = TestFunction(T)
            uh = Function(V)
            self.uh[isub] = uh
            for s in internal_sides:
                marker = SIDE_TO_MARKER[s]
                ln = assemble(tau * ds_m(marker)).dat.data_ro.copy()
                idx = np.nonzero(np.abs(ln) > 1e-14)[0]
                xm = assemble(SpatialCoordinate(mesh)[0] * tau
                              * ds_m(marker)).dat.data_ro[idx] / ln[idx]
                ym = assemble(SpatialCoordinate(mesh)[1] * tau
                              * ds_m(marker)).dat.data_ro[idx] / ln[idx]
                coord = ym if s in (1, 3) else xm     # lados verticais: y
                order = np.argsort(coord)
                sign = +1.0 if s in (1, 2) else -1.0  # right/top: n = +e
                self.trace[(isub, s)] = {
                    "form": dot(uh, n_facet) * tau * ds_m(marker),
                    "idx": idx, "len": ln[idx], "order": order, "sign": sign,
                }

            # Traço de fluxo nas faces de SAÍDA (lado outflow externo).
            for s_name, marker in (("left", 1), ("right", 2),
                                   ("bottom", 3), ("top", 4)):
                if problem.side(s_name).kind != "outflow":
                    continue
                flag_idx = {"bottom": 0, "right": 1, "top": 2, "left": 3}[s_name]
                if flag_side[flag_idx]:      # só bordas externas reais
                    continue
                ln = assemble(tau * ds_m(marker)).dat.data_ro.copy()
                idx = np.nonzero(np.abs(ln) > 1e-14)[0]
                if idx.size == 0:
                    continue
                cm = (assemble(SpatialCoordinate(mesh)[1] * tau * ds_m(marker))
                      .dat.data_ro[idx] / ln[idx]) if s_name in ("left", "right") \
                    else (assemble(SpatialCoordinate(mesh)[0] * tau * ds_m(marker))
                          .dat.data_ro[idx] / ln[idx])
                order = np.argsort(cm)
                sign = +1.0 if s_name in ("right", "top") else -1.0
                self.out_trace[(isub, s_name)] = {
                    "form": dot(uh, n_facet) * tau * ds_m(marker),
                    "idx": idx, "len": ln[idx], "order": order, "sign": sign}

        # Montador global
        self.assembler = None
        if self.rank == 0:
            self.assembler = GlobalAssembler(
                cfg, self.skel, self.dofm, ispace, robin,
                basis_per_sides=None, n_facets=None, dss=None, meshes=None,
                Vs=None, Qs=None, face_mass={}, cofunc_flux={}, cofunc_pres={})

        # ── Offline do sistema global de interface (spec §5, Fase 1) ──
        # A_Γ depende só das MsBF/massas de face/cofunções: com K = δt e a
        # geometria fixas, ela é a MESMA em todos os passos de tempo. Montar e
        # fatorar aqui evita refazer, por passo, O(N_B²) produtos internos por
        # subdomínio mais uma fatoração esparsa.
        my_mat = {isub: subdomain_matrix_triplets(
            cfg, self.skel, self.dofm, isub,
            self.ws[isub]["homogeneous"], self.ws[isub]["face_mass"],
            self.ws[isub]["cofunc_flux"], self.ws[isub]["cofunc_pres"])
            for isub in self.my_subs}
        gathered_mat = comm.gather(my_mat, root=0)

        self.n_gamma = self.dofm.n_total
        self.A_gamma = None
        self.lu_gamma = None
        if self.rank == 0:
            parts = []
            for d in gathered_mat:
                parts.extend(d[k] for k in sorted(d.keys()))
            nonempty = [t for t in parts if t[0].size > 0]
            rows = np.concatenate([t[0] for t in nonempty])
            cols = np.concatenate([t[1] for t in nonempty])
            vals = np.concatenate([t[2] for t in nonempty])
            A = sp.coo_matrix((vals, (rows, cols)),
                              shape=(self.n_gamma, self.n_gamma)).tocsc()
            A.sum_duplicates()
            self.A_gamma = A
            if cfg.global_solver == "iterative" and self.n_gamma > cfg.iterative_threshold:
                self.lu_gamma = None      # caminho iterativo: resolve por passo
            else:
                self.lu_gamma = spla.splu(A)

        self.last_X = None
        self.last_pT = {}
        self.last_uT = {}
        self.last_boundary_fluxes = {}

    # -- offline: sistema global de interface --------------------------------

    def solve_interface(self, b):
        """Resolve A_Γ X = b com a fatoração do offline (rank 0) e difunde X.

        `b` é significativo apenas no rank 0; os demais podem passar None.
        """
        if self.rank == 0:
            if self.lu_gamma is not None:
                X = self.lu_gamma.solve(b)
            else:
                from mrcm.global_assembly import _solve_iterative
                X = _solve_iterative(self.A_gamma, b, self.cfg)
        else:
            X = None
        return self.comm.bcast(X, root=0)

    # -- online ------------------------------------------------------------

    def project(self, divw):
        """Resolve a projeção MRCM para a fonte ∇·w* (array Nglobx×Ngloby).

        Retorna `fluxes`: dict com os fluxos médios (Eq. 21) por interface,
          ("v", i, iy) -> array nlocy  (interface vertical entre colunas i-1|i
                                        de subdomínios, orientação +x)
          ("h", ix, j) -> array nlocx  (interface horizontal, orientação +y)
        disponível de forma idêntica em todos os ranks.
        """
        from firedrake import assemble
        cfg = self.cfg
        comm = self.comm

        # 1. Parte particular (fonte atualizada) — paralelo, sem comunicação.
        particular = {}
        for isub in self.my_subs:
            ic, jc = self.cell_map[isub]
            self.src_fn[isub].dat.data[:] = divw[ic, jc]
            particular[isub] = self.solvers[isub].solve(self.L_forms[isub])

        # 2. RHS local -> redução ao rank 0 -> solve com a LU do offline.
        #    Só b muda de um passo para o outro (spec §5, Fase 1).
        b_local = np.zeros(self.n_gamma, dtype=np.float64)
        for isub in self.my_subs:
            b_idx, b_vals = subdomain_rhs(
                cfg, self.skel, self.dofm, isub, particular[isub],
                self.ws[isub]["cofunc_flux"], self.ws[isub]["cofunc_pres"])
            np.add.at(b_local, b_idx, b_vals)
        b = np.zeros(self.n_gamma, dtype=np.float64) if self.rank == 0 else None
        comm.Reduce(b_local, b, op=MPI_SUM, root=0)
        X = self.solve_interface(b)
        self.last_X = X

        # 3. Reconstrução distribuída + traços de fluxo nas interfaces.
        my_traces = {}
        for isub in self.my_subs:
            pT, uT = reconstruct_subdomain(
                particular[isub], self.ws[isub]["homogeneous"], X,
                self.ws[isub]["V"], self.ws[isub]["Q"],
                sparsity_cutoff=cfg.reconstruction_sparsity_cutoff)
            self.last_pT[isub], self.last_uT[isub] = pT, uT
            self.uh[isub].assign(uT)
            for (jsub, s), tr in self.trace.items():
                if jsub != isub:
                    continue
                vals = assemble(tr["form"]).dat.data_ro[tr["idx"]]
                flux = tr["sign"] * (vals / tr["len"])[tr["order"]]
                my_traces[(isub, s)] = flux

        # 4. Média dos fluxos (Eq. 21) — allgather (arrays pequenos: nloc
        #    valores por lado de interface).
        all_tr = {}
        for d in comm.allgather(my_traces):
            all_tr.update(d)

        fluxes = {}
        for iy in range(cfg.Ndomy):          # interfaces verticais
            for i in range(1, cfg.Ndomx):
                sl = (i - 1) + iy * cfg.Ndomx   # subdomínio à esquerda
                sr = i + iy * cfg.Ndomx
                fluxes[("v", i, iy)] = 0.5 * (all_tr[(sl, 1)] + all_tr[(sr, 3)])
        for j in range(1, cfg.Ndomy):        # interfaces horizontais
            for ix in range(cfg.Ndomx):
                sb = ix + (j - 1) * cfg.Ndomx   # subdomínio abaixo
                st = ix + j * cfg.Ndomx
                fluxes[("h", ix, j)] = 0.5 * (all_tr[(sb, 2)] + all_tr[(st, 0)])

        # fluxos de saída (v·n reconstruído nas faces externas outflow)
        my_out = {}
        for isub in self.my_subs:
            self.uh[isub].assign(self.last_uT[isub])
            for (jsub, s_name), tr in self.out_trace.items():
                if jsub != isub:
                    continue
                vals = assemble(tr["form"]).dat.data_ro[tr["idx"]]
                my_out[(isub, s_name)] = tr["sign"] * (vals / tr["len"])[tr["order"]]
        self.last_boundary_fluxes = {}
        for d in comm.allgather(my_out):
            self.last_boundary_fluxes.update(d)

        return fluxes


# ===========================================================================
