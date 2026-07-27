"""von Kármán sobre decomposição quadtree: conservação e estrutura."""
import numpy as np
import pytest

from mrcm.config import MRCMConfig
from mprm.decomposition import (Decomposition, plan_quadtree_geometry,
                                build_obstacle_tree)
from mprm.problem import von_karman_problem, square_cylinder
from mprm.transport import mac_divergence, solid_faces
from mprm.flow import run_flow


def _setup_pequeno():
    """Canal pequeno o bastante para um smoke test (~30s).

    Ly=6D, N_by=3 (ímpar, família (a)) -> H0=2D > D. Níveis 0-1, nloc=(8,4),
    h=D/4, malha 32x24. O corpo cabe na folha base [0,2]x[2,4].
    """
    g = plan_quadtree_geometry(D=1.0, Ly=6.0, N_bx=4, N_by=3,
                               max_level=1, nloc_min=4, family="a")
    qt = build_obstacle_tree(g, radii=(3.0,))
    problem = von_karman_problem(
        Lx=g["Lx"], Ly=g["Ly"], Nx=g["Nx"], Ny=g["Ny"], D=g["D"],
        xc=g["xc"], yc=g["yc"], U_in=1.0, Re=100.0,
        inflow="parabolic", walls="slip", inflow_norm="max")
    d = Decomposition.from_quadtree(qt, h=g["h"], Lx=g["Lx"], Ly=g["Ly"],
                                    solid=problem.solid)
    return g, qt, problem, d


def test_arvore_e_valida_para_o_obstaculo():
    g, qt, problem, d = _setup_pequeno()
    assert len(d.leaves) > g["N_bx"] * g["N_by"], "a árvore não refinou nada"
    niveis = sorted({lf.level for lf in d.leaves})
    assert niveis == [0, 1], f"níveis obtidos: {niveis}"
    d.check_fluid_connectivity(problem.solid)     # não levanta


def test_von_karman_quadtree_conserva_massa():
    g, qt, problem, d = _setup_pequeno()
    r = run_flow(problem, Ndomx=g["N_bx"], Ndomy=g["N_by"], nloc=g["nloc"][0],
                 T=0.2, decomposition=d, verbose=False,
                 obstacle_metrics=False, U_ref=1.0, D=g["D"])
    dv = np.abs(mac_divergence(r["u"], r["v"], g["h"], g["h"]))
    dv_fluido = dv[~problem.solid].max()
    assert dv_fluido < 1e-11, f"max|div u| no fluido = {dv_fluido:.3e}"


def test_von_karman_quadtree_respeita_o_obstaculo():
    g, qt, problem, d = _setup_pequeno()
    r = run_flow(problem, Ndomx=g["N_bx"], Ndomy=g["N_by"], nloc=g["nloc"][0],
                 T=0.2, decomposition=d, verbose=False,
                 obstacle_metrics=False, U_ref=1.0, D=g["D"])
    fx, fy = solid_faces(problem.solid)
    assert np.abs(r["u"][fx]).max() == 0.0
    assert np.abs(r["v"][fy]).max() == 0.0


def test_resultado_reporta_a_decomposicao():
    g, qt, problem, d = _setup_pequeno()
    r = run_flow(problem, Ndomx=g["N_bx"], Ndomy=g["N_by"], nloc=g["nloc"][0],
                 T=0.05, decomposition=d, verbose=False,
                 obstacle_metrics=False, U_ref=1.0, D=g["D"])
    info = r["decomposition"]
    assert info["N_sub"] == len(d.leaves)
    assert info["n_gamma_edges"] == len(d.gamma_edges())
    assert info["H_min"] < info["H_max"], "a árvore deveria ser heterogênea"
    assert info["n_gamma_dofs"] > 0


# ═══════════════════════════════════════════════════════════════════════
# MPI (spec §7)
# ═══════════════════════════════════════════════════════════════════════
import json, os, shutil, subprocess, sys as _sys


def _rodar(nranks):
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "mpi_quadtree_run.py")
    cmd = ([_sys.executable, script] if nranks == 1
           else ["mpiexec", "-n", str(nranks), _sys.executable, script])
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=900,
                         cwd=os.path.dirname(os.path.dirname(script)))
    assert out.returncode == 0, f"falhou com {nranks} rank(s):\n{out.stderr[-3000:]}"
    linha = [l for l in out.stdout.splitlines() if l.startswith("JSON ")]
    assert linha, f"sem saída JSON:\n{out.stdout[-2000:]}"
    return json.loads(linha[-1][5:])


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("mpiexec") is None, reason="mpiexec ausente")
def test_sequencial_e_mpi_concordam_sob_quadtree():
    seq = _rodar(1)
    par = _rodar(2)
    assert seq["N_sub"] == par["N_sub"]
    for chave in ("u_norm", "v_norm", "u_sum"):
        ref = abs(seq[chave]) or 1.0
        err = abs(seq[chave] - par[chave]) / ref
        assert err < 1e-10, f"{chave}: erro relativo {err:.3e} (seq vs 2 ranks)"
    assert par["div_max"] < 1e-11


def test_cavidade_uniforme_continua_igual():
    """Regressão: o caminho sem `decomposition` não pode ter mudado."""
    from mprm.flow import run_cavity
    r = run_cavity(Re=100.0, Ndom=2, nloc=8, T=0.05, verbose=False)
    dv = np.abs(mac_divergence(r["u"], r["v"], 1.0 / 16, 1.0 / 16)).max()
    assert dv < 1e-11, f"max|div u| = {dv:.3e}"
