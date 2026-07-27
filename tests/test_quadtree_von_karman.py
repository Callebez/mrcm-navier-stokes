"""Estudo comparativo quadtree vs cartesiano — estrutura e schema."""
import numpy as np
import pytest

from mprm.benchmarks.quadtree_vk import (build_case, run_case,
                                         uniform_counterpart, run_study)

CHAVES_CONFIG = {
    "label", "N_sub", "n_gamma_edges", "n_gamma_dofs", "H_min", "H_max",
    "levels", "cond_A_gamma", "t_offline_s", "t_online_per_step_s",
    "Cd_mean", "Cd_std", "Cl_amp", "St", "div_max", "defect_max", "err_u_rel",
}


def _caso_pequeno(family="a"):
    if family == "a":
        return build_case(family="a", D=1.0, Ly=6.0, N_bx=4, N_by=3,
                          max_level=1, nloc_min=4, radii=(3.0,),
                          Re=100.0, U_in=1.0)
    return build_case(family="b", D=1.0, Ly=4.0, N_bx=4, N_by=2,
                      max_level=3, nloc_min=4, radii=(4.0, 2.0, 1.0),
                      Re=100.0, U_in=1.0)


def test_build_case_familia_a():
    c = _caso_pequeno("a")
    assert c["geom"]["family"] == "a"
    assert c["decomp"].h == pytest.approx(c["geom"]["h"])
    assert c["problem"].solid.any()
    assert len(c["decomp"].leaves) >= 12


def test_build_case_familia_b_corpo_e_uniao_de_folhas():
    c = _caso_pequeno("b")
    g = c["geom"]
    D, xc, yc = g["D"], g["xc"], g["yc"]
    dentro = [lf for lf in c["decomp"].leaves
              if lf.x0 >= xc - D / 2 - 1e-12 and lf.x1 <= xc + D / 2 + 1e-12
              and lf.y0 >= yc - D / 2 - 1e-12 and lf.y1 <= yc + D / 2 + 1e-12]
    assert len(dentro) == 4


def test_uniform_counterpart_casa_o_alvo():
    c = _caso_pequeno("a")
    u = uniform_counterpart(c, target="N_sub")
    assert u["decomp"].is_uniform
    assert u["decomp"].h == pytest.approx(c["decomp"].h), \
        "o baseline precisa ter o MESMO h para a comparação ser honesta"
    alvo = len(c["decomp"].leaves)
    obtido = len(u["decomp"].leaves)
    assert abs(obtido - alvo) <= alvo, f"alvo {alvo}, obtido {obtido}"


def test_run_case_produz_o_schema():
    c = _caso_pequeno("a")
    r = run_case(c, T=0.1, comm=None, degreeU=1, degreeP=3, alpha=1.0,
                 verbose=False)
    assert CHAVES_CONFIG.issubset(set(r)), \
        f"faltando: {CHAVES_CONFIG - set(r)}"
    assert r["div_max"] < 1e-11
    assert r["t_offline_s"] > 0.0


def test_run_study_monta_o_json():
    cases = [_caso_pequeno("a")]
    out = run_study(cases, T=0.1, comm=None, degreeU=1, degreeP=3, alpha=1.0,
                    with_uniform_baseline=True, with_fine_reference=False,
                    verbose=False)
    assert set(out) >= {"geometry", "flow", "spaces", "configs"}
    assert len(out["configs"]) >= 2, "estudo sem baseline uniforme"
    for cfgd in out["configs"]:
        assert CHAVES_CONFIG.issubset(set(cfgd))
