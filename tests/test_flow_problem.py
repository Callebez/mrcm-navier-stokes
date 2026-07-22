"""Testes de unidade da abstração de problema (puro NumPy, sem firedrake)."""
import numpy as np
import pytest

from mprm_core import (SideBC, FlowProblem, square_cylinder,
                       cavity_problem, von_karman_problem)


def test_sidebc_valida_kind():
    with pytest.raises(ValueError):
        SideBC("banana")
    with pytest.raises(ValueError):
        SideBC("inflow")  # inflow exige profile


def test_square_cylinder_mask_shape_e_centro():
    Nx, Ny, Lx, Ly, D = 40, 20, 4.0, 2.0, 0.5
    xc, yc = 1.0, 1.0
    mask = square_cylinder(Nx, Ny, Lx, Ly, D, xc, yc)
    assert mask.shape == (Nx, Ny)
    assert mask.dtype == bool
    # célula do centro é sólida; um canto distante é fluido
    hx, hy = Lx / Nx, Ly / Ny
    ic, jc = int(xc / hx), int(yc / hy)
    assert mask[ic, jc]
    assert not mask[0, 0]
    # área ~ D*D (erro de discretização: até ~2 células por lado)
    area = mask.sum() * hx * hy
    assert abs(area - D * D) < 2 * (hx + hy) * D


def test_flowproblem_props():
    p = cavity_problem(N=12, Re=100.0, lid_velocity=1.0)
    assert (p.Nx, p.Ny) == (12, 12)
    assert p.hx == pytest.approx(1.0 / 12)
    assert p.is_impermeable
    assert not p.has_outflow
    assert p.side("top").kind == "wall"
    assert p.side("top").velocity == 1.0
    assert p.nu == pytest.approx(1.0 / 100.0)
    assert p.solid is None


def test_von_karman_problem_estrutura():
    p = von_karman_problem(Lx=4.0, Ly=2.0, Nx=40, Ny=20, D=0.5,
                           xc=1.0, yc=1.0, U_in=1.0, Re=100.0,
                           inflow="uniform", walls="wall")
    assert p.side("left").kind == "inflow"
    assert p.side("right").kind == "outflow"
    assert p.side("top").kind == "wall"
    assert p.has_outflow
    assert not p.is_impermeable
    assert p.solid is not None and p.solid.any()
    assert p.nu == pytest.approx(1.0 * 0.5 / 100.0)  # nu = U_in*D/Re


def test_von_karman_inflow_uniform_profile():
    p = von_karman_problem(Lx=4.0, Ly=2.0, Nx=40, Ny=20, D=0.5,
                           xc=1.0, yc=1.0, U_in=1.5, Re=100.0,
                           inflow="uniform")
    prof = p.side("left").profile
    # perfil uniforme: função (ou constante) devolve U_in em qualquer y
    val = prof(1.0) if callable(prof) else prof
    assert val == pytest.approx(1.5)
