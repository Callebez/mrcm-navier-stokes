"""Regressão end-to-end: run_cavity --quick e run_von_karman smoke."""
import numpy as np
import pytest

from mprm_core import run_cavity, run_von_karman
from mprm_core import mac_divergence


def test_run_cavity_quick_conserva_massa():
    r = run_cavity(Re=100.0, Ndom=3, nloc=8, T=0.2, degreeU=1, degreeP=1,
                   basis_kind="legendre_P", method="mprm", verbose=False)
    assert r["divmax"] < 1e-10           # div ~ machine precision (cavidade)
    assert np.isfinite(r["u"]).all()


def test_run_cavity_fine_vs_mprm_proximos():
    common = dict(Re=100.0, Ndom=3, nloc=8, T=0.2, verbose=False)
    a = run_cavity(method="mprm", degreeU=1, degreeP=1,
                   basis_kind="legendre_P", **common)
    b = run_cavity(method="fine", **common)
    num = np.sum((a["u"] - b["u"])**2) + np.sum((a["v"] - b["v"])**2)
    den = np.sum(b["u"]**2) + np.sum(b["v"]**2)
    assert np.sqrt(num / den) < 5e-2     # concordância grosseira em T curto
