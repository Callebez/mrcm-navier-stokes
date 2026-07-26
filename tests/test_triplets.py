"""Equivalência da separação matriz/RHS das triplas do sistema de interface.

A parte-matriz é constante no tempo (depende só das MsBF e das cofunções);
a parte-RHS depende da fonte. Este teste garante que a soma das duas
reproduz exatamente `compute_subdomain_triplets` do mrcm.
"""
import numpy as np
import pytest

from mrcm.config import MRCMConfig
from mrcm.global_assembly import compute_subdomain_triplets
from mprm.triplets import subdomain_matrix_triplets, subdomain_rhs
from mprm.projector import MPrMProjector


def _projector(Ndom=2, nloc=8):
    cfg = MRCMConfig(Ndomx=Ndom, Ndomy=Ndom, nlocx=nloc, nlocy=nloc,
                     degreeU=1, degreeP=1, alpha=1.0,
                     basis_kind="legendre_P",
                     mixed_family="BDM", mixed_order=0,
                     executor_kind="sequential")
    return cfg, MPrMProjector(cfg, dt=0.01)


def test_matriz_mais_rhs_reproduz_compute_subdomain_triplets():
    cfg, proj = _projector()
    rng = np.random.default_rng(0)
    divw = rng.standard_normal((cfg.Nglobx, cfg.Ngloby))

    # Parte particular com uma fonte concreta (o que o passo online faz).
    from firedrake import assemble  # noqa: F401  (garante Firedrake carregado)
    for isub in proj.my_subs:
        ic, jc = proj.cell_map[isub]
        proj.src_fn[isub].dat.data[:] = divw[ic, jc]
    particular = {isub: proj.solvers[isub].solve(proj.L_forms[isub])
                  for isub in proj.my_subs}

    for isub in proj.my_subs:
        ws = proj.ws[isub]
        ref = compute_subdomain_triplets(
            cfg, proj.skel, proj.dofm, isub,
            particular[isub], ws["homogeneous"],
            ws["face_mass"], ws["cofunc_flux"], ws["cofunc_pres"])
        rows, cols, vals = subdomain_matrix_triplets(
            cfg, proj.skel, proj.dofm, isub,
            ws["homogeneous"], ws["face_mass"],
            ws["cofunc_flux"], ws["cofunc_pres"])
        b_idx, b_vals = subdomain_rhs(
            cfg, proj.skel, proj.dofm, isub,
            particular[isub], ws["cofunc_flux"], ws["cofunc_pres"])

        assert np.array_equal(rows, ref[0]), f"rows diferem em isub={isub}"
        assert np.array_equal(cols, ref[1]), f"cols diferem em isub={isub}"
        assert np.allclose(vals, ref[2], atol=1e-14, rtol=0)
        assert np.array_equal(b_idx, ref[3]), f"b_idx diferem em isub={isub}"
        assert np.allclose(b_vals, ref[4], atol=1e-14, rtol=0)


def test_parte_matriz_independe_da_fonte():
    """O ponto da Fase 1: mudar a fonte não muda a matriz."""
    cfg, proj = _projector()
    rng = np.random.default_rng(1)

    def _matriz_para(divw):
        for isub in proj.my_subs:
            ic, jc = proj.cell_map[isub]
            proj.src_fn[isub].dat.data[:] = divw[ic, jc]
            proj.solvers[isub].solve(proj.L_forms[isub])
        ws = proj.ws[0]
        return subdomain_matrix_triplets(
            cfg, proj.skel, proj.dofm, 0,
            ws["homogeneous"], ws["face_mass"],
            ws["cofunc_flux"], ws["cofunc_pres"])

    a = _matriz_para(rng.standard_normal((cfg.Nglobx, cfg.Ngloby)))
    b = _matriz_para(rng.standard_normal((cfg.Nglobx, cfg.Ngloby)))
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])
    assert np.allclose(a[2], b[2], atol=1e-14, rtol=0)
