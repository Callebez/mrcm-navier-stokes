"""Projetor MRCM: regressão do caminho impermeável + validação do
assert de obstáculo interior. (usa firedrake — mais lento)"""
import numpy as np
import pytest

from mrcm.config import MRCMConfig
from mprm_core import (MPrMProjector, assert_solid_interior_to_subdomains,
                       von_karman_problem)


def _cfg(Ndom=2, nloc=8, dU=1, dP=1):
    return MRCMConfig(Ndomx=Ndom, Ndomy=Ndom, nlocx=nloc, nlocy=nloc,
                      degreeU=dU, degreeP=dP, alpha=1.0,
                      basis_kind="legendre_P",
                      mixed_family="BDM", mixed_order=0,
                      executor_kind="sequential")


def test_projector_impermeavel_problem_none_roda():
    cfg = _cfg()
    proj = MPrMProjector(cfg, dt=1.0)          # problem=None -> impermeável
    assert proj.problem.is_impermeable
    divw = np.zeros((cfg.Nglobx, cfg.Ngloby))
    fluxes = proj.project(divw)
    # fonte nula -> todos os fluxos de interface ~ 0
    for arr in fluxes.values():
        assert np.allclose(arr, 0.0, atol=1e-8)


def test_assert_obstaculo_interior():
    cfg = _cfg(Ndom=2, nloc=8)   # subdomínios de 8 células; interface em x=8
    # obstáculo que ATRAVESSA a interface central deve falhar
    solid = np.zeros((16, 16), dtype=bool)
    solid[7:10, 6:10] = True     # cruza a coluna 8 (interface)
    with pytest.raises(ValueError):
        assert_solid_interior_to_subdomains(solid, cfg)
    # obstáculo contido no subdomínio inferior-esquerdo: ok
    solid2 = np.zeros((16, 16), dtype=bool)
    solid2[2:5, 2:5] = True
    assert_solid_interior_to_subdomains(solid2, cfg)  # não levanta
