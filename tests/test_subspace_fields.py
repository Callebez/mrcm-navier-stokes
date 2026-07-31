"""E11a — campos de entrada por subespaço da decomposição de Helmholtz-Hodge.

Verifica a construção de `hh_build_w_mode`, que permite anular na entrada a
parcela solenoidal ou a irrotacional. Não mede o método: mede o instrumento.
"""
import numpy as np

from mprm.benchmarks.hh import hh_build_w, hh_build_w_mode
from mprm.transport import mac_divergence


def test_modo_mixed_reproduz_a_construcao_atual():
    """Retrocompatibilidade exata com hh_build_w."""
    w1a, w2a, u1a, u2a = hh_build_w(24)
    w1b, w2b, u1b, u2b = hh_build_w_mode(24, mode="mixed")
    assert np.allclose(w1a, w1b) and np.allclose(w2a, w2b)
    assert np.allclose(u1a, u1b) and np.allclose(u2a, u2b)


def test_modo_gradient_tem_parcela_solenoidal_nula():
    w1, w2, u1, u2 = hh_build_w_mode(24, mode="gradient")
    assert np.allclose(u1, 0.0) and np.allclose(u2, 0.0)
    # entrada é gradiente puro: sua divergência NÃO é nula
    assert np.abs(mac_divergence(w1, w2, 1 / 24, 1 / 24)).max() > 1e-3


def test_modo_solenoidal_tem_entrada_de_divergencia_nula():
    N = 48
    w1, w2, u1, u2 = hh_build_w_mode(N, mode="solenoidal")
    assert np.allclose(w1, u1) and np.allclose(w2, u2)
    # o campo de Olshanskii é solenoidal: div discreta -> 0 com o refino
    d = np.abs(mac_divergence(w1, w2, 1 / N, 1 / N)).max()
    assert d < 1e-2, f"div discreta do campo analítico: {d:.2e}"


def test_impermeabilidade_em_todos_os_modos():
    for m in ("mixed", "gradient", "solenoidal"):
        w1, w2, _, _ = hh_build_w_mode(16, mode=m)
        assert np.allclose(w1[0, :], 0.0) and np.allclose(w1[-1, :], 0.0), m
        assert np.allclose(w2[:, 0], 0.0) and np.allclose(w2[:, -1], 0.0), m


def test_modo_invalido_e_rejeitado():
    try:
        hh_build_w_mode(16, mode="irrotational")
    except ValueError as exc:
        assert "irrotational" in str(exc)
    else:
        raise AssertionError("mode inválido deveria levantar ValueError")
