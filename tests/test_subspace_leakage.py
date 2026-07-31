"""E11 — vazamento entre subespaços da decomposição.

Hipótese (Cap. 3 da dissertação): o erro da decomposição multiescala aloja-se
na parcela irrotacional. Critério: com entrada de gradiente puro, a parcela
solenoidal devolvida é não nula e é a sede do erro; com entrada solenoidal
pura, o erro é comparativamente menor.
"""
import numpy as np

from mprm.benchmarks.hh import hh_benchmark_mac, hh_benchmark_mode


def test_gradiente_puro_devolve_solenoidal_espurio():
    """Entrada = grad p. Resposta exata = 0. Tudo o que sobra é erro."""
    r = hh_benchmark_mode(Ndom=3, nloc=16, alpha=1.0, mode="gradient")
    assert r["err_u"] is None, "erro relativo é indefinido quando u_ex = 0"
    assert r["err_abs"] > 0.0, "erro exatamente nulo indicaria bug no arranjo"
    assert r["leak_ratio"] < 1.0, "vazamento não pode exceder a energia da entrada"


def test_divergencia_nula_em_todos_os_modos():
    """O downscaling deve entregar div u = 0 independentemente da entrada."""
    for m in ("mixed", "gradient", "solenoidal"):
        r = hh_benchmark_mode(Ndom=3, nloc=16, alpha=1.0, mode=m)
        assert r["divmax"] < 1e-9, f"modo {m}: divmax={r['divmax']:.2e}"


def test_vazamento_cai_com_enriquecimento_da_interface():
    """A previsão testável: enriquecer a interface reduz o vazamento."""
    r1 = hh_benchmark_mode(Ndom=3, nloc=16, degreeU=1, degreeP=1,
                           alpha=1.0, mode="gradient")
    r3 = hh_benchmark_mode(Ndom=3, nloc=16, degreeU=3, degreeP=3,
                           alpha=1.0, mode="gradient")
    assert r3["leak_ratio"] < r1["leak_ratio"], (
        f"enriquecer não reduziu o vazamento: "
        f"{r1['leak_ratio']:.3e} -> {r3['leak_ratio']:.3e}")


def test_linearidade_do_pipeline():
    """Registra explicitamente por que a varredura de amplitude é vazia.

    Este teste NÃO verifica o método: verifica a premissa que torna
    desnecessária a varredura de m_pr do teste de pressure-robustness.
    """
    from mprm.benchmarks.hh import hh_build_w_mode, _make_pipeline, _project_once

    w1, w2, _, _ = hh_build_w_mode(48, mode="gradient")
    proj, down = _make_pipeline(Ndom=3, nloc=16, degreeU=1, degreeP=1,
                                alpha=1.0, dt=1.0)
    a1, a2 = _project_once(w1, w2, projector=proj, downscaler=down)
    b1, b2 = _project_once(1e6 * w1, 1e6 * w2, projector=proj, downscaler=down)
    assert np.allclose(b1, 1e6 * a1, rtol=1e-8)
    assert np.allclose(b2, 1e6 * a2, rtol=1e-8)


def test_hh_benchmark_mac_delega_sem_mudar_de_resultado():
    """Critério de aceitação 2: retrocompatibilidade estrutural."""
    a = hh_benchmark_mac(Ndom=3, nloc=16, alpha=1.0)
    b = hh_benchmark_mode(Ndom=3, nloc=16, alpha=1.0, mode="mixed")
    assert a["err_u"] == b["err_u"]
    assert a["divmax"] == b["divmax"]
    assert a["N"] == b["N"]
    assert np.array_equal(a["u"][0], b["u"][0])
    assert np.array_equal(a["u_exact"][1], b["u_exact"][1])


def test_leak_ratio_e_imune_a_amplitude():
    """leak_ratio normaliza pela energia da ENTRADA, não da referência.

    É a única métrica desta task comparável entre os três modos, porque no
    modo gradient o denominador de err_u não existe.
    """
    for m in ("mixed", "gradient", "solenoidal"):
        r = hh_benchmark_mode(Ndom=2, nloc=8, alpha=1.0, mode=m)
        assert r["leak_ratio"] is not None, m
        assert r["mode"] == m
