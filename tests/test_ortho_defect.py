"""E13 — defeito de ortogonalidade da decomposição multiescala.

Hipótese (sec:erro-decomposicao da dissertação): as duas parcelas da
decomposição de Helmholtz-Hodge multiescala NÃO são ortogonais, e o resíduo
é a integral, sobre o esqueleto, do salto do produto entre pressão e fluxo
normal. Critério: o defeito medido no pipeline multiescala é ordens de
grandeza maior que o da projeção em malha fina, que não tem esqueleto.

Esta é a quantidade que o capítulo de resultados promete medir DIRETAMENTE
(eq:ErroOrtogonalidade) e que nenhuma execução havia calculado.
"""
import numpy as np

from mprm.benchmarks.hh import (hh_build_w_mode, hh_ortho_defect,
                                hh_benchmark_mode)


def test_cosseno_pequeno_para_o_par_analitico():
    """u de Olshanskii e grad p̃ são ortogonais no contínuo.

    Verifica o INSTRUMENTO, não o método: o cosseno discreto do par exato
    deve ser pequeno e cair com o refino (erro de quadratura).
    """
    cos = {}
    for N in (24, 48, 96):
        _, _, u1, u2 = hh_build_w_mode(N, mode="solenoidal")
        g1, g2, _, _ = hh_build_w_mode(N, mode="gradient")
        _, c = hh_ortho_defect(u1, u2, g1, g2, 1.0 / N)
        cos[N] = abs(c)
    assert cos[96] < 1e-2, f"piso de quadratura alto demais: {cos[96]:.2e}"
    assert cos[96] < cos[24], f"não caiu com o refino: {cos}"


def test_cosseno_imune_a_amplitude():
    """A propriedade que torna o cosseno superior a err_u e a leak_ratio."""
    _, _, u1, u2 = hh_build_w_mode(32, mode="solenoidal")
    g1, g2, _, _ = hh_build_w_mode(32, mode="gradient")
    _, c1 = hh_ortho_defect(u1, u2, g1, g2, 1.0 / 32)
    _, c2 = hh_ortho_defect(1e6 * u1, 1e6 * u2, 1e-3 * g1, 1e-3 * g2, 1.0 / 32)
    assert np.isclose(c1, c2, rtol=1e-10)


def test_defeito_e_maior_no_multiescala_que_na_malha_fina():
    """A hipótese central: o defeito vem do esqueleto.

    A malha fina global é um único subdomínio — não tem esqueleto, logo
    eq:ErroOrtogonalidade não tem sobre o que integrar e só resta o piso de
    quadratura. O multiescala com Ndom=3 tem 4 interfaces internas.
    """
    from mprm.flow import FineProjector

    Ndom, nloc = 3, 16
    N = Ndom * nloc
    h = 1.0 / N
    w1, w2, _, _ = hh_build_w_mode(N, mode="mixed")

    uf1, uf2 = FineProjector(N, N, h, h, 1.0).step(w1, w2)
    _, cos_fino = hh_ortho_defect(uf1, uf2, w1 - uf1, w2 - uf2, h)

    r = hh_benchmark_mode(Ndom=Ndom, nloc=nloc, alpha=1.0, mode="mixed")
    cos_ms = r["ortho_cos"]

    assert abs(cos_ms) > 10.0 * abs(cos_fino), (
        f"defeito multiescala não excede o da malha fina: "
        f"{cos_ms:.3e} vs {cos_fino:.3e}")


def test_cosseno_definido_no_modo_gradient():
    """err_u é None por falta de denominador; ortho_cos não é."""
    r = hh_benchmark_mode(Ndom=3, nloc=16, alpha=1.0, mode="gradient")
    assert r["err_u"] is None
    assert r["ortho_cos"] is not None
    assert abs(r["ortho_cos"]) <= 1.0, "cosseno fora de [-1, 1]"


# ═══════════════════════════════════════════════════════════════════════════
# Lado do esqueleto e a identidade (Task 7)
# ═══════════════════════════════════════════════════════════════════════════

def test_last_potential_e_opt_in():
    """O caminho quente (von Kármán, cavidade) não paga pelo potencial."""
    from mrcm.config import MRCMConfig
    from mprm.downscaler import MACDownscaler

    cfg = MRCMConfig(Ndomx=2, Ndomy=2, nlocx=8, nlocy=8,
                     degreeU=1, degreeP=1, alpha=1.0,
                     mixed_family="BDM", mixed_order=0,
                     executor_kind="sequential")
    d = MACDownscaler(cfg, 1.0)
    assert d.keep_potential is False
    assert d.last_potential == {}


def test_psi_invariante_a_constante_global():
    """O salto não vê a constante global remanescente da ancoragem."""
    from mprm.benchmarks.hh import hh_ortho_skeleton

    Ndom, nloc = 3, 16
    N = Ndom * nloc
    h = 1.0 / N
    rng = np.random.default_rng(0)
    uh1 = rng.standard_normal((N + 1, N))
    uh2 = rng.standard_normal((N, N + 1))
    psi = rng.standard_normal((N, N))
    a = hh_ortho_skeleton(uh1, uh2, psi, Ndom, nloc, h, 1.0)
    b = hh_ortho_skeleton(uh1, uh2, psi + 7.25, Ndom, nloc, h, 1.0)
    assert np.isclose(a, b, rtol=1e-12, atol=1e-12)


def test_esqueleto_nulo_sem_interfaces():
    """Ndom=1 não tem esqueleto: eq:ErroOrtogonalidade não tem suporte."""
    from mprm.benchmarks.hh import hh_ortho_skeleton

    nloc = 16
    rng = np.random.default_rng(1)
    uh1 = rng.standard_normal((nloc + 1, nloc))
    uh2 = rng.standard_normal((nloc, nloc + 1))
    psi = rng.standard_normal((nloc, nloc))
    assert hh_ortho_skeleton(uh1, uh2, psi, 1, nloc, 1.0 / nloc, 1.0) == 0.0


def test_identidade_volume_esqueleto():
    """H2 — as duas leituras de eq:ErroOrtogonalidade concordam.

    Tolerância inicial deliberadamente frouxa: o número medido é o
    resultado, não o critério.
    """
    r = hh_benchmark_mode(Ndom=3, nloc=16, alpha=1.0, mode="mixed")
    assert r["ortho_skel"] is not None
    assert r["ortho_identity_rel"] < 0.2, (
        f"volume={r['ortho_abs']:.6e}  esqueleto={r['ortho_skel']:.6e}  "
        f"rel={r['ortho_identity_rel']:.3e}")


def test_operador_multiescala_e_idempotente():
    """A dissociação, no nível do OPERADOR.

    sec:erro-decomposicao afirma que divergência nula e ortogonalidade se
    dissociam. Aqui isso aparece na forma mais nítida possível: o operador
    multiescala É uma projeção (P∘P = P até a precisão da máquina), só não
    é a projeção ORTOGONAL. Projeta — mas torto.
    """
    from mprm.benchmarks.hh import (_make_pipeline, _project_once, _l2_mac)

    Ndom, nloc = 3, 16
    N = Ndom * nloc
    h = 1.0 / N
    w1, w2, _, _ = hh_build_w_mode(N, mode="mixed")
    proj, down = _make_pipeline(Ndom, nloc, 1, 1, 1.0, 1.0)

    p1, p2 = _project_once(w1, w2, projector=proj, downscaler=down)
    pp1, pp2 = _project_once(p1, p2, projector=proj, downscaler=down)
    rel = _l2_mac(pp1 - p1, pp2 - p2, h) / _l2_mac(p1, p2, h)
    assert rel < 1e-9, f"operador não é idempotente: ‖P²w−Pw‖/‖Pw‖ = {rel:.2e}"


def test_no_modo_gradient_o_cosseno_nao_e_independente():
    """CAVEAT quantitativo: em modo gradient, cos e leak são a mesma coisa.

    A saída espúria é ortogonal à entrada (<u_h, w> = 0), o que força
        cos θ = −leak / sqrt(1 + leak²)
    por álgebra. Nesse modo o defeito de ortogonalidade NÃO acrescenta
    informação a leak_ratio — o valor independente aparece no modo mixed e,
    sobretudo, em arranjos sem solução exata, onde leak_ratio nem existe.
    """
    from mprm.benchmarks.hh import (_make_pipeline, _project_once, _dot_mac)

    Ndom, nloc = 3, 16
    N = Ndom * nloc
    h = 1.0 / N
    w1, w2, _, _ = hh_build_w_mode(N, mode="gradient")
    proj, down = _make_pipeline(Ndom, nloc, 1, 1, 1.0, 1.0)
    u1, u2 = _project_once(w1, w2, projector=proj, downscaler=down)
    assert abs(_dot_mac(u1, u2, w1, w2, h)) < 1e-12, "saída não é ⊥ à entrada"

    g = hh_benchmark_mode(Ndom=Ndom, nloc=nloc, alpha=1.0, mode="gradient")
    leak = g["leak_ratio"]
    previsto = leak / np.sqrt(1.0 + leak * leak)
    assert np.isclose(abs(g["ortho_cos"]), previsto, rtol=1e-8), (
        f"cos={g['ortho_cos']:.8e}  previsto={previsto:.8e}")

    # ... e no modo mixed as duas métricas de fato divergem
    m = hh_benchmark_mode(Ndom=Ndom, nloc=nloc, alpha=1.0, mode="mixed")
    razao = abs(m["ortho_cos"]) / m["leak_ratio"]
    assert razao < 0.5, f"esperava divergência no modo mixed, razão={razao:.3f}"


def test_identidade_melhora_com_refino():
    """Se a identidade é discreta, o resíduo cai com nloc."""
    lo = hh_benchmark_mode(Ndom=3, nloc=8, alpha=1.0, mode="mixed")
    hi = hh_benchmark_mode(Ndom=3, nloc=24, alpha=1.0, mode="mixed")
    assert hi["ortho_identity_rel"] < lo["ortho_identity_rel"], (
        f"{lo['ortho_identity_rel']:.3e} -> {hi['ortho_identity_rel']:.3e}")
