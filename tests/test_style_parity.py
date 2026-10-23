"""A copia do estilo e consistente. Dois repositorios, UMA dissertacao."""

import os
from pathlib import Path

import pytest

# Conhecido do spec: \the\textwidth = 455,0 pt = 6,2960 in
FULL = 6.30
HALF = 3.05

REF_KEYS = {"analitica", "fina_sem_decomposicao", "monolitica_mais_fina"}

COPIA = Path(__file__).resolve().parents[1] / "mprm" / "plotting.py"
ORIGEM = Path(os.environ.get(
    "MRCM_FIREDRAKE", Path.home() / "mrcm" / "mrcm-firedrake")
) / "studies" / "_harness" / "plotting.py"


def test_a_copia_do_estilo_e_byte_a_byte():
    """O plano pede copia byte-identica; o teste anterior so conferia que os
    NOMES da API existiam, de modo que uma mudanca de paleta ou de
    `_ajustar_rotulos_log` num repositorio so passaria despercebida.

    A unica diferenca legitima entre os dois repositorios e o destino das
    figuras, e ela vive em `mprm/paths.py`, FORA do arquivo copiado."""
    if not ORIGEM.exists():
        pytest.skip(f"{ORIGEM} ausente (defina MRCM_FIREDRAKE)")
    assert COPIA.read_bytes() == ORIGEM.read_bytes(), (
        f"{COPIA} divergiu de {ORIGEM}; refaca a copia com `cp`")


def test_a_copia_expoe_a_api_publica():
    import mprm.plotting as p
    for nome in ("C", "REF", "REF_LABEL", "PALETTE", "FIG_W", "M", "LS",
                 "new_figure", "curve_panel", "stacked_bars", "panel_tag",
                 "save_figure", "shared_legend", "add_slope_triangle",
                 "slope_reference", "regime_annotations"):
        assert hasattr(p, nome), f"ausente: {nome}"
    assert not hasattr(p, "standard_suptitle")


def test_a_largura_da_figura_e_a_textwidth_da_classe():
    import mprm.plotting as p
    assert p.FIG_W["full"] == FULL
    assert p.FIG_W["half"] == HALF


def test_as_cores_de_referencia_tem_nomes_explicitos():
    import mprm.plotting as p
    assert set(p.REF) == REF_KEYS
    assert set(p.REF_LABEL) == REF_KEYS
    for v in p.REF_LABEL.values():
        assert "contra a solu" in v


def test_new_figure_usa_a_largura_travada():
    import mprm.plotting as p
    fig, _ = p.new_figure(1, 1)
    assert round(fig.get_figwidth(), 2) == FULL


def test_new_figure_recusa_figsize_explicito():
    import mprm.plotting as p
    import pytest
    with pytest.raises(ValueError):
        p.new_figure(1, 1, figsize=(9, 9))


def test_a_paleta_e_ordinal_e_tem_5_fases():
    import mprm.plotting as p
    assert len(p.PHASES) == 5
    # Ordenadas: luminosidade monótona, verificada pela ordem no dicionário C
    assert p.C["phase0"] == "#c6dbef"
    assert p.C["phase1"] == "#79abdf"
    assert p.C["phase4_5"] == "#1d4e7f"
