"""Adaptador de caminhos para a cópia do módulo de estilo.

`mprm/plotting.py` é cópia BYTE A BYTE de
`mrcm-firedrake/studies/_harness/plotting.py`, e o teste
`tests/test_style_parity.py` verifica isso. A única coisa que difere entre
os dois repositórios é onde a figura é gravada, e é por isso que ela mora
aqui, e não dentro do módulo copiado: enquanto a diferença estiver contida
neste arquivo, a cópia pode ser refeita com `cp` e o teste de paridade
continua sendo uma verificação de verdade.

Lá `plots_dir(study_id)` devolve `studies/<study_id>/plots/`; aqui o
repositório tem um só destino de figuras, de modo que `study_id` é aceito e
ignorado.
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
PLOTS = RAIZ / "figures" / "plots"


def plots_dir(study_id: str) -> Path:
    """O destino das figuras. `study_id` existe pela assinatura comum."""
    PLOTS.mkdir(parents=True, exist_ok=True)
    return PLOTS
