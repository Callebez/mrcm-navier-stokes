"""
tests/params.py — Interface de parsing de parâmetros dos testes MPrM.

Cada teste (`test_*.py`) declara suas flags específicas via o callback
`extra_args`; as flags comuns a todos os testes (decomposição de domínio,
espaços de interface, α, fluido, tempo, saída) são adicionadas
automaticamente por `build_parser`. Isso evita duplicar `argparse` em cada
arquivo de teste e mantém os nomes de flag consistentes entre eles.

Uso típico num arquivo `test_*.py`:

    from params import CommonParams, parse_args, parse_float_list

    def main(argv=None):
        defaults = CommonParams(Ndom=3, nloc=30)          # overrides do teste
        args = parse_args(
            "Estudo 1: HH vs α (Fig. 6, Lages §3.1)",
            defaults=defaults,
            extra_args=lambda p: p.add_argument(
                "--alphas", default="logspace:-4,4,9",
                help="lista ou 'logspace:ini,fim,n'"),
        )
        alphas = parse_float_list(args.alphas)
        ...
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from typing import Callable, Optional

import numpy as np


@dataclass
class CommonParams:
    """Parâmetros compartilhados por todos os testes MPrM.

    Cada teste instancia isto com os *defaults* que fazem sentido para ele
    (p.ex. a cavidade usa `degreeP=0`; o HH usa `degreeU=degreeP=k` num
    laço) e repassa a `parse_args` — os valores viram o default de cada
    flag, e a linha de comando pode sobrescrevê-los.
    """

    # Domínio
    Lx: float = 1.0
    Ly: float = 1.0

    # Decomposição MRCM
    Ndom: int = 3          # subdomínios por direção (N_H)
    nloc: int = 16         # células finas por subdomínio

    # Espaços de interface (MRCM)
    degreeU: int = 1
    degreeP: int = 1
    # legendre_P por padrão: hierárquica, preserva π₀ (constante) mesmo com
    # graus assimétricos — nodal_Q trunca a base de Lagrange do grau máximo
    # e perde a constante quando degreeU != degreeP (ver docs/05, achado D3').
    basis_kind: str = "legendre_P"

    # Robin
    alpha: float = 1.0

    # Fluido / tempo (usado pelos testes de cavidade)
    Re: float = 100.0
    lid_velocity: float = 1.0
    T: float = 8.0
    dt: Optional[float] = None   # None ⇒ CFL automático (ver mprm_core.run_cavity)

    # Execução / saída
    outdir: str = "results_alpha"
    quick: bool = False    # versão reduzida (smoke test) — cada teste decide o que reduzir
    verbose: bool = True


def build_parser(description: str,
                 defaults: Optional[CommonParams] = None,
                 extra_args: Optional[Callable[[argparse.ArgumentParser], None]] = None,
                 ) -> argparse.ArgumentParser:
    """Monta um `ArgumentParser` com as flags comuns + as específicas do teste.

    ``extra_args``, se fornecido, recebe o parser e adiciona as flags que só
    aquele teste usa (ex.: ``--alphas``, ``--ks``).
    """
    d = defaults if defaults is not None else CommonParams()
    p = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g_dom = p.add_argument_group("domínio e decomposição")
    g_dom.add_argument("--Lx", type=float, default=d.Lx)
    g_dom.add_argument("--Ly", type=float, default=d.Ly)
    g_dom.add_argument("--Ndom", type=int, default=d.Ndom,
                       help="subdomínios por direção (N_H)")
    g_dom.add_argument("--nloc", type=int, default=d.nloc,
                       help="células finas por subdomínio")

    g_if = p.add_argument_group("espaços de interface e Robin")
    g_if.add_argument("--degreeU", type=int, default=d.degreeU,
                      help="grau do espaço de fluxo U_H (k_U = grau+1 DOFs/interface)")
    g_if.add_argument("--degreeP", type=int, default=d.degreeP,
                      help="grau do espaço de pressão P_H (k_P = grau+1 DOFs/interface)")
    g_if.add_argument("--basis-kind", dest="basis_kind", default=d.basis_kind,
                      choices=["nodal_Q", "monomial_P", "legendre_P"])
    g_if.add_argument("--alpha", type=float, default=d.alpha,
                      help="parâmetro de Robin α (β = α·L/K)")

    g_fl = p.add_argument_group("fluido e tempo (testes de cavidade)")
    g_fl.add_argument("--Re", type=float, default=d.Re)
    g_fl.add_argument("--lid-velocity", dest="lid_velocity", type=float,
                      default=d.lid_velocity)
    g_fl.add_argument("--T", type=float, default=d.T, help="tempo final")
    g_fl.add_argument("--dt", type=float, default=d.dt,
                      help="passo de tempo fixo (default: CFL automático)")

    g_io = p.add_argument_group("execução e saída")
    g_io.add_argument("--outdir", default=d.outdir)
    g_io.add_argument("--quick", action="store_true", default=d.quick,
                      help="versão reduzida (smoke test)")
    g_io.add_argument("--quiet", dest="verbose", action="store_false",
                      default=d.verbose, help="suprime prints por caso")

    if extra_args is not None:
        extra_args(p)
    return p


def parse_args(description: str,
               defaults: Optional[CommonParams] = None,
               extra_args: Optional[Callable[[argparse.ArgumentParser], None]] = None,
               argv=None) -> argparse.Namespace:
    """Atalho: `build_parser(...).parse_args(argv)`."""
    return build_parser(description, defaults, extra_args).parse_args(argv)


def common_params_from_args(args: argparse.Namespace) -> CommonParams:
    """Extrai os campos de `CommonParams` de um Namespace já parseado —
    útil para repassar como um objeto único em vez de campo a campo."""
    return CommonParams(**{
        f: getattr(args, f) for f in CommonParams.__dataclass_fields__
        if hasattr(args, f)
    })


def parse_float_list(spec: str) -> list[float]:
    """Converte uma especificação textual numa lista de floats.

    Formatos aceitos:
      "1e-4,1e-2,1,1e2"      — lista explícita separada por vírgula
      "logspace:-4,4,9"      — np.logspace(-4, 4, 9)  (início/fim são expoentes)
      "linspace:0,1,5"       — np.linspace(0, 1, 5)
    """
    spec = spec.strip()
    if spec.startswith("logspace:"):
        lo, hi, n = spec[len("logspace:"):].split(",")
        return [float(v) for v in np.logspace(float(lo), float(hi), int(n))]
    if spec.startswith("linspace:"):
        lo, hi, n = spec[len("linspace:"):].split(",")
        return [float(v) for v in np.linspace(float(lo), float(hi), int(n))]
    return [float(v) for v in spec.split(",") if v.strip()]


def parse_int_list(spec: str) -> list[int]:
    """Converte uma especificação textual numa lista de ints.

    Formatos aceitos:
      "1,2,3,4"     — lista explícita separada por vírgula
      "range:1,5"   — list(range(1, 5))
    """
    spec = spec.strip()
    if spec.startswith("range:"):
        lo, hi = spec[len("range:"):].split(",")
        return list(range(int(lo), int(hi)))
    return [int(v) for v in spec.split(",") if v.strip()]


def with_overrides(defaults: CommonParams, **kw) -> CommonParams:
    """`dataclasses.replace` com nome mais legível no call site dos testes."""
    return replace(defaults, **kw)
