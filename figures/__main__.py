#!/usr/bin/env python3
"""Regenera as figuras do bloco de Navier-Stokes a partir dos JSON.

    python -m figures

Cada PDF ganha um `.prov.txt` ao lado, gerado a partir de `figspec_ns.py`.
Antes estes arquivos eram escritos a mao e diziam "ver plot_ghia.py" no lugar
dos campos consumidos, o que os tornava inuteis para quem escreve o capitulo.
"""
import runpy
import subprocess
import sys
from pathlib import Path

from figures import figspec_ns

MODULOS = ["plot_olshanskii", "plot_par_misto", "plot_ghia"]
RAIZ = Path(__file__).resolve().parents[1]
PLOTS = RAIZ / "figures" / "plots"


def git_sha() -> str:
    """SHA curto do HEAD, com sufixo `-dirty` se houver alteracao RASTREADA."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=RAIZ, text=True, stderr=subprocess.DEVNULL).strip()
        sujo = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=RAIZ, text=True, stderr=subprocess.DEVNULL).strip()
        return f"{sha}-dirty" if sujo else sha
    except Exception:
        return "(sem git)"


def write_provenance(fig: figspec_ns.Figure) -> Path:
    linhas = [
        f"figura:  {fig.slug}",
        f"secao:   {fig.section}",
        "estudo:  NavierStokes (bloco de projecao)",
        "fontes:",
        *[f"  - {s}" for s in fig.sources],
        # Aqui ha um SHA so, e nao dois como no lado de Darcy: neste
        # repositorio o JSON da campanha nao carrega `git_sha`, de modo que o
        # unico SHA recuperavel e o da arvore que desenhou a figura.
        f"git_sha (figura, arvore que desenhou): {git_sha()}",
        "",
        f"erro medido: {fig.reference}",
        f"eixo x:  {fig.x}",
        f"eixo y:  {fig.y}",
        f"paineis: {fig.panels or '(painel unico)'}",
        "",
        "campos consumidos:",
        *[f"  - {c}" for c in fig.fields],
        "",
        "excluidos pelo crivo:",
        *[f"  - {c}" for c in fig.excluded],
        "",
        "o que a legenda precisa dizer:",
        f"  {fig.caption_hint}",
    ]
    destino = PLOTS / f"{fig.slug}.prov.txt"
    destino.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return destino


def main() -> int:
    base = Path(__file__).resolve().parent
    for nome in MODULOS:
        print(f"── {nome}")
        try:
            runpy.run_path(str(base / f"{nome}.py"), run_name="__main__")
        except SystemExit as exc:
            if exc.code:
                return int(exc.code)

    faltando = []
    for fig in figspec_ns.FIGURES:
        if not (PLOTS / f"{fig.slug}.pdf").exists():
            faltando.append(fig.slug)
            continue
        write_provenance(fig)
        print(f"   {fig.slug}.pdf + proveniencia")
    if faltando:
        print(f"   AUSENTES: {', '.join(faltando)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
