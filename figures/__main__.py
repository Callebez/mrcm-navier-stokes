#!/usr/bin/env python3
"""Regenera as figuras do bloco de Navier-Stokes a partir dos JSON.

    python -m figures
"""
import runpy
import sys
from pathlib import Path

MODULOS = ["plot_olshanskii", "plot_par_misto", "plot_ghia"]


def main() -> int:
    base = Path(__file__).resolve().parent
    for nome in MODULOS:
        print(f"── {nome}")
        try:
            runpy.run_path(str(base / f"{nome}.py"), run_name="__main__")
        except SystemExit as exc:
            if exc.code:
                return int(exc.code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
