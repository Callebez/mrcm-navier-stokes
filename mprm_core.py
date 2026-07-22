"""mprm_core.py — wrapper retrocompatível.

Redireciona todos os imports para o pacote `mprm`.
Código novo deve usar `from mprm import ...` diretamente.
"""
from mprm import *  # noqa: F401, F403
