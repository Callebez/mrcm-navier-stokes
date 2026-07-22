"""Insere os paths necessários para o pytest importar mprm_core, common,
params e o pacote mrcm — independente do rootdir. Sem isto, `from common
import ...` e `import mprm_core` falham sob pytest (o import direto por
`python tests/test_x.py` continua funcionando via script-dir)."""
import os
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(TESTS_DIR)
MRCM_REPO = os.path.expanduser("~/mrcm/mrcm-firedrake")
for _p in (TESTS_DIR, ROOT_DIR, MRCM_REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)
