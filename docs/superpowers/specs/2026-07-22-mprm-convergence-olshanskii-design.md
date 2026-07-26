# MPrM — Estudo de Convergência Olshanskii — Design Spec

> **Plano:** `docs/superpowers/plans/2026-07-22-mprm-convergence-olshanskii.md`
> **Branch:** `mprm-convergence-olshanskii`
> **Data:** 2026-07-22

Especificação de design para o módulo `mprm/benchmarks/convergence.py` e os
scripts de teste que implementam o estudo de convergência do problema de
Olshanskii.

---

## 1. Arquitetura do módulo de convergência

### 1.1 Separação de responsabilidades

```
mprm/benchmarks/convergence.py   — funções puras de estudo (sem I/O)
tests/test_convergence_*.py      — scripts executáveis (CLI + I/O + plot)
results_alpha/convergence_olshanskii/ — saída JSON/PNG
docs/12-estudo-convergencia-olshanskii.md — writeup manual
```

O módulo `convergence.py` **não faz I/O** (não escreve arquivos, não plota).
Ele recebe parâmetros, executa os jobs do MRCM e retorna dicionários de
resultados. Os scripts de teste em `tests/` são responsáveis por parsing de
argumentos, formatação de saída, salvamento de JSON e geração de gráficos.

### 1.2 API do módulo `convergence.py`

```python
from typing import Optional
import numpy as np

def convergence_curve_hh(
    comm,
    Ndoms: list[int],
    nloc: int,
    degreeU: int,
    degreeP: int,
    alpha: float,
    basis_kind: str = "legendre_P",
    include_raw_mrcm: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Modo Hh: nloc fixo, Ndom varre → H e h proporcionais.

    Parameters
    ----------
    include_raw_mrcm : bool
        Se True, inclui também o erro da reconstrução BDM₁–DG₀ crua
        (sem downscaling) em cada nível, no campo `err_raw_mrcm`.

    Returns
    -------
    dict com estrutura:
        meta: dict com parâmetros do estudo
        cases: list[dict] — um por Ndom, cada um com:
            Ndom, H, h, N, err_vs_exact, err_vs_fine,
            err_fine_vs_exact, divmax, defect, wall_s
            (e err_raw_mrcm se include_raw_mrcm=True)
        rate_exact: dict {rate, r_squared, Hs, errors} ou None
        rate_fine: dict {rate, r_squared, Hs, errors} ou None
        rate_raw: dict (se include_raw_mrcm) ou None
    """

def convergence_curve_H(
    comm,
    Ndoms: list[int],
    h_fine_target: float,
    degreeU: int,
    degreeP: int,
    alpha: float,
    basis_kind: str = "legendre_P",
    verbose: bool = True,
) -> dict:
    """
    Modo H (multiescala): h ~ fixo, Ndom varia → H→0.

    Para cada Ndom, calcula nloc = round(1/(h_fine_target * Ndom))
    e N = Ndom * nloc. A malha fina real h = 1/N difere ligeiramente
    do target devido ao arredondamento; o h real é registrado.

    Returns
    -------
    dict com mesma estrutura de convergence_curve_hh,
    com campo adicional h_target nos metadados.
    """

def convergence_curve_h(
    comm,
    Ndom: int,
    nlocs: list[int],
    degreeU: int,
    degreeP: int,
    alpha: float,
    basis_kind: str = "legendre_P",
    verbose: bool = True,
) -> dict:
    """
    Modo h (saturação): H fixo (Ndom fixo), nloc varre → h→0.

    Returns
    -------
    dict: cases ordenados por h decrescente.
        rate_exact e rate_fine medidos vs h (não vs H).
    """

def convergence_curve_p(
    comm,
    Ndom: int,
    nloc: int,
    ks: list[int],
    alpha: float,
    basis_kind: str = "legendre_P",
    asymmetric: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Modo p (grau): malha fixa, graus crescentes.

    Se asymmetric=False, usa (P_k, U_k) para cada k.
    Se asymmetric=True, ks deve ser lista de pares [(degU, degP), ...].

    Returns
    -------
    dict: cases ordenados por k (ou por degU+degP).
        Sem rate (convergência-p não é algébrica).
    """
```

### 1.3 Reconstrução BDM₁–DG₀ crua (sem downscaling)

Para o experimento crucial T10, é necessário avaliar o campo de velocidade
BDM₁–DG₀ diretamente na malha MAC, sem passar pelo downscaling.

**Estratégia de implementação:**

O `MPrMProjector.project()` resolve o sistema global de interface e reconstrói
os campos locais. O campo `u_h` resultante é um `firedrake.Function` no espaço
BDM₁. Para avaliar nas faces MAC:

1. Para faces verticais (componente x): avalia-se `u_h[0]` nos pontos médios
   das faces verticais da malha MAC.
2. Para faces horizontais (componente y): avalia-se `u_h[1]` nos pontos médios
   das faces horizontais.

Como BDM₁ tem 1 DOF de fluxo normal por aresta, e a malha Firedrake é uma
triangulação não estruturada que cobre o domínio, a avaliação ponto a ponto
é feita com `firedrake.Function.at()`:

```python
def evaluate_bdm1_on_mac_faces(u_h, Nx, Ny, Lx=1.0, Ly=1.0):
    """Avalia campo BDM₁ nas faces da malha MAC Nx×Ny."""
    hx, hy = Lx / Nx, Ly / Ny
    # Faces verticais: componentes x nos pontos (xf, yc)
    xf = np.linspace(0.0, Lx, Nx + 1)
    yc = (np.arange(Ny) + 0.5) * hy
    Xv, Yv = np.meshgrid(xf, yc, indexing="ij")
    u1_mac = np.array(u_h.at(Xv.flatten(), Yv.flatten())[0]).reshape(Nx+1, Ny)

    # Faces horizontais: componentes y nos pontos (xc, yf)
    xc = (np.arange(Nx) + 0.5) * hx
    yf = np.linspace(0.0, Ly, Ny + 1)
    Xh, Yh = np.meshgrid(xc, yf, indexing="ij")
    u2_mac = np.array(u_h.at(Xh.flatten(), Yh.flatten())[1]).reshape(Nx, Ny+1)

    return u1_mac, u2_mac
```

**Cuidado com contorno:** Nos contornos impermeáveis, `u·n = 0` é imposto
fortemente. A avaliação com `.at()` deve respeitar isso; caso contrário,
aplicar a máscara de contorno manualmente.

### 1.4 Isolamento do Firedrake

Seguindo a restrição global de que `import mprm` não deve puxar Firedrake,
o módulo `convergence.py` importa Firedrake **apenas dentro das funções** que
o utilizam (padrão já estabelecido em `mprm_core.py` e `mprm/projector.py`).

```python
# convergence.py — imports no topo
from __future__ import annotations
import time
import numpy as np

# imports de Firedrake são locais às funções que montam o MRCM
```

---

## 2. Design dos scripts de teste

### 2.1 Interface de linha de comando unificada

Cada script de teste em `tests/` segue o padrão estabelecido por
`tests/test_convergence_hh.py`:

- Usa `tests/params.py` (`CommonParams`, `parse_args`, `parse_int_list`,
  `parse_float_list`) para parsing.
- Flags padrão: `--quick` (smoke test com poucos níveis), `--outdir` (diretório
  de saída), `--no-plot` (suprimir gráficos).
- Flags específicas: `--Ndoms`, `--nloc`, `--degreeU`, `--degreeP`, `--alpha`,
  `--alphas`, `--ks`, etc.
- Saída: JSON em `results_alpha/convergence_olshanskii/` + PNG opcional.

### 2.2 Exemplo: `test_convergence_hh_alpha.py`

```python
"""
tests/test_convergence_hh_alpha.py — Modo Hh × α (Experimento E1).

Varre α ∈ {0.01, 0.1, 1, 10, 100, 1000} com (P₁,U₁), nloc=16,
Ndom ∈ {2,4,8,16}. Para cada α, ajusta a taxa de convergência log-log
e gera uma curva no gráfico comparativo.

Uso:
  python tests/test_convergence_hh_alpha.py
  python tests/test_convergence_hh_alpha.py --quick
  mpiexec -n 4 python tests/test_convergence_hh_alpha.py
"""

def run_study(comm, alphas, Ndoms, nloc, degreeU, degreeP, ...) -> dict:
    """Para cada α, chama convergence_curve_hh e coleta taxa."""
    results = {"meta": ..., "curves": []}
    for a in alphas:
        curve = convergence_curve_hh(comm, Ndoms, nloc, degreeU, degreeP, a)
        results["curves"].append({"alpha": a, **curve})
    return results
```

### 2.3 Geração de gráficos

Cada script gera UM gráfico principal (PNG) com:
- Escala log-log: eixo X = H (ou h, ou k), eixo Y = erro relativo.
- Anotação da taxa de convergência e R² em cada curva.
- Cores consistentes com `tests/common.py` (`COLORS`, `MARKERS`).
- Grade de referência (O(H¹), O(H²), O(H³)) em linha tracejada.

---

## 3. Estrutura dos arquivos JSON de saída

### 3.1 Schema unificado

Cada arquivo JSON de saída segue o schema:

```json
{
  "meta": {
    "study": "hh_convergence_Hh_alpha",
    "date": "2026-07-22T...",
    "n_ranks": 4,
    "nloc": 16,
    "degreeU": 1,
    "degreeP": 1,
    "basis_kind": "legendre_P",
    "Ndoms": [2, 4, 8, 16],
    "dt": 1.0,
    "reference_exact": "analytic Olshanskii field",
    "reference_fine": "undecomposed MAC projection, same N"
  },
  "curves": [
    {
      "alpha": 0.01,
      "rate_exact": {"rate": 1.98, "r_squared": 0.9999, "Hs": [...], "errors": [...]},
      "rate_fine": {"rate": 1.98, "r_squared": 0.9999, "Hs": [...], "errors": [...]},
      "cases": [
        {"Ndom": 2, "H": 0.5, "h": 0.03125, "N": 32,
         "err_vs_exact": 0.504, "err_vs_fine": 0.504,
         "divmax": 3.8e-12, "defect": 2.9e-14, "wall_s": 2.3}
      ]
    }
  ]
}
```

Para experimentos que não produzem curvas múltiplas (E4, E5, E6), o campo
`curves` é omitido e `rate_*` e `cases` ficam no nível raiz (formato atual
de `convergence_hh.json`).

### 3.2 Nomenclatura de arquivos

```
results_alpha/convergence_olshanskii/
├── e1_hh_alpha.json
├── e1_hh_alpha.png
├── e2_hh_degrees_sym.json
├── e2_hh_degrees_sym.png
├── e3_hh_degrees_asym.json
├── e3_hh_degrees_asym.png
├── e4_H_multiscale.json
├── e4_H_multiscale.png
├── e5_h_saturation.json
├── e5_h_saturation.png
├── e6_p_degree.json
├── e6_p_degree.png
├── e7_hh_nloc.json
├── e7_hh_nloc.png
├── e10_downscaling_on_off.json
├── e10_downscaling_on_off.png
└── convergence_olshanskii_report.md
```

---

## 4. Script de sumário automático

### 4.1 `scripts/generate_convergence_report.py`

Lê todos os JSONs em `results_alpha/convergence_olshanskii/` e gera:

1. **Tabela Markdown** com todos os experimentos:
   ```
   | Exp | Modo | Parâmetros | Njobs | Taxa | R² | Tempo (s) |
   |-----|------|-----------|-------|------|----|-----------|
   | E1  | Hh×α | (P₁,U₁), nloc=16 | 24 | 1.99 | 1.000 | 8234 |
   ```

2. **Figura consolidada** (grid 2×3) com as 6 curvas principais.

3. **Seção de achados** destacando:
   - Melhor combinação (α, graus) para erro mínimo
   - Regiões de superconvergência (taxa > esperada)
   - Regiões de degradação (taxa < esperada)
   - Confirmação/refutação da hipótese do downscaling

---

## 5. Considerações de desempenho

### 5.1 Custo computacional por job

O custo de cada job é dominado pelo **offline do MRCM** (fatoração das matrizes
locais + montagem do sistema de interface). Tempos típicos (1 core):

| Ndom | nloc | N | Graus | Offline (s) | Online (s) |
|------|------|---|-------|-------------|------------|
| 4 | 16 | 64 | (1,1) | ~1.5 | ~0.02 |
| 8 | 16 | 128 | (1,1) | ~4 | ~0.05 |
| 16 | 16 | 256 | (1,1) | ~20 | ~0.15 |
| 4 | 16 | 64 | (2,2) | ~3 | ~0.05 |
| 8 | 16 | 128 | (2,2) | ~12 | ~0.15 |
| 16 | 16 | 256 | (2,2) | ~80 | ~0.5 |
| 4 | 16 | 64 | (3,3) | ~10 | ~0.15 |
| 16 | 16 | 256 | (3,3) | ~300 | ~2 |

### 5.2 Estratégia de paralelização

- **MPI (4 ranks):** `mpiexec -n 4 python tests/test_convergence_*.py`
  - O `mrcm` com `executor_kind="mpi"` distribui a montagem/solução do sistema
    de interface entre os ranks.
  - Speedup esperado: ~3× com 4 ranks para Ndom ≥ 4.
- **Embarassingly parallel por parâmetro:** Os experimentos E1–E3 têm loops
  externos em α ou graus — cada iteração é independente e pode ser distribuída
  em jobs separados (ex.: GNU parallel).

### 5.3 Memória

- Ndom=16, nloc=16, (P₃,U₃): ~3000 DOFs de interface → matriz densa ~72 MB.
  Cabe folgadamente em RAM.
- Ndom=2, nloc=128 (modo H): ~16k triângulos por subdomínio → matriz local
  ~200k × 200k (esparsa). Memória ~500 MB por subdomínio. Ainda viável.

---

## 6. Testes de regressão

### 6.1 Teste de não-regressão do baseline

```python
# tests/test_convergence_regression.py (adicionar ao existente)
def test_hh_baseline_reproduced():
    """E1 com α=1 deve reproduzir convergence_hh.json (taxa 1.99 ± 0.05)."""
    r = convergence_curve_hh(comm, [2,4,8,16], 16, 1, 1, 1.0)
    assert abs(r["rate_fine"]["rate"] - 2.0) < 0.1
    assert r["rate_fine"]["r_squared"] > 0.99
```

### 6.2 Smoke tests (quick mode)

Cada script implementa `--quick`:
- Ndom ∈ {2, 4} (2 níveis — suficiente para verificar que o código roda, mas
  não para taxa confiável).
- Tempo total < 30 s por script.
- Não salva JSON (só imprime no stdout).

---

## 7. Extensibilidade futura

O design prevê extensões que estão fora do escopo deste plano mas que podem
ser adicionadas posteriormente:

1. **Modo Hh para campos não-suaves:** Substituir o campo C^∞ de Olshanskii
   por um campo com singularidade (ex.: canto, fonte pontual) e medir a
   degradação da taxa.
2. **Modo Hh com obstáculo:** Usar o problema de von Kármán (após correções
   de acurácia) como benchmark de convergência para escoamentos com obstáculo.
3. **Convergência temporal:** Medir a taxa de convergência no tempo do esquema
   de Chorin (teoricamente O(δt) para a pressão, O(δt²) para a velocidade).
4. **Comparação com outros resolvedores:** DG0 vs RT0, BDM₁ vs BDM₂,
   downscaler FV vs downscaler FEM.
