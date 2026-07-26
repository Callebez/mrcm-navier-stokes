# Relatório — Tarefa 1: Separar as triplas do sistema de interface em parte-matriz e parte-RHS

## 1. Identificação
- **Tarefa:** 1 — Separar as triplas do sistema de interface em parte-matriz e parte-RHS
- **Data:** 2026-07-26
- **Commit antes:** `577d9bb` (`git rev-parse --short HEAD` antes de começar, na branch
  `feature/quadtree-mprm`)
- **Commit depois:** preenchido após `git log --oneline -1` logo após o commit do Passo 6
- **Arquivos criados:**
  - `mprm/triplets.py`
  - `tests/test_triplets.py`
  - `docs/superpowers/reports/01-triplets.md` (este arquivo)
- **Arquivos modificados:** nenhum arquivo existente foi alterado nesta tarefa.

## 2. Estado herdado
Herdado de `docs/superpowers/reports/00-baseline.md`: branch `feature/quadtree-mprm`
criada a partir de `2816cfd`, com o `ImportError` de `FineProjector` e o teste de
transporte desatualizado já corrigidos. Baseline real da suíte: **19 passed**
(não 21 — §C do plano está incorreto nesse número, ver seção 7 do relatório 00).
Nada da arquitetura nova existia: sem `mprm/triplets.py`, sem `mprm/decomposition.py`;
`MPrMProjector` remontava `A_Γ` a cada passo.

Assinatura pública consumida nesta tarefa, copiada verbatim do código
(`mrcm/global_assembly.py:238-320`, fora deste repositório — Restrição Global #4,
não modificado):

```python
def compute_subdomain_triplets(cfg: "MRCMConfig", skeleton: "Skeleton",
                               dof_mgr: "DOFManager", el: int,
                               particular_el, homogeneous_el,
                               face_mass_el: dict, cofunc_flux_el: dict,
                               cofunc_pres_el: dict):
    """Returns (rows, cols, vals, b_idx, b_vals) numpy arrays."""
```

Também consumidos, verbatim:

```python
# mrcm/topology.py:506-527
def gamma_incidences(self, sub: int) -> "list[GammaIncidence]":
    ...  # devolve GammaIncidence(side, seg, ell, bbox)

# mrcm/topology.py:600-604
def flux_index(self, edge: int, idf: int) -> int:
    return edge * self.ndofsU + idf

def pressure_index(self, edge: int, idf: int) -> int:
    return self.NunkU + edge * self.ndofsP + idf
```

E de `mprm/projector.py:72-129` (dentro deste repositório): atributos
`self.skel` (`Skeleton`), `self.dofm` (`DOFManager`), `self.my_subs`,
`self.ws` (workspace por subdomínio com chaves `"homogeneous"`, `"face_mass"`,
`"cofunc_flux"`, `"cofunc_pres"`), `self.solvers`, `self.src_fn`, `self.L_forms`,
`self.cell_map`.

## 3. Regressão ANTES da execução

Comando exato:

```bash
source ~/venv-firedrake/bin/activate
python -m pytest tests/ -q
```

| Suíte | Baseline esperado (herdado da Tarefa 0) | Obtido antes desta tarefa | Igual? |
|---|---|---|---|
| `pytest tests/ -q` | 19 passed | 19 passed (confirmado rodando com `--ignore=tests/test_triplets.py`, já que o arquivo já estava criado nesta sessão de trabalho) | sim |

Nota: o baseline real herdado é **19 passed**, não os 21 que o texto do brief cita
como referência (o brief foi escrito antes da Tarefa 0 corrigir o número de §C do
plano). Toda comparação nesta tarefa usa 19 como base, conforme
`docs/superpowers/reports/00-baseline.md` §9.

## 4. O que foi feito

1. Lido `mrcm/global_assembly.py:238-320` (função `compute_subdomain_triplets`) e
   confirmado, termo a termo, que:
   - `rows/cols/vals` dependem apenas de `homogeneous_el`, `face_mass_el` e das
     cofunções (`cofunc_flux_el`, `cofunc_pres_el`) — nenhuma referência a
     `particular_el` nesse trecho.
   - `b_idx/b_vals` dependem apenas de `particular_el[0]` (via `u_bar_arr`) e das
     cofunções — nenhuma referência a `homogeneous_el` ou `face_mass_el`.
   - A ordem de preenchimento de `b_idx` é: para cada incidência Γ (`inc` no laço
     `gamma_incidences`), primeiro todos os `idf` de fluxo (`ndofsU` entradas),
     depois todos os `idf` de pressão (`ndofsP` entradas) — intercalado *dentro*
     do laço de incidências, não agrupado por tipo ao final.
2. Criado `tests/test_triplets.py` exatamente como no brief (Step 1), sem
   alterações — as assinaturas e nomes de atributos do `MPrMProjector` batiam
   com o código real.
3. Rodado o teste e confirmado o `ModuleNotFoundError` esperado (Step 2).
4. Criado `mprm/triplets.py` copiando o corpo do brief (Step 3) verbatim: as
   duas funções reproduzem exatamente a lógica de `compute_subdomain_triplets`,
   apenas particionada — `subdomain_matrix_triplets` acumula `rows/cols/vals`
   com o mesmo laço duplo (incidências × homogêneas), e `subdomain_rhs`
   acumula `b_idx/b_vals` com o mesmo laço de incidências, preservando a ordem
   fluxo-antes-de-pressão dentro de cada incidência (ver a "Atenção à ordem" do
   brief, Step 4 — o código do brief já respeitava essa ordem sem necessidade
   de ajuste).
5. Nenhum ajuste foi necessário em relação ao texto verbatim do brief: a
   implementação copiada passou de primeira, sem divergência de ordenação em
   `b_idx` nem de valores em `vals`/`b_vals` (ver seção 5 — diffs bit-idênticos,
   `max|diff| = 0.0`).
6. Rodada a suíte completa (Step 5): `21 passed` (19 do baseline + 2 novos),
   sem nenhuma regressão nos 19 testes pré-existentes.

## 5. Testes criados nesta tarefa

### `test_matriz_mais_rhs_reproduz_compute_subdomain_triplets`
- **Arquivo:** `tests/test_triplets.py:52`
- **O que valida:** que `subdomain_matrix_triplets` + `subdomain_rhs`, somados,
  reproduzem exatamente `(rows, cols, vals, b_idx, b_vals)` de
  `compute_subdomain_triplets`, para os 4 subdomínios de uma decomposição
  cartesiana 2×2, com uma fonte concreta (`∇·w*` amostrado de
  `np.random.default_rng(0)`).
- **Resultado esperado (escrito ANTES de rodar):** `rows`/`cols`/`b_idx` bit-idênticos
  (`np.array_equal`); `vals`/`b_vals` iguais dentro de `atol=1e-14, rtol=0`.
- **Execução 1 (antes da implementação):**
  ```
  ModuleNotFoundError: No module named 'mprm.triplets'
  ```
  (mensagem literal obtida com `python -m pytest tests/test_triplets.py -q`).
- **Execução 2 (após a implementação):** `2 passed in 3.22s` (`PASS` para os dois
  testes deste arquivo).
- **Valor numérico obtido:** `max|vals - ref| = 0.0` e `max|b_vals - ref| = 0.0`
  para os 4 subdomínios (`isub=0..3`), medido em execução isolada fora do
  pytest (script ad hoc, ver seção 8). `rows`, `cols` e `b_idx` bit-idênticos
  (`np.array_equal` `True`) para todos os subdomínios; `len(rows)=64`,
  `len(b_idx)=8` por subdomínio nesta configuração (`Ndomx=Ndomy=2`,
  `degreeU=degreeP=1`).

### `test_parte_matriz_independe_da_fonte`
- **Arquivo:** `tests/test_triplets.py:86`
- **O que valida:** que `subdomain_matrix_triplets` do subdomínio 0 é idêntica
  para duas fontes diferentes (`∇·w*` amostrado de `np.random.default_rng(1)`,
  duas realizações), confirmando que a parte-matriz não depende da fonte — o
  ponto central da Fase 1.
- **Resultado esperado (escrito ANTES de rodar):** `rows`/`cols` bit-idênticos
  entre as duas fontes; `vals` iguais dentro de `atol=1e-14, rtol=0`.
- **Execução 1 (antes da implementação):** mesmo `ModuleNotFoundError: No module
  named 'mprm.triplets'` (falha de coleta do módulo inteiro, já que o import
  está no topo do arquivo).
- **Execução 2 (após a implementação):** `PASS`.
- **Valor numérico obtido:** `max|vals_a - vals_b| = 0.0` (bit-idêntico, não
  apenas dentro da tolerância) — esperado, já que a mesma sequência de
  operações de ponto flutuante é executada, mudando apenas o `Function`
  `particular_el` que a parte-matriz nunca lê.

## 6. Regressão DEPOIS da execução

Comando exato:

```bash
source ~/venv-firedrake/bin/activate
python -m pytest tests/ -q
```

Saída resumida:

```
21 passed, 3 warnings in 8.40s
```

| Suíte | Antes | Depois | Igual? |
|---|---|---|---|
| `pytest tests/ -q` | 19 passed | 21 passed (19 + 2 novos) | sim — sem regressão, +2 testes |

Confirmado também `19 passed` ao rodar com `--ignore=tests/test_triplets.py`,
isolando que os 2 testes novos são exatamente a diferença. Nenhum teste
pré-existente mudou de status. Os 3 warnings são os mesmos `UmfpackWarning`
("(almost) singular matrix") já presentes no baseline da Tarefa 0
(`tests/test_projector.py::test_projector_impermeavel_problem_none_roda`,
`tests/test_run_regression.py::test_run_cavity_quick_conserva_massa`,
`tests/test_run_regression.py::test_run_cavity_fine_vs_mprm_proximos`) — não
introduzidos por esta tarefa.

## 7. Desvios do plano

Nenhum desvio de implementação: o código de `mprm/triplets.py` e de
`tests/test_triplets.py` foi copiado verbatim do brief e passou sem exigir
nenhum ajuste (nem de ordenação de `b_idx`, nem de valores). O único desvio é
o já documentado na Tarefa 0 e herdado por esta: o brief cita "23 passed"
(21 do baseline antigo + 2 novos) como resultado esperado do Step 5; o valor
real obtido é **21 passed** (19 do baseline real + 2 novos), consistente com
`docs/superpowers/reports/00-baseline.md` §7 e §9.

## 8. Referências consultadas

- **Código interno:** `mrcm/global_assembly.py:238-320` (`~/mrcm/mrcm-firedrake`,
  fora deste repositório) — função `compute_subdomain_triplets` a dividir;
  confirmado que `rows/cols/vals` só usam `homogeneous_el`/`face_mass_el`/
  cofunções e que `b_idx/b_vals` só usam `particular_el`/cofunções; confirmada a
  ordem de preenchimento de `b_idx` (fluxo antes de pressão, dentro do laço de
  incidências).
- **Código interno:** `mrcm/topology.py:506-527` — `Skeleton.gamma_incidences`,
  retorna `list[GammaIncidence]`.
- **Código interno:** `mrcm/topology.py:64-75` — `GammaIncidence` (NamedTuple
  `side, seg, ell, bbox`).
- **Código interno:** `mrcm/topology.py:582-611` — `DOFManager`: `flux_index`,
  `pressure_index`, `NunkU`, `n_total`.
- **Código interno:** `mprm/projector.py:72-129` — atributos de
  `MPrMProjector` consumidos pelo teste (`skel`, `dofm`, `my_subs`, `ws`,
  `solvers`, `src_fn`, `L_forms`, `cell_map`).
- **Documento interno:**
  `.superpowers/sdd/2026-07-25-mprm-quadtree-von-karman/task-1-brief.md`
  (íntegra) — código-fonte verbatim de teste e implementação.
- **Documento interno:**
  `docs/superpowers/plans/2026-07-25-mprm-quadtree-von-karman.md` §B
  (linhas 46-121, formato deste relatório), §C (linhas 125-156, baseline —
  superado pelo valor real de 19 registrado na Tarefa 0).
- **Documento interno:** `docs/superpowers/reports/00-baseline.md` — baseline
  real herdado (19 passed) e estado da branch antes desta tarefa.
- **Artigo:** Guiraldello et al., *The Multiscale Robin Coupled Method*,
  Eqs. 58–59 — citado no brief como origem das linhas de pressão (Eq. 58) e de
  fluxo (Eq. 59) do sistema de interface; não foi necessário reabrir o PDF
  (`~/Dissertação/Refs/MRCM.pdf`) porque os comentários já existentes em
  `mrcm/global_assembly.py:253-256` citam as mesmas equações e o código foi
  conferido termo a termo contra elas.

## 9. Estado para a próxima tarefa

- **O que existe agora que antes não existia:**
  ```python
  # mprm/triplets.py
  def subdomain_matrix_triplets(cfg, skeleton, dof_mgr, el,
                                homogeneous_el, face_mass_el,
                                cofunc_flux_el, cofunc_pres_el) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
      """(rows, cols, vals) COO — int32, int32, float64. Independente da fonte."""

  def subdomain_rhs(cfg, skeleton, dof_mgr, el, particular_el,
                    cofunc_flux_el, cofunc_pres_el) -> tuple[np.ndarray, np.ndarray]:
      """(b_idx, b_vals) — int32, float64. Muda a cada passo com a fonte."""
  ```
- **O que a próxima tarefa deve consumir:** `mprm/triplets.py` (as duas funções
  acima) para montar `A_Γ` uma única vez no offline (spec §5 Fase 1, segunda
  metade) e fatorá-la (ex. `scipy.sparse.linalg.splu`), chamando
  `subdomain_rhs` a cada passo apenas para o vetor `b_Γ`. `compute_subdomain_triplets`
  do `mrcm` continua existindo e intocada — não removê-la, apenas parar de
  chamá-la no caminho online do `MPrMProjector`.
- **Armadilhas descobertas:** nenhuma armadilha de fato — o brief já continha o
  código correto (incluindo a ordenação de `b_idx`) e nenhum ajuste foi
  necessário. A única atenção real é não confundir o "23 passed" do texto do
  brief (herdado do "21" antigo de §C) com o alvo real, que é 19+2=21.
- **O que ainda NÃO existe:** `mprm/decomposition.py` não existe. `A_Γ` ainda
  não é montada nem fatorada uma única vez no offline — `mprm/triplets.py`
  apenas separa o cálculo das triplas; `MPrMProjector` ainda não foi alterado
  para consumir `subdomain_matrix_triplets`/`subdomain_rhs` em vez de
  `compute_subdomain_triplets` (isso é objeto da Tarefa 2). Nenhuma
  funcionalidade de quadtree foi tocada.
