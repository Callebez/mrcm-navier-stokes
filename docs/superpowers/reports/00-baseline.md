# Relatório — Tarefa 0: Baseline verde, branch e infraestrutura de relatório

## 1. Identificação
- **Tarefa:** 0 — Baseline verde, branch e infraestrutura de relatório
- **Data:** 2026-07-26
- **Commit antes:** `2816cfd` (`git rev-parse --short HEAD` antes de começar, na branch
  `feature/quadtree-mprm` recém-criada a partir do HEAD do worktree)
- **Commit depois:** este próprio commit (o sha não pode ser conhecido antes de
  commitar o arquivo que o registraria; obtenha com `git log --oneline -1` logo
  após o Passo 7 do brief)
- **Arquivos criados:** `docs/superpowers/reports/00-baseline.md` (este arquivo)
- **Arquivos modificados:**
  - `mprm/__init__.py:49` — adicionada `FineProjector` ao bloco de reexports
    `from mprm.flow import (...)` (linhas 47-53)
  - `tests/test_transport.py:50-53` — `mac_transport(u, v, prob, dt)` passou a
    `mac_transport(u, v, prob, dt, convective_order=1)`, com comentário
    explicando o motivo

## 2. Estado herdado
Não há relatório anterior — esta é a primeira tarefa do plano. O estado herdado
é o próprio repositório na branch `mprm-opt-parallel` (commit `2816cfd`), sem
nenhuma infraestrutura de relatório em `docs/superpowers/reports/` (diretório
inexistente antes desta tarefa).

Assinaturas públicas consumidas nesta tarefa, copiadas verbatim do código:

```python
# mprm/flow.py:87-96
class FineProjector:
    """Referência: projeção pela malha fina global (Poisson MAC 5 pontos,
    Neumann homogêneo, pino de pressão) — o 'fine mesh solver' do artigo."""

    def __init__(self, Nx, Ny, hx, hy, dt):
        ...

    def step(self, w1, w2):
        ...
```

```python
# mprm/transport.py:96
def mac_transport(u, v, problem: "FlowProblem", dt, convective_order=2, limiter="minmod"):
```

## 3. Regressão ANTES da execução

Comando exato (Passo 2 do brief):

```bash
source ~/venv-firedrake/bin/activate
python -m pytest tests/ -q
```

Saída resumida (execução simples, sem flags de coleta):

```
ERROR tests/test_alpha_hh.py
ERROR tests/test_convergence_hh.py
!!!!!!!!!!!!!!!!!!! Interrupted: 2 errors during collection !!!!!!!!!!!!!!!!!!!!
2 errors in 0.54s
```

Ambos os erros: `ImportError: cannot import name 'FineProjector' from 'mprm_core'`
(`mprm_core.py` faz `from mprm import *`; a classe existe em `mprm/flow.py:87`
mas não estava no bloco de reexports de `mprm/__init__.py:47-52`).

Para comparar com a metodologia de medição usada em §C do plano (que ignora os
2 módulos com erro de coleta), reexecutei com `--continue-on-collection-errors`
antes de aplicar qualquer correção (via `git stash` temporário dos dois fixes já
escritos, seguido de `git stash pop` — nenhuma mudança foi perdida):

```
FAILED tests/test_transport.py::test_cavity_transporte_bit_identico_ao_legado
ERROR tests/test_alpha_hh.py
ERROR tests/test_convergence_hh.py
1 failed, 18 passed, 3 warnings, 2 errors in 8.23s
```

| Suíte | Baseline esperado (§C, medido em `fd62b2d`) | Obtido antes desta tarefa (`2816cfd`) | Igual? |
|---|---|---|---|
| `pytest tests/ -q` (execução simples) | `19 tests collected, 2 errors` → sessão interrompida | `2 errors during collection` → `Interrupted` | sim |
| `pytest tests/ -q --continue-on-collection-errors` | `1 failed, 18 passed, 3 warnings` | `1 failed, 18 passed, 3 warnings, 2 errors` | sim |

O estado pré-existente bate exatamente com o que §C do plano registra. Os 18
passed + 1 failed = 19 testes reais existentes na árvore `tests/`; os 2 erros
de coleta não escondiam testes adicionais (ver seção 7).

## 4. O que foi feito

1. Criada a branch `feature/quadtree-mprm` a partir do commit `2816cfd`
   (`git checkout -b feature/quadtree-mprm`), conforme Global Constraint #3 do
   plano.
2. Reproduzidas as duas falhas pré-existentes (seção 3).
3. Adicionada `FineProjector` ao bloco `from mprm.flow import (...)` em
   `mprm/__init__.py`, tornando-a importável via `mprm_core` (que faz
   `from mprm import *`) e via `from mprm import FineProjector`.
4. Corrigida a chamada em `test_cavity_transporte_bit_identico_ao_legado` para
   fixar `convective_order=1`, já que `_wstar_legacy` é upwind de 1ª ordem e o
   default de `mac_transport` virou `convective_order=2` no commit `4e40213`
   (o teste media equivalência de estêncil, e passou a comparar coisas
   diferentes sem essa fixação).
5. Rodada a suíte completa: resultado `19 passed, 3 warnings`, não os
   `21 passed` que §C do plano declara como baseline-alvo.
6. Investigada a divergência 19 vs. 21 antes de prosseguir (não presumida):
   contei `grep -rn "^def test_" tests/*.py` → exatamente 19 ocorrências em
   toda a árvore `tests/`. Os 4 arquivos com prefixo `test_` que não geram
   nenhum item de coleta (`test_alpha_hh.py`, `test_alpha_cavity.py`,
   `test_convergence_hh.py`, `test_hipoteses_shedding.py`) contêm apenas
   funções de script de estudo paramétrico (`run_study`, `_run_one_case`,
   `plot_results`, `main`, `run_one`), nenhuma função `def test_*` nem classe
   `Test*` — confirmado também por `pytest --collect-only` retornando
   `0 items`/`no tests collected` para eles, sem erro. Conferido que essa
   composição de arquivos é idêntica em todos os commits recentes
   (`2816cfd`, `fd62b2d`, `a2b145b`, `f476269`, `920a798` — `git ls-tree`),
   ou seja, não é uma anomalia momentânea deste worktree.
7. Resolução adotada pelo orquestrador, aplicando a regra §A do plano ("se algo
   no plano contradisser o código real, o código real vence: registre a
   divergência e prossiga com a correção mínima"): o baseline real é
   `19 passed`, não `21 passed`. §C do plano está incorreto nesse número
   específico — presumia que corrigir os 2 erros de coleta revelaria 2 testes
   adicionais, mas os módulos em questão nunca continham testes pytest.
8. Escrito este relatório e preparado o commit único do Passo 7.

## 5. Testes criados nesta tarefa

Nenhum. Esta tarefa não cria testes novos — corrige dois problemas
pré-existentes (reexport ausente de `FineProjector`; teste de transporte
desatualizado) em testes que já existiam em `tests/test_alpha_hh.py`,
`tests/test_convergence_hh.py` (indiretamente, via `mprm/__init__.py`) e
`tests/test_transport.py`. Ver seção 4.

## 6. Regressão DEPOIS da execução

Comando exato:

```bash
source ~/venv-firedrake/bin/activate
python -m pytest tests/ -q
```

Saída resumida:

```
19 passed, 3 warnings in 7.51s
```

| Suíte | Antes | Depois | Igual? |
|---|---|---|---|
| `pytest tests/ -q` (execução simples) | `2 errors during collection` → `Interrupted`, 0 testes rodados | `19 passed, 3 warnings` | não — de quebrado para verde |
| `pytest tests/ -q --continue-on-collection-errors` | `1 failed, 18 passed, 3 warnings, 2 errors` | `19 passed, 3 warnings` (0 erros de coleta) | não — melhorou |

Testes cujo status mudou nominalmente:
- `tests/test_alpha_hh.py`, `tests/test_convergence_hh.py`: de `ERROR` na
  coleta (sessão inteira interrompida) para coleta bem-sucedida com `0 items`
  cada — não contêm funções de teste, são scripts de estudo paramétrico; sua
  correção elimina o erro de importação mas não adiciona testes à contagem.
- `tests/test_transport.py::test_cavity_transporte_bit_identico_ao_legado`: de
  `FAILED` (comparava upwind 1ª ordem contra `mac_transport` com default
  `convective_order=2`) para `PASSED` (fixado `convective_order=1`).
- Os demais 18 testes: sem mudança de status, já passavam antes e depois.

**Nota explícita:** §C do plano declara `21 passed` como baseline-alvo; o valor
real após esta tarefa é `19 passed`. Causa: 4 arquivos `tests/test_*.py`
(`test_alpha_hh.py`, `test_alpha_cavity.py`, `test_convergence_hh.py`,
`test_hipoteses_shedding.py`) são scripts de estudo paramétrico sem nenhuma
função `test_*`, não testes pytest — confirmado por `grep -rn '^def test_'` em
`tests/*.py` (19 ocorrências) e estável em toda a linhagem recente do branch.

## 7. Desvios do plano

O plano (§C) declara `21 passed` como "baseline alvo, válido a partir da Tarefa
0 e usado por todas as demais". O valor real obtido após corrigir exatamente as
duas falhas descritas no brief (reexport de `FineProjector`; ordem convectiva
fixada no teste legado) é `19 passed`, não `21`.

Causa raiz: §C presumia que, uma vez corrigido o `ImportError`, os dois módulos
`tests/test_alpha_hh.py` e `tests/test_convergence_hh.py` passariam a
contribuir testes à contagem. Na prática, nenhum dos dois — nem os outros dois
arquivos com prefixo `test_` presentes na árvore (`test_alpha_cavity.py`,
`test_hipoteses_shedding.py`) — contém qualquer função `def test_*` ou classe
`Test*`; são scripts de estudo paramétrico (funções `run_study`,
`_run_one_case`, `_run_one_level`, `_fit_rate`, `plot_results`, `main`,
`run_one`) que compartilham o diretório `tests/` e a convenção de nome
`test_*.py`, mas não são testes pytest. `pytest --collect-only` confirma
`0 items` para os quatro, sem erro, tanto antes quanto depois da correção do
`ImportError`.

Contagem real de testes pytest em toda a árvore (`grep -rn "^def test_"
tests/*.py`): 19, distribuídas em `test_downscaler_obstacle.py` (2),
`test_flow_problem.py` (5), `test_forces_strouhal.py` (3), `test_projector.py`
(2), `test_run_regression.py` (2), `test_transport.py` (4), `test_von_karman.py`
(1). Essa composição é idêntica nos commits `2816cfd`, `fd62b2d`, `a2b145b`,
`f476269` e `920a798` (`git ls-tree -r --name-only <sha> -- tests/`), ou seja, a
divergência não é causada por este worktree nem por qualquer ação desta tarefa
— é uma inconsistência do próprio texto do plano (§C) contra o código real.

Aplicando a regra §A do plano ("se algo no plano contradisser o código real, o
código real vence: registre a divergência ... e prossiga com a correção
mínima"), o orquestrador resolveu adotar `19 passed` como baseline real. Esta
tarefa prossegue com essa correção mínima (documentar e seguir em frente) em
vez de travar. Toda tarefa seguinte deve comparar contra `19 + N` (onde `N` é o
número de testes que a própria tarefa cria), não contra `21 + N`.

## 8. Referências consultadas

- **Código interno:** `mprm/flow.py:87-132` — classe `FineProjector`
  (`__init__(self, Nx, Ny, hx, hy, dt)`, `step(self, w1, w2)`), origem da
  necessidade do reexport.
- **Código interno:** `mprm/__init__.py:47-52` — bloco de reexports de
  `mprm.flow`, onde `FineProjector` estava ausente.
- **Código interno:** `mprm_core.py:1-6` — wrapper retrocompatível
  (`from mprm import *`); confirma que o reexport em `mprm/__init__.py` é
  suficiente para satisfazer `from mprm_core import FineProjector`.
- **Código interno:** `tests/test_transport.py:42-53` — teste desatualizado
  (`test_cavity_transporte_bit_identico_ao_legado`), estêncil `_wstar_legacy`.
- **Código interno:** `mprm/transport.py:96-112` — assinatura de
  `mac_transport(u, v, problem, dt, convective_order=2, limiter="minmod")` e
  docstring do parâmetro `convective_order`.
- **Código interno:** `tests/conftest.py:1-13` — mecanismo de `sys.path`;
  descartado como causa da divergência de contagem (não há hooks de coleta ou
  parametrização).
- **Documento interno:**
  `docs/superpowers/plans/2026-07-25-mprm-quadtree-von-karman.md` §A (linhas
  31-43, regra "código real vence"), §B (linhas 46-121, formato deste
  relatório), §C (linhas 125-156, baseline de testes e a tabela dos 3
  problemas pré-existentes).
- **Documento interno:**
  `.superpowers/sdd/2026-07-25-mprm-quadtree-von-karman/task-0-brief.md`
  (íntegra) — passos exatos executados nesta tarefa.
- **Histórico git:** `git ls-tree -r --name-only <sha> -- tests/` para
  `2816cfd`, `fd62b2d`, `a2b145b`, `f476269`, `920a798` — composição idêntica
  da árvore `tests/` em toda a linhagem recente. `git log --oneline -- tests/
  test_alpha_hh.py tests/test_convergence_hh.py` → introduzidos em `aefb134`,
  modificados em `920a798` ("estudo paramétrico + testes de hipóteses de
  shedding").

## 9. Estado para a próxima tarefa

Baseline verde: 19 passed com `python -m pytest tests/ -q` (não 21 — ver seção 7, §C do plano está incorreto). Toda tarefa seguinte compara contra 19 + N. Nada da arquitetura nova existe ainda: não há mprm/triplets.py nem mprm/decomposition.py. MPrMProjector ainda remonta A_Γ a cada passo.
