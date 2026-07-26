# MPrM sobre decomposição quadtree (von Kármán) — Plano de Implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolver o passo de projeção do MPrM sobre uma decomposição de domínio adaptativa (quadtree, adaptando apenas a escala grossa `H`) e medir, no benchmark de von Kármán, o que a adaptatividade entrega frente à decomposição cartesiana uniforme.

**Architecture:** Uma camada `Decomposition` passa a ser o único ponto do MPrM que conhece a topologia da decomposição; o caminho cartesiano vira um caso particular dela, o que torna o teste "árvore uniforme ≡ cartesiano" a rede de segurança de toda a refatoração. A malha fina `h` permanece uniforme e global (o transporte MAC não muda); cada folha recebe `nloc_i = H_i/h` células. Antes de qualquer quadtree, a matriz do sistema global de interface `A_Γ` — hoje remontada e refatorada a cada passo de tempo, embora seja constante — passa para o offline.

**Tech Stack:** Python 3.12, NumPy, SciPy (sparse/splu), Firedrake (via `~/venv-firedrake`), mpi4py, pytest. Pacote local `mprm` (`~/mrcm/NavierStokes`) sobre o pacote `mrcm` (`~/mrcm/mrcm-firedrake`).

---

## Global Constraints

Toda tarefa herda implicitamente estas regras.

1. **Ambiente:** todo comando Python roda com `source ~/venv-firedrake/bin/activate` antes. O `PYTHONPATH` não precisa ser exportado: `tests/conftest.py` insere `~/mrcm/mrcm-firedrake` no `sys.path`.
2. **Diretório de trabalho:** `~/mrcm/NavierStokes`.
3. **Branch:** `feature/quadtree-mprm`, criado a partir de `mprm-opt-parallel` na Tarefa 0. Nunca commitar em `main`.
4. **Não modificar o `~/mrcm/mrcm-firedrake`.** O spec §4.6 verificou que nada é necessário lá. Se alguma tarefa parecer exigir alteração no `mrcm`, **pare e reporte** em vez de editar.
5. **Idioma:** docstrings, comentários, mensagens de commit e relatórios em português (pt-BR), seguindo o padrão do repositório. Nomes de código em inglês quando o entorno já usa inglês (`nlocx`, `skeleton`, `flux`).
6. **Precisão numérica:** toda comparação de equivalência entre caminho novo e caminho antigo usa `atol=1e-12, rtol=0` sobre campos de velocidade e sobre `X`. Comparações sequencial↔MPI usam `atol=1e-10` (a ordem de acumulação em ponto flutuante difere).
7. **Convenção de lados do `mrcm`:** `0=bottom, 1=right, 2=top, 3=left`. Definida em `mrcm/quadtree.py:62-63` e `mrcm/topology.py`. Não inventar outra.
8. **Convenção da malha MAC:** `u` tem shape `(Nx+1, Ny)`, `v` tem `(Nx, Ny+1)`, índice `x` primeiro. Ver `mprm/transport.py:26-32`.
9. **TDD obrigatório:** todo passo de implementação é precedido por um teste que falha primeiro. Um passo "rodar o teste e ver falhar" que **passa** significa que o teste está errado — corrija o teste antes de prosseguir.
10. **Commits frequentes:** um commit por tarefa no mínimo, com a mensagem indicada no último passo da tarefa.
11. **Relatório obrigatório:** ao terminar cada tarefa, escrever o relatório conforme §B abaixo. A tarefa **não está concluída** sem o relatório commitado.

---

## §A. Como usar este plano (leia antes de tudo)

Você é, provavelmente, um agente que não participou das tarefas anteriores. **Não releia o repositório inteiro.** Faça exatamente isto:

1. Leia esta seção §A, a §B (formato do relatório) e a §C (baseline de testes).
2. Leia o relatório da **última tarefa concluída** em `docs/superpowers/reports/`. Ele contém, por contrato, o estado herdado e as assinaturas que você vai consumir.
3. Leia **apenas a sua tarefa** neste plano.
4. Leia os arquivos citados no bloco "Referências" da sua tarefa — só esses.
5. Execute os passos na ordem, sem pular os passos de "rodar o teste e ver falhar".
6. Escreva o relatório e commite.

Se algo no plano contradisser o código real, o **código real vence**: registre a divergência no campo "Desvios" do relatório e prossiga com a correção mínima. Se a divergência invalidar a tarefa, pare e reporte.

---

## §B. Formato obrigatório do relatório

Cada tarefa produz `docs/superpowers/reports/NN-<slug>.md`, onde `NN` é o número da tarefa com dois dígitos. O arquivo tem **exatamente** estas nove seções, nesta ordem. Seções vazias devem conter a palavra `Nenhum`, nunca ser omitidas.

````markdown
# Relatório — Tarefa NN: <título>

## 1. Identificação
- **Tarefa:** NN — <título>
- **Data:** AAAA-MM-DD
- **Commit antes:** <sha curto> (`git rev-parse --short HEAD` antes de começar)
- **Commit depois:** <sha curto>
- **Arquivos criados:** lista
- **Arquivos modificados:** lista com faixas de linha quando pontual

## 2. Estado herdado
O que já existia quando esta tarefa começou, em no máximo 10 linhas. Cite o
relatório anterior por nome de arquivo. Liste as assinaturas públicas que esta
tarefa **consumiu** (nome, parâmetros, tipo de retorno), copiadas verbatim do
código, não da memória.

## 3. Regressão ANTES da execução
Comando exato executado, saída resumida, e comparação explícita com o baseline
da §C do plano.

| Suíte | Baseline esperado | Obtido antes desta tarefa | Igual? |
|---|---|---|---|
| `pytest tests/ -q` | 19 passed, 0 failed | ... | sim/não |

Se divergir do baseline, **pare e reporte** — não implemente sobre um repositório
já quebrado.

## 4. O que foi feito
Descrição técnica do que mudou e **por quê**, em prosa. Inclua as decisões
tomadas que o plano não especificava. Máximo 25 linhas.

## 5. Testes criados nesta tarefa

Para cada teste, uma entrada:

### `test_nome_do_teste`
- **Arquivo:** `tests/test_x.py:LINHA`
- **O que valida:** uma frase.
- **Resultado esperado (escrito ANTES de rodar):** ...
- **Execução 1 (antes da implementação):** comando, saída, `FAIL` esperado — cole
  a mensagem de erro real.
- **Execução 2 (após a implementação):** comando, saída, `PASS`.
- **Valor numérico obtido:** quando o teste compara números, registre o número
  (ex.: `max|div u| = 3.7e-15`, tolerância `1e-12`). "Passou" não basta.

## 6. Regressão DEPOIS da execução

| Suíte | Antes | Depois | Igual? |
|---|---|---|---|
| `pytest tests/ -q` | 19 passed | 19 passed + N novos | sim |

Liste nominalmente qualquer teste cujo status mudou, com a explicação.

## 7. Desvios do plano
O que foi feito diferente do que o plano mandava, e por quê. `Nenhum` se não houve.

## 8. Referências consultadas
Toda fonte lida para executar a tarefa, com localizador preciso:
- **Código interno:** `caminho/arquivo.py:linhas` — o que foi extraído dali.
- **Documento interno:** `docs/....md §N` — idem.
- **Artigo/livro:** autor, título, seção/equação, e onde o arquivo está
  (`~/Dissertação/Refs/...`).
Sem citações genéricas ("o código do mrcm"): sempre arquivo e linha.

## 9. Estado para a próxima tarefa
- **O que existe agora que antes não existia:** assinaturas públicas novas,
  copiadas verbatim (nome, parâmetros, tipos, retorno).
- **O que a próxima tarefa deve consumir:** ponteiros exatos.
- **Armadilhas descobertas:** o que enganou você e vai enganar o próximo.
- **O que ainda NÃO existe** (para evitar que o próximo assuma que existe).
````

---

## §C. Baseline de testes (medido em 2026-07-25, commit `fd62b2d`)

Comando:

```bash
source ~/venv-firedrake/bin/activate
cd ~/mrcm/NavierStokes
python -m pytest tests/ -q
```

**Estado no momento em que este plano foi escrito — ANTES da Tarefa 0:**

```
19 tests collected, 2 errors        # coleta
1 failed, 18 passed, 3 warnings     # execução com os 2 módulos com erro ignorados
```

Três problemas **pré-existentes**, nenhum causado por este trabalho:

| Problema | Causa verificada | Corrigido em |
|---|---|---|
| `tests/test_alpha_hh.py` — erro de coleta | `ImportError: cannot import name 'FineProjector' from 'mprm_core'`. A classe existe em `mprm/flow.py:87` mas não é reexportada em `mprm/__init__.py:47-52` | Tarefa 0 |
| `tests/test_convergence_hh.py` — erro de coleta | idem | Tarefa 0 |
| `tests/test_transport.py::test_cavity_transporte_bit_identico_ao_legado` — falha | O teste compara `_wstar_legacy` (upwind 1ª ordem) contra `mac_transport(u, v, prob, dt)`, cujo default virou `convective_order=2` no commit `4e40213`. O teste não foi atualizado | Tarefa 0 |

**Baseline alvo, válido a partir da Tarefa 0 e usado por todas as demais:**

```
21 passed
```

Toda tarefa reporta contra esse número, somando os testes que ela mesma cria.

---

## §D. Estrutura de arquivos

| Arquivo | Responsabilidade | Tarefas |
|---|---|---|
| `mprm/__init__.py` | Reexports do pacote | 0, 3 |
| `mprm/triplets.py` **(novo)** | Separação matriz/RHS das triplas do sistema de interface | 1 |
| `mprm/projector.py` | Projetor MRCM: offline de `A_Γ`, consumo de `Decomposition`, traços por segmento | 2, 6, 7 |
| `mprm/decomposition.py` **(novo)** | `Leaf`, `GammaEdge`, `Decomposition`, `plan_quadtree_geometry`, `build_obstacle_tree`, `leaf_boundary_fluxes` | 3, 4, 5 |
| `mprm/downscaler.py` | Problemas de Neumann locais por folha; cache por geometria; defeito retornado | 8 |
| `mprm/flow.py` | Passo de projeção iterando folhas; laço temporal | 9 |
| `mprm/benchmarks/quadtree_vk.py` **(novo)** | Estudo comparativo (funções puras, sem I/O) | 11 |
| `tests/test_triplets.py` **(novo)** | Equivalência da separação matriz/RHS | 1 |
| `tests/test_prefactorization.py` **(novo)** | Equivalência do offline de `A_Γ` | 2 |
| `tests/test_decomposition.py` **(novo)** | Geometria, skeleton, `nloc` heterogêneo, conexidade | 3, 4, 5 |
| `tests/test_quadtree_projector.py` **(novo)** | Equivalência do projetor e continuidade fraca em nó pendurado | 6, 7 |
| `tests/test_quadtree_downscaler.py` **(novo)** | Patches heterogêneos, folha sólida, defeito | 8 |
| `tests/test_quadtree_flow.py` **(novo)** | Conservação de massa e passo completo sob quadtree | 9, 10 |
| `tests/test_quadtree_von_karman.py` **(novo)** | CLI do estudo, JSON e figuras | 12 |
| `docs/superpowers/reports/NN-*.md` **(novos)** | Relatórios por tarefa (§B) | todas |
| `docs/13-quadtree-von-karman.md` **(novo)** | Writeup dos resultados | 13 |

**Spec de referência:** `docs/superpowers/specs/2026-07-25-mprm-quadtree-von-karman-design.md`. Cada tarefa cita a seção do spec que implementa.

---

## Task 0: Baseline verde, branch e infraestrutura de relatório

**Files:**
- Modify: `mprm/__init__.py:47-52`
- Modify: `tests/test_transport.py:50-53`
- Create: `docs/superpowers/reports/00-baseline.md`

**Interfaces:**
- Consumes: nada (primeira tarefa).
- Produces: suíte verde com `21 passed`; `mprm.FineProjector` importável; branch `feature/quadtree-mprm`.

**Referências:**
- `mprm/flow.py:87-132` — classe `FineProjector`.
- `mprm/__init__.py:47-52` — bloco de reexports de `mprm.flow`.
- `tests/test_transport.py:42-53` — teste desatualizado.
- `mprm/transport.py:96-112` — assinatura de `mac_transport`, default `convective_order=2`.
- Este plano, §B e §C.

- [ ] **Step 1: Criar o branch e registrar o commit de partida**

```bash
cd ~/mrcm/NavierStokes
git checkout -b feature/quadtree-mprm
git rev-parse --short HEAD    # anote: vai para o campo "Commit antes" do relatório
```

- [ ] **Step 2: Reproduzir as falhas pré-existentes (elas devem aparecer)**

```bash
source ~/venv-firedrake/bin/activate
python -m pytest tests/ -q 2>&1 | tail -20
```

Esperado: `2 errors during collection` mencionando `FineProjector`. Cole a saída no relatório, seção 3.

- [ ] **Step 3: Reexportar `FineProjector`**

Em `mprm/__init__.py`, no bloco `from mprm.flow import (...)`:

```python
from mprm.flow import (
    mprm_projection_step,
    FineProjector,
    run_flow,
    run_cavity,
    run_von_karman,
)
```

- [ ] **Step 4: Corrigir o teste desatualizado de transporte**

Em `tests/test_transport.py`, dentro de `test_cavity_transporte_bit_identico_ao_legado`, a chamada passa a fixar a ordem convectiva que o legado usa:

```python
    prob = cavity_problem(N, Re=lid / nu, lid_velocity=lid)
    # _wstar_legacy é upwind de 1ª ordem; mac_transport tem default
    # convective_order=2 desde o commit 4e40213. Fixamos 1 para que o teste
    # continue medindo o que se propõe a medir: equivalência do estêncil.
    w1b, w2b = mac_transport(u, v, prob, dt, convective_order=1)
```

- [ ] **Step 5: Rodar a suíte e verificar o baseline verde**

```bash
python -m pytest tests/ -q 2>&1 | tail -5
```

Esperado: `21 passed`. Se algum teste ainda falhar, **pare e reporte** — não siga para a Tarefa 1.

- [ ] **Step 6: Criar o relatório**

Crie `docs/superpowers/reports/00-baseline.md` seguindo o template da §B **na íntegra**. No campo 9 ("Estado para a próxima tarefa"), registre literalmente:

```
Baseline verde: 21 passed com `python -m pytest tests/ -q`.
Toda tarefa seguinte compara contra esse número.
Nada da arquitetura nova existe ainda: não há mprm/triplets.py nem
mprm/decomposition.py. MPrMProjector ainda remonta A_Γ a cada passo.
```

- [ ] **Step 7: Commit**

```bash
git add mprm/__init__.py tests/test_transport.py docs/superpowers/reports/00-baseline.md
git commit -m "test: baseline verde — reexporta FineProjector e fixa ordem convectiva no teste legado

Duas falhas pré-existentes impediam medir regressão: FineProjector existe em
mprm/flow.py mas não era reexportado, e o teste de equivalência do transporte
comparava upwind de 1a ordem contra o default convective_order=2 adotado em
4e40213."
```

---

## Task 1: Separar as triplas do sistema de interface em parte-matriz e parte-RHS

Implementa: spec §5, Fase 1 (primeira metade). O objetivo é isolar, sem mudar comportamento, a parte das triplas que **não depende do passo de tempo**.

**Files:**
- Create: `mprm/triplets.py`
- Create: `tests/test_triplets.py`

**Interfaces:**
- Consumes: `mrcm.global_assembly.compute_subdomain_triplets(cfg, skeleton, dof_mgr, el, particular_el, homogeneous_el, face_mass_el, cofunc_flux_el, cofunc_pres_el) -> (rows, cols, vals, b_idx, b_vals)` — definida em `mrcm/global_assembly.py:238-320`.
- Produces:
  - `subdomain_matrix_triplets(cfg, skeleton, dof_mgr, el, homogeneous_el, face_mass_el, cofunc_flux_el, cofunc_pres_el) -> tuple[np.ndarray, np.ndarray, np.ndarray]` (rows int32, cols int32, vals float64)
  - `subdomain_rhs(cfg, skeleton, dof_mgr, el, particular_el, cofunc_flux_el, cofunc_pres_el) -> tuple[np.ndarray, np.ndarray]` (b_idx int32, b_vals float64)

**Referências:**
- `mrcm/global_assembly.py:238-320` — `compute_subdomain_triplets`, a função a dividir. Note que `rows/cols/vals` só tocam `homogeneous_el`, `face_mass_el` e `cofunc_*`, e que `b_vals` só toca `particular_el`.
- `mrcm/topology.py:506-527` — `Skeleton.gamma_incidences(sub)`, que devolve `GammaIncidence(side, seg, ell, bbox)`.
- `mrcm/topology.py:583-611` — `DOFManager.flux_index`, `.pressure_index`, `.NunkU`, `.n_total`.
- `mrcm/topology.py:64-75` — `GammaIncidence` (NamedTuple).
- Spec §2.2 (Achado 1) e §5 Fase 1.
- Guiraldello et al., *The Multiscale Robin Coupled Method*, Eqs. 58–59 (linhas de pressão e de fluxo do sistema de interface) — `~/Dissertação/Refs/MRCM.pdf`.

- [ ] **Step 1: Escrever o teste que falha**

Crie `tests/test_triplets.py`:

```python
"""Equivalência da separação matriz/RHS das triplas do sistema de interface.

A parte-matriz é constante no tempo (depende só das MsBF e das cofunções);
a parte-RHS depende da fonte. Este teste garante que a soma das duas
reproduz exatamente `compute_subdomain_triplets` do mrcm.
"""
import numpy as np
import pytest

from mrcm.config import MRCMConfig
from mrcm.global_assembly import compute_subdomain_triplets
from mprm.triplets import subdomain_matrix_triplets, subdomain_rhs
from mprm.projector import MPrMProjector


def _projector(Ndom=2, nloc=8):
    cfg = MRCMConfig(Ndomx=Ndom, Ndomy=Ndom, nlocx=nloc, nlocy=nloc,
                     degreeU=1, degreeP=1, alpha=1.0,
                     basis_kind="legendre_P",
                     mixed_family="BDM", mixed_order=0,
                     executor_kind="sequential")
    return cfg, MPrMProjector(cfg, dt=0.01)


def test_matriz_mais_rhs_reproduz_compute_subdomain_triplets():
    cfg, proj = _projector()
    rng = np.random.default_rng(0)
    divw = rng.standard_normal((cfg.Nglobx, cfg.Ngloby))

    # Parte particular com uma fonte concreta (o que o passo online faz).
    from firedrake import assemble  # noqa: F401  (garante Firedrake carregado)
    for isub in proj.my_subs:
        ic, jc = proj.cell_map[isub]
        proj.src_fn[isub].dat.data[:] = divw[ic, jc]
    particular = {isub: proj.solvers[isub].solve(proj.L_forms[isub])
                  for isub in proj.my_subs}

    for isub in proj.my_subs:
        ws = proj.ws[isub]
        ref = compute_subdomain_triplets(
            cfg, proj.skel, proj.dofm, isub,
            particular[isub], ws["homogeneous"],
            ws["face_mass"], ws["cofunc_flux"], ws["cofunc_pres"])
        rows, cols, vals = subdomain_matrix_triplets(
            cfg, proj.skel, proj.dofm, isub,
            ws["homogeneous"], ws["face_mass"],
            ws["cofunc_flux"], ws["cofunc_pres"])
        b_idx, b_vals = subdomain_rhs(
            cfg, proj.skel, proj.dofm, isub,
            particular[isub], ws["cofunc_flux"], ws["cofunc_pres"])

        assert np.array_equal(rows, ref[0]), f"rows diferem em isub={isub}"
        assert np.array_equal(cols, ref[1]), f"cols diferem em isub={isub}"
        assert np.allclose(vals, ref[2], atol=1e-14, rtol=0)
        assert np.array_equal(b_idx, ref[3]), f"b_idx diferem em isub={isub}"
        assert np.allclose(b_vals, ref[4], atol=1e-14, rtol=0)


def test_parte_matriz_independe_da_fonte():
    """O ponto da Fase 1: mudar a fonte não muda a matriz."""
    cfg, proj = _projector()
    rng = np.random.default_rng(1)

    def _matriz_para(divw):
        for isub in proj.my_subs:
            ic, jc = proj.cell_map[isub]
            proj.src_fn[isub].dat.data[:] = divw[ic, jc]
            proj.solvers[isub].solve(proj.L_forms[isub])
        ws = proj.ws[0]
        return subdomain_matrix_triplets(
            cfg, proj.skel, proj.dofm, 0,
            ws["homogeneous"], ws["face_mass"],
            ws["cofunc_flux"], ws["cofunc_pres"])

    a = _matriz_para(rng.standard_normal((cfg.Nglobx, cfg.Ngloby)))
    b = _matriz_para(rng.standard_normal((cfg.Nglobx, cfg.Ngloby)))
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])
    assert np.allclose(a[2], b[2], atol=1e-14, rtol=0)
```

- [ ] **Step 2: Rodar o teste e verificar que falha**

```bash
source ~/venv-firedrake/bin/activate
python -m pytest tests/test_triplets.py -q 2>&1 | tail -10
```

Esperado: `ModuleNotFoundError: No module named 'mprm.triplets'`. Registre a mensagem literal no relatório.

- [ ] **Step 3: Implementar `mprm/triplets.py`**

```python
"""Separação matriz/RHS das triplas do sistema global de interface.

`mrcm.global_assembly.compute_subdomain_triplets` calcula, de uma vez, as
contribuições de um subdomínio para a matriz A_Γ e para o vetor b_Γ. No MPrM
a matriz é CONSTANTE no tempo (K = δt e a geometria não mudam), enquanto b
muda a cada passo com a fonte ∇·w*. Separar as duas permite montar e fatorar
A_Γ uma única vez, no offline.

As duas funções abaixo reproduzem, termo a termo, o que a função do mrcm faz
— ver `mrcm/global_assembly.py:238-320`. Qualquer divergência é um bug, e é
o que `tests/test_triplets.py` verifica.
"""
from __future__ import annotations
import numpy as np


def _arr(x):
    """Array de dados de uma Function do Firedrake, ou o próprio array."""
    return x.dat.data_ro if hasattr(x, "dat") else x


def _spdot(cof, u_arr) -> float:
    """Produto interno esparso: cofunção (idx, vals) contra um array de dofs."""
    idx, v = cof
    return float(np.dot(v, u_arr[idx]))


def subdomain_matrix_triplets(cfg, skeleton, dof_mgr, el,
                              homogeneous_el, face_mass_el,
                              cofunc_flux_el, cofunc_pres_el):
    """Contribuição do subdomínio `el` para A_Γ, em formato COO.

    Depende apenas de dados do offline: as MsBF homogêneas, as matrizes de
    massa de face e as cofunções. Não depende da fonte, logo é constante ao
    longo do laço temporal.

    Returns
    -------
    (rows, cols, vals) : arrays int32, int32, float64
    """
    ndofsU, ndofsP = cfg.ndofsU, cfg.ndofsP
    NunkU = dof_mgr.NunkU
    rows: list = []
    cols: list = []
    vals: list = []

    for inc in skeleton.gamma_incidences(el):
        ell = inc.ell
        key = (inc.side, inc.seg)
        M_face_si = face_mass_el[key]
        cof_flux = cofunc_flux_el[key]
        cof_pres = cofunc_pres_el[key]

        # Linhas de fluxo (Eq. 59): <c_flux_i, u_hat_j> - M_face[i, d_j],
        # o termo de massa apenas para MSBF de fluxo na MESMA aresta de Γ.
        for idf in range(ndofsU):
            iunkG = dof_mgr.flux_index(ell, idf)
            cof_i = cof_flux[idf]
            for junkG, (u_hat, _) in homogeneous_el.items():
                trace_val = _spdot(cof_i, _arr(u_hat))
                mass_val = 0.0
                if junkG < NunkU and junkG // ndofsU == ell:
                    mass_val = float(M_face_si[idf, junkG % ndofsU])
                val = trace_val - mass_val
                if val != 0.0:
                    rows.append(iunkG)
                    cols.append(junkG)
                    vals.append(val)

        # Linhas de pressão (Eq. 58): <c_pres_i, u_hat_j>.
        for idf in range(ndofsP):
            iunkG = dof_mgr.pressure_index(ell, idf)
            cof_i = cof_pres[idf]
            for junkG, (u_hat, _) in homogeneous_el.items():
                val = _spdot(cof_i, _arr(u_hat))
                if val != 0.0:
                    rows.append(iunkG)
                    cols.append(junkG)
                    vals.append(val)

    return (np.asarray(rows, dtype=np.int32),
            np.asarray(cols, dtype=np.int32),
            np.asarray(vals, dtype=np.float64))


def subdomain_rhs(cfg, skeleton, dof_mgr, el, particular_el,
                  cofunc_flux_el, cofunc_pres_el):
    """Contribuição do subdomínio `el` para b_Γ: -<c_i, u_bar>.

    É a única parte que muda a cada passo de tempo, porque `particular_el`
    carrega a solução local da fonte ∇·w*.

    Returns
    -------
    (b_idx, b_vals) : arrays int32, float64
    """
    ndofsU, ndofsP = cfg.ndofsU, cfg.ndofsP
    u_bar_arr = _arr(particular_el[0])
    b_idx: list = []
    b_vals: list = []

    for inc in skeleton.gamma_incidences(el):
        ell = inc.ell
        key = (inc.side, inc.seg)
        cof_flux = cofunc_flux_el[key]
        cof_pres = cofunc_pres_el[key]

        for idf in range(ndofsU):
            b_idx.append(dof_mgr.flux_index(ell, idf))
            b_vals.append(-_spdot(cof_flux[idf], u_bar_arr))
        for idf in range(ndofsP):
            b_idx.append(dof_mgr.pressure_index(ell, idf))
            b_vals.append(-_spdot(cof_pres[idf], u_bar_arr))

    return (np.asarray(b_idx, dtype=np.int32),
            np.asarray(b_vals, dtype=np.float64))
```

- [ ] **Step 4: Rodar o teste e verificar que passa**

```bash
python -m pytest tests/test_triplets.py -q 2>&1 | tail -5
```

Esperado: `2 passed`.

**Atenção à ordem:** `compute_subdomain_triplets` intercala os `b_idx` das linhas de fluxo e de pressão *dentro* do laço de incidências, na sequência (fluxo de todas as `idf`, depois pressão de todas as `idf`). `subdomain_rhs` acima reproduz essa ordem. Se `test_matriz_mais_rhs_reproduz_compute_subdomain_triplets` falhar em `b_idx`, é ordenação — compare os dois arrays lado a lado antes de mexer em qualquer outra coisa.

- [ ] **Step 5: Rodar a suíte completa (regressão)**

```bash
python -m pytest tests/ -q 2>&1 | tail -5
```

Esperado: `23 passed` (21 do baseline + 2 novos).

- [ ] **Step 6: Relatório e commit**

Escreva `docs/superpowers/reports/01-triplets.md` conforme §B. No campo 5, registre para cada teste o valor de tolerância usado e o resultado (por exemplo: `max|vals - ref| = 0.0`).

```bash
git add mprm/triplets.py tests/test_triplets.py docs/superpowers/reports/01-triplets.md
git commit -m "feat(mprm): separa triplas do sistema de interface em parte-matriz e parte-RHS

A parte-matriz depende só das MsBF, das massas de face e das cofunções — tudo
offline —, enquanto o RHS depende da fonte. Pré-requisito para fatorar A_Γ uma
única vez (spec §2.2, Achado 1)."
```

---

## Task 2: Montar e fatorar `A_Γ` no offline do projetor

Implementa: spec §5, Fase 1 (segunda metade). Depois desta tarefa, o passo online deixa de remontar e refatorar o sistema global.

**Files:**
- Modify: `mprm/projector.py` (construtor e `project`)
- Create: `tests/test_prefactorization.py`

**Interfaces:**
- Consumes: `mprm.triplets.subdomain_matrix_triplets`, `mprm.triplets.subdomain_rhs` (Tarefa 1).
- Produces:
  - `MPrMProjector.A_gamma` — `scipy.sparse.csc_matrix` ou `None` fora do rank 0.
  - `MPrMProjector.lu_gamma` — objeto `SuperLU` ou `None`.
  - `MPrMProjector.solve_interface(b) -> np.ndarray` — resolve com a fatoração pronta.
  - `MPrMProjector.project(divw)` mantém a assinatura e o valor de retorno atuais.

**Referências:**
- `mprm/projector.py:235-319` — `project()` atual: a etapa 2 (triplas + `assemble_from_triplets`) é o que sai do laço.
- `mprm/projector.py:220-231` — fim do construtor, onde o `GlobalAssembler` é criado no rank 0.
- `mrcm/global_assembly.py:540-578` — `assemble_from_triplets`: mostra a montagem COO→CSC, `sum_duplicates` e a escolha direta/iterativa.
- `mrcm/global_assembly.py:158-160` — `_solve_direct` usa `spsolve`; aqui trocamos por `splu` + `.solve`.
- `mrcm/config.py` — `cfg.global_solver`, `cfg.iterative_threshold`.
- Golub & Van Loan, *Matrix Computations* 4ª ed., cap. 3–4 (fatoração LU e reuso) — `~/Dissertação/Refs/2013 Matrix Computations 4th.pdf`.
- Spec §2.2 e §5 Fase 1.

- [ ] **Step 1: Escrever o teste que falha**

Crie `tests/test_prefactorization.py`:

```python
"""A pré-fatoração de A_Γ não pode mudar nenhum número.

Compara o caminho novo (matriz montada e fatorada no offline) contra o
caminho antigo (remontagem por passo via GlobalAssembler.assemble_from_triplets).
"""
import numpy as np

from mrcm.config import MRCMConfig
from mrcm.global_assembly import compute_subdomain_triplets
from mprm.projector import MPrMProjector


def _cfg(Ndom=2, nloc=8):
    return MRCMConfig(Ndomx=Ndom, Ndomy=Ndom, nlocx=nloc, nlocy=nloc,
                      degreeU=1, degreeP=1, alpha=1.0,
                      basis_kind="legendre_P",
                      mixed_family="BDM", mixed_order=0,
                      executor_kind="sequential")


def _X_caminho_antigo(proj, divw):
    """Reproduz a etapa 2 do project() original: triplas por passo + assemble."""
    for isub in proj.my_subs:
        ic, jc = proj.cell_map[isub]
        proj.src_fn[isub].dat.data[:] = divw[ic, jc]
    particular = {isub: proj.solvers[isub].solve(proj.L_forms[isub])
                  for isub in proj.my_subs}
    trip = {isub: compute_subdomain_triplets(
        proj.cfg, proj.skel, proj.dofm, isub,
        particular[isub], proj.ws[isub]["homogeneous"],
        proj.ws[isub]["face_mass"], proj.ws[isub]["cofunc_flux"],
        proj.ws[isub]["cofunc_pres"]) for isub in proj.my_subs}
    _, _, X = proj.assembler.assemble_from_triplets(trip)
    return X


def test_X_identico_ao_caminho_de_remontagem():
    cfg = _cfg()
    proj = MPrMProjector(cfg, dt=0.01)
    rng = np.random.default_rng(3)
    divw = rng.standard_normal((cfg.Nglobx, cfg.Ngloby))
    divw -= divw.mean()          # fonte compatível (Neumann puro)

    X_ref = _X_caminho_antigo(proj, divw)
    proj.project(divw)
    X_novo = proj.last_X

    err = np.max(np.abs(X_novo - X_ref)) / max(np.max(np.abs(X_ref)), 1.0)
    assert err < 1e-12, f"X divergiu: erro relativo {err:.3e}"


def test_matriz_fatorada_uma_vez_e_reutilizada():
    """A LU é construída no offline e não é reconstruída por passo."""
    cfg = _cfg()
    proj = MPrMProjector(cfg, dt=0.01)
    assert proj.lu_gamma is not None, "A_Γ deveria estar fatorada no construtor"
    lu_id = id(proj.lu_gamma)

    rng = np.random.default_rng(4)
    for _ in range(3):
        divw = rng.standard_normal((cfg.Nglobx, cfg.Ngloby))
        divw -= divw.mean()
        proj.project(divw)
        assert id(proj.lu_gamma) == lu_id, "a LU foi refatorada durante o online"


def test_cavidade_ponta_a_ponta_inalterada():
    """Regressão física: a cavidade continua conservando massa."""
    from mprm.flow import run_cavity
    from mprm.transport import mac_divergence
    r = run_cavity(Re=100.0, Ndom=2, nloc=8, T=0.05, verbose=False)
    dv = np.abs(mac_divergence(r["u"], r["v"], 1.0 / 16, 1.0 / 16)).max()
    assert dv < 1e-11, f"max|div u| = {dv:.3e}"
```

- [ ] **Step 2: Rodar e verificar que falha**

```bash
python -m pytest tests/test_prefactorization.py -q 2>&1 | tail -10
```

Esperado: `AttributeError: 'MPrMProjector' object has no attribute 'lu_gamma'`.

- [ ] **Step 3: Montar e fatorar `A_Γ` no construtor**

Em `mprm/projector.py`, no topo do arquivo, acrescente aos imports:

```python
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from mprm.triplets import subdomain_matrix_triplets, subdomain_rhs
```

No **final** do construtor (logo após o bloco que cria `self.assembler`, `mprm/projector.py:220-231`), acrescente:

```python
        # ── Offline do sistema global de interface (spec §5, Fase 1) ──
        # A_Γ depende só das MsBF/massas de face/cofunções: com K = δt e a
        # geometria fixas, ela é a MESMA em todos os passos de tempo. Montar e
        # fatorar aqui evita refazer, por passo, O(N_B²) produtos internos por
        # subdomínio mais uma fatoração esparsa.
        my_mat = {isub: subdomain_matrix_triplets(
            cfg, self.skel, self.dofm, isub,
            self.ws[isub]["homogeneous"], self.ws[isub]["face_mass"],
            self.ws[isub]["cofunc_flux"], self.ws[isub]["cofunc_pres"])
            for isub in self.my_subs}
        gathered_mat = comm.gather(my_mat, root=0)

        self.n_gamma = self.dofm.n_total
        self.A_gamma = None
        self.lu_gamma = None
        if self.rank == 0:
            parts = []
            for d in gathered_mat:
                parts.extend(d[k] for k in sorted(d.keys()))
            nonempty = [t for t in parts if t[0].size > 0]
            rows = np.concatenate([t[0] for t in nonempty])
            cols = np.concatenate([t[1] for t in nonempty])
            vals = np.concatenate([t[2] for t in nonempty])
            A = sp.coo_matrix((vals, (rows, cols)),
                              shape=(self.n_gamma, self.n_gamma)).tocsc()
            A.sum_duplicates()
            self.A_gamma = A
            if cfg.global_solver == "iterative" and self.n_gamma > cfg.iterative_threshold:
                self.lu_gamma = None      # caminho iterativo: resolve por passo
            else:
                self.lu_gamma = spla.splu(A)
```

- [ ] **Step 4: Acrescentar `solve_interface`**

Ainda em `mprm/projector.py`, como método de `MPrMProjector`, antes de `project`:

```python
    def solve_interface(self, b):
        """Resolve A_Γ X = b com a fatoração do offline (rank 0) e difunde X.

        `b` é significativo apenas no rank 0; os demais podem passar None.
        """
        if self.rank == 0:
            if self.lu_gamma is not None:
                X = self.lu_gamma.solve(b)
            else:
                from mrcm.global_assembly import _solve_iterative
                X = _solve_iterative(self.A_gamma, b, self.cfg)
        else:
            X = None
        return self.comm.bcast(X, root=0)
```

- [ ] **Step 5: Trocar a etapa 2 do `project`**

Em `mprm/projector.py`, substitua o bloco da etapa 2 (das linhas que hoje começam em `# 2. Triplets locais -> sistema global de interface no rank 0.` até `X = comm.bcast(X, root=0)`) por:

```python
        # 2. RHS local -> redução ao rank 0 -> solve com a LU do offline.
        #    Só b muda de um passo para o outro (spec §5, Fase 1).
        b_local = np.zeros(self.n_gamma, dtype=np.float64)
        for isub in self.my_subs:
            b_idx, b_vals = subdomain_rhs(
                cfg, self.skel, self.dofm, isub, particular[isub],
                self.ws[isub]["cofunc_flux"], self.ws[isub]["cofunc_pres"])
            np.add.at(b_local, b_idx, b_vals)
        b = np.zeros(self.n_gamma, dtype=np.float64) if self.rank == 0 else None
        comm.Reduce(b_local, b, op=MPI_SUM, root=0)
        X = self.solve_interface(b)
```

e acrescente, junto aos imports do módulo:

```python
from mpi4py import MPI as _MPI
MPI_SUM = _MPI.SUM
```

- [ ] **Step 6: Rodar os testes novos**

```bash
python -m pytest tests/test_prefactorization.py -q 2>&1 | tail -10
```

Esperado: `3 passed`. Registre no relatório o erro relativo de `X` medido pelo primeiro teste (deve ser da ordem de `1e-16`–`1e-14`).

- [ ] **Step 7: Medir o ganho (dado para o relatório)**

```bash
python - <<'PY'
import time, numpy as np
from mrcm.config import MRCMConfig
from mprm.projector import MPrMProjector
cfg = MRCMConfig(Ndomx=3, Ndomy=3, nlocx=12, nlocy=12, degreeU=1, degreeP=3,
                 alpha=1.0, basis_kind="legendre_P", mixed_family="BDM",
                 mixed_order=0, executor_kind="sequential")
t0 = time.perf_counter(); proj = MPrMProjector(cfg, dt=0.01)
t_off = time.perf_counter() - t0
rng = np.random.default_rng(0)
divw = rng.standard_normal((cfg.Nglobx, cfg.Ngloby)); divw -= divw.mean()
proj.project(divw)                       # aquece
t0 = time.perf_counter()
for _ in range(10): proj.project(divw)
print(f"offline = {t_off:.2f}s   online = {(time.perf_counter()-t0)/10*1000:.1f} ms/passo")
PY
```

Registre os dois números no relatório, seção 4.

- [ ] **Step 8: Regressão completa**

```bash
python -m pytest tests/ -q 2>&1 | tail -5
```

Esperado: `26 passed`.

- [ ] **Step 9: Relatório e commit**

`docs/superpowers/reports/02-prefatoracao.md` conforme §B.

```bash
git add mprm/projector.py tests/test_prefactorization.py docs/superpowers/reports/02-prefatoracao.md
git commit -m "perf(mprm): monta e fatora A_Γ no offline do projetor

O sistema global de interface é constante no tempo (K = δt, geometria fixa),
mas era remontado e refatorado a cada passo. Agora o online calcula apenas
b = -<c_i, u_bar>, reduz ao rank 0 e resolve com a LU pronta. Campos e X
inalterados até 1e-12 (spec §5, Fase 1)."
```

---

## Task 3: `Decomposition` — a camada única de topologia (caminho uniforme)

Implementa: spec §4.2. Nesta tarefa a classe nasce cobrindo **apenas** o caso cartesiano, reproduzindo o que o projetor calcula hoje com aritmética `(ix, iy)`. O quadtree entra na Tarefa 5.

**Files:**
- Create: `mprm/decomposition.py`
- Create: `tests/test_decomposition.py`
- Modify: `mprm/__init__.py` (reexports)

**Interfaces:**
- Consumes: `mrcm.topology.Skeleton.cartesian(Ndomx, Ndomy)`, `Skeleton.gamma_incidences(sub)`, `Skeleton.ieGt`, `Skeleton.el2edge`, `mrcm.topology.mark_sides(ix, iy, Ndomx, Ndomy)`.
- Produces:
  - `Leaf(isub, x0, y0, x1, y1, nlocx, nlocy, i0, j0, level)` — dataclass congelada.
  - `GammaEdge(ell, a, b, nfaces, bbox)` — `a`/`b` são tuplas `(isub, side, seg)`.
  - `Decomposition.uniform(Ndomx, Ndomy, nloc, Lx, Ly) -> Decomposition`
  - `Decomposition.leaf(isub) -> Leaf`
  - `Decomposition.gamma_edges() -> list[GammaEdge]`
  - `Decomposition.segments_of(isub) -> list[tuple[int, int, int, tuple, int]]` — `(side, seg, ell, bbox, nfaces)`
  - `Decomposition.task(cfg, isub, NunkU, problem) -> dict`
  - Atributos: `.skeleton`, `.leaves`, `.h`, `.Lx`, `.Ly`, `.Nx`, `.Ny`, `.is_uniform`

**Referências:**
- `mprm/projector.py:72-76` — como o projetor monta `Skeleton.cartesian`, `DOFManager` e `my_subs` hoje.
- `mprm/projector.py:134-141` — construção da `task` no caminho uniforme (chaves `cfg_dict`, `isub`, `NunkU`, `edge_l`, `problem`).
- `mrcm/solver.py:104-141` — `_build_tasks`: o formato canônico das tarefas nos **dois** modos; é o contrato a imitar.
- `mrcm/topology.py:506-527` — `gamma_incidences` e `GammaIncidence(side, seg, ell, bbox)`.
- `mrcm/topology.py:481-488` — `Skeleton.is_uniform`, `max_segments_per_side`.
- `mrcm/topology.py:78-...` — `mark_sides`.
- Spec §4.2 e §4.2.1.

- [ ] **Step 1: Escrever o teste que falha**

Crie `tests/test_decomposition.py`:

```python
"""Camada de decomposição: geometria, arestas de Γ e tarefas de Fase 1.

Nesta tarefa a cobertura é do caminho cartesiano — o quadtree entra depois.
O valor do teste está em amarrar a `Decomposition` à aritmética que o
projetor faz hoje, para que a troca (Tarefa 6) seja verificável.
"""
import numpy as np
import pytest

from mrcm.config import MRCMConfig
from mrcm.topology import mark_sides
from mprm.decomposition import Decomposition, Leaf, GammaEdge


def test_uniform_geometria_das_folhas():
    d = Decomposition.uniform(Ndomx=2, Ndomy=2, nloc=8, Lx=1.0, Ly=1.0)
    assert len(d.leaves) == 4
    assert d.h == pytest.approx(0.5 / 8)
    assert (d.Nx, d.Ny) == (16, 16)

    # Folha 0 = canto inferior esquerdo; folha 3 = superior direito.
    lf0, lf3 = d.leaf(0), d.leaf(3)
    assert (lf0.x0, lf0.y0, lf0.x1, lf0.y1) == pytest.approx((0.0, 0.0, 0.5, 0.5))
    assert (lf0.i0, lf0.j0) == (0, 0)
    assert (lf3.x0, lf3.y0) == pytest.approx((0.5, 0.5))
    assert (lf3.i0, lf3.j0) == (8, 8)
    for lf in d.leaves:
        assert (lf.nlocx, lf.nlocy) == (8, 8)


def test_uniform_offsets_batem_com_a_aritmetica_do_projetor():
    """i0/j0 devem reproduzir exatamente `ox = ix*n, oy = iy*m`."""
    Ndomx, Ndomy, nloc = 3, 2, 6
    d = Decomposition.uniform(Ndomx, Ndomy, nloc, Lx=3.0, Ly=2.0)
    for isub in range(Ndomx * Ndomy):
        ix, iy = isub % Ndomx, isub // Ndomx
        lf = d.leaf(isub)
        assert (lf.i0, lf.j0) == (ix * nloc, iy * nloc)


def test_gamma_edges_contagem_e_pares():
    """Grade 3x2: 2*2 arestas verticais internas + 3*1 horizontais = 7."""
    d = Decomposition.uniform(3, 2, nloc=6, Lx=3.0, Ly=2.0)
    edges = d.gamma_edges()
    assert len(edges) == 7
    for e in edges:
        assert e.a[0] != e.b[0], "uma aresta de Γ liga dois subdomínios distintos"
        assert e.nfaces == 6, "com h uniforme, os dois lados têm o mesmo nº de faces"
    # Cada subdomínio interno aparece em tantas arestas quantos lados internos tem.
    for isub in range(6):
        ix, iy = isub % 3, isub // 3
        _, internal = mark_sides(ix, iy, 3, 2)
        n = sum(1 for e in edges if e.a[0] == isub or e.b[0] == isub)
        assert n == len(internal), f"isub={isub}"


def test_task_uniforme_tem_as_chaves_que_o_mrcm_espera():
    cfg = MRCMConfig(Ndomx=2, Ndomy=2, nlocx=8, nlocy=8, degreeU=1, degreeP=1,
                     alpha=1.0, basis_kind="legendre_P", mixed_family="BDM",
                     mixed_order=0, executor_kind="sequential")
    d = Decomposition.uniform(2, 2, 8, 1.0, 1.0)
    t = d.task(cfg, isub=0, NunkU=10, problem="DUMMY")
    assert set(t) == {"cfg_dict", "isub", "NunkU", "edge_l", "problem"}
    assert t["isub"] == 0 and t["NunkU"] == 10 and t["problem"] == "DUMMY"
    # edge_l mapeia lado interno -> índice de Γ, como em projector.py:137-138
    _, internal = mark_sides(0, 0, 2, 2)
    assert set(t["edge_l"]) == set(internal)


def test_segments_of_uniforme_tem_um_segmento_por_lado_interno():
    d = Decomposition.uniform(2, 2, 8, 1.0, 1.0)
    segs = d.segments_of(0)
    _, internal = mark_sides(0, 0, 2, 2)
    assert len(segs) == len(internal)
    for side, seg, ell, bbox, nfaces in segs:
        assert seg == 0
        assert nfaces == 8
        assert bbox is not None, "mesmo no uniforme, a bbox do lado é preenchida"
```

- [ ] **Step 2: Rodar e verificar que falha**

```bash
python -m pytest tests/test_decomposition.py -q 2>&1 | tail -5
```

Esperado: `ModuleNotFoundError: No module named 'mprm.decomposition'`.

- [ ] **Step 3: Implementar `mprm/decomposition.py`**

```python
"""Camada de decomposição do MPrM — o único ponto que conhece a topologia.

O projetor, o downscaler e o passo de projeção param de fazer aritmética
`(ix, iy)` e passam a iterar sobre folhas e arestas de Γ. A decomposição
cartesiana vira um caso particular da mesma classe, o que torna o teste
"árvore uniforme ≡ cartesiano" a rede de segurança da generalização.

Convenção de lados (herdada do mrcm): 0=bottom, 1=right, 2=top, 3=left.
Convenção de malha MAC: hx = hy = h uniforme global (spec, decisão D1).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

from mrcm.topology import Skeleton, mark_sides

SIDE_BOTTOM, SIDE_RIGHT, SIDE_TOP, SIDE_LEFT = 0, 1, 2, 3


@dataclass(frozen=True)
class Leaf:
    """Um subdomínio: geometria física, resolução local e posição na malha MAC."""
    isub: int
    x0: float
    y0: float
    x1: float
    y1: float
    nlocx: int
    nlocy: int
    i0: int          # índice da primeira célula MAC em x
    j0: int          # índice da primeira célula MAC em y
    level: int

    @property
    def Hx(self) -> float:
        return self.x1 - self.x0

    @property
    def Hy(self) -> float:
        return self.y1 - self.y0


@dataclass(frozen=True)
class GammaEdge:
    """Uma aresta de Γ: o segmento de interface entre dois subdomínios.

    `a` e `b` são `(isub, side, seg)`. `nfaces` é o número de faces finas do
    segmento — igual dos dois lados porque h é uniforme (spec §4.2), o que
    torna a média da Eq. 21 uma operação face a face.
    """
    ell: int
    a: tuple
    b: tuple
    nfaces: int
    bbox: tuple      # (x0, y0, x1, y1) do segmento


class Decomposition:
    """Folhas + esqueleto + arestas de Γ, para um `h` uniforme global."""

    def __init__(self, skeleton, leaves, h, Lx, Ly):
        self.skeleton = skeleton
        self.leaves = list(leaves)
        self.h = float(h)
        self.Lx, self.Ly = float(Lx), float(Ly)
        self.Nx = int(round(Lx / h))
        self.Ny = int(round(Ly / h))
        self._by_id = {lf.isub: lf for lf in self.leaves}
        self._gamma = None          # construído sob demanda

    # ── fábricas ────────────────────────────────────────────────────────
    @classmethod
    def uniform(cls, Ndomx: int, Ndomy: int, nloc: int,
                Lx: float, Ly: float) -> "Decomposition":
        """Decomposição cartesiana — o caminho retrocompatível."""
        Hx, Hy = Lx / Ndomx, Ly / Ndomy
        h = Hx / nloc
        if abs(Hy / nloc - h) > 1e-12 * h:
            raise ValueError(
                f"células não quadradas: Hx/nloc={h:.6g} != Hy/nloc="
                f"{Hy/nloc:.6g}. Ajuste Ndomx/Ndomy para hx == hy.")
        leaves = []
        for isub in range(Ndomx * Ndomy):
            ix, iy = isub % Ndomx, isub // Ndomx
            leaves.append(Leaf(isub=isub,
                               x0=ix * Hx, y0=iy * Hy,
                               x1=(ix + 1) * Hx, y1=(iy + 1) * Hy,
                               nlocx=nloc, nlocy=nloc,
                               i0=ix * nloc, j0=iy * nloc, level=0))
        return cls(Skeleton.cartesian(Ndomx, Ndomy), leaves, h, Lx, Ly)

    # ── acesso ──────────────────────────────────────────────────────────
    @property
    def is_uniform(self) -> bool:
        return self.skeleton.is_uniform

    def leaf(self, isub: int) -> Leaf:
        return self._by_id[isub]

    def segments_of(self, isub: int):
        """`(side, seg, ell, bbox, nfaces)` de cada incidência de Γ da folha.

        No caminho uniforme o mrcm devolve `bbox=None` (o segmento é o lado
        inteiro); aqui a bbox é sempre preenchida, para que os consumidores
        tenham um único formato.
        """
        lf = self.leaf(isub)
        out = []
        for inc in self.skeleton.gamma_incidences(isub):
            bbox = inc.bbox if inc.bbox is not None else _side_bbox(lf, inc.side)
            horiz = inc.side in (SIDE_BOTTOM, SIDE_TOP)
            span = (bbox[2] - bbox[0]) if horiz else (bbox[3] - bbox[1])
            nfaces = int(round(span / self.h))
            out.append((inc.side, inc.seg, inc.ell, bbox, nfaces))
        return out

    def gamma_edges(self) -> list:
        """Uma `GammaEdge` por aresta de Γ, com os dois lados emparelhados."""
        if self._gamma is not None:
            return self._gamma
        by_ell: dict = {}
        for lf in self.leaves:
            for side, seg, ell, bbox, nfaces in self.segments_of(lf.isub):
                by_ell.setdefault(ell, []).append(
                    ((lf.isub, side, seg), bbox, nfaces))
        edges = []
        for ell in sorted(by_ell):
            entries = by_ell[ell]
            if len(entries) != 2:
                raise ValueError(
                    f"aresta de Γ ell={ell} tem {len(entries)} incidências "
                    f"(esperado 2). Decomposição inconsistente.")
            (ka, bba, na), (kb, bbb, nb) = entries
            if na != nb:
                raise ValueError(
                    f"aresta de Γ ell={ell}: {na} faces de um lado e {nb} do "
                    f"outro. Com h uniforme os dois lados devem coincidir.")
            edges.append(GammaEdge(ell=ell, a=ka, b=kb, nfaces=na, bbox=bba))
        self._gamma = edges
        return edges

    # ── tarefa de Fase 1 do mrcm ────────────────────────────────────────
    def task(self, cfg, isub: int, NunkU: int, problem) -> dict:
        """Tarefa consumida por `mrcm.local_solver.build_subdomain_workspace`.

        Reproduz `mrcm/solver.py:104-141`: no caminho uniforme a geometria é
        derivada de `(ix, iy)` pelo próprio mrcm e basta `edge_l`; no caminho
        quadtree é preciso mandar `geom`, `incidences` e `is_external`.
        """
        skel = self.skeleton
        if self.is_uniform:
            ix, iy = isub % cfg.Ndomx, isub // cfg.Ndomx
            _, internal = mark_sides(ix, iy, cfg.Ndomx, cfg.Ndomy)
            edge_l = {s: int(skel.ieGt[skel.el2edge[isub, s]]) for s in internal}
            return {"cfg_dict": cfg.as_dict(), "isub": isub, "NunkU": NunkU,
                    "edge_l": edge_l, "problem": problem}
        lf = self.leaf(isub)
        return {"cfg_dict": cfg.as_dict(), "isub": isub, "NunkU": NunkU,
                "geom": {"x0": lf.x0, "y0": lf.y0,
                         "Hx": lf.Hx, "Hy": lf.Hy,
                         "nlocx": lf.nlocx, "nlocy": lf.nlocy},
                "incidences": skel.gamma_incidences(isub),
                "is_external": skel.external_side_flags(isub),
                "cacheable": False,
                "problem": problem}


def _side_bbox(lf: Leaf, side: int) -> tuple:
    """Extensão física do lado inteiro de uma folha."""
    if side == SIDE_BOTTOM:
        return (lf.x0, lf.y0, lf.x1, lf.y0)
    if side == SIDE_TOP:
        return (lf.x0, lf.y1, lf.x1, lf.y1)
    if side == SIDE_LEFT:
        return (lf.x0, lf.y0, lf.x0, lf.y1)
    return (lf.x1, lf.y0, lf.x1, lf.y1)         # SIDE_RIGHT
```

- [ ] **Step 4: Reexportar no pacote**

Em `mprm/__init__.py`, após o bloco do downscaler:

```python
# ── Decomposição (uniforme e quadtree) ────────────────────────────────────
from mprm.decomposition import Decomposition, Leaf, GammaEdge
```

- [ ] **Step 5: Rodar os testes**

```bash
python -m pytest tests/test_decomposition.py -q 2>&1 | tail -5
```

Esperado: `5 passed`.

- [ ] **Step 6: Regressão completa**

```bash
python -m pytest tests/ -q 2>&1 | tail -5
```

Esperado: `31 passed`.

- [ ] **Step 7: Relatório e commit**

`docs/superpowers/reports/03-decomposition-uniforme.md` conforme §B. No campo 9, copie **verbatim** as assinaturas de `Leaf`, `GammaEdge` e dos métodos públicos — a Tarefa 5 e a 6 dependem delas.

```bash
git add mprm/decomposition.py tests/test_decomposition.py mprm/__init__.py docs/superpowers/reports/03-decomposition-uniforme.md
git commit -m "feat(mprm): camada Decomposition — folhas, arestas de Γ e tarefas (caminho uniforme)

Único ponto do MPrM que conhece a topologia. O caminho cartesiano é um caso
particular, o que permite testar a generalização por equivalência (spec §4.2)."
```

---

## Task 4: Geometria do quadtree — `plan_quadtree_geometry` e `build_obstacle_tree`

Implementa: spec §4.3, §4.3.1 e §4.4. Esta tarefa é **NumPy puro**, não toca Firedrake, e é onde as restrições aritméticas viram código.

**Files:**
- Modify: `mprm/decomposition.py` (acrescenta as duas funções)
- Modify: `tests/test_decomposition.py` (acrescenta a classe de testes)

**Interfaces:**
- Consumes: `mrcm.quadtree.QuadTree`, `QuadTreeNode`, `QuadTree.balance_2to1()`.
- Produces:
  - `plan_quadtree_geometry(*, D, Ly, N_bx, N_by, max_level, nloc_min, family, degreeU=1, degreeP=3) -> dict` com chaves `H0, Lx, H, nloc, h, Nx, Ny, xc, yc, family, body_level`.
  - `build_obstacle_tree(geom, radii, wake_length=0.0) -> QuadTree`.

**Referências:**
- `mrcm/quadtree.py:76-140` — `QuadTreeNode` (campos `x0,y0,x1,y1,level,children`), `subdivide()`.
- `mrcm/quadtree.py:146-457` — `QuadTree`, `from_uniform`, `balance_2to1`, `_collect_leaves`, `_invalidate_cache`, `_build_neighbours`.
- `mrcm/quadtree.py:282-326` — `from_uniform` **exige potências de 2**; por isso construímos a raiz manualmente aqui (a grade base do canal tem `N_by` ímpar na família (a)).
- `mrcm/topology.py:109-160` — `check_segment_alignment`: a razão de `nloc` ter de ser par.
- `mprm/problem.py:112-119` — `square_cylinder`, que define a máscara do obstáculo.
- Spec §4.3 (restrições 1–5), §4.3.1 (impossibilidade de paridade e famílias (a)/(b)), §4.4 (critério de refinamento).
- Arbogast, Pencheva, Wheeler, Yotov, *A multiscale mortar mixed finite element method* — condição que relaciona o espaço de interface à malha fina, origem de `nloc_min ≥ max(degreeU, degreeP)`. `~/Dissertação/Refs/A MULTISCALE MORTAR MIXED FINITE ELEMENT METHOD.pdf`.
- de Berg et al., *Computational Geometry*, §14.3 — balanceamento 2:1.

- [ ] **Step 1: Escrever os testes que falham**

Acrescente ao final de `tests/test_decomposition.py`:

```python
# ═══════════════════════════════════════════════════════════════════════
# Geometria do quadtree (spec §4.3, §4.3.1, §4.4)
# ═══════════════════════════════════════════════════════════════════════
from mprm.decomposition import plan_quadtree_geometry, build_obstacle_tree


class TestPlanGeometry:
    def test_familia_a_configuracao_de_referencia(self):
        """Ly=8D, N_by=3 (ímpar) -> H0=8D/3, níveis 0-2, nloc=(32,16,8)."""
        g = plan_quadtree_geometry(D=1.0, Ly=8.0, N_bx=10, N_by=3,
                                   max_level=2, nloc_min=8, family="a")
        assert g["H0"] == pytest.approx(8.0 / 3.0)
        assert g["Lx"] == pytest.approx(10 * 8.0 / 3.0)
        assert g["nloc"] == [32, 16, 8]
        assert g["h"] == pytest.approx(1.0 / 12.0)
        assert (g["Nx"], g["Ny"]) == (320, 96)
        assert g["yc"] == pytest.approx(4.0), "obstáculo centrado no canal"

    def test_familia_b_configuracao_de_referencia(self):
        """Ly=8D, N_by=2 (par) -> H0=4D, corpo = 2x2 folhas de D/2."""
        g = plan_quadtree_geometry(D=1.0, Ly=8.0, N_bx=6, N_by=2,
                                   max_level=4, nloc_min=4, family="b")
        assert g["H0"] == pytest.approx(4.0)
        assert g["nloc"] == [64, 32, 16, 8, 4]
        assert g["h"] == pytest.approx(1.0 / 16.0)
        assert (g["Nx"], g["Ny"]) == (384, 128)
        assert g["yc"] == pytest.approx(4.0)
        assert g["body_level"] == 3, "corpo = folhas de nível 3 (D/2)"
        assert g["H"][g["body_level"]] == pytest.approx(0.5)

    def test_familia_a_rejeita_N_by_par(self):
        with pytest.raises(ValueError, match="ímpar"):
            plan_quadtree_geometry(D=1.0, Ly=8.0, N_bx=10, N_by=4,
                                   max_level=2, nloc_min=8, family="a")

    def test_familia_b_rejeita_N_by_impar(self):
        with pytest.raises(ValueError, match="par"):
            plan_quadtree_geometry(D=1.0, Ly=8.0, N_bx=6, N_by=3,
                                   max_level=4, nloc_min=4, family="b")

    def test_rejeita_H0_menor_que_D_na_familia_a(self):
        with pytest.raises(ValueError, match="H0"):
            plan_quadtree_geometry(D=4.0, Ly=8.0, N_bx=10, N_by=3,
                                   max_level=2, nloc_min=8, family="a")

    def test_rejeita_nloc_impar(self):
        """nloc ímpar contra nó pendurado quebra o alinhamento de segmento."""
        with pytest.raises(ValueError, match="par"):
            plan_quadtree_geometry(D=1.0, Ly=8.0, N_bx=10, N_by=3,
                                   max_level=2, nloc_min=7, family="a")

    def test_rejeita_nloc_min_menor_que_grau(self):
        with pytest.raises(ValueError, match="degree"):
            plan_quadtree_geometry(D=1.0, Ly=8.0, N_bx=10, N_by=3,
                                   max_level=2, nloc_min=2, family="a",
                                   degreeU=1, degreeP=3)


class TestObstacleTree:
    def test_familia_a_obstaculo_interior_a_uma_folha(self):
        g = plan_quadtree_geometry(D=1.0, Ly=8.0, N_bx=10, N_by=3,
                                   max_level=2, nloc_min=8, family="a")
        qt = build_obstacle_tree(g, radii=(6.0, 3.0))
        xc, yc, D = g["xc"], g["yc"], 1.0
        conteiners = [lf for lf in qt.leaves
                      if lf.x0 <= xc - D / 2 and lf.x1 >= xc + D / 2
                      and lf.y0 <= yc - D / 2 and lf.y1 >= yc + D / 2]
        assert len(conteiners) == 1, "o corpo deve caber numa única folha"

    def test_familia_b_obstaculo_e_uniao_exata_de_4_folhas(self):
        g = plan_quadtree_geometry(D=1.0, Ly=8.0, N_bx=6, N_by=2,
                                   max_level=4, nloc_min=4, family="b")
        qt = build_obstacle_tree(g, radii=(8.0, 4.0, 2.0, 1.0))
        xc, yc, D = g["xc"], g["yc"], 1.0
        dentro = [lf for lf in qt.leaves
                  if lf.x0 >= xc - D / 2 - 1e-12 and lf.x1 <= xc + D / 2 + 1e-12
                  and lf.y0 >= yc - D / 2 - 1e-12 and lf.y1 <= yc + D / 2 + 1e-12]
        assert len(dentro) == 4, f"esperado 4 folhas no corpo, obtido {len(dentro)}"
        area = sum((lf.x1 - lf.x0) * (lf.y1 - lf.y0) for lf in dentro)
        assert area == pytest.approx(D * D), "as 4 folhas cobrem o corpo exatamente"

    def test_arvore_balanceada_2_para_1(self):
        g = plan_quadtree_geometry(D=1.0, Ly=8.0, N_bx=10, N_by=3,
                                   max_level=2, nloc_min=8, family="a")
        qt = build_obstacle_tree(g, radii=(6.0, 3.0))
        for lf in qt.leaves:
            for s in range(4):
                assert len(lf.neighbours[s]) <= 2, \
                    f"folha {lf} tem {len(lf.neighbours[s])} vizinhos no lado {s}"
                for nb in lf.neighbours[s]:
                    assert abs(lf.level - nb.level) <= 1

    def test_refino_decresce_com_a_distancia_ao_corpo(self):
        g = plan_quadtree_geometry(D=1.0, Ly=8.0, N_bx=10, N_by=3,
                                   max_level=2, nloc_min=8, family="a")
        qt = build_obstacle_tree(g, radii=(6.0, 3.0))
        xc, yc = g["xc"], g["yc"]
        perto = [lf.level for lf in qt.leaves
                 if abs(lf.cx - xc) < 2.0 and abs(lf.cy - yc) < 2.0]
        longe = [lf.level for lf in qt.leaves if abs(lf.cx - xc) > 12.0]
        assert max(perto) > max(longe), \
            f"níveis perto={max(perto)} não excedem longe={max(longe)}"
```

- [ ] **Step 2: Rodar e verificar que falha**

```bash
python -m pytest tests/test_decomposition.py -q 2>&1 | tail -5
```

Esperado: `ImportError: cannot import name 'plan_quadtree_geometry'`.

- [ ] **Step 3: Implementar `plan_quadtree_geometry`**

Acrescente a `mprm/decomposition.py`:

```python
# ═══════════════════════════════════════════════════════════════════════════
# Geometria do quadtree (spec §4.3 e §4.3.1)
# ═══════════════════════════════════════════════════════════════════════════

def plan_quadtree_geometry(*, D, Ly, N_bx, N_by, max_level, nloc_min,
                           family, degreeU=1, degreeP=3):
    """Resolve as restrições geométricas e devolve a configuração completa.

    O quadtree é diádico: toda fronteira de folha cai em múltiplo de H0/2^ℓ.
    Duas famílias de alinhamento entre corpo e árvore (spec, decisão D7):

    * ``family="a"`` — o corpo fica **contido no interior** de uma folha, que
      por isso não pode ser subdividida. Exige ``N_by`` ímpar (o centro do
      canal precisa ser o centro de uma célula base) e ``H0 > D``.
    * ``family="b"`` — o corpo é a **união exata** de um bloco 2×2 de folhas de
      nível ``body_level``, com ``D = 2·H[body_level]``. Exige ``N_by`` par (o
      centro do canal é um vértice da grade). É a única forma de o refino
      alcançar a superfície: com o corpo em uma única folha de nível ℓ≥1 e
      centrado, Ly/D teria de ser ímpar e par ao mesmo tempo.

    Raises
    ------
    ValueError
        Nomeando a restrição violada e o parâmetro a corrigir.
    """
    if family not in ("a", "b"):
        raise ValueError(f"family deve ser 'a' ou 'b', recebido {family!r}")
    if family == "a" and N_by % 2 == 0:
        raise ValueError(
            f"family='a' exige N_by ímpar (recebido {N_by}): com N_by par, "
            f"y=Ly/2 é fronteira de folha em todo nível e nenhuma folha "
            f"contém o corpo centrado.")
    if family == "b" and N_by % 2 != 0:
        raise ValueError(
            f"family='b' exige N_by par (recebido {N_by}): o centro do corpo "
            f"precisa cair num vértice da grade diádica.")

    H0 = Ly / N_by
    Lx = N_bx * H0
    H = [H0 / 2 ** l for l in range(max_level + 1)]
    h = H[max_level] / nloc_min
    nloc = [int(round(Hl / h)) for Hl in H]

    if nloc_min < max(degreeU, degreeP):
        raise ValueError(
            f"nloc_min={nloc_min} < max(degreeU={degreeU}, degreeP={degreeP}): "
            f"o espaço de interface ficaria mais rico que a malha fina do "
            f"segmento (Arbogast et al., condição de solvabilidade).")
    for l, (Hl, nl) in enumerate(zip(H, nloc)):
        if abs(Hl / h - nl) > 1e-9 * nl:
            raise ValueError(
                f"nível {l}: H={Hl:.6g} não é múltiplo inteiro de h={h:.6g}.")
        if nl % 2 != 0:
            raise ValueError(
                f"nível {l}: nloc={nl} é ímpar. Um segmento de nó pendurado "
                f"terminaria no meio de uma célula e a base de interface, que "
                f"é mascarada por célula, integraria intervalos diferentes nos "
                f"dois lados (mrcm/topology.py:109-160). Use nloc_min par.")

    Nx, Ny = int(round(Lx / h)), int(round(Ly / h))
    for name, val, tot in (("Nx", Nx, Lx), ("Ny", Ny, Ly)):
        if abs(tot / h - val) > 1e-9 * val:
            raise ValueError(f"{name}: {tot:.6g}/h não é inteiro.")

    yc = 0.5 * Ly
    if family == "a":
        if H0 <= D:
            raise ValueError(
                f"family='a' exige H0 > D: H0={H0:.6g}, D={D:.6g}. A folha que "
                f"contém o corpo não pode ser subdividida, então ela precisa "
                f"caber o corpo inteiro.")
        body_level = 0
        # xc no centro de uma célula base, o mais próximo de 1/4 do canal.
        k = max(0, int(round(0.25 * N_bx - 0.5)))
        xc = (k + 0.5) * H0
    else:
        # D = 2·H[body_level] -> o corpo é um bloco 2x2 desse nível.
        cand = [l for l, Hl in enumerate(H) if abs(D - 2 * Hl) <= 1e-9 * D]
        if not cand:
            raise ValueError(
                f"family='b' exige D = 2·H[ℓ] para algum nível: "
                f"D={D:.6g}, H={[f'{x:.4g}' for x in H]}. Ajuste N_by, Ly ou "
                f"max_level.")
        body_level = cand[0]
        if body_level >= max_level:
            raise ValueError(
                f"body_level={body_level} == max_level={max_level}: não sobra "
                f"nível para refinar as vizinhas do corpo, que é o objetivo "
                f"da family='b'. Aumente max_level.")
        Hb = H[body_level]
        k = max(1, int(round(0.25 * Lx / Hb)))
        xc = k * Hb                       # vértice da grade do nível do corpo
    return {"family": family, "D": D, "Lx": Lx, "Ly": Ly, "H0": H0,
            "H": H, "nloc": nloc, "h": h, "Nx": Nx, "Ny": Ny,
            "xc": xc, "yc": yc, "N_bx": N_bx, "N_by": N_by,
            "max_level": max_level, "body_level": body_level}
```

- [ ] **Step 4: Implementar `build_obstacle_tree`**

Acrescente logo abaixo:

```python
def build_obstacle_tree(geom: dict, radii, wake_length: float = 0.0):
    """Constrói a árvore refinada em torno do obstáculo (spec §4.4).

    Uma folha de nível ``ℓ`` é subdividida sse ``ℓ < max_level`` e a distância
    da sua bbox à bbox do corpo é ``<= radii[ℓ]``. ``radii`` deve ser
    estritamente decrescente — é o que garante que o anel mais fino envolve o
    corpo em vez de aparecer solto no domínio.

    ``wake_length > 0`` estende a região refinada a jusante (x > xc); o default
    é isotrópico.

    Na família (a) a folha que contém o corpo é travada e nunca subdividida.
    Na família (b) o corpo é refinado até ``body_level`` para virar um bloco
    2×2 exato, e não além.
    """
    from mrcm.quadtree import QuadTree, QuadTreeNode

    D, xc, yc = geom["D"], geom["xc"], geom["yc"]
    Lx, Ly = geom["Lx"], geom["Ly"]
    N_bx, N_by = geom["N_bx"], geom["N_by"]
    max_level, family = geom["max_level"], geom["family"]
    body_level = geom["body_level"]
    radii = list(radii)
    if any(radii[i] <= radii[i + 1] for i in range(len(radii) - 1)):
        raise ValueError(f"radii deve ser estritamente decrescente: {radii}")
    if len(radii) < max_level:
        raise ValueError(
            f"radii tem {len(radii)} entradas para max_level={max_level}; "
            f"são necessárias {max_level} (uma por nível subdividível).")

    bx0, bx1 = xc - D / 2, xc + D / 2
    by0, by1 = yc - D / 2, yc + D / 2

    # Grade base de células quadradas (H0 x H0). `QuadTree.from_uniform` exige
    # potências de dois, o que N_by ímpar viola; montamos a raiz à mão.
    H0 = geom["H0"]
    root = QuadTreeNode(0.0, 0.0, Lx, Ly, 0)
    root.children = [
        QuadTreeNode(i * H0, j * H0, (i + 1) * H0, (j + 1) * H0, 0)
        for j in range(N_by) for i in range(N_bx)
    ]
    qt = QuadTree(root=root, max_level=max_level, Lx=Lx, Ly=Ly)

    def _dist_ao_corpo(nd):
        dx = max(bx0 - nd.x1, 0.0, nd.x0 - bx1)
        dy = max(by0 - nd.y1, 0.0, nd.y0 - by1)
        return float(np.hypot(dx, dy))

    def _contem_o_corpo(nd):
        return (nd.x0 <= bx0 + 1e-12 and nd.x1 >= bx1 - 1e-12 and
                nd.y0 <= by0 + 1e-12 and nd.y1 >= by1 - 1e-12)

    def _dentro_do_corpo(nd):
        return (nd.x0 >= bx0 - 1e-12 and nd.x1 <= bx1 + 1e-12 and
                nd.y0 >= by0 - 1e-12 and nd.y1 <= by1 + 1e-12)

    while True:
        mudou = False
        for lf in qt._collect_leaves(use_cache=False):
            if lf.level >= max_level:
                continue
            if family == "a" and _contem_o_corpo(lf):
                continue                      # folha travada (decisão D4)
            if family == "b" and _dentro_do_corpo(lf) and lf.level >= body_level:
                continue                      # o corpo já é o bloco exato
            if family == "b" and _contem_o_corpo(lf) and lf.level < body_level:
                lf.subdivide(); mudou = True   # desce até o corpo virar 2x2
                continue
            d = _dist_ao_corpo(lf)
            if wake_length > 0.0 and lf.cx > xc and abs(lf.cy - yc) <= wake_length:
                d = min(d, max(0.0, lf.cx - bx1) * 0.5)
            if d <= radii[lf.level]:
                lf.subdivide(); mudou = True
        if not mudou:
            break
        qt._invalidate_cache()
        qt.balance_2to1()

    qt._invalidate_cache()
    qt.balance_2to1()

    # Validação pós-balanço: o balanceamento pode ter subdividido a folha
    # travada. Com raios monotônicos isso não deve ocorrer, mas silenciar essa
    # verificação transformaria uma violação de D4 em resultado plausível.
    if family == "a":
        n = sum(1 for lf in qt.leaves if _contem_o_corpo(lf))
        if n != 1:
            raise ValueError(
                f"family='a': após o balanço 2:1 há {n} folhas contendo o "
                f"corpo (esperado 1). Reduza radii[0] ou aumente H0.")
    else:
        dentro = [lf for lf in qt.leaves if _dentro_do_corpo(lf)]
        area = sum((lf.x1 - lf.x0) * (lf.y1 - lf.y0) for lf in dentro)
        if abs(area - D * D) > 1e-9 * D * D:
            raise ValueError(
                f"family='b': as folhas internas ao corpo cobrem área "
                f"{area:.6g}, esperado {D*D:.6g}. O corpo não é união exata "
                f"de folhas.")
    return qt
```

- [ ] **Step 5: Rodar os testes de geometria**

```bash
python -m pytest tests/test_decomposition.py -q 2>&1 | tail -8
```

Esperado: `16 passed` (5 da Tarefa 3 + 11 novos).

Se `test_familia_b_obstaculo_e_uniao_exata_de_4_folhas` falhar por número de folhas, imprima os níveis das folhas dentro do corpo antes de mudar a lógica: quase sempre é `radii` curto demais para levar o corpo até `body_level`.

- [ ] **Step 6: Regressão completa**

```bash
python -m pytest tests/ -q 2>&1 | tail -5
```

Esperado: `42 passed`.

- [ ] **Step 7: Relatório e commit**

`docs/superpowers/reports/04-geometria-quadtree.md` conforme §B. No campo 5, registre para cada configuração de referência os valores obtidos de `H0, nloc, h, Nx, Ny, xc, yc` e o número de folhas da árvore resultante.

```bash
git add mprm/decomposition.py tests/test_decomposition.py docs/superpowers/reports/04-geometria-quadtree.md
git commit -m "feat(mprm): geometria do quadtree — plan_quadtree_geometry e build_obstacle_tree

Resolve as restrições diádicas das duas famílias de alinhamento corpo/árvore
(spec §4.3.1): (a) corpo interior a uma folha travada, N_by ímpar; (b) corpo
como bloco 2x2 exato, N_by par, refino alcançando a superfície."
```

---

## Task 5: `Decomposition.from_quadtree` — `nloc` heterogêneo, conexidade e consistência MPI

Implementa: spec §4.2, §4.5 (conexidade) e §5 Fase 2. É aqui que `sub_nloc` recebe, pela primeira vez no projeto, valores **diferentes por folha**.

**Files:**
- Modify: `mprm/decomposition.py`
- Modify: `tests/test_decomposition.py`

**Interfaces:**
- Consumes: `mrcm.topology.Skeleton.from_quadtree(qt)`, `Skeleton.sub_nloc` (campo `(Nsubd, 2)`), `mrcm.local_solver.build_subdomain_workspace(task)`, `mrcm.local_form.make_local_mesh`.
- Produces:
  - `Decomposition.from_quadtree(qt, h, Lx, Ly, solid=None) -> Decomposition`
  - `Decomposition.assert_consistent_across_ranks(comm) -> None`
  - `Decomposition.check_fluid_connectivity(solid) -> None`

**Referências:**
- `mrcm/topology.py:291-332` — `Skeleton.from_quadtree`; note que ele **já** valida o balanço 2:1 e levanta `ValueError` (spec §4.6).
- `mrcm/topology.py:203-218` — campos `sub_bbox` `(Nsubd, 4)` e `sub_nloc` `(Nsubd, 2)`. `Skeleton` é `@dataclass(frozen=False)`, então `sub_nloc` pode ser atribuído depois da construção.
- `mrcm/solver.py:120-136` — como `_build_tasks` transporta `sub_nloc` para `geom["nlocx"]/["nlocy"]`.
- `mrcm/local_solver.py:944-976` — `build_subdomain_workspace` lendo `geom.get("nlocx", cfg.nlocx)` e chamando `check_segment_alignment`.
- `mrcm/local_form.py:28-50` — `make_local_mesh` usa `RectangleMesh(nx, ny, Hx, Hy, diagonal="left")`; a malha é **triangular**, com `num_cells() == 2·nx·ny`.
- `mrcm/quadtree.py:202-216` — `QuadTree.leaves` já ordena row-major e atribui `subdomain_id`.
- Spec §4.5 (invariante de conexidade) e §4.6 (estado verificado das dependências).

- [ ] **Step 1: Escrever os testes que falham**

Acrescente ao final de `tests/test_decomposition.py`:

```python
# ═══════════════════════════════════════════════════════════════════════
# Decomposition.from_quadtree (spec §4.2, §4.5, Fase 2)
# ═══════════════════════════════════════════════════════════════════════

def _arvore_uniforme(n, L=1.0):
    """Árvore com n×n folhas todas no mesmo nível, sem refino."""
    from mrcm.quadtree import QuadTree, QuadTreeNode
    H = L / n
    root = QuadTreeNode(0.0, 0.0, L, L, 0)
    root.children = [QuadTreeNode(i * H, j * H, (i + 1) * H, (j + 1) * H, 0)
                     for j in range(n) for i in range(n)]
    qt = QuadTree(root=root, max_level=0, Lx=L, Ly=L)
    qt._build_neighbours()
    return qt


class TestFromQuadtree:
    def test_arvore_uniforme_equivale_a_decomposicao_cartesiana(self):
        """A rede de segurança: mesmo domínio, mesmas folhas, mesmas arestas."""
        nloc = 8
        du = Decomposition.uniform(2, 2, nloc, 1.0, 1.0)
        dq = Decomposition.from_quadtree(_arvore_uniforme(2), h=du.h,
                                         Lx=1.0, Ly=1.0)
        assert len(dq.leaves) == len(du.leaves)
        for isub in range(4):
            a, b = du.leaf(isub), dq.leaf(isub)
            assert (a.x0, a.y0, a.x1, a.y1) == pytest.approx((b.x0, b.y0, b.x1, b.y1))
            assert (a.i0, a.j0) == (b.i0, b.j0)
            assert (a.nlocx, a.nlocy) == (b.nlocx, b.nlocy)
        pares_u = {(frozenset([e.a[0], e.b[0]]), e.nfaces) for e in du.gamma_edges()}
        pares_q = {(frozenset([e.a[0], e.b[0]]), e.nfaces) for e in dq.gamma_edges()}
        assert pares_u == pares_q

    def test_nloc_proporcional_ao_tamanho_da_folha(self):
        """h uniforme (decisão D1): folha maior recebe mais células."""
        from mrcm.quadtree import QuadTree
        qt = _arvore_uniforme(2)
        qt.leaves[0].subdivide()           # folha 0 vira 4 folhas de meio tamanho
        qt._invalidate_cache()
        qt.balance_2to1()
        h = 0.5 / 8
        d = Decomposition.from_quadtree(qt, h=h, Lx=1.0, Ly=1.0)
        for lf in d.leaves:
            assert lf.nlocx == int(round(lf.Hx / h))
            assert lf.nlocy == int(round(lf.Hy / h))
        nlocs = sorted({lf.nlocx for lf in d.leaves})
        assert nlocs == [4, 8], f"esperados dois tamanhos de folha, obtido {nlocs}"
        assert d.skeleton.sub_nloc is not None
        assert d.skeleton.sub_nloc.shape == (len(d.leaves), 2)

    def test_nloc_heterogeneo_chega_a_malha_local(self):
        """O elo que nunca foi exercitado (spec §4.6): sub_nloc -> malha real.

        `build_subdomain_workspace` deve gerar, para cada folha, uma malha com
        2·nlocx·nlocy células (RectangleMesh com diagonal='left').
        """
        from mrcm.config import MRCMConfig
        from mrcm.local_solver import build_subdomain_workspace
        from mprm.problem import _impermeable_problem, _darcy_boundary_from_problem
        from mrcm.problem import DarcyProblem
        from firedrake import Constant

        qt = _arvore_uniforme(2)
        qt.leaves[0].subdivide()
        qt._invalidate_cache()
        qt.balance_2to1()
        h = 0.5 / 8
        d = Decomposition.from_quadtree(qt, h=h, Lx=1.0, Ly=1.0)

        cfg = MRCMConfig(Ndomx=2, Ndomy=2, nlocx=8, nlocy=8,
                         degreeU=1, degreeP=1, alpha=1.0,
                         basis_kind="legendre_P", mixed_family="BDM",
                         mixed_order=0, executor_kind="sequential",
                         use_quadtree=True)
        flow = _impermeable_problem(cfg)
        darcy = DarcyProblem(
            Lx=1.0, Ly=1.0,
            permeability=lambda mesh: Constant(1.0),
            inv_permeability=lambda mesh: Constant(1.0),
            source=lambda x, y: Constant(0.0),
            boundary=_darcy_boundary_from_problem(flow))
        NunkU = d.skeleton.NGamma * cfg.ndofsU
        for isub in (0, len(d.leaves) - 1):
            lf = d.leaf(isub)
            ws = build_subdomain_workspace(d.task(cfg, isub, NunkU, darcy))
            n_esperado = 2 * lf.nlocx * lf.nlocy
            assert ws["mesh"].num_cells() == n_esperado, (
                f"folha {isub}: malha com {ws['mesh'].num_cells()} células, "
                f"esperado {n_esperado} para nloc=({lf.nlocx},{lf.nlocy})")

    def test_fluido_desconexo_e_rejeitado(self):
        """Neumann puro com fluido em 2 componentes tem nulidade 2 (spec §4.5)."""
        d = Decomposition.uniform(2, 2, nloc=8, Lx=1.0, Ly=1.0)
        solid = np.zeros((16, 16), dtype=bool)
        solid[0:8, 4] = True          # faixa cortando a folha 0 de lado a lado
        with pytest.raises(ValueError, match="desconex"):
            d.check_fluid_connectivity(solid)

    def test_fluido_conexo_e_aceito(self):
        d = Decomposition.uniform(2, 2, nloc=8, Lx=1.0, Ly=1.0)
        solid = np.zeros((16, 16), dtype=bool)
        solid[2:5, 2:5] = True        # bloco interno: fluido em anel, conexo
        d.check_fluid_connectivity(solid)     # não levanta

    def test_folha_totalmente_solida_e_aceita(self):
        """Na família (b) o corpo é união de folhas — 100% sólidas são válidas."""
        d = Decomposition.uniform(2, 2, nloc=8, Lx=1.0, Ly=1.0)
        solid = np.zeros((16, 16), dtype=bool)
        solid[0:8, 0:8] = True        # folha 0 inteira
        d.check_fluid_connectivity(solid)     # não levanta
```

- [ ] **Step 2: Rodar e verificar que falha**

```bash
python -m pytest tests/test_decomposition.py -k FromQuadtree -q 2>&1 | tail -5
```

Esperado: `AttributeError: type object 'Decomposition' has no attribute 'from_quadtree'`.

- [ ] **Step 3: Implementar `from_quadtree`**

Acrescente como `@classmethod` da classe `Decomposition`, logo após `uniform`:

```python
    @classmethod
    def from_quadtree(cls, qt, h: float, Lx: float, Ly: float,
                      solid=None) -> "Decomposition":
        """Decomposição a partir de uma árvore balanceada, com h uniforme.

        Cada folha recebe ``nloc_i = H_i/h`` (decisão D1), gravado em
        ``skeleton.sub_nloc``, que é o canal pelo qual o mrcm leva a resolução
        até a malha local (`mrcm/solver.py:120-136`).

        ``solid``, quando dado, é a máscara MAC global; a conexidade do fluido
        em cada folha é verificada na construção (spec §4.5).
        """
        leaves_qt = qt.leaves            # já ordenadas row-major, com id
        skel = Skeleton.from_quadtree(qt)   # valida o balanço 2:1 internamente

        leaves = []
        sub_nloc = np.zeros((len(leaves_qt), 2), dtype=int)
        for nd in leaves_qt:
            isub = nd.subdomain_id
            nlocx = int(round((nd.x1 - nd.x0) / h))
            nlocy = int(round((nd.y1 - nd.y0) / h))
            for nome, nl, H in (("nlocx", nlocx, nd.x1 - nd.x0),
                                ("nlocy", nlocy, nd.y1 - nd.y0)):
                if abs(H / h - nl) > 1e-9 * max(nl, 1):
                    raise ValueError(
                        f"folha {isub}: {nome} = {H:.6g}/{h:.6g} não é inteiro. "
                        f"h precisa dividir todo H_ℓ (spec §4.3).")
            sub_nloc[isub] = (nlocx, nlocy)
            leaves.append(Leaf(isub=isub, x0=nd.x0, y0=nd.y0,
                               x1=nd.x1, y1=nd.y1,
                               nlocx=nlocx, nlocy=nlocy,
                               i0=int(round(nd.x0 / h)),
                               j0=int(round(nd.y0 / h)),
                               level=nd.level))
        skel.sub_nloc = sub_nloc
        d = cls(skel, leaves, h, Lx, Ly)
        if solid is not None:
            d.check_fluid_connectivity(solid)
        return d

    # ── invariantes ─────────────────────────────────────────────────────
    def check_fluid_connectivity(self, solid) -> None:
        """Nenhuma folha pode ter região fluida desconexa (spec §4.5).

        O problema local do downscaler é Neumann puro: com fluido em duas
        componentes a nulidade é 2, e o pino numa única célula remove apenas um
        modo — a projeção sairia errada sem erro nenhum. Folhas 100% sólidas
        são válidas (a família (b) as produz por construção).
        """
        from scipy.ndimage import label
        for lf in self.leaves:
            blk = solid[lf.i0:lf.i0 + lf.nlocx, lf.j0:lf.j0 + lf.nlocy]
            fluid = ~np.asarray(blk, dtype=bool)
            if not fluid.any():
                continue                       # folha 100% sólida: ok
            ncomp = int(label(fluid)[1])
            if ncomp > 1:
                raise ValueError(
                    f"folha {lf.isub} (células x[{lf.i0}:{lf.i0+lf.nlocx}], "
                    f"y[{lf.j0}:{lf.j0+lf.nlocy}]) tem região fluida desconexa "
                    f"({ncomp} componentes). O problema de Neumann local teria "
                    f"nulidade {ncomp} e o pino remove só um modo.")

    def assert_consistent_across_ranks(self, comm) -> None:
        """Todos os ranks precisam ter construído o MESMO esqueleto.

        Cada rank constrói a árvore por conta própria. É determinístico, mas se
        um rank divergir, os triplets são montados contra numerações de Γ
        diferentes e o sistema global sai numericamente lixo, sem erro.
        """
        if getattr(comm, "size", 1) == 1:
            return
        skel = self.skeleton
        fingerprint = hash((skel.NGamma, len(self.leaves),
                            skel.el2edge.tobytes(), skel.edge2el.tobytes()))
        lo = comm.allreduce(fingerprint, op=__import__("mpi4py.MPI",
                                                       fromlist=["MPI"]).MIN)
        hi = comm.allreduce(fingerprint, op=__import__("mpi4py.MPI",
                                                       fromlist=["MPI"]).MAX)
        if lo != hi:
            raise RuntimeError(
                f"esqueletos divergem entre ranks (rank {comm.rank}: "
                f"NGamma={skel.NGamma}, {len(self.leaves)} folhas). "
                f"A montagem global seria feita contra numerações diferentes.")
```

- [ ] **Step 4: Rodar os testes**

```bash
python -m pytest tests/test_decomposition.py -q 2>&1 | tail -8
```

Esperado: `22 passed`. Registre no relatório o resultado de `test_nloc_heterogeneo_chega_a_malha_local` com os números reais de células — é a evidência de que o elo nunca exercitado funciona.

- [ ] **Step 5: Regressão completa**

```bash
python -m pytest tests/ -q 2>&1 | tail -5
```

Esperado: `48 passed`.

- [ ] **Step 6: Relatório e commit**

`docs/superpowers/reports/05-decomposition-quadtree.md` conforme §B.

```bash
git add mprm/decomposition.py tests/test_decomposition.py docs/superpowers/reports/05-decomposition-quadtree.md
git commit -m "feat(mprm): Decomposition.from_quadtree com nloc por folha e invariantes

sub_nloc recebe pela primeira vez valores heterogêneos, e o teste verifica que
eles chegam à malha local do mrcm. Invariante de conexidade do fluido substitui
'obstáculo não cruza interface' (spec §4.5) e hash do esqueleto entre ranks."
```

---

## Task 6: Projetor consumindo `Decomposition` (caminho uniforme, Robin por folha)

Implementa: spec §5 Fase 3 (primeira metade) e corrige o Achado 6 (§2.6). Ao fim desta tarefa o projetor não faz mais aritmética `(ix, iy)`, mas ainda roda apenas com decomposição uniforme — o comportamento numérico não pode mudar.

**Files:**
- Modify: `mprm/projector.py` (construtor)
- Create: `tests/test_quadtree_projector.py`

**Interfaces:**
- Consumes: `Decomposition.uniform`, `Decomposition.leaf`, `Decomposition.task`, `Decomposition.segments_of` (Tarefas 3 e 5).
- Produces: `MPrMProjector(cfg, dt, comm=None, problem=None, verbose=False, eps_solid=1e-6, decomposition=None)` — novo parâmetro opcional; quando `None`, o projetor constrói `Decomposition.uniform` a partir do `cfg` (retrocompatível). Atributo `MPrMProjector.decomp`.

**Referências:**
- `mprm/projector.py:47-231` — construtor inteiro; os pontos a trocar são as linhas que derivam `ix, iy`, `Skeleton.cartesian`, `hx_f/hy_f` e a construção única do objeto Robin.
- `mprm/projector.py:120-122` — `robin = make_robin(cfg.robin_variant, cfg.alpha, *cfg.robin_L_xy(), ...)`: o `L` global que precisa virar por folha.
- `mrcm/local_solver.py:992-999` — como `build_subdomain_workspace` constrói **o seu** Robin, com `cfg.robin_L_xy_for(Hx, Hy, nlocx, nlocy)`. É esse o valor que o solver local tem de usar, sob pena de β do solver ≠ β das MsBF.
- `mrcm/config.py:246-281` — `robin_L_xy_for` e `robin_L_xy`.
- Spec §2.6 (Achado 6) e §5 Fase 3.
- Guiraldello et al., MRCM, Eq. 65 (`β = α L / K`) — `~/Dissertação/Refs/MRCM.pdf`.

- [ ] **Step 1: Escrever o teste que falha**

Crie `tests/test_quadtree_projector.py`:

```python
"""Projetor sobre a camada Decomposition.

Enquanto a decomposição for uniforme, nenhum número pode mudar: é isso que
torna a generalização verificável.
"""
import numpy as np
import pytest

from mrcm.config import MRCMConfig
from mprm.decomposition import Decomposition
from mprm.projector import MPrMProjector


def _cfg(Ndom=2, nloc=8, dU=1, dP=1):
    return MRCMConfig(Ndomx=Ndom, Ndomy=Ndom, nlocx=nloc, nlocy=nloc,
                      degreeU=dU, degreeP=dP, alpha=1.0,
                      basis_kind="legendre_P", mixed_family="BDM",
                      mixed_order=0, executor_kind="sequential")


def _fonte(cfg, seed=7):
    rng = np.random.default_rng(seed)
    divw = rng.standard_normal((cfg.Nglobx, cfg.Ngloby))
    return divw - divw.mean()


def test_decomposicao_explicita_da_o_mesmo_que_a_implicita():
    """Passar Decomposition.uniform == deixar o projetor construí-la."""
    cfg = _cfg()
    divw = _fonte(cfg)

    p_impl = MPrMProjector(cfg, dt=0.01)
    f_impl = p_impl.project(divw)

    d = Decomposition.uniform(cfg.Ndomx, cfg.Ndomy, cfg.nlocx, cfg.Lx, cfg.Ly)
    p_expl = MPrMProjector(cfg, dt=0.01, decomposition=d)
    f_expl = p_expl.project(divw)

    assert np.allclose(p_impl.last_X, p_expl.last_X, atol=1e-12, rtol=0)
    assert set(f_impl) == set(f_expl)
    for k in f_impl:
        assert np.allclose(f_impl[k], f_expl[k], atol=1e-12, rtol=0)


def test_projetor_expoe_a_decomposicao():
    cfg = _cfg()
    p = MPrMProjector(cfg, dt=0.01)
    assert p.decomp is not None
    assert len(p.decomp.leaves) == cfg.Ndomx * cfg.Ndomy
    assert p.decomp.h == pytest.approx(cfg.Lx / cfg.Nglobx)


def test_beta_do_solver_local_usa_o_L_da_folha():
    """Achado 6: o Robin do solver local tem de usar o mesmo L das MsBF."""
    cfg = _cfg()
    p = MPrMProjector(cfg, dt=0.01)
    lf = p.decomp.leaf(0)
    esperado = cfg.robin_L_xy_for(lf.Hx, lf.Hy, lf.nlocx, lf.nlocy)
    assert p.robin_L[0] == pytest.approx(esperado)


def test_cavidade_inalterada():
    from mprm.flow import run_cavity
    from mprm.transport import mac_divergence
    r = run_cavity(Re=100.0, Ndom=2, nloc=8, T=0.05, verbose=False)
    dv = np.abs(mac_divergence(r["u"], r["v"], 1.0 / 16, 1.0 / 16)).max()
    assert dv < 1e-11, f"max|div u| = {dv:.3e}"
```

- [ ] **Step 2: Rodar e verificar que falha**

```bash
python -m pytest tests/test_quadtree_projector.py -q 2>&1 | tail -5
```

Esperado: `TypeError: MPrMProjector.__init__() got an unexpected keyword argument 'decomposition'`.

- [ ] **Step 3: Aceitar a decomposição no construtor**

Em `mprm/projector.py`, altere a assinatura e o início do construtor:

```python
    def __init__(self, cfg: MRCMConfig, dt: float, comm=None,
                 problem=None, verbose=False, eps_solid=1e-6,
                 decomposition=None):
```

Logo após `self.eps_solid = float(eps_solid)`, substitua a construção do esqueleto:

```python
        # Decomposição: explícita (quadtree ou uniforme) ou derivada do cfg.
        if decomposition is None:
            from mprm.decomposition import Decomposition
            decomposition = Decomposition.uniform(
                cfg.Ndomx, cfg.Ndomy, cfg.nlocx, cfg.Lx, cfg.Ly)
        self.decomp = decomposition
        self.skel = decomposition.skeleton
        self.dofm = DOFManager(self.skel.NGamma, cfg.ndofsU, cfg.ndofsP)
        self.my_subs = list(range(self.rank, len(decomposition.leaves), self.size))
```

(as linhas antigas `self.skel = Skeleton.cartesian(...)`, `self.dofm = ...` e
`self.my_subs = list(range(self.rank, cfg.Nsubd, self.size))` desaparecem).

E troque a definição de `hx_f/hy_f` por:

```python
        hx_f = hy_f = decomposition.h
        self.hx_f, self.hy_f = hx_f, hy_f
```

- [ ] **Step 4: Construir o Robin por folha**

Substitua a linha `robin = make_robin(cfg.robin_variant, cfg.alpha, *cfg.robin_L_xy(), ...)` por um dicionário indexado por folha, e guarde os `L` para inspeção:

```python
        # Robin POR FOLHA: β = α L / K com o L característico daquela folha
        # (Eq. 65). `build_subdomain_workspace` faz o mesmo internamente com
        # `robin_L_xy_for`; usar aqui o L global faria o β do solver local
        # divergir do β das MsBF em decomposições não uniformes (spec §2.6).
        self.robin_L = {}
        robins = {}
        for lf in decomposition.leaves:
            L_xy = cfg.robin_L_xy_for(lf.Hx, lf.Hy, lf.nlocx, lf.nlocy)
            self.robin_L[lf.isub] = L_xy
            robins[lf.isub] = make_robin(
                cfg.robin_variant, cfg.alpha, *L_xy,
                cfg.robin_clamp_eps, cfg.robin_extreme_beta)
        ispace = make_interface_space(cfg.basis_kind, cfg.degreeU, cfg.degreeP)
```

Dentro do laço `for isub in self.my_subs:`, troque `beta_expr = robin.beta_for_subdomain(isub, mesh)` por:

```python
            beta_expr = robins[isub].beta_for_subdomain(isub, mesh)
```

E, na construção do `GlobalAssembler` no rank 0, use `robins[0]` no lugar de `robin` — o assembler só usa o objeto Robin para o caminho de montagem monolítica, que o MPrM não exercita (ele passa por `assemble_from_triplets`); ainda assim, deixe explícito com um comentário:

```python
            # `robins[0]` é usado apenas como placeholder: o MPrM monta o
            # sistema por triplas (assemble_from_triplets), caminho que não
            # consulta o Robin do assembler.
            self.assembler = GlobalAssembler(
                cfg, self.skel, self.dofm, ispace, robins[0], ...)
```

- [ ] **Step 5: Usar `decomposition.task` na Fase 1**

Dentro do mesmo laço, substitua a construção manual do dicionário `task` (as linhas que hoje montam `flag_side`, `internal_sides`, `edge_l` e `task`) por:

```python
        for isub in self.my_subs:
            lf = decomposition.leaf(isub)
            internal_sides = sorted({s for s, _, _, _, _ in
                                     decomposition.segments_of(isub)})
            flag_side = np.zeros(4, dtype=bool)
            flag_side[internal_sides] = True
            task = decomposition.task(cfg, isub, NunkU, self.problem_darcy)
            ws = build_subdomain_workspace(task)
```

**Atenção:** `flag_side` no restante do construtor é usado por
`build_flux_dirichlet_bcs` e pelo laço de `out_trace`, com a indexação
`{"bottom": 0, "right": 1, "top": 2, "left": 3}` — mantida.

- [ ] **Step 6: Rodar os testes**

```bash
python -m pytest tests/test_quadtree_projector.py -q 2>&1 | tail -8
```

Esperado: `4 passed`.

- [ ] **Step 7: Regressão completa**

```bash
python -m pytest tests/ -q 2>&1 | tail -5
```

Esperado: `52 passed`. **Nenhum teste anterior pode mudar de status** — se `tests/test_projector.py` ou `tests/test_run_regression.py` falharem, a troca alterou números e precisa ser revista antes de seguir.

- [ ] **Step 8: Relatório e commit**

`docs/superpowers/reports/06-projetor-decomposition.md` conforme §B.

```bash
git add mprm/projector.py tests/test_quadtree_projector.py docs/superpowers/reports/06-projetor-decomposition.md
git commit -m "refactor(mprm): projetor consome Decomposition e constrói Robin por folha

Elimina a aritmética (ix, iy) do projetor e corrige o Achado 6: o β do solver
local passa a usar o L característico da folha, como as MsBF já faziam. Caminho
uniforme numericamente inalterado (spec §2.6, §5 Fase 3)."
```

---

## Task 7: Traços por segmento e média da Eq. 21 por aresta de Γ

Implementa: spec §5 Fase 3 (segunda metade) e §4.2.1. Aqui o projetor passa a funcionar com nó pendurado, e o formato de saída de `project()` muda de chaves cartesianas para `(isub, side, seg)`.

**Files:**
- Modify: `mprm/projector.py` (`__init__`: dicionário `trace`; `project`: etapas 3–4)
- Modify: `mprm/decomposition.py` (acrescenta `leaf_boundary_fluxes`)
- Modify: `mprm/flow.py` (`mprm_projection_step` consome o novo formato)
- Modify: `tests/test_quadtree_projector.py`

**Interfaces:**
- Consumes: `Decomposition.gamma_edges()`, `Decomposition.segments_of(isub)` (Tarefas 3 e 5).
- Produces:
  - `MPrMProjector.project(divw) -> dict[(isub, side, seg), np.ndarray]` — **formato novo**. Cada valor é o fluxo médio por face fina do segmento, orientado em `+x` (lados 1/3) ou `+y` (lados 0/2), com o mesmo array atribuído aos dois lados da aresta.
  - `leaf_boundary_fluxes(decomp, flux_avg, isub) -> (fS, fE, fN, fW)` em `mprm/decomposition.py`.

**Referências:**
- `mprm/projector.py:177-196` — construção atual do dicionário `trace`, com `idx`, `len`, `order`, `sign`. É onde entra a máscara por segmento.
- `mprm/projector.py:281-305` — etapas 3 e 4 do `project`: extração dos traços e média da Eq. 21 com chaves `("v", i, iy)` / `("h", ix, j)`.
- `mprm/flow.py:44-70` — consumo atual das chaves cartesianas em `mprm_projection_step`.
- `mprm/downscaler.py:101-147` — `solve_patch(divw_loc, fW, fE, fS, fN, solid_loc)`: a orientação de `fW/fE` é `+x` e a de `fS/fN` é `+y` (ver `rhs[0,:] += fW*hy` e `vt1[0,:] = fW`).
- `mrcm/topology.py:203-212` — `edge_bbox` `(Nedges, 4)`, a extensão física de cada segmento.
- Lages, MPrM, Eq. 21 (média dos traços de fluxo nas interfaces) — `~/Dissertação/Refs/The_multiscale_projection_method_for_the_solution_of_incompressible_Navier_Stokes_equations.pdf`.
- Spec §4.2.1 e §5 Fase 3.

- [ ] **Step 1: Escrever os testes que falham**

Acrescente a `tests/test_quadtree_projector.py`:

```python
# ═══════════════════════════════════════════════════════════════════════
# Traços por segmento e média por aresta de Γ (spec §5 Fase 3)
# ═══════════════════════════════════════════════════════════════════════
from mprm.decomposition import leaf_boundary_fluxes


def _arvore_com_no_pendurado(L=1.0):
    """2x2 folhas com a folha 0 subdividida: 4 arestas com nó pendurado."""
    from mrcm.quadtree import QuadTree, QuadTreeNode
    H = L / 2
    root = QuadTreeNode(0.0, 0.0, L, L, 0)
    root.children = [QuadTreeNode(i * H, j * H, (i + 1) * H, (j + 1) * H, 0)
                     for j in range(2) for i in range(2)]
    qt = QuadTree(root=root, max_level=1, Lx=L, Ly=L)
    qt._build_neighbours()
    qt.leaves[0].subdivide()
    qt._invalidate_cache()
    qt.balance_2to1()
    return qt


def test_project_devolve_chaves_por_segmento():
    cfg = _cfg()
    p = MPrMProjector(cfg, dt=0.01)
    f = p.project(_fonte(cfg))
    for k, v in f.items():
        assert isinstance(k, tuple) and len(k) == 3, f"chave inesperada: {k}"
        isub, side, seg = k
        assert 0 <= side <= 3 and seg in (0, 1)
        assert isinstance(v, np.ndarray)
    # Os dois lados de cada aresta recebem o MESMO array (Eq. 21).
    for e in p.decomp.gamma_edges():
        assert np.array_equal(f[e.a], f[e.b]), f"aresta {e.ell} assimétrica"


def test_leaf_boundary_fluxes_posiciona_segmentos():
    cfg = _cfg()
    p = MPrMProjector(cfg, dt=0.01)
    f = p.project(_fonte(cfg))
    for lf in p.decomp.leaves:
        fS, fE, fN, fW = leaf_boundary_fluxes(p.decomp, f, lf.isub)
        assert fS.shape == (lf.nlocx,) and fN.shape == (lf.nlocx,)
        assert fW.shape == (lf.nlocy,) and fE.shape == (lf.nlocy,)
        # Lados externos ficam zerados (parede/entrada: ṽ·n = 0).
        if lf.x0 == 0.0:
            assert np.allclose(fW, 0.0)


def test_quadtree_com_no_pendurado_continuidade_fraca():
    """Salto de fluxo desprezível através das arestas penduradas.

    Roda com degreeU=2/degreeP=2: o modo constante (k=0) é idêntico nas duas
    parametrizações de base e não detectaria erro de segmento — foi essa a
    lição da RevisaoQuadtree.md do mrcm.
    """
    from mprm.decomposition import Decomposition
    cfg = MRCMConfig(Ndomx=2, Ndomy=2, nlocx=8, nlocy=8,
                     degreeU=2, degreeP=2, alpha=1.0,
                     basis_kind="legendre_P", mixed_family="BDM",
                     mixed_order=0, executor_kind="sequential",
                     use_quadtree=True)
    qt = _arvore_com_no_pendurado()
    d = Decomposition.from_quadtree(qt, h=0.25 / 4, Lx=1.0, Ly=1.0)
    p = MPrMProjector(cfg, dt=0.01, decomposition=d)

    rng = np.random.default_rng(11)
    divw = rng.standard_normal((d.Nx, d.Ny))
    divw -= divw.mean()
    f = p.project(divw)

    escala = max(np.abs(np.concatenate([v for v in f.values()])).max(), 1e-30)
    for e in p.decomp.gamma_edges():
        salto = np.abs(p.raw_traces[e.a] - p.raw_traces[e.b]).max()
        assert salto <= 1e-8 * escala + 1e-12, (
            f"aresta ell={e.ell} ({e.a} | {e.b}): salto {salto:.3e} "
            f"contra escala de fluxo {escala:.3e}")
```

- [ ] **Step 2: Rodar e verificar que falha**

```bash
python -m pytest tests/test_quadtree_projector.py -k "segmento or pendurado or boundary" -q 2>&1 | tail -8
```

Esperado: falha na primeira asserção de `test_project_devolve_chaves_por_segmento` (as chaves ainda são `("v", i, iy)`).

- [ ] **Step 3: Filtrar os traços por segmento no construtor**

Em `mprm/projector.py`, substitua o laço `for s in internal_sides:` que popula `self.trace` por:

```python
            # Traços por SEGMENTO: os dois segmentos de um lado compartilham o
            # mesmo marcador `ds`, então a separação é pela bbox do segmento.
            for side, seg, ell, bbox, nfaces in decomposition.segments_of(isub):
                marker = SIDE_TO_MARKER[side]
                ln = assemble(tau * ds_m(marker)).dat.data_ro.copy()
                idx = np.nonzero(np.abs(ln) > 1e-14)[0]
                xm = assemble(SpatialCoordinate(mesh)[0] * tau
                              * ds_m(marker)).dat.data_ro[idx] / ln[idx]
                ym = assemble(SpatialCoordinate(mesh)[1] * tau
                              * ds_m(marker)).dat.data_ro[idx] / ln[idx]
                horiz = side in (0, 2)
                coord = xm if horiz else ym
                lo = bbox[0] if horiz else bbox[1]
                hi = bbox[2] if horiz else bbox[3]
                keep = (coord >= lo - 0.25 * self.hx_f) & \
                       (coord <= hi + 0.25 * self.hx_f)
                idx, coord = idx[keep], coord[keep]
                if idx.size != nfaces:
                    raise ValueError(
                        f"folha {isub} lado {side} seg {seg}: {idx.size} faces "
                        f"finas no segmento, esperado {nfaces}. Alinhamento de "
                        f"segmento ou h inconsistente.")
                order = np.argsort(coord)
                sign = +1.0 if side in (1, 2) else -1.0   # right/top: n = +e
                self.trace[(isub, side, seg)] = {
                    "form": dot(uh, n_facet) * tau * ds_m(marker),
                    "idx": idx, "len": ln[idx], "order": order, "sign": sign,
                }
```

- [ ] **Step 4: Trocar a média da Eq. 21 no `project`**

Em `mprm/projector.py`, no `project`, substitua o bloco de extração (dentro do laço de reconstrução) e todo o bloco de média por:

```python
            for (jsub, side, seg), tr in self.trace.items():
                if jsub != isub:
                    continue
                vals = assemble(tr["form"]).dat.data_ro[tr["idx"]]
                my_traces[(isub, side, seg)] = \
                    tr["sign"] * (vals / tr["len"])[tr["order"]]

        # 4. Média dos fluxos por aresta de Γ (Eq. 21). Com h uniforme os dois
        #    lados têm o mesmo número de faces finas, então é média face a face
        #    — inclusive nas arestas com nó pendurado.
        all_tr = {}
        for d in comm.allgather(my_traces):
            all_tr.update(d)
        self.raw_traces = all_tr

        fluxes = {}
        for e in self.decomp.gamma_edges():
            media = 0.5 * (all_tr[e.a] + all_tr[e.b])
            fluxes[e.a] = media
            fluxes[e.b] = media
```

- [ ] **Step 5: Implementar `leaf_boundary_fluxes`**

Acrescente ao final de `mprm/decomposition.py`:

```python
def leaf_boundary_fluxes(decomp: "Decomposition", flux_avg: dict, isub: int):
    """Fluxos dos 4 lados da folha, montados a partir dos segmentos de Γ.

    Sob quadtree um lado pode ter dois segmentos, cada um confrontando uma
    folha vizinha diferente. Como h é uniforme, cada segmento cobre um bloco
    contíguo e exato de faces finas do lado, e a montagem é concatenação — sem
    interpolação nem restrição.

    Returns
    -------
    (fS, fE, fN, fW) : arrays de tamanho (nlocx, nlocy, nlocx, nlocy).
        Orientação: fS/fN em +y, fW/fE em +x — a mesma que
        `MACDownscaler.solve_patch` espera.
    """
    lf = decomp.leaf(isub)
    f = {0: np.zeros(lf.nlocx), 1: np.zeros(lf.nlocy),
         2: np.zeros(lf.nlocx), 3: np.zeros(lf.nlocy)}
    for side, seg, ell, bbox, nfaces in decomp.segments_of(isub):
        val = flux_avg.get((isub, side, seg))
        if val is None:                       # lado externo: ṽ·n = 0
            continue
        horiz = side in (0, 2)
        origin = lf.x0 if horiz else lf.y0
        start = bbox[0] if horiz else bbox[1]
        off = int(round((start - origin) / decomp.h))
        if off < 0 or off + val.size > f[side].size:
            raise ValueError(
                f"folha {isub} lado {side} seg {seg}: offset {off} + "
                f"{val.size} faces excede o lado ({f[side].size}).")
        f[side][off:off + val.size] = val
    return f[0], f[1], f[2], f[3]
```

- [ ] **Step 6: Adaptar `mprm_projection_step` ao novo formato**

Em `mprm/flow.py`, dentro de `mprm_projection_step`, substitua o bloco que hoje lê `fluxes.get(("v", ix, iy), zeros_m)` (quatro linhas) por:

```python
    from mprm.decomposition import leaf_boundary_fluxes
    decomp = projector.decomp
    ...
    for isub in projector.my_subs:
        lf = decomp.leaf(isub)
        ox, oy = lf.i0, lf.j0
        n_i, m_i = lf.nlocx, lf.nlocy
        fS, fE, fN, fW = leaf_boundary_fluxes(decomp, fluxes, isub)
```

e ajuste os overrides de saída e o recorte do patch para usar `ox, oy, n_i, m_i`
no lugar de `ix*n, iy*m, n, m`. As linhas de atualização global passam a ser:

```python
            u[ox:ox + n_i + 1, oy:oy + m_i] = w1[ox:ox + n_i + 1, oy:oy + m_i] - vt1
            v[ox:ox + n_i, oy:oy + m_i + 1] = w2[ox:ox + n_i, oy:oy + m_i + 1] - vt2
```

**Nota:** nesta tarefa o `MACDownscaler` ainda assume `nloc` fixo; como todos os
testes aqui usam decomposição uniforme ou uma árvore cujas folhas o downscaler
ainda não precisa resolver, isso é suficiente. A Tarefa 8 remove a limitação.

- [ ] **Step 7: Rodar os testes**

```bash
python -m pytest tests/test_quadtree_projector.py -q 2>&1 | tail -10
```

Esperado: `7 passed`. Registre no relatório o valor do salto medido em
`test_quadtree_com_no_pendurado_continuidade_fraca` (deve ser da ordem de
`1e-18`–`1e-15` contra uma escala de fluxo `O(1e-2)`).

- [ ] **Step 8: Regressão completa**

```bash
python -m pytest tests/ -q 2>&1 | tail -5
```

Esperado: `55 passed`.

- [ ] **Step 9: Relatório e commit**

`docs/superpowers/reports/07-tracos-por-segmento.md` conforme §B. No campo 9, deixe explícito que **o formato de retorno de `project()` mudou** e qual é o novo.

```bash
git add mprm/projector.py mprm/decomposition.py mprm/flow.py tests/test_quadtree_projector.py docs/superpowers/reports/07-tracos-por-segmento.md
git commit -m "feat(mprm): traços de fluxo por segmento e média da Eq. 21 por aresta de Γ

project() passa a devolver dict[(isub, side, seg)] e a média percorre as
arestas do esqueleto, o que funciona com nó pendurado sem operador de
restrição — consequência de h uniforme (spec §4.2.1, §5 Fase 3)."
```

---

## Task 8: Downscaler por folha — cache por geometria, defeito retornado, folha sólida

Implementa: spec §5 Fase 4 e §4.5 (item 1).

**Files:**
- Modify: `mprm/downscaler.py`
- Create: `tests/test_quadtree_downscaler.py`

**Interfaces:**
- Produces:
  - `MACDownscaler(cfg, dt, h=None)` — `h` opcional; quando dado, substitui `cfg.Hx/cfg.nlocx`.
  - `MACDownscaler.solve_patch(divw_loc, fW, fE, fS, fN, solid_loc=None) -> (vt1, vt2, defect)` — **retorno de 3 elementos** (antes 2). O tamanho do patch vem do shape de `divw_loc`, não mais do `cfg`.
  - `MACDownscaler.last_defect` permanece, agora como máximo sobre a última chamada de `reset_defect()`.
  - `MACDownscaler.reset_defect()`.

**Referências:**
- `mprm/downscaler.py:25-58` — construtor com a matriz única; passa a ser cache.
- `mprm/downscaler.py:60-99` — `_build_masked_lu` e `_get_masked`: a chave do cache ganha `(n, m)`.
- `mprm/downscaler.py:101-154` — `solve_patch`; a linha `abs(defect)/(hx*hy*nfluid)` é a que divide por zero quando a folha é 100 % sólida.
- `mprm/flow.py:68-70` — chamador atual, que desempacota dois valores.
- Spec §4.5 e §5 Fase 4.

- [ ] **Step 1: Escrever os testes que falham**

Crie `tests/test_quadtree_downscaler.py`:

```python
"""Downscaler com patches de tamanhos diferentes e folhas sólidas."""
import numpy as np
import pytest

from mrcm.config import MRCMConfig
from mprm.downscaler import MACDownscaler


def _cfg(nloc=8):
    return MRCMConfig(Ndomx=2, Ndomy=2, nlocx=nloc, nlocy=nloc,
                      degreeU=1, degreeP=1, alpha=1.0,
                      basis_kind="legendre_P", mixed_family="BDM",
                      mixed_order=0, executor_kind="sequential")


def _neumann_compativel(n, m, h, seed=0):
    """Fonte com integral nula e fluxos de contorno nulos."""
    rng = np.random.default_rng(seed)
    divw = rng.standard_normal((n, m))
    return divw - divw.mean()


def test_patches_de_tamanhos_diferentes():
    d = MACDownscaler(_cfg(), dt=0.01, h=0.5 / 8)
    for (n, m) in ((8, 8), (4, 4), (16, 16)):
        divw = _neumann_compativel(n, m, d.h)
        vt1, vt2, defect = d.solve_patch(divw, np.zeros(m), np.zeros(m),
                                         np.zeros(n), np.zeros(n))
        assert vt1.shape == (n + 1, m)
        assert vt2.shape == (n, m + 1)
        assert abs(defect) < 1e-12, f"patch {n}x{m}: defeito {defect:.3e}"


def test_cache_reutiliza_fatoracao_por_geometria():
    d = MACDownscaler(_cfg(), dt=0.01, h=0.5 / 8)
    divw8 = _neumann_compativel(8, 8, d.h)
    d.solve_patch(divw8, np.zeros(8), np.zeros(8), np.zeros(8), np.zeros(8))
    n_lu = len(d._lu_cache)
    d.solve_patch(divw8, np.zeros(8), np.zeros(8), np.zeros(8), np.zeros(8))
    assert len(d._lu_cache) == n_lu, "refatorou uma geometria já vista"
    divw4 = _neumann_compativel(4, 4, d.h)
    d.solve_patch(divw4, np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4))
    assert len(d._lu_cache) == n_lu + 1, "não fatorou a nova geometria"


def test_folha_totalmente_solida_devolve_zero_sem_nan():
    """spec §4.5 item 1: nfluid = 0 dividia por zero no defeito."""
    d = MACDownscaler(_cfg(), dt=0.01, h=0.5 / 8)
    n = m = 8
    solid = np.ones((n, m), dtype=bool)
    vt1, vt2, defect = d.solve_patch(np.zeros((n, m)), np.zeros(m), np.zeros(m),
                                     np.zeros(n), np.zeros(n), solid_loc=solid)
    assert np.all(vt1 == 0.0) and np.all(vt2 == 0.0)
    assert np.isfinite(defect) and defect == 0.0


def test_solucao_de_patch_uniforme_inalterada():
    """Regressão numérica: o caminho sem sólido continua dando o mesmo ṽ."""
    d = MACDownscaler(_cfg(), dt=0.01, h=0.5 / 8)
    n = m = 8
    divw = _neumann_compativel(n, m, d.h, seed=5)
    fW = np.linspace(-1.0, 1.0, m) * 1e-3
    fE = np.linspace(1.0, -1.0, m) * 1e-3
    fS = np.zeros(n)
    fN = np.zeros(n)
    vt1, vt2, defect = d.solve_patch(divw, fW, fE, fS, fN)
    # Condições de contorno impostas exatamente nas faces externas do patch.
    assert np.allclose(vt1[0, :], fW, atol=1e-14, rtol=0)
    assert np.allclose(vt1[-1, :], fE, atol=1e-14, rtol=0)
    assert np.allclose(vt2[:, 0], fS, atol=1e-14, rtol=0)
    assert np.allclose(vt2[:, -1], fN, atol=1e-14, rtol=0)


def test_defeito_e_por_chamada_nao_acumulado():
    d = MACDownscaler(_cfg(), dt=0.01, h=0.5 / 8)
    n = m = 8
    divw = np.ones((n, m))          # fonte incompatível: defeito não nulo
    _, _, defect_ruim = d.solve_patch(divw, np.zeros(m), np.zeros(m),
                                      np.zeros(n), np.zeros(n))
    assert abs(defect_ruim) > 0.0
    divw_ok = _neumann_compativel(n, m, d.h)
    _, _, defect_bom = d.solve_patch(divw_ok, np.zeros(m), np.zeros(m),
                                     np.zeros(n), np.zeros(n))
    assert abs(defect_bom) < 1e-12, (
        f"o defeito da chamada anterior vazou: {defect_bom:.3e}")
```

- [ ] **Step 2: Rodar e verificar que falha**

```bash
python -m pytest tests/test_quadtree_downscaler.py -q 2>&1 | tail -8
```

Esperado: `TypeError: MACDownscaler.__init__() got an unexpected keyword argument 'h'`.

- [ ] **Step 3: Reescrever `mprm/downscaler.py`**

Substitua o corpo da classe por:

```python
class MACDownscaler:
    """Problemas locais de Neumann (Eq. 22) na malha MAC, por folha.

    Sob decomposição quadtree as folhas têm tamanhos diferentes, então a
    matriz FV de 5 pontos deixa de ser única: é cacheada por
    ``(n, m, padrão de sólido)``. Como os níveis de refinamento são poucos, o
    número de fatorações distintas é pequeno.

    O defeito de compatibilidade é **retornado** por chamada. O antigo atributo
    acumulado nunca era reiniciado entre passos e, com folhas de tamanhos
    diferentes, misturava escalas de normalização.
    """

    def __init__(self, cfg, dt: float, h: float = None):
        self.cfg = cfg
        self.dt = float(dt)
        self.h = float(h) if h is not None else cfg.Hx / cfg.nlocx
        self.hx = self.hy = self.h
        self._lu_cache = {}       # (n, m, solid_key) -> (splu, pin)
        self.last_defect = 0.0

    def reset_defect(self):
        """Zera o acumulador de defeito (chamar a cada passo de tempo)."""
        self.last_defect = 0.0

    # ── matrizes ────────────────────────────────────────────────────────
    def _build_lu(self, n, m, solid):
        """Fatoração FV de 5 pontos para o patch (n, m) e um padrão de sólido.

        Células sólidas viram p = 0 (linha identidade, desacopladas) e as faces
        sólidas somem do estêncil (no-flow). O pino da pressão vai na primeira
        célula de FLUIDO — o sistema é Neumann puro.
        """
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        dt, hx, hy = self.dt, self.hx, self.hy

        def lid(i, j):
            return i + j * n

        rows, cols, vals = [], [], []
        for j in range(m):
            for i in range(n):
                c = lid(i, j)
                if solid is not None and solid[i, j]:
                    rows.append(c); cols.append(c); vals.append(1.0)
                    continue
                diag = 0.0
                for (di, dj, d, ln) in ((1, 0, hx, hy), (-1, 0, hx, hy),
                                        (0, 1, hy, hx), (0, -1, hy, hx)):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < n and 0 <= jj < m and \
                            not (solid is not None and solid[ii, jj]):
                        w = dt * ln / d
                        rows.append(c); cols.append(lid(ii, jj))
                        vals.append(w); diag -= w
                rows.append(c); cols.append(c); vals.append(diag)
        A = sp.csr_matrix((vals, (rows, cols)), shape=(n * m, n * m)).tolil()
        if solid is None:
            pin = 0
        else:
            pin = int(np.argmin(solid.flatten(order="F")))
        A[pin, :] = 0.0
        A[pin, pin] = 1.0
        return spla.splu(A.tocsc()), pin

    def _get_lu(self, n, m, solid):
        key = (n, m, None if solid is None else solid.tobytes())
        if key not in self._lu_cache:
            self._lu_cache[key] = self._build_lu(n, m, solid)
        return self._lu_cache[key]

    # ── solve ───────────────────────────────────────────────────────────
    def solve_patch(self, divw_loc, fW, fE, fS, fN, solid_loc=None):
        """Resolve o Neumann local (Eq. 22) e devolve ``(ṽ₁, ṽ₂, defeito)``.

        O tamanho do patch vem de ``divw_loc.shape`` — não do cfg —, o que é o
        que permite folhas de tamanhos diferentes.
        """
        dt, hx, hy = self.dt, self.hx, self.hy
        n, m = divw_loc.shape
        solid = None if solid_loc is None else \
            np.asarray(solid_loc, dtype=bool).reshape(n, m)

        # Folha 100% sólida: nada a resolver. Sem esta guarda, nfluid = 0 e a
        # normalização do defeito divide por zero (spec §4.5, item 1).
        if solid is not None and solid.all():
            self.last_defect = max(self.last_defect, 0.0)
            return np.zeros((n + 1, m)), np.zeros((n, m + 1)), 0.0

        rhs = (divw_loc * hx * hy).copy()
        rhs[0, :] += fW * hy
        rhs[-1, :] -= fE * hy
        rhs[:, 0] += fS * hx
        rhs[:, -1] -= fN * hx

        lu, pin = self._get_lu(n, m, solid)
        if solid is None:
            b = rhs.flatten(order="F")
            defect = float(b.sum())
            norm = hx * hy * n * m
            b -= defect / b.size
            b[pin] = 0.0
            p = lu.solve(b).reshape((n, m), order="F")
        else:
            rhs[solid] = 0.0
            b = rhs.flatten(order="F")
            fluid = ~solid.flatten(order="F")
            nfluid = int(fluid.sum())
            defect = float(b[fluid].sum())
            norm = hx * hy * nfluid
            b[fluid] -= defect / nfluid
            b[~fluid] = 0.0
            b[pin] = 0.0
            p = lu.solve(b).reshape((n, m), order="F")
            p[solid] = 0.0

        defect_rel = abs(defect) / norm
        self.last_defect = max(self.last_defect, defect_rel)

        vt1 = np.empty((n + 1, m))
        vt1[1:-1, :] = dt * (p[1:, :] - p[:-1, :]) / hx
        vt1[0, :], vt1[-1, :] = fW, fE
        vt2 = np.empty((n, m + 1))
        vt2[:, 1:-1] = dt * (p[:, 1:] - p[:, :-1]) / hy
        vt2[:, 0], vt2[:, -1] = fS, fN

        if solid is not None:
            fx = np.zeros((n + 1, m), bool); fx[:-1, :] |= solid; fx[1:, :] |= solid
            fy = np.zeros((n, m + 1), bool); fy[:, :-1] |= solid; fy[:, 1:] |= solid
            vt1[fx] = 0.0
            vt2[fy] = 0.0
        return vt1, vt2, defect_rel
```

- [ ] **Step 4: Atualizar o chamador**

Em `mprm/flow.py`, dentro de `mprm_projection_step`, o desempacotamento passa a ser de três valores e o defeito é agregado:

```python
        vt1, vt2, defect = downscaler.solve_patch(
            divw[ox:ox + n_i, oy:oy + m_i], fW, fE, fS, fN,
            solid_loc=solid_loc)
        my_patch[isub] = (vt1, vt2)
        my_defects[isub] = defect
```

Declare `my_defects = {}` junto de `my_patch` e, após o `allgather`, guarde
`projector.last_defects = {...}` com o dicionário reunido (o laço temporal usa
isso para reportar por folha).

Em `mprm/flow.py`, na construção do downscaler dentro de `run_flow`, passe o `h`:

```python
        downscaler = MACDownscaler(cfg, dt, h=projector.decomp.h)
```

- [ ] **Step 5: Rodar os testes**

```bash
python -m pytest tests/test_quadtree_downscaler.py -q 2>&1 | tail -8
```

Esperado: `6 passed`. Registre no relatório os defeitos numéricos medidos.

- [ ] **Step 6: Regressão completa**

```bash
python -m pytest tests/ -q 2>&1 | tail -5
```

Esperado: `61 passed`.

- [ ] **Step 7: Relatório e commit**

`docs/superpowers/reports/08-downscaler-por-folha.md` conforme §B. Deixe claro no campo 9 que **`solve_patch` mudou de assinatura de retorno**.

```bash
git add mprm/downscaler.py mprm/flow.py tests/test_quadtree_downscaler.py docs/superpowers/reports/08-downscaler-por-folha.md
git commit -m "feat(mprm): downscaler por folha — cache por geometria e defeito retornado

O tamanho do patch passa a vir do shape da fonte, a fatoração é cacheada por
(n, m, padrão de sólido) e o defeito é devolvido por chamada em vez de
acumulado num atributo nunca reiniciado. Guarda para folha 100% sólida, que a
família (b) produz por construção (spec §4.5, §5 Fase 4)."
```

---

## Task 9: von Kármán rodando sobre quadtree, ponta a ponta

Implementa: spec §5 Fase 4 (conclusão) e §6. É a primeira tarefa que produz **física** sobre decomposição adaptativa.

**Files:**
- Modify: `mprm/flow.py` (`run_flow`, `run_von_karman`)
- Create: `tests/test_quadtree_flow.py`

**Interfaces:**
- Produces:
  - `run_flow(problem, Ndomx, Ndomy, nloc, T, ..., decomposition=None)` — novo parâmetro; quando dado, `Ndomx/Ndomy/nloc` são ignorados na construção do projetor e usados apenas para o `MRCMConfig`.
  - `run_von_karman(..., decomposition=None)`.
  - O dicionário de retorno ganha a chave `"decomposition"` com `{"N_sub", "n_gamma_edges", "n_gamma_dofs", "H_min", "H_max", "levels"}`.

**Referências:**
- `mprm/flow.py:151-241` — `run_flow`: construção de `cfg`, `MPrMProjector`, `MACDownscaler` e o laço temporal.
- `mprm/flow.py:28-84` — `mprm_projection_step`, já adaptado nas Tarefas 7 e 8.
- `mprm/problem.py:134-170` — `von_karman_problem(Lx, Ly, Nx, Ny, D, xc, yc, U_in, Re, inflow, walls, inflow_norm)`.
- `mprm/problem.py:173-204` — `assert_solid_interior_to_subdomains`: sob decomposição quadtree ela **não se aplica** (foi escrita para a grade cartesiana); a verificação equivalente é `Decomposition.check_fluid_connectivity` (spec §4.5).
- `mprm/forces.py:15-40` — `ChannelFinePoisson`, a referência de malha fina.
- Spec §6 (fluxo de dados do passo) e §5 Fase 4.

- [ ] **Step 1: Escrever os testes que falham**

Crie `tests/test_quadtree_flow.py`:

```python
"""von Kármán sobre decomposição quadtree: conservação e estrutura."""
import numpy as np
import pytest

from mrcm.config import MRCMConfig
from mprm.decomposition import (Decomposition, plan_quadtree_geometry,
                                build_obstacle_tree)
from mprm.problem import von_karman_problem, square_cylinder
from mprm.transport import mac_divergence, solid_faces
from mprm.flow import run_flow


def _setup_pequeno():
    """Canal pequeno o bastante para um smoke test (~30s).

    Ly=6D, N_by=3 (ímpar, família (a)) -> H0=2D > D. Níveis 0-1, nloc=(8,4),
    h=D/4, malha 32x24. O corpo cabe na folha base [0,2]x[2,4].
    """
    g = plan_quadtree_geometry(D=1.0, Ly=6.0, N_bx=4, N_by=3,
                               max_level=1, nloc_min=4, family="a")
    qt = build_obstacle_tree(g, radii=(3.0,))
    problem = von_karman_problem(
        Lx=g["Lx"], Ly=g["Ly"], Nx=g["Nx"], Ny=g["Ny"], D=g["D"],
        xc=g["xc"], yc=g["yc"], U_in=1.0, Re=100.0,
        inflow="parabolic", walls="slip", inflow_norm="max")
    d = Decomposition.from_quadtree(qt, h=g["h"], Lx=g["Lx"], Ly=g["Ly"],
                                    solid=problem.solid)
    return g, qt, problem, d


def test_arvore_e_valida_para_o_obstaculo():
    g, qt, problem, d = _setup_pequeno()
    assert len(d.leaves) > g["N_bx"] * g["N_by"], "a árvore não refinou nada"
    niveis = sorted({lf.level for lf in d.leaves})
    assert niveis == [0, 1], f"níveis obtidos: {niveis}"
    d.check_fluid_connectivity(problem.solid)     # não levanta


def test_von_karman_quadtree_conserva_massa():
    g, qt, problem, d = _setup_pequeno()
    r = run_flow(problem, Ndomx=g["N_bx"], Ndomy=g["N_by"], nloc=g["nloc"][0],
                 T=0.2, decomposition=d, verbose=False,
                 obstacle_metrics=False, U_ref=1.0, D=g["D"])
    dv = np.abs(mac_divergence(r["u"], r["v"], g["h"], g["h"]))
    dv_fluido = dv[~problem.solid].max()
    assert dv_fluido < 1e-11, f"max|div u| no fluido = {dv_fluido:.3e}"


def test_von_karman_quadtree_respeita_o_obstaculo():
    g, qt, problem, d = _setup_pequeno()
    r = run_flow(problem, Ndomx=g["N_bx"], Ndomy=g["N_by"], nloc=g["nloc"][0],
                 T=0.2, decomposition=d, verbose=False,
                 obstacle_metrics=False, U_ref=1.0, D=g["D"])
    fx, fy = solid_faces(problem.solid)
    assert np.abs(r["u"][fx]).max() == 0.0
    assert np.abs(r["v"][fy]).max() == 0.0


def test_resultado_reporta_a_decomposicao():
    g, qt, problem, d = _setup_pequeno()
    r = run_flow(problem, Ndomx=g["N_bx"], Ndomy=g["N_by"], nloc=g["nloc"][0],
                 T=0.05, decomposition=d, verbose=False,
                 obstacle_metrics=False, U_ref=1.0, D=g["D"])
    info = r["decomposition"]
    assert info["N_sub"] == len(d.leaves)
    assert info["n_gamma_edges"] == len(d.gamma_edges())
    assert info["H_min"] < info["H_max"], "a árvore deveria ser heterogênea"
    assert info["n_gamma_dofs"] > 0


def test_cavidade_uniforme_continua_igual():
    """Regressão: o caminho sem `decomposition` não pode ter mudado."""
    from mprm.flow import run_cavity
    r = run_cavity(Re=100.0, Ndom=2, nloc=8, T=0.05, verbose=False)
    dv = np.abs(mac_divergence(r["u"], r["v"], 1.0 / 16, 1.0 / 16)).max()
    assert dv < 1e-11, f"max|div u| = {dv:.3e}"
```

- [ ] **Step 2: Rodar e verificar que falha**

```bash
python -m pytest tests/test_quadtree_flow.py -q 2>&1 | tail -8
```

Esperado: `TypeError: run_flow() got an unexpected keyword argument 'decomposition'`.

- [ ] **Step 3: Aceitar a decomposição em `run_flow`**

Em `mprm/flow.py`, acrescente `decomposition=None` à assinatura de `run_flow` e, no bloco `if method == "mprm":`:

```python
    if method == "mprm":
        cfg = MRCMConfig(Ndomx=Ndomx, Ndomy=Ndomy, nlocx=nloc, nlocy=nloc,
                         degreeU=degreeU, degreeP=degreeP, alpha=alpha,
                         basis_kind=basis_kind, Lx=problem.Lx, Ly=problem.Ly,
                         mixed_family="BDM", mixed_order=0,
                         executor_kind="sequential",
                         use_quadtree=(decomposition is not None
                                       and not decomposition.is_uniform))
        projector = MPrMProjector(cfg, dt, comm=comm, problem=problem,
                                  eps_solid=eps_solid,
                                  decomposition=decomposition)
        downscaler = MACDownscaler(cfg, dt, h=projector.decomp.h)
```

- [ ] **Step 4: Reportar a decomposição no retorno**

Ainda em `run_flow`, antes do `return out`:

```python
    if method == "mprm":
        d = projector.decomp
        Hs = [lf.Hx for lf in d.leaves]
        out["decomposition"] = {
            "N_sub": len(d.leaves),
            "n_gamma_edges": len(d.gamma_edges()),
            "n_gamma_dofs": int(projector.dofm.n_total),
            "H_min": float(min(Hs)), "H_max": float(max(Hs)),
            "levels": sorted({lf.level for lf in d.leaves}),
            "is_uniform": bool(d.is_uniform),
        }
```

- [ ] **Step 5: Repassar em `run_von_karman`**

Acrescente `decomposition=None` à assinatura de `run_von_karman` e passe adiante
na chamada a `run_flow`. Quando `decomposition` é dado, **não** chame
`assert_solid_interior_to_subdomains`: essa verificação foi escrita para a grade
cartesiana e a equivalente sob quadtree é `check_fluid_connectivity`, feita na
construção da `Decomposition`. Em `mprm/projector.py`, ajuste a guarda existente:

```python
        if problem.solid is not None and self.decomp.is_uniform:
            assert_solid_interior_to_subdomains(problem.solid, cfg, strict=False)
```

- [ ] **Step 6: Rodar os testes**

```bash
python -m pytest tests/test_quadtree_flow.py -q 2>&1 | tail -8
```

Esperado: `5 passed`. Registre no relatório: número de folhas da árvore, níveis,
`H_min/H_max`, `n_gamma_edges` e o `max|div u|` medido.

- [ ] **Step 7: Regressão completa**

```bash
python -m pytest tests/ -q 2>&1 | tail -5
```

Esperado: `66 passed`.

- [ ] **Step 8: Relatório e commit**

`docs/superpowers/reports/09-von-karman-quadtree.md` conforme §B.

```bash
git add mprm/flow.py mprm/projector.py tests/test_quadtree_flow.py docs/superpowers/reports/09-von-karman-quadtree.md
git commit -m "feat(mprm): von Kármán sobre decomposição quadtree, ponta a ponta

run_flow aceita uma Decomposition e reporta a estrutura no resultado. Massa
conservada em 1e-12 no fluido com árvore adaptativa (spec §5 Fase 4, §6)."
```

---

## Task 10: Equivalência sequencial ↔ MPI sob quadtree

Implementa: spec §7 (consistência entre ranks) e §8 (teste 9).

**Files:**
- Create: `tests/mpi_quadtree_run.py` (script auxiliar, executado sob `mpiexec`)
- Modify: `tests/test_quadtree_flow.py` (teste que dispara o subprocesso)
- Modify: `mprm/projector.py` (chamar a verificação de consistência)

**Interfaces:**
- Consumes: `Decomposition.assert_consistent_across_ranks(comm)` (Tarefa 5).
- Produces: nenhum símbolo novo de biblioteca.

**Referências:**
- `mprm/projector.py:74` — distribuição round-robin `my_subs`.
- `mprm/projector.py:261-269, 288-292` — `gather`/`allgather` do caminho paralelo.
- `docs/10-paralelizacao-mprm.md` — convenções de MPI já adotadas no projeto.
- `RevisaoQuadtree.md §2.3` (em `~/mrcm/mrcm-firedrake`) — o modo de falha que a verificação previne: esqueletos divergentes produzem lixo silencioso.
- Spec §7.

- [ ] **Step 1: Escrever o script auxiliar**

Crie `tests/mpi_quadtree_run.py`:

```python
#!/usr/bin/env python3
"""Roda von Kármán sobre quadtree e imprime um resumo em JSON no rank 0.

Usado por `tests/test_quadtree_flow.py` para comparar sequencial e MPI. Cada
rank constrói a árvore por conta própria — é exatamente o que a verificação de
consistência do esqueleto precisa exercitar.
"""
import json
import sys
import numpy as np

from mpi4py import MPI
from mprm.decomposition import (Decomposition, plan_quadtree_geometry,
                                build_obstacle_tree)
from mprm.problem import von_karman_problem
from mprm.transport import mac_divergence
from mprm.flow import run_flow


def main():
    comm = MPI.COMM_WORLD
    g = plan_quadtree_geometry(D=1.0, Ly=6.0, N_bx=4, N_by=3,
                               max_level=1, nloc_min=4, family="a")
    qt = build_obstacle_tree(g, radii=(3.0,))
    problem = von_karman_problem(
        Lx=g["Lx"], Ly=g["Ly"], Nx=g["Nx"], Ny=g["Ny"], D=g["D"],
        xc=g["xc"], yc=g["yc"], U_in=1.0, Re=100.0,
        inflow="parabolic", walls="slip", inflow_norm="max")
    d = Decomposition.from_quadtree(qt, h=g["h"], Lx=g["Lx"], Ly=g["Ly"],
                                    solid=problem.solid)
    r = run_flow(problem, Ndomx=g["N_bx"], Ndomy=g["N_by"], nloc=g["nloc"][0],
                 T=0.2, decomposition=d, comm=comm, verbose=False,
                 obstacle_metrics=False, U_ref=1.0, D=g["D"])
    if comm.rank == 0:
        dv = np.abs(mac_divergence(r["u"], r["v"], g["h"], g["h"]))
        print("JSON " + json.dumps({
            "ranks": comm.size,
            "u_norm": float(np.linalg.norm(r["u"])),
            "v_norm": float(np.linalg.norm(r["v"])),
            "u_sum": float(r["u"].sum()),
            "div_max": float(dv[~problem.solid].max()),
            "N_sub": r["decomposition"]["N_sub"],
        }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Escrever o teste que falha**

Acrescente a `tests/test_quadtree_flow.py`:

```python
import json
import os
import shutil
import subprocess
import sys


def _rodar(nranks):
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "mpi_quadtree_run.py")
    cmd = ([sys.executable, script] if nranks == 1
           else ["mpiexec", "-n", str(nranks), sys.executable, script])
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=900,
                         cwd=os.path.dirname(os.path.dirname(script)))
    assert out.returncode == 0, f"falhou com {nranks} rank(s):\n{out.stderr[-3000:]}"
    linha = [l for l in out.stdout.splitlines() if l.startswith("JSON ")]
    assert linha, f"sem saída JSON:\n{out.stdout[-2000:]}"
    return json.loads(linha[-1][5:])


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("mpiexec") is None, reason="mpiexec ausente")
def test_sequencial_e_mpi_concordam_sob_quadtree():
    seq = _rodar(1)
    par = _rodar(2)
    assert seq["N_sub"] == par["N_sub"]
    for chave in ("u_norm", "v_norm", "u_sum"):
        ref = abs(seq[chave]) or 1.0
        err = abs(seq[chave] - par[chave]) / ref
        assert err < 1e-10, f"{chave}: erro relativo {err:.3e} (seq vs 2 ranks)"
    assert par["div_max"] < 1e-11
```

- [ ] **Step 3: Rodar e verificar que falha ou passa por acidente**

```bash
python -m pytest tests/test_quadtree_flow.py -k mpi -q 2>&1 | tail -15
```

Se passar de primeira, **ainda assim** execute o Step 4: a verificação de
consistência precisa existir mesmo quando a árvore é determinística hoje.
Registre no relatório qual dos dois casos ocorreu.

- [ ] **Step 4: Chamar a verificação de consistência no projetor**

Em `mprm/projector.py`, logo após `self.decomp = decomposition`:

```python
        # Cada rank constrói a árvore por conta própria; se um divergir, os
        # triplets são montados contra numerações de Γ diferentes e o sistema
        # global sai numericamente errado, sem erro nenhum.
        self.decomp.assert_consistent_across_ranks(comm)
```

- [ ] **Step 5: Registrar o marcador `slow`**

Crie `pytest.ini` na raiz do repositório:

```ini
[pytest]
markers =
    slow: testes que disparam subprocessos MPI ou laços temporais longos
```

- [ ] **Step 6: Rodar o teste MPI e a suíte**

```bash
python -m pytest tests/test_quadtree_flow.py -q 2>&1 | tail -5
python -m pytest tests/ -q 2>&1 | tail -5
```

Esperado: `67 passed` no total. Registre no relatório os valores de `u_norm`
sequencial e com 2 ranks, e o erro relativo medido.

- [ ] **Step 7: Relatório e commit**

`docs/superpowers/reports/10-mpi-quadtree.md` conforme §B.

```bash
git add tests/mpi_quadtree_run.py tests/test_quadtree_flow.py mprm/projector.py pytest.ini docs/superpowers/reports/10-mpi-quadtree.md
git commit -m "test(mprm): equivalência sequencial/MPI sob quadtree + verificação de esqueleto

Cada rank constrói a árvore independentemente; a verificação por hash impede o
modo de falha silencioso descrito na RevisaoQuadtree.md §2.3 (spec §7)."
```

---

## Task 11: Módulo do estudo comparativo

Implementa: spec §5 Fase 5 (funções puras). Sem I/O: nada de arquivos, plots ou `print` fora do modo verboso — o mesmo contrato de `mprm/benchmarks/convergence.py`.

**Files:**
- Create: `mprm/benchmarks/quadtree_vk.py`
- Create: `tests/test_quadtree_von_karman.py` (só a parte de teste; a CLI vem na Tarefa 12)

**Interfaces:**
- Produces:
  - `build_case(family, D, Ly, N_bx, N_by, max_level, nloc_min, radii, Re, U_in) -> dict` — geometria + problema + decomposição, sem resolver nada.
  - `run_case(case, T, comm, degreeU, degreeP, alpha, verbose) -> dict` — resultado de uma configuração no schema do spec §5.
  - `uniform_counterpart(case, target) -> dict` — decomposição cartesiana comparável (`target` é `"n_gamma"` ou `"N_sub"`).
  - `run_study(cases, T, comm, ...) -> dict` — JSON completo (`geometry`, `flow`, `spaces`, `configs`).

**Referências:**
- `mprm/benchmarks/convergence.py:1-60` — o padrão de módulo puro a seguir (`_make_meta`, retorno em dicionário).
- `mprm/forces.py:217-260` — `obstacle_forces(u, v, p, problem, U_ref, D, u_prev, v_prev, dt)`.
- `mprm/forces.py` — `strouhal(...)` e `DiagnosticPressureSolver`.
- `mprm/flow.py:151-241` — `run_flow` e as chaves do dicionário de retorno (`u`, `v`, `dt`, `nsteps`, `divmax`, `defect`, `t_wall`, `cl_history`, `probe_history`, `decomposition`).
- Spec §5 Fase 5 (as 5 configurações, as métricas e o schema JSON — copie o schema verbatim de lá).
- Lages, MPrM, §3.5 e Tabelas 2–3 (valores de referência `C_D = 1,33`, `C_L = 0,263`, `St = 0,121`).
- Breuer et al. (2000), IJHFF 21:186–196 (`C_D = 1,38`–`1,47`, `St = 0,130`–`0,145`). **Não está em `~/Dissertação/Refs`** — citar pelos números registrados em `docs/11-plano-validacao-von-karman.md §1.2`.

- [ ] **Step 1: Escrever o teste que falha**

Crie `tests/test_quadtree_von_karman.py`:

```python
"""Estudo comparativo quadtree vs cartesiano — estrutura e schema."""
import numpy as np
import pytest

from mprm.benchmarks.quadtree_vk import (build_case, run_case,
                                         uniform_counterpart, run_study)

CHAVES_CONFIG = {
    "label", "N_sub", "n_gamma_edges", "n_gamma_dofs", "H_min", "H_max",
    "levels", "cond_A_gamma", "t_offline_s", "t_online_per_step_s",
    "Cd_mean", "Cd_std", "Cl_amp", "St", "div_max", "defect_max", "err_u_rel",
}


def _caso_pequeno(family="a"):
    if family == "a":
        return build_case(family="a", D=1.0, Ly=6.0, N_bx=4, N_by=3,
                          max_level=1, nloc_min=4, radii=(3.0,),
                          Re=100.0, U_in=1.0)
    return build_case(family="b", D=1.0, Ly=4.0, N_bx=4, N_by=2,
                      max_level=3, nloc_min=4, radii=(4.0, 2.0, 1.0),
                      Re=100.0, U_in=1.0)


def test_build_case_familia_a():
    c = _caso_pequeno("a")
    assert c["geom"]["family"] == "a"
    assert c["decomp"].h == pytest.approx(c["geom"]["h"])
    assert c["problem"].solid.any()
    assert len(c["decomp"].leaves) >= 12


def test_build_case_familia_b_corpo_e_uniao_de_folhas():
    c = _caso_pequeno("b")
    g = c["geom"]
    D, xc, yc = g["D"], g["xc"], g["yc"]
    dentro = [lf for lf in c["decomp"].leaves
              if lf.x0 >= xc - D / 2 - 1e-12 and lf.x1 <= xc + D / 2 + 1e-12
              and lf.y0 >= yc - D / 2 - 1e-12 and lf.y1 <= yc + D / 2 + 1e-12]
    assert len(dentro) == 4


def test_uniform_counterpart_casa_o_alvo():
    c = _caso_pequeno("a")
    u = uniform_counterpart(c, target="N_sub")
    assert u["decomp"].is_uniform
    assert u["decomp"].h == pytest.approx(c["decomp"].h), \
        "o baseline precisa ter o MESMO h para a comparação ser honesta"
    alvo = len(c["decomp"].leaves)
    obtido = len(u["decomp"].leaves)
    assert abs(obtido - alvo) <= alvo, f"alvo {alvo}, obtido {obtido}"


def test_run_case_produz_o_schema():
    c = _caso_pequeno("a")
    r = run_case(c, T=0.1, comm=None, degreeU=1, degreeP=3, alpha=1.0,
                 verbose=False)
    assert CHAVES_CONFIG.issubset(set(r)), \
        f"faltando: {CHAVES_CONFIG - set(r)}"
    assert r["div_max"] < 1e-11
    assert r["cond_A_gamma"] > 0.0
    assert r["t_offline_s"] > 0.0


def test_run_study_monta_o_json():
    cases = [_caso_pequeno("a")]
    out = run_study(cases, T=0.1, comm=None, degreeU=1, degreeP=3, alpha=1.0,
                    with_uniform_baseline=True, with_fine_reference=False,
                    verbose=False)
    assert set(out) >= {"geometry", "flow", "spaces", "configs"}
    assert len(out["configs"]) >= 2, "estudo sem baseline uniforme"
    for cfgd in out["configs"]:
        assert CHAVES_CONFIG.issubset(set(cfgd))
```

- [ ] **Step 2: Rodar e verificar que falha**

```bash
python -m pytest tests/test_quadtree_von_karman.py -q 2>&1 | tail -5
```

Esperado: `ModuleNotFoundError: No module named 'mprm.benchmarks.quadtree_vk'`.

- [ ] **Step 3: Implementar o módulo**

Crie `mprm/benchmarks/quadtree_vk.py`:

```python
"""Estudo comparativo: decomposição quadtree vs cartesiana no von Kármán.

Funções puras — sem I/O, sem plot, sem escrita de arquivo. O script
`tests/test_quadtree_von_karman.py` cuida de CLI, JSON e figuras, seguindo o
mesmo contrato de `mprm/benchmarks/convergence.py`.

Métricas por configuração (spec §5, Fase 5): C_D médio, amplitude de C_L, St,
erro relativo contra a referência de malha fina, max|∇·u|, defeito de
compatibilidade, N_sub, n_Γ, min/max H, condicionamento de A_Γ, e tempos de
offline e online separados.
"""
from __future__ import annotations
import time
from datetime import datetime
import numpy as np

from mprm.decomposition import (Decomposition, plan_quadtree_geometry,
                                build_obstacle_tree)
from mprm.problem import von_karman_problem
from mprm.flow import run_flow
from mprm.forces import strouhal


def build_case(*, family, D, Ly, N_bx, N_by, max_level, nloc_min, radii,
               Re=100.0, U_in=1.0, wake_length=0.0, label=None) -> dict:
    """Geometria + problema + decomposição, sem resolver nada."""
    g = plan_quadtree_geometry(D=D, Ly=Ly, N_bx=N_bx, N_by=N_by,
                               max_level=max_level, nloc_min=nloc_min,
                               family=family)
    qt = build_obstacle_tree(g, radii=radii, wake_length=wake_length)
    problem = von_karman_problem(
        Lx=g["Lx"], Ly=g["Ly"], Nx=g["Nx"], Ny=g["Ny"], D=g["D"],
        xc=g["xc"], yc=g["yc"], U_in=U_in, Re=Re,
        inflow="parabolic", walls="slip", inflow_norm="max")
    decomp = Decomposition.from_quadtree(qt, h=g["h"], Lx=g["Lx"], Ly=g["Ly"],
                                         solid=problem.solid)
    return {"label": label or f"quadtree_{family}_L{max_level}",
            "geom": g, "problem": problem, "decomp": decomp,
            "Re": Re, "U_in": U_in}


def uniform_counterpart(case: dict, target: str = "n_gamma",
                        label: str = None) -> dict:
    """Decomposição cartesiana comparável, com o MESMO h e o MESMO problema.

    `target` é "N_sub" ou "n_gamma". Fixada a malha global, as decomposições
    cartesianas com células quadradas formam uma família discreta: `Nx/Ndomx`
    e `Ny/Ndomy` precisam ser inteiros iguais. Percorremos os divisores comuns
    e escolhemos o que mais se aproxima do alvo — a diferença residual é
    reportada em `target_gap`, porque não há como zerá-la em geral (spec §5).
    """
    g, decomp = case["geom"], case["decomp"]
    Nx, Ny = g["Nx"], g["Ny"]
    alvo = (len(decomp.leaves) if target == "N_sub"
            else len(decomp.gamma_edges()))

    melhor, melhor_erro = None, None
    for nloc in range(2, min(Nx, Ny) + 1, 2):
        if Nx % nloc or Ny % nloc:
            continue
        Ndomx, Ndomy = Nx // nloc, Ny // nloc
        d = Decomposition.uniform(Ndomx, Ndomy, nloc, g["Lx"], g["Ly"])
        valor = (Ndomx * Ndomy if target == "N_sub" else len(d.gamma_edges()))
        erro = abs(valor - alvo)
        if melhor_erro is None or erro < melhor_erro:
            melhor, melhor_erro = (d, Ndomx, Ndomy, nloc, valor), erro
    if melhor is None:
        raise ValueError(
            f"nenhuma decomposição cartesiana com células quadradas para "
            f"Nx={Nx}, Ny={Ny}.")
    d, Ndomx, Ndomy, nloc, valor = melhor
    return {"label": label or f"cartesian_{Ndomx}x{Ndomy}",
            "geom": g, "problem": case["problem"], "decomp": d,
            "Re": case["Re"], "U_in": case["U_in"],
            "Ndomx": Ndomx, "Ndomy": Ndomy, "nloc": nloc,
            "target": target, "target_value": valor, "target_gap": melhor_erro}


def _cond_estimado(A) -> float:
    """Estimativa barata do número de condição de A_Γ (norma 1)."""
    import scipy.sparse.linalg as spla
    if A is None:
        return float("nan")
    try:
        return float(np.linalg.cond(A.toarray(), p=1)) if A.shape[0] <= 2000 \
            else float(spla.norm(A, 1) * spla.norm(spla.inv(A.tocsc()), 1))
    except Exception:
        return float("nan")


def run_case(case: dict, T: float, comm=None, degreeU=1, degreeP=3,
             alpha=1.0, basis_kind="legendre_P", u_ref=None, v_ref=None,
             verbose=False) -> dict:
    """Roda uma configuração e devolve o dicionário do schema (spec §5)."""
    g, problem, decomp = case["geom"], case["problem"], case["decomp"]
    Ndomx = case.get("Ndomx", g["N_bx"])
    Ndomy = case.get("Ndomy", g["N_by"])
    nloc = case.get("nloc", g["nloc"][0])

    t0 = time.perf_counter()
    r = run_flow(problem, Ndomx=Ndomx, Ndomy=Ndomy, nloc=nloc, T=T,
                 decomposition=decomp, degreeU=degreeU, degreeP=degreeP,
                 alpha=alpha, basis_kind=basis_kind, comm=comm,
                 verbose=verbose, obstacle_metrics=True,
                 U_ref=case["U_in"], D=g["D"])
    t_total = time.perf_counter() - t0

    info = r["decomposition"]
    cl = r.get("cl_history")
    if cl is not None and len(cl) > 4:
        metade = len(cl) // 2
        Cd = cl[metade:, 1]
        Cl = cl[metade:, 2]
        Cd_mean, Cd_std = float(Cd.mean()), float(Cd.std())
        Cl_amp = float(0.5 * (Cl.max() - Cl.min()))
        # strouhal(signal, dt, U_ref, D, ...) — ver mprm/forces.py:286.
        # O sinal é amostrado a cada passo (obstacle_metrics_every=1), logo o
        # dt de amostragem é o dt do laço.
        St = float(strouhal(cl[:, 2], r["dt"], case["U_in"], g["D"]))
    else:
        Cd_mean = Cd_std = Cl_amp = St = float("nan")

    err_u = float("nan")
    if u_ref is not None:
        num = np.sum((r["u"] - u_ref) ** 2) + np.sum((r["v"] - v_ref) ** 2)
        den = np.sum(u_ref ** 2) + np.sum(v_ref ** 2)
        err_u = float(np.sqrt(num / den))

    t_online = t_total / max(r["nsteps"], 1)
    return {"label": case["label"],
            "N_sub": info["N_sub"],
            "n_gamma_edges": info["n_gamma_edges"],
            "n_gamma_dofs": info["n_gamma_dofs"],
            "H_min": info["H_min"], "H_max": info["H_max"],
            "levels": info["levels"],
            "cond_A_gamma": float("nan"),      # preenchido abaixo se possível
            "t_offline_s": float(r.get("t_offline", 0.0)),
            "t_online_per_step_s": float(t_online),
            "Cd_mean": Cd_mean, "Cd_std": Cd_std,
            "Cl_amp": Cl_amp, "St": St,
            "div_max": float(r["divmax"]),
            "defect_max": float(r["defect"]),
            "err_u_rel": err_u,
            "target_gap": case.get("target_gap", 0),
            "u": r["u"], "v": r["v"]}


def run_study(cases, T, comm=None, degreeU=1, degreeP=3, alpha=1.0,
              basis_kind="legendre_P", with_uniform_baseline=True,
              with_fine_reference=True, verbose=False) -> dict:
    """Executa todas as configurações e monta o JSON do spec §5."""
    g = cases[0]["geom"]
    problem = cases[0]["problem"]

    u_ref = v_ref = None
    if with_fine_reference:
        r_fine = run_flow(problem, Ndomx=1, Ndomy=1, nloc=1, T=T,
                          method="fine", comm=comm, verbose=verbose,
                          obstacle_metrics=False, U_ref=cases[0]["U_in"],
                          D=g["D"])
        u_ref, v_ref = r_fine["u"], r_fine["v"]

    todos = list(cases)
    if with_uniform_baseline:
        for alvo in ("n_gamma", "N_sub"):
            todos.append(uniform_counterpart(cases[0], target=alvo))

    configs = []
    for c in todos:
        res = run_case(c, T=T, comm=comm, degreeU=degreeU, degreeP=degreeP,
                       alpha=alpha, basis_kind=basis_kind,
                       u_ref=u_ref, v_ref=v_ref, verbose=verbose)
        res.pop("u", None)
        res.pop("v", None)
        configs.append(res)

    return {"meta": {"date": datetime.now().isoformat(timespec="seconds"),
                     "n_ranks": getattr(comm, "size", 1)},
            "geometry": {k: g[k] for k in
                         ("family", "Lx", "Ly", "D", "h", "Nx", "Ny",
                          "xc", "yc", "H0", "nloc", "max_level")},
            "flow": {"Re": cases[0]["Re"], "U_in": cases[0]["U_in"],
                     "inflow": "parabolic", "inflow_norm": "max",
                     "walls": "slip", "T": T},
            "spaces": {"degreeU": degreeU, "degreeP": degreeP,
                       "alpha": alpha, "basis_kind": basis_kind},
            "configs": configs}
```

- [ ] **Step 4: Preencher `cond_A_gamma` e `t_offline`**

Duas ligações que faltam:

1. Em `mprm/flow.py`, dentro de `run_flow`, meça o tempo de construção do
   projetor e grave em `out["t_offline"]`:

```python
        t_off0 = time.perf_counter()
        projector = MPrMProjector(...)
        downscaler = MACDownscaler(cfg, dt, h=projector.decomp.h)
        t_offline = time.perf_counter() - t_off0
```
   e, no dicionário de saída, `out["t_offline"] = t_offline` (use `0.0` no
   caminho `method="fine"`).

2. Em `run_flow`, exponha a matriz para o estudo:
   `out["A_gamma"] = projector.A_gamma` **não** — isso carregaria a matriz para
   dentro do JSON. Em vez disso, grave o condicionamento já reduzido a número:

```python
        out["cond_A_gamma"] = _cond_A_gamma(projector)
```

   com o helper em `mprm/flow.py`:

```python
def _cond_A_gamma(projector) -> float:
    """Estimativa do número de condição de A_Γ (norma 1), ou nan.

    É a métrica que responde ao risco das folhas sólidas na família (b)
    (spec §4.5, item 3).
    """
    A = getattr(projector, "A_gamma", None)
    if A is None or A.shape[0] > 4000:
        return float("nan")
    try:
        return float(np.linalg.cond(A.toarray(), p=1))
    except Exception:
        return float("nan")
```

   e em `quadtree_vk.run_case`, troque a linha de `cond_A_gamma` por
   `float(r.get("cond_A_gamma", float("nan")))`. Remova então o helper
   `_cond_estimado` do módulo de benchmark — ele deixa de ser usado.

- [ ] **Step 5: Rodar os testes**

```bash
python -m pytest tests/test_quadtree_von_karman.py -q 2>&1 | tail -8
```

Esperado: `5 passed`.

- [ ] **Step 6: Regressão completa**

```bash
python -m pytest tests/ -q 2>&1 | tail -5
```

Esperado: `72 passed`.

- [ ] **Step 7: Relatório e commit**

`docs/superpowers/reports/11-modulo-estudo.md` conforme §B.

```bash
git add mprm/benchmarks/quadtree_vk.py mprm/flow.py tests/test_quadtree_von_karman.py docs/superpowers/reports/11-modulo-estudo.md
git commit -m "feat(mprm): módulo do estudo comparativo quadtree vs cartesiano

Funções puras que constroem casos, escolhem o baseline cartesiano comparável
(mesmo h) e produzem o schema JSON do spec §5, incluindo cond(A_Γ) — a métrica
que responde ao risco das folhas sólidas."
```

---

## Task 12: CLI do estudo — JSON, tabela e figuras

Implementa: spec §5 Fase 5 (camada de I/O). Segue o padrão dos demais scripts de `tests/`: pytest + `main()` executável, com `common.py` cuidando de estilo, JSON e figuras.

**Files:**
- Modify: `tests/test_quadtree_von_karman.py` (acrescenta `main()` e helpers de I/O)

**Interfaces:**
- Consumes: `mprm.benchmarks.quadtree_vk.{build_case, uniform_counterpart, run_study}` (Tarefa 11); `tests/common.py:{get_comm, apply_plot_style, banner, make_meta, save_json, save_figure, COLORS, INK, DEFAULT_OUTDIR}`; `tests/params.py:{parse_args, common_params_from_args}`.
- Produces: `results_alpha/quadtree_von_karman.json` e `results_alpha/quadtree_von_karman.png`.

**Referências:**
- `tests/common.py:37,65,79-141` — `DEFAULT_OUTDIR`, `get_comm()`, `COLORS`, `INK`, `apply_plot_style()`, `banner(msg, comm)`, `make_meta(study, comm, **fields)`, `save_json(results, outdir, filename, comm)`, `save_figure(fig, outdir, filename)`.
- `tests/params.py:77-146` — `build_parser`, `parse_args(description, ...)`, `common_params_from_args(args)`.
- `tests/test_von_karman.py` — o modelo de script a imitar (pytest + `main()` + `if __name__ == "__main__"`).
- `docs/von-karman-guia-execucao.md` — convenções de execução já documentadas.
- Spec §5 Fase 5 (schema JSON).

- [ ] **Step 1: Escrever o teste que falha**

Acrescente a `tests/test_quadtree_von_karman.py`:

```python
def test_tabela_de_resultados_formata_todas_as_configs(tmp_path):
    from tests.test_quadtree_von_karman import _tabela   # auto-import
    out = {"configs": [
        {"label": "quadtree_a_L1", "N_sub": 18, "n_gamma_edges": 24,
         "n_gamma_dofs": 144, "H_min": 1.0, "H_max": 2.0, "levels": [0, 1],
         "cond_A_gamma": 1.2e3, "t_offline_s": 3.4,
         "t_online_per_step_s": 0.02, "Cd_mean": 1.31, "Cd_std": 0.01,
         "Cl_amp": 0.26, "St": 0.121, "div_max": 1e-14, "defect_max": 2e-15,
         "err_u_rel": 0.03, "target_gap": 0},
    ]}
    txt = _tabela(out)
    assert "quadtree_a_L1" in txt
    assert "1.31" in txt and "0.121" in txt
    assert txt.count("\n") >= 3, "tabela sem cabeçalho ou sem linhas"
```

- [ ] **Step 2: Rodar e verificar que falha**

```bash
python -m pytest tests/test_quadtree_von_karman.py -k tabela -q 2>&1 | tail -5
```

Esperado: `ImportError: cannot import name '_tabela'`.

- [ ] **Step 3: Implementar a camada de I/O**

Acrescente ao final de `tests/test_quadtree_von_karman.py`:

```python
# ═══════════════════════════════════════════════════════════════════════
# CLI: execução do estudo, JSON, tabela e figura
# ═══════════════════════════════════════════════════════════════════════
_COLUNAS = [
    ("label", "configuração", "{:<22}"),
    ("N_sub", "N_sub", "{:>6}"),
    ("n_gamma_dofs", "n_Γ", "{:>7}"),
    ("H_min", "H_min", "{:>7.3f}"),
    ("H_max", "H_max", "{:>7.3f}"),
    ("Cd_mean", "C_D", "{:>7.3f}"),
    ("Cl_amp", "C_L amp", "{:>8.3f}"),
    ("St", "St", "{:>6.3f}"),
    ("err_u_rel", "err u", "{:>9.2e}"),
    ("div_max", "max|div|", "{:>9.1e}"),
    ("cond_A_gamma", "cond A_Γ", "{:>9.1e}"),
    ("t_offline_s", "offline s", "{:>10.1f}"),
    ("t_online_per_step_s", "online s/passo", "{:>14.4f}"),
]


def _tabela(out: dict) -> str:
    """Tabela textual das configurações — vai para o stdout e para a dissertação."""
    cab = "  ".join(rot for _, rot, _ in _COLUNAS)
    linhas = [cab, "-" * len(cab)]
    for c in out["configs"]:
        celulas = []
        for chave, _, fmt in _COLUNAS:
            v = c.get(chave)
            try:
                celulas.append(fmt.format(v))
            except (TypeError, ValueError):
                celulas.append(fmt.replace(".3f", "").replace(".2e", "")
                               .replace(".1e", "").replace(".4f", "")
                               .replace(".1f", "").format(str(v)))
        linhas.append("  ".join(celulas))
    return "\n".join(linhas)


def _figura(out: dict):
    """Barras comparativas: erro contra a referência fina e custo por passo."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from common import apply_plot_style, COLORS, INK

    apply_plot_style()
    cfgs = out["configs"]
    rotulos = [c["label"] for c in cfgs]
    x = np.arange(len(cfgs))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].bar(x, [c["err_u_rel"] for c in cfgs], color=COLORS[1])
    axes[0].set_ylabel(r"$\|u-u_{ref}\|_2/\|u_{ref}\|_2$")
    axes[0].set_title("Erro contra a malha fina")

    axes[1].bar(x, [c["n_gamma_dofs"] for c in cfgs], color=COLORS[2])
    axes[1].set_ylabel(r"$n_\Gamma$")
    axes[1].set_title("Graus de liberdade de interface")

    axes[2].bar(x, [c["t_online_per_step_s"] for c in cfgs], color=COLORS[3])
    axes[2].set_ylabel("s / passo")
    axes[2].set_title("Custo online")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(rotulos, rotation=30, ha="right", color=INK)
    fig.tight_layout()
    return fig


def main():
    import sys
    from common import (get_comm, banner, make_meta, save_json, save_figure,
                        DEFAULT_OUTDIR)
    from params import parse_args

    def _flags(p):
        """Flags específicas deste estudo. `--quick` já vem do parser comum
        (tests/params.py:121)."""
        p.add_argument("--family", default="a", choices=["a", "b"],
                       help="alinhamento corpo/árvore (spec §4.3.1)")

    # parse_args(description, defaults=None, extra_args=callable, argv=None)
    # — ver tests/params.py:130.
    args = parse_args("Estudo comparativo: quadtree vs cartesiano (von Kármán)",
                      extra_args=_flags)
    comm = get_comm()
    rapido = args.quick

    if rapido:
        caso = build_case(family=args.family, D=1.0, Ly=6.0, N_bx=4, N_by=3,
                          max_level=1, nloc_min=4, radii=(3.0,),
                          Re=100.0, U_in=1.0) if args.family == "a" else \
               build_case(family="b", D=1.0, Ly=4.0, N_bx=4, N_by=2,
                          max_level=3, nloc_min=4, radii=(4.0, 2.0, 1.0),
                          Re=100.0, U_in=1.0)
        T = 1.0
    elif args.family == "a":
        # Configuração de referência do spec §4.3.
        caso = build_case(family="a", D=1.0, Ly=8.0, N_bx=10, N_by=3,
                          max_level=2, nloc_min=8, radii=(6.0, 3.0),
                          Re=100.0, U_in=1.0)
        T = 40.0
    else:
        # Configuração de referência da família (b), spec §4.3.1.
        caso = build_case(family="b", D=1.0, Ly=8.0, N_bx=6, N_by=2,
                          max_level=4, nloc_min=4,
                          radii=(8.0, 4.0, 2.0, 1.0), Re=100.0, U_in=1.0)
        T = 40.0

    banner(f"Estudo quadtree (família {args.family}) — "
           f"malha {caso['geom']['Nx']}x{caso['geom']['Ny']}, "
           f"{len(caso['decomp'].leaves)} folhas, T={T}", comm)

    out = run_study([caso], T=T, comm=comm, degreeU=1, degreeP=3, alpha=1.0,
                    with_uniform_baseline=True,
                    with_fine_reference=not rapido,
                    verbose=(comm.rank == 0))
    out["meta"].update(make_meta("quadtree_von_karman", comm,
                                 family=args.family, quick=rapido))

    if comm.rank == 0:
        print(_tabela(out), flush=True)
        save_json(out, DEFAULT_OUTDIR, "quadtree_von_karman.json", comm=comm)
        save_figure(_figura(out), DEFAULT_OUTDIR, "quadtree_von_karman.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

**Nota:** o parser comum de `tests/params.py` já define `--Lx`, `--Ly`,
`--Ndom`, `--nloc`, `--degreeU`, `--degreeP`, `--alpha`, `--Re`, `--T`, `--dt`,
`--outdir`, `--quick` e `--quiet`. Não os redefina em `_flags` — `argparse`
levanta `ArgumentError` em opção duplicada.

- [ ] **Step 4: Rodar o teste da tabela**

```bash
python -m pytest tests/test_quadtree_von_karman.py -q 2>&1 | tail -5
```

Esperado: `6 passed`.

- [ ] **Step 5: Rodar a CLI no modo rápido (fumaça de ponta a ponta)**

```bash
python tests/test_quadtree_von_karman.py --quick 2>&1 | tail -20
ls -la results_alpha/quadtree_von_karman.json results_alpha/quadtree_von_karman.png
```

Esperado: tabela impressa com pelo menos 3 configurações, JSON e PNG gravados.
Cole a tabela no relatório.

- [ ] **Step 6: Regressão completa**

```bash
python -m pytest tests/ -q 2>&1 | tail -5
```

Esperado: `73 passed`.

- [ ] **Step 7: Relatório e commit**

`docs/superpowers/reports/12-cli-estudo.md` conforme §B.

```bash
git add tests/test_quadtree_von_karman.py docs/superpowers/reports/12-cli-estudo.md
git commit -m "feat(mprm): CLI do estudo quadtree — JSON, tabela e figuras

Segue o padrão dos demais scripts de tests/ (pytest + main()), com common.py
cuidando de estilo, JSON e figuras."
```

---

## Task 13: Executar o estudo e escrever os resultados

Implementa: spec §9 (critérios de sucesso). Esta tarefa **não muda código de biblioteca** — ela produz os números e o texto.

**Files:**
- Create: `docs/13-quadtree-von-karman.md`
- Create (gerados): `results_alpha/quadtree_von_karman_a.json`, `..._b.json`, PNGs

**Interfaces:**
- Consumes: a CLI da Tarefa 12.
- Produces: nenhum símbolo de código.

**Referências:**
- Spec §5 Fase 5 (as configurações), §9 (critérios de sucesso), §10 (riscos a verificar empiricamente).
- Lages, MPrM, §3.5, Tabelas 2–3 — valores de referência (`C_D = 1,33`, `C_L = 0,263`, `St = 0,121`). `~/Dissertação/Refs/The_multiscale_projection_method_for_the_solution_of_incompressible_Navier_Stokes_equations.pdf`.
- `docs/11-plano-validacao-von-karman.md §1.2` — Breuer et al. (2000): `C_D = 1,38`–`1,47`, `St = 0,130`–`0,145`.
- `docs/07-diagnostico-von-karman.md` — os desvios de setup já diagnosticados, para comparar com o que a geometria nova corrige.
- `docs/12-estudo-parametros-teoria.md` — formato dos writeups anteriores.

- [ ] **Step 1: Rodar a família (a) completa**

```bash
source ~/venv-firedrake/bin/activate
cd ~/mrcm/NavierStokes
mpiexec -n 4 python tests/test_quadtree_von_karman.py --family a 2>&1 | tee /tmp/estudo_a.log
cp results_alpha/quadtree_von_karman.json results_alpha/quadtree_von_karman_a.json
cp results_alpha/quadtree_von_karman.png results_alpha/quadtree_von_karman_a.png
```

Se exceder o tempo aceitável, use a contingência do spec §10: reduza um nível de
refinamento (`max_level=1`, `radii=(6.0,)`), **não** `nloc_min`. Registre a
mudança no relatório.

- [ ] **Step 2: Rodar a família (b) completa**

```bash
mpiexec -n 4 python tests/test_quadtree_von_karman.py --family b 2>&1 | tee /tmp/estudo_b.log
cp results_alpha/quadtree_von_karman.json results_alpha/quadtree_von_karman_b.json
cp results_alpha/quadtree_von_karman.png results_alpha/quadtree_von_karman_b.png
```

- [ ] **Step 3: Verificar os invariantes físicos antes de interpretar**

```bash
python - <<'PY'
import json
for fam in ("a", "b"):
    d = json.load(open(f"results_alpha/quadtree_von_karman_{fam}.json"))
    print(f"--- família {fam} ---")
    for c in d["configs"]:
        print(f"{c['label']:<24} div={c['div_max']:.2e} "
              f"defeito={c['defect_max']:.2e} cond={c['cond_A_gamma']:.2e}")
PY
```

Todo `div_max` precisa estar abaixo de `1e-11`. Se algum não estiver, **pare**:
o resultado não é interpretável e há um bug a investigar antes.

- [ ] **Step 4: Escrever `docs/13-quadtree-von-karman.md`**

Estrutura obrigatória do documento:

1. **Configurações executadas** — tabela com geometria, `h`, malha global,
   `N_sub`, `n_Γ`, `H_min/H_max` de cada configuração das duas famílias.
2. **Invariantes** — `max|div u|`, defeito de compatibilidade e `cond(A_Γ)`
   por configuração. É a evidência de que os números seguintes significam algo.
3. **Métricas físicas** — `C_D`, amplitude de `C_L`, `St`, com as referências
   de Lages (1,33 / 0,263 / 0,121) e Breuer (1,38–1,47 / 0,130–0,145) na mesma
   tabela.
4. **Adaptativo vs uniforme** — a `n_Γ` igual e a `N_sub` igual, com o
   `target_gap` explícito, porque a família cartesiana disponível é discreta.
5. **Famílias (a) vs (b)** — o refino chegando à superfície compensa? Responda
   com número, e reporte `cond(A_Γ)` das duas: é o custo previsto para as
   folhas sólidas (spec §4.5, item 3).
6. **Limitações** — o que a execução mostrou e o spec não previa.

**Regra de honestidade:** se o adaptativo não superar o uniforme, ou se a
família (b) não superar a (a), escreva isso como resultado. O spec §9.5 declara
esse desfecho publicável de antemão, justamente para que a redação não fique
tentada a maquiar.

- [ ] **Step 5: Rodar a suíte uma última vez**

```bash
python -m pytest tests/ -q 2>&1 | tail -5
```

Esperado: `73 passed`.

- [ ] **Step 6: Relatório e commit**

`docs/superpowers/reports/13-resultados.md` conforme §B. No campo 5, em vez de
testes, registre as execuções: comando, tempo de parede, e as métricas obtidas.

```bash
git add docs/13-quadtree-von-karman.md results_alpha/quadtree_von_karman_*.json results_alpha/quadtree_von_karman_*.png docs/superpowers/reports/13-resultados.md
git commit -m "docs(mprm): resultados do estudo quadtree vs cartesiano no von Kármán

Duas famílias de alinhamento corpo/árvore, comparadas a n_Γ e a N_sub iguais
contra a referência de malha fina, com os invariantes (div, defeito, cond A_Γ)
reportados antes das métricas físicas."
```

---

## §E. Autorrevisão do plano

Feita pelo autor do plano antes da entrega.

**1. Cobertura do spec**

| Seção do spec | Tarefa que implementa |
|---|---|
| §2.2 Achado 1 (remontagem de `A_Γ`) | 1, 2 |
| §2.6 Achado 6 (β do Robin por folha) | 6 |
| §4.2 camada `Decomposition` | 3, 5 |
| §4.2.1 composição de fluxos por segmento | 7 |
| §4.3 restrições geométricas, família (a) | 4 |
| §4.3.1 família (b), bloco 2×2 | 4, 11, 13 |
| §4.4 critério de refinamento | 4 |
| §4.5 folhas sólidas e conexidade | 5, 8 |
| §4.6 dependências do `mrcm` (nenhuma mudança lá) | 5 (teste de `nloc` heterogêneo) |
| §5 Fase 1 | 1, 2 |
| §5 Fase 2 | 3, 4, 5 |
| §5 Fase 3 | 6, 7 |
| §5 Fase 4 | 8, 9 |
| §5 Fase 5 | 11, 12, 13 |
| §5.6 relação com o plano de validação | 13 (writeup) |
| §7 validações e consistência MPI | 4, 5, 10 |
| §8 testes 1–12 | 1–10 |
| §9 critérios de sucesso | 13 |

**2. Itens do spec deliberadamente fora deste plano:** os Achados 2 (traço
pré-montado) e 3 (`Allgatherv`) do spec §2.3–2.4, listados na própria §11 do
spec como fora de escopo. Não há tarefa para eles, e isso é intencional.

**3. Consistência de assinaturas entre tarefas**

- `subdomain_matrix_triplets` / `subdomain_rhs` (T1) → consumidas em T2 com a mesma ordem de parâmetros.
- `Decomposition.uniform/from_quadtree/leaf/segments_of/gamma_edges/task` (T3, T5) → consumidas em T6, T7, T8, T9, T11 com os mesmos nomes.
- `leaf_boundary_fluxes(decomp, flux_avg, isub) -> (fS, fE, fN, fW)` (T7) → consumida em T7 (`mprm/flow.py`) na mesma ordem de retorno que `solve_patch` espera.
- `solve_patch(...) -> (vt1, vt2, defect)` (T8) → o chamador é atualizado na mesma tarefa; nenhuma tarefa posterior desempacota dois valores.
- `project(divw) -> dict[(isub, side, seg)]` (T7) → consumido por `leaf_boundary_fluxes` e por `tests/test_quadtree_projector.py`; o formato antigo não sobrevive em nenhum lugar.
- `run_flow(..., decomposition=None)` (T9) → usado em T11 e T12.

**4. Contagem de testes acumulada** (para conferência rápida do baseline em cada tarefa): T0 `21` · T1 `23` · T2 `26` · T3 `31` · T4 `42` · T5 `48` · T6 `52` · T7 `55` · T8 `61` · T9 `66` · T10 `67` · T11 `72` · T12–T13 `73`.

Se a sua contagem divergir, não "conserte" o número: investigue. Um teste a
menos costuma ser um `skipif` disparando (por exemplo, `mpiexec` ausente na
Tarefa 10), e isso deve ser registrado no relatório em vez de escondido.

---

## §F. Referências completas

Consolidação de tudo que o plano cita, para que nenhum agente precise procurar.

**Spec e documentos internos**
- `docs/superpowers/specs/2026-07-25-mprm-quadtree-von-karman-design.md` — spec deste trabalho.
- `docs/superpowers/specs/2026-07-25-mprm-quadtree-von-karman-parecer.md` — parecer externo; ver §4.6 do spec para o que foi verificado e rejeitado.
- `docs/01-resumo-mprm.md` — resumo do artigo do MPrM com a numeração de equações usada aqui.
- `docs/07-diagnostico-von-karman.md` — desvios de setup do benchmark.
- `docs/10-paralelizacao-mprm.md` — convenções de MPI.
- `docs/11-plano-validacao-von-karman.md` — valores de referência de Lages e Breuer.
- `docs/von-karman-guia-execucao.md` — convenções de execução.
- `~/mrcm/mrcm-firedrake/RevisaoQuadtree.md` — revisão crítica do quadtree (**anterior** às correções).
- `~/mrcm/mrcm-firedrake/handoffs/2026-07-25-quadtree-session-review.md` — correções aplicadas; prevalece sobre a revisão acima.
- `~/mrcm/mrcm-firedrake/quadtree-design.md` — decisões de projeto do quadtree (D1–D9 do `mrcm`).

**Código de referência (arquivo:linhas)**
- `mrcm/global_assembly.py:238-320` (triplas), `:540-578` (montagem e solve), `:158-160` (`_solve_direct`).
- `mrcm/topology.py:64-75` (`GammaIncidence`), `:109-160` (`check_segment_alignment`), `:203-218` (`sub_bbox`, `sub_nloc`), `:291-332` (`from_quadtree` e validação 2:1), `:481-488` (`is_uniform`), `:506-538` (`gamma_incidences`, `external_side_flags`), `:583-611` (`DOFManager`).
- `mrcm/quadtree.py:62-63` (lados), `:76-140` (`QuadTreeNode`), `:146-457` (`QuadTree`, `balance_2to1`), `:282-326` (`from_uniform` exige potências de 2).
- `mrcm/solver.py:104-141` (`_build_tasks`, o contrato das tarefas).
- `mrcm/local_solver.py:930-1000` (`build_subdomain_workspace`, leitura de `geom["nlocx"]` e Robin por folha).
- `mrcm/local_form.py:28-50` (`make_local_mesh`, malha triangular com `2·nx·ny` células).
- `mrcm/config.py:227-232` (`ndofsU = degreeU+1`, `ndofsP = degreeP+1`), `:246-281` (`robin_L_xy_for`).
- `mprm/projector.py`, `mprm/downscaler.py`, `mprm/flow.py`, `mprm/transport.py`, `mprm/problem.py`, `mprm/forces.py` — alvos das modificações, com as faixas citadas em cada tarefa.

**Bibliografia (com localização do arquivo)**

*Formulação:*
- Lages, C. *The multiscale projection method for the solution of incompressible Navier–Stokes equations.* Eqs. 8–10, 13, 15–23; §3.5. `~/Dissertação/Refs/The_multiscale_projection_method_for_the_solution_of_incompressible_Navier_Stokes_equations.pdf`
- Guiraldello, Ausas, Sousa, Pereira, Buscaglia. *The Multiscale Robin Coupled Method for flows in porous media.* Eqs. 58–59, 65. `~/Dissertação/Refs/MRCM.pdf`
- Guiraldello et al. *Interface spaces for the Multiscale Robin Coupled Method.* `~/Dissertação/Refs/Interface Spaces - MRCM.pdf`
- Quarteroni, Saleri, Veneziani. *Factorization methods for the numerical approximation of Navier–Stokes equations.* `~/Dissertação/Refs/Factorization methods for the numerical approximation of Navier–Stokes equations.pdf`

*Interfaces não conformes:*
- Arbogast, Cowsar, Wheeler, Yotov. *Mixed finite element methods on nonmatching multiblock grids.* `~/Dissertação/Refs/Mixed finite element methods on nonmatching multiblock grids.pdf`
- Arbogast, Pencheva, Wheeler, Yotov. *A multiscale mortar mixed finite element method.* Condição de solvabilidade → `nloc_min ≥ max(degreeU, degreeP)`. `~/Dissertação/Refs/A MULTISCALE MORTAR MIXED FINITE ELEMENT METHOD.pdf`
- Bernardi, Maday, Patera. *Domain decomposition by the mortar element method.* `~/Dissertação/Refs/`
- Ganis, Yotov. *Implementation of a mortar mixed finite element method using a multiscale flux basis.* Padrão offline/online. `~/Dissertação/Refs/`

*Árvores e adaptatividade:*
- Olshanskii, Terekhov, Vassilevski (2013). *An octree-based solver for the incompressible Navier–Stokes equations.* Computers & Fluids 84:231–246. `~/Dissertação/Refs/Olshanskii2013_octree_solver.pdf` — **usar o PDF**: a conversão em `Refs/markdown/` está corrompida (contém texto de outro artigo).
- *A multimesh finite element method for the Navier–Stokes equations based on projection methods.* `~/Dissertação/Refs/`
- de Berg et al. *Computational Geometry*, §14.3 — balanceamento 2:1. **Não está em `Refs`**; citado pelo `mrcm/quadtree.py:36-42`.
- Popinet (2003). *Gerris.* JCP 190. **Não está em `Refs`** — a obter para a redação.

*Elementos mistos e álgebra:*
- Brezzi, Fortin. *Mixed and hybrid finite element methods.* `~/Dissertação/Refs/Mixed and Hybrid Finite Element Methods - Brezzi.pdf`
- Arnold, Falk, Winther (2006). `~/Dissertação/Refs/arnold2006.pdf`
- Golub, Van Loan. *Matrix Computations*, 4ª ed. — fatoração LU e reuso, base da Fase 1. `~/Dissertação/Refs/2013 Matrix Computations 4th.pdf`

*Validação:*
- Breuer, Bernsdorf, Zeiser, Durst (2000). IJHFF 21:186–196. **Não está em `Refs`** — números registrados em `docs/11-plano-validacao-von-karman.md §1.2`.
- Ghia, Ghia, Shin. *High-Re solutions for incompressible flow…* — cavidade. `~/Dissertação/Refs/High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method.pdf`
- Benchmark de Helmholtz–Hodge de Olshanskii — já implementado em `mprm/benchmarks/hh.py`.

---
