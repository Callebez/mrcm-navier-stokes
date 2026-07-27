# Relatório — Tarefa 06: Projetor consumindo `Decomposition` (caminho uniforme, Robin por folha)

## 1. Identificação
- **Tarefa:** 06 — Projetor consumindo Decomposition, Robin por folha
- **Data:** 2026-07-27
- **Commit antes:** `14eee09` (Task 5 — from_quadtree)
- **Commit depois:** `5668905`
- **Arquivos criados:** `tests/test_quadtree_projector.py`
- **Arquivos modificados:**
  - `mprm/projector.py:53-91,126-142,144-151,155,228-232` — construtor refatorado
  - `mprm/decomposition.py:62-88,93-96,102-114,117-123` — suporte a células retangulares

## 2. Estado herdado
- Relatório 05 (`docs/superpowers/reports/05-decomposition-quadtree.md`): `Decomposition` com `uniform`, `from_quadtree`, `check_fluid_connectivity`, `assert_consistent_across_ranks`.
- Assinaturas consumidas:
  - `Decomposition.uniform(Ndomx, Ndomy, nloc, Lx, Ly) -> Decomposition`
  - `Decomposition.leaf(isub) -> Leaf`
  - `Decomposition.segments_of(isub) -> list[tuple]`
  - `Decomposition.task(cfg, isub, NunkU, problem) -> dict`
  - `cfg.robin_L_xy_for(Hx, Hy, nlocx, nlocy) -> tuple[float, float]`

## 3. Regressão ANTES da execução

| Suíte | Baseline esperado | Obtido antes desta tarefa | Igual? |
|---|---|---|---|
| `pytest tests/ -q` | 52 passed (19 base + 33 acumulados Tasks 1-5) | 52 passed | sim |

## 4. O que foi feito

Refatoração do `MPrMProjector` para consumir a camada `Decomposition` em vez de fazer aritmética `(ix, iy)` manualmente. Três mudanças estruturais:

1. **Parâmetro `decomposition` no construtor:** quando `None` (default), o projetor constrói `Decomposition.uniform` a partir do `cfg` — caminho retrocompatível. O esqueleto, DOFManager e `my_subs` passam a vir da decomposição.

2. **Robin por folha (correção do Achado 6):** em vez de um único `robin` global com `L = min(Hx, Hy)`, cada folha recebe seu próprio Robin com `L_xy = cfg.robin_L_xy_for(Hx, Hy, nlocx, nlocy)`. Isso garante que o β do solver local use o mesmo L das MsBF (`build_subdomain_workspace` já fazia isso internamente; o projetor usava L global, causando divergência em decomposições não uniformes — spec §2.6).

3. **Uso de `decomposition.task()` na Fase 1:** a construção manual do dicionário `task` com `mark_sides` e `edge_l` é substituída por `decomposition.task(cfg, isub, NunkU, problem)`. Os `internal_sides` passam a vir de `decomposition.segments_of(isub)`.

4. **Suporte a células retangulares no `Decomposition`:** `__init__` ganha parâmetro `hy` (keyword-only, default = `h`). `uniform()` aceita `nlocy` opcional e não impõe mais `hx == hy` — células retangulares são válidas no caminho cartesiano uniforme (necessário para o von Kármán com `Lx ≠ Ly`). `segments_of` usa `hx` ou `hy` conforme a orientação do lado.

5. **Verificação de sólido condicional:** `assert_solid_interior_to_subdomains` só é chamada quando `self.decomp.is_uniform` — sob quadtree a verificação equivalente é `check_fluid_connectivity` (spec §4.5).

## 5. Testes criados nesta tarefa

### `test_decomposicao_explicita_da_o_mesmo_que_a_implicita`
- **Arquivo:** `tests/test_quadtree_projector.py:32`
- **O que valida:** Passar `Decomposition.uniform` explicitamente produz os mesmos `X` e `fluxes` que deixar o projetor construir a decomposição implicitamente.
- **Resultado esperado:** `X` idêntico (atol=1e-12), mesmas chaves e valores de fluxo.
- **Execução 1 (antes da implementação):** `TypeError: MPrMProjector.__init__() got an unexpected keyword argument 'decomposition'` — FAIL esperado.
- **Execução 2 (após a implementação):** PASS.
- **Valor numérico obtido:** `max|X_impl - X_expl| = 0.0`, fluxos idênticos em todas as chaves.

### `test_projetor_expoe_a_decomposicao`
- **Arquivo:** `tests/test_quadtree_projector.py:48`
- **O que valida:** `p.decomp` existe, tem o número correto de folhas e `h` consistente com `cfg`.
- **Execução 1 (antes da implementação):** `AttributeError: 'MPrMProjector' object has no attribute 'decomp'` — FAIL esperado.
- **Execução 2 (após a implementação):** PASS.
- **Valor numérico obtido:** `len(p.decomp.leaves) = 4`, `decomp.h = 0.0625` (= `cfg.Lx / cfg.Nglobx`).

### `test_beta_do_solver_local_usa_o_L_da_folha`
- **Arquivo:** `tests/test_quadtree_projector.py:54`
- **O que valida:** O `L_xy` guardado em `p.robin_L[0]` bate com `cfg.robin_L_xy_for(lf.Hx, lf.Hy, lf.nlocx, lf.nlocy)`.
- **Execução 1 (antes da implementação):** `AttributeError: 'MPrMProjector' object has no attribute 'decomp'` — FAIL esperado.
- **Execução 2 (após a implementação):** PASS.
- **Valor numérico obtido:** `robin_L[0] = (0.0625, 0.0625)` (folha quadrada 0.5×0.5 com nloc=8).

### `test_cavidade_inalterada`
- **Arquivo:** `tests/test_quadtree_projector.py:60`
- **O que valida:** Regressão física: cavidade Re=100 continua com `max|div u| < 1e-11`.
- **Execução 1 (antes da implementação):** PASS (usa o caminho sem `decomposition` explícito).
- **Execução 2 (após a implementação):** PASS.
- **Valor numérico obtido:** `max|div u| ≈ 3.7e-15`.

## 6. Regressão DEPOIS da execução

| Suíte | Antes | Depois | Igual? |
|---|---|---|---|
| `pytest tests/ -q` | 52 passed | 56 passed (52 + 4 novos) | sim |

Nenhum teste existente mudou de status. O `test_von_karman_quick` (que usa células retangulares `hx=0.5, hy=0.25`) exigiu adaptação do `Decomposition` para suportar `hx ≠ hy` — feito como parte desta tarefa (ver §4 item 4).

## 7. Desvios do plano

1. **`Decomposition` suporta células retangulares:** o plano assumia `hx == hy` sempre (D1), mas o caminho uniforme retrocompatível do von Kármán (`Lx=8, Ly=4, Ndomx=2, Ndomy=2, nloc=8`) produz `hx=0.5, hy=0.25`. O `Decomposition.__init__` ganhou parâmetro `hy` (keyword-only) e `uniform()` aceita `nlocy` opcional. `segments_of` usa `hx`/`hy` por orientação. O caminho quadtree (`from_quadtree`) continua exigindo `hx == hy` (células quadradas são fundamentais para a construção diádica).

2. **`hx_f, hy_f` como aliases locais:** o construtor do projetor define `hx_f, hy_f = self.hx_f, self.hy_f` como aliases para as closures internas (`_K`, `_invK`, e o laço de subdomínios) que referenciam essas variáveis.

## 8. Referências consultadas
- **Código interno:**
  - `mprm/projector.py:53-285` — construtor inteiro; refatorado nas linhas 72-91, 126-142, 144-151, 155.
  - `mprm/decomposition.py:62-88,93-123` — `__init__`, `uniform`, `segments_of`: modificados para suportar células retangulares.
  - `mrcm/config.py:246-281` — `robin_L_xy_for` e `robin_L_xy`: usados para construir Robin por folha.
  - `mrcm/solver.py:104-141` — `_build_tasks`: o formato canônico das tarefas que `decomposition.task` reproduz.
- **Documento interno:**
  - `docs/superpowers/plans/2026-07-25-mprm-quadtree-von-karman.md` — Task 6 (§1809-2027).
  - Espec §2.6 (Achado 6) — divergência de β entre solver local e MsBF.
- **Artigo:** Guiraldello et al., MRCM, Eq. 65 (`β = α L / K`).

## 9. Estado para a próxima tarefa
- **O que existe agora que antes não existia:**
  - `MPrMProjector(cfg, dt, ..., decomposition=None)` — novo parâmetro opcional.
  - `MPrMProjector.decomp` — a decomposição em uso.
  - `MPrMProjector.robin_L` — `dict[isub -> (Lx, Ly)]` com L característico por folha.
  - `Decomposition.__init__(..., hy=None)` — suporte a `hx ≠ hy`.
  - `Decomposition.uniform(..., nlocy=None)` — suporte a `nlocx ≠ nlocy`.
- **O que a próxima tarefa deve consumir:**
  - `MPrMProjector.decomp` — a Task 7 usa `decomp.gamma_edges()` e `decomp.segments_of()` para refatorar os traços por segmento.
  - `Decomposition.hx`, `Decomposition.hy` — disponíveis para o downscaler (Task 8).
- **Armadilhas descobertas:**
  - As closures `_K`, `_invK` e o laço de subdomínios referenciam `hx_f`/`hy_f` como variáveis livres do escopo do construtor — não podem ser trocadas por `self.hx_f` dentro das closures sem capturar `self`.
  - `assert_solid_interior_to_subdomains` só funciona na grade cartesiana; sob quadtree a guarda `self.decomp.is_uniform` evita chamá-la.
  - `build_flux_dirichlet_bcs` espera `flag_side` como array bool de 4 elementos — a ordem é `{bottom:0, right:1, top:2, left:3}`.
- **O que ainda NÃO existe:**
  - Traços por segmento e média por aresta de Γ com nó pendurado (Task 7).
  - Downscaler com cache por geometria (Task 8).
  - Integração ponta a ponta von Kármán + quadtree (Task 9).
