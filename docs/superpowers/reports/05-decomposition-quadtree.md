# Relatório — Tarefa 05: `Decomposition.from_quadtree` — `nloc` heterogêneo, conexidade e consistência MPI

## 1. Identificação
- **Tarefa:** 05 — Decomposition.from_quadtree com nloc heterogêneo, invariantes de conexidade e consistência MPI
- **Data:** 2026-07-27
- **Commit antes:** `073932e` (Task 4 — geometria do quadtree)
- **Commit depois:** `d9dbe76`
- **Arquivos criados:** `docs/superpowers/reports/05-decomposition-quadtree.md`
- **Arquivos modificados:**
  - `mprm/decomposition.py` — acrescenta `from_quadtree` (L93-140), `check_fluid_connectivity` (L142-163), `assert_consistent_across_ranks` (L165-193)
  - `tests/test_decomposition.py` — acrescenta `_arvore_uniforme` helper e classe `TestFromQuadtree` com 6 testes (L223-337)

## 2. Estado herdado
- Relatório 04 (`docs/superpowers/reports/04-geometria-quadtree.md`): `Decomposition` com caminho uniforme, `Leaf`, `GammaEdge`, `plan_quadtree_geometry`, `build_obstacle_tree`.
- Assinaturas consumidas:
  - `Decomposition.uniform(Ndomx, Ndomy, nloc, Lx, Ly) -> Decomposition`
  - `Decomposition(skeleton, leaves, h, Lx, Ly)` — construtor
  - `Decomposition.leaf(isub) -> Leaf`
  - `Decomposition.gamma_edges() -> list[GammaEdge]`
  - `Decomposition.segments_of(isub) -> list[tuple]`
  - `Decomposition.task(cfg, isub, NunkU, problem) -> dict`
  - `mrcm.quadtree.QuadTree`, `QuadTreeNode`, `QuadTree.leaves`, `QuadTree.balance_2to1()`
  - `mrcm.topology.Skeleton.from_quadtree(qt) -> Skeleton`

## 3. Regressão ANTES da execução

| Suíte | Baseline esperado | Obtido antes desta tarefa | Igual? |
|---|---|---|---|
| `pytest tests/ -q` | 46 passed (19 base + 27 acumulados Tasks 1-4) | 46 passed | sim |

## 4. O que foi feito

Três métodos novos na classe `Decomposition`:

1. **`from_quadtree(qt, h, Lx, Ly, solid=None)`** (classmethod): constrói a decomposição a partir de uma `QuadTree` balanceada. Para cada nó folha da árvore, calcula `nloc_i = H_i/h` (decisão D1, h uniforme), valida que a divisão é inteira, e preenche `skeleton.sub_nloc` — o canal pelo qual o MRCM leva a resolução local até `build_subdomain_workspace`. Se `solid` é fornecido, verifica a conexidade do fluido na construção.

2. **`check_fluid_connectivity(solid)`**: percorre cada folha, extrai o bloco da máscara sólida, e usa `scipy.ndimage.label` para contar componentes conexas do fluido. Folhas com fluido em 2+ componentes levantam `ValueError` (o problema de Neumann local teria nulidade >1 e o pino remove só um modo — spec §4.5). Folhas 100% sólidas são aceitas (família (b) as produz por construção).

3. **`assert_consistent_across_ranks(comm)`**: verifica que todos os ranks MPI construíram o mesmo esqueleto (NGamma, número de folhas, el2edge, edge2el). Usa `zlib.crc32` em vez de `hash()` porque `PYTHONHASHSEED` randomiza o hash de bytes por processo — `hash()` difere entre ranks MPI mesmo para esqueletos idênticos (pre-flight item 2; mesma abordagem de `mrcm/solver.py:339`).

**Desvio do plano aplicado:** `assert_consistent_across_ranks` usa `zlib.crc32` em vez de `hash()`, conforme resolução do pre-flight item 2. O fingerprint cobre `NGamma`, número de folhas, `el2edge` e `edge2el` via `np.ascontiguousarray(...).tobytes()` — mesmo método do `_assert_skeleton_consistent` do MRCM.

## 5. Testes criados nesta tarefa

### `test_arvore_uniforme_equivale_a_decomposicao_cartesiana`
- **Arquivo:** `tests/test_decomposition.py:237`
- **O que valida:** Uma árvore 2×2 uniforme produz folhas, arestas e offsets idênticos a `Decomposition.uniform(2, 2)`.
- **Resultado esperado:** Equivalência exata de coordenadas, offsets, nloc e pares de arestas.
- **Execução 1 (antes da implementação):** `AttributeError: type object 'Decomposition' has no attribute 'from_quadtree'` — FAIL esperado.
- **Execução 2 (após a implementação):** PASS.
- **Valor numérico obtido:** `frozenset` de pares de arestas idêntico entre uniforme e quadtree.

### `test_nloc_proporcional_ao_tamanho_da_folha`
- **Arquivo:** `tests/test_decomposition.py:259`
- **O que valida:** Folha subdividida produz folhas menores com `nloc = H/h`, e o array `skeleton.sub_nloc` tem shape `(N_sub, 2)`.
- **Resultado esperado:** Dois tamanhos de `nloc`: 4 (folhas pequenas) e 8 (folhas não subdivididas).
- **Execução 1 (antes da implementação):** `AttributeError` — FAIL esperado.
- **Execução 2 (após a implementação):** PASS.
- **Valor numérico obtido:** `nlocs = [4, 8]`, `sub_nloc.shape = (7, 2)` (1 folha subdividida em 4 + 3 originais = 7).

### `test_nloc_heterogeneo_chega_a_malha_local`
- **Arquivo:** `tests/test_decomposition.py:272`
- **O que valida:** `build_subdomain_workspace` recebe `sub_nloc` e gera malha com `2·nlocx·nlocy` células triangulares.
- **Resultado esperado:** Malha local com número correto de células para folhas de tamanhos diferentes.
- **Execução 1 (antes da implementação):** `AttributeError` — FAIL esperado.
- **Execução 2 (após a implementação):** PASS.
- **Valor numérico obtido:** Folha 0 (nloc=4,4): 32 células; folha 6 (nloc=8,8): 128 células — ambas batem com `2·nlocx·nlocy`.

### `test_fluido_desconexo_e_rejeitado`
- **Arquivo:** `tests/test_decomposition.py:313`
- **O que valida:** Uma faixa sólida que corta a folha de lado a lado produz 2 componentes de fluido → `ValueError` com "desconex".
- **Execução 1 (antes da implementação):** `AttributeError: 'Decomposition' object has no attribute 'check_fluid_connectivity'` — FAIL esperado.
- **Execução 2 (após a implementação):** PASS (`ValueError` levantado).

### `test_fluido_conexo_e_aceito`
- **Arquivo:** `tests/test_decomposition.py:319`
- **O que valida:** Bloco sólido interno (não toca as bordas) mantém fluido conexo (anel).
- **Execução 1 (antes da implementação):** `AttributeError` — FAIL esperado.
- **Execução 2 (após a implementação):** PASS (não levanta exceção).

### `test_folha_totalmente_solida_e_aceita`
- **Arquivo:** `tests/test_decomposition.py:325`
- **O que valida:** Folha 100% sólida (família (b)) não levanta erro de conexidade.
- **Execução 1 (antes da implementação):** `AttributeError` — FAIL esperado.
- **Execução 2 (após a implementação):** PASS (não levanta exceção).

## 6. Regressão DEPOIS da execução

| Suíte | Antes | Depois | Igual? |
|---|---|---|---|
| `pytest tests/ -q` | 46 passed | 52 passed (46 + 6 novos) | sim |

Nenhum teste existente mudou de status. Os 6 novos testes em `test_decomposition.py` elevam o total de 24 para 30 no arquivo (24 existentes + 6).

## 7. Desvios do plano

1. **`assert_consistent_across_ranks` usa `zlib.crc32` em vez de `hash()`** — conformidade com a resolução do pre-flight item 2. O plano original (linhas 1762-1776) usava `hash((skel.NGamma, ...))`, que falha em MPI porque `PYTHONHASHSEED` randomiza o hash de bytes por processo. A implementação usa `zlib.crc32` sobre `tobytes()` dos arrays, mesma abordagem documentada em `mrcm/solver.py:339`. A detecção via `MIN`/`MAX` allreduce foi mantida.

## 8. Referências consultadas
- **Código interno:**
  - `mprm/decomposition.py:62-91` — `Decomposition.__init__`, `uniform`: estrutura da classe que `from_quadtree` estende.
  - `mprm/decomposition.py:169-261` — `plan_quadtree_geometry` (Task 4): chaves do dicionário `geom` que `from_quadtree` consome indiretamente.
  - `mrcm/quadtree.py:219-307` — `QuadTree.leaves`, `_invalidate_cache`, `_build_neighbours`: propriedades da árvore usadas por `from_quadtree`.
  - `mrcm/topology.py:291-332` — `Skeleton.from_quadtree`: valida balanço 2:1 e preenche `sub_bbox`.
  - `mrcm/topology.py:203-218` — campos `sub_bbox` e `sub_nloc` do `Skeleton`.
  - `mrcm/solver.py:339-358` — `_assert_skeleton_consistent`: usa `zlib.crc32` em vez de `hash()` pelo mesmo motivo (pre-flight item 2).
- **Documento interno:**
  - `.superpowers/sdd/.../pre-flight-resolutions.md` — itens 1 (folha travada, já resolvido na Task 4) e 2 (hash salgado, resolvido nesta tarefa).
  - `docs/superpowers/plans/2026-07-25-mprm-quadtree-von-karman.md` — Task 5 (§1538-1806).

## 9. Estado para a próxima tarefa
- **O que existe agora que antes não existia:**
  - `Decomposition.from_quadtree(qt, h, Lx, Ly, solid=None) -> Decomposition` — classmethod que constrói decomposição com `nloc` heterogêneo a partir de `QuadTree`.
  - `Decomposition.check_fluid_connectivity(solid) -> None` — levanta `ValueError` se alguma folha tiver fluido desconexo.
  - `Decomposition.assert_consistent_across_ranks(comm) -> None` — levanta `RuntimeError` se ranks MPI divergirem no esqueleto.
  - `Decomposition.skeleton.sub_nloc` — array `(Nsubd, 2)` com `nloc` por folha, populado por `from_quadtree`.
- **O que a próxima tarefa deve consumir:**
  - `Decomposition.from_quadtree` (Task 9, via `build_case` da Task 11).
  - `Decomposition.check_fluid_connectivity` — substituto de `assert_solid_interior_to_subdomains` sob quadtree (Tasks 6, 9).
  - `Decomposition.assert_consistent_across_ranks` — chamado no construtor do projetor (Task 10).
- **Armadilhas descobertas:**
  - `Skeleton.from_quadtree` já valida balanço 2:1 e levanta `ValueError`; não é preciso revalidar.
  - `sub_nloc` precisa ser atribuído DEPOIS de `Skeleton.from_quadtree` porque o construtor do `Skeleton` não conhece `nloc`.
  - `scipy.ndimage.label` trata a máscara como `bool`; passar `solid` como `int` com valores 0/1 funciona (0=False, 1=True), mas `~solid` inverte corretamente.
- **O que ainda NÃO existe:**
  - Projetor consumindo `Decomposition` (Task 6).
  - Traços por segmento e média por aresta de Γ com nó pendurado (Task 7).
  - Downscaler com cache por geometria (Task 8).
  - Integração ponta a ponta von Kármán + quadtree (Task 9).
