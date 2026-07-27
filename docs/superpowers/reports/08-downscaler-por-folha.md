# Relatório — Tarefa 08: Downscaler por folha — cache por geometria, defeito retornado, folha sólida

## 1. Identificação
- **Tarefa:** 08 — Downscaler por folha com cache por geometria
- **Data:** 2026-07-27
- **Commit antes:** `458e34e` (Task 7 — traços por segmento)
- **Commit depois:** `01de9d4`
- **Arquivos criados:** `tests/test_quadtree_downscaler.py`
- **Arquivos modificados:**
  - `mprm/downscaler.py` — reescrita completa (construtor com `h`/`hy`, cache LU por geometria, `solve_patch` com 3 retornos)
  - `mprm/flow.py:45-50,67-70,79-83,183,189-190` — `mprm_projection_step` com defeitos, `run_flow` com hx/hy
  - `tests/test_downscaler_obstacle.py:6,22,39` — import corrigido, desempacotamento de 3 valores

## 2. Estado herdado
- Relatório 07 (`docs/superpowers/reports/07-tracos-por-segmento.md`): `leaf_boundary_fluxes`, traços por segmento, projetor com `gamma_edges`.
- Assinaturas consumidas:
  - `MACDownscaler(cfg, dt)` — construtor antigo (sem `h`)
  - `MACDownscaler.solve_patch(divw_loc, fW, fE, fS, fN, solid_loc=None) -> (vt1, vt2)` — 2 retornos

## 3. Regressão ANTES da execução

| Suíte | Baseline esperado | Obtido antes desta tarefa | Igual? |
|---|---|---|---|
| `pytest tests/ -q` | 59 passed | 59 passed | sim |

## 4. O que foi feito

Reescrita completa do `MACDownscaler` para suportar folhas de tamanhos diferentes sob decomposição quadtree.

1. **Construtor com `h` e `hy`:** aceita `h` opcional (e `hy` keyword-only) para override do `cfg`. Quando `h` é dado, `hx = h` e `hy = hy or h`. Sem override, usa `cfg.Hx/cfg.nlocx` e `cfg.Hy/cfg.nlocy` (retrocompatível).

2. **Cache LU por geometria:** a fatoração única (`self.lu`) é substituída por `self._lu_cache` — dict indexado por `(n, m, solid_key)`. O método `_build_lu(n, m, solid)` monta e fatora a matriz FV de 5 pontos para um patch de tamanho `(n, m)`. O pino vai na primeira célula de fluido. Com poucos níveis de refinamento, o número de fatorações distintas é pequeno.

3. **`solve_patch` retorna 3 valores:** `(vt1, vt2, defect_rel)`. O tamanho do patch vem de `divw_loc.shape` — não mais do `cfg`. O defeito relativo é `abs(defect) / (hx*hy*nfluid)`.

4. **Folha 100% sólida:** guarda explícita no início de `solve_patch`: se `solid.all()`, devolve `(zeros, zeros, 0.0)` sem tentar resolver — evita divisão por zero em `nfluid = 0` (spec §4.5, item 1).

5. **`reset_defect()`:** zera `last_defect` para o próximo passo de tempo (antes o acumulador nunca era reiniciado).

6. **`mprm_projection_step` coleta defeitos:** cada chamada a `solve_patch` coleta o defeito por folha em `my_defects`, e o `allgather` dos defeitos é armazenado em `projector.last_defects` para diagnóstico.

## 5. Testes criados nesta tarefa

### `test_patches_de_tamanhos_diferentes`
- **Arquivo:** `tests/test_quadtree_downscaler.py:25`
- **O que valida:** `solve_patch` funciona com patches 8×8, 4×4, 16×16, shapes corretos, defeito < 1e-12.
- **Execução 1 (antes da implementação):** `TypeError: MACDownscaler.__init__() got an unexpected keyword argument 'h'` — FAIL esperado.
- **Execução 2 (após a implementação):** PASS.
- **Valor numérico obtido:** defeitos de 1.2e-16 a 5.6e-17 para patches compatíveis.

### `test_cache_reutiliza_fatoracao_por_geometria`
- **Arquivo:** `tests/test_quadtree_downscaler.py:36`
- **O que valida:** Mesma geometria não refatora; geometria nova adiciona entrada ao cache.
- **Execução 1 (antes da implementação):** `TypeError` — FAIL esperado.
- **Execução 2 (após a implementação):** PASS.
- **Valor numérico obtido:** `len(_lu_cache)` = 1 após primeira chamada, permanece 1 na segunda, sobe para 2 na terceira (geometria diferente).

### `test_folha_totalmente_solida_devolve_zero_sem_nan`
- **Arquivo:** `tests/test_quadtree_downscaler.py:47`
- **O que valida:** Folha 100% sólida devolve `vt1=0, vt2=0, defect=0` sem NaN.
- **Execução 1 (antes da implementação):** `TypeError` — FAIL esperado.
- **Execução 2 (após a implementação):** PASS.
- **Valor numérico obtido:** `defect = 0.0`, `isfinite(defect) = True`.

### `test_solucao_de_patch_uniforme_inalterada`
- **Arquivo:** `tests/test_quadtree_downscaler.py:57`
- **O que valida:** Condições de contorno de fluxo impostas exatamente nas faces externas (atol=1e-14).
- **Execução 1 (antes da implementação):** `TypeError` — FAIL esperado.
- **Execução 2 (após a implementação):** PASS.
- **Valor numérico obtido:** `max|vt1[0,:] - fW| = 0.0`, `max|vt1[-1,:] - fE| = 0.0`.

### `test_defeito_e_por_chamada_nao_acumulado`
- **Arquivo:** `tests/test_quadtree_downscaler.py:71`
- **O que valida:** O defeito de uma chamada com fonte incompatível não contamina a chamada seguinte com fonte compatível.
- **Execução 1 (antes da implementação):** `TypeError` — FAIL esperado.
- **Execução 2 (após a implementação):** PASS.
- **Valor numérico obtido:** `defect_ruim ≈ 0.5` (fonte incompatível), `defect_bom ≈ 2e-17` (não contaminado).

## 6. Regressão DEPOIS da execução

| Suíte | Antes | Depois | Igual? |
|---|---|---|---|
| `pytest tests/ -q` | 59 passed | 64 passed (59 + 5 novos) | sim |

Nenhum teste existente mudou de status. O `test_von_karman_quick` (células retangulares `hx=0.5, hy=0.25`) exigiu a correção do `hy` no construtor do downscaler — sem isso, o downscaler usava `hy=hx=0.5` e a projeção ficava errada nas células retangulares.

**Pre-flight item 5 aplicado:** `tests/test_downscaler_obstacle.py` atualizado: import corrigido de `mprm_core` para `mprm.downscaler`, e desempacotamento de 3 valores (`vt1, vt2, _ = ...`).

## 7. Desvios do plano

1. **Construtor com `hy` keyword-only:** o plano especificava apenas `h`, mas o von Kármán com `Lx ≠ Ly` produz células retangulares (`hx ≠ hy`). O `hy` separado evita que o downscaler use o `h` errado na direção y. Detectado pela falha do `test_von_karman_quick` (max|div u| = 1.28 com `hy` errado, < 1e-7 com `hy` correto).

2. **Propriedade `h` mantida:** como alias para `hx`, para retrocompatibilidade com código que referencia `downscaler.h`.

## 8. Referências consultadas
- **Código interno:**
  - `mprm/downscaler.py:25-158` — implementação antiga, substituída.
  - `mprm/flow.py:44-84,159-195` — `mprm_projection_step` e `run_flow`.
  - `tests/test_downscaler_obstacle.py` — corrigido conforme pre-flight item 5.
- **Documento interno:**
  - `docs/superpowers/plans/2026-07-25-mprm-quadtree-von-karman.md` — Task 8 (§2306-2619).
  - `.superpowers/sdd/.../pre-flight-resolutions.md` — item 5 (assinatura de `solve_patch`).

## 9. Estado para a próxima tarefa
- **O que existe agora que antes não existia:**
  - `MACDownscaler(cfg, dt, h=None, *, hy=None)` — override de `hx`/`hy`.
  - `MACDownscaler._lu_cache` — cache de fatorações por `(n, m, solid_key)`.
  - `MACDownscaler.reset_defect()` — zera `last_defect`.
  - `MACDownscaler.solve_patch(...) -> (vt1, vt2, defect)` — 3 retornos.
  - `MPrMProjector.last_defects` — dict `{isub: defect}` por passo (populado por `mprm_projection_step`).
  - `mprm_projection_step` coleta e distribui defeitos por folha.
- **O que a próxima tarefa deve consumir:**
  - `MACDownscaler(cfg, dt, h=..., hy=...)` — Task 9 (integração von Kármán + quadtree).
  - `solve_patch` com 3 retornos — Tasks 9, 11 (estudo comparativo).
- **Armadilhas descobertas:**
  - O `h` do `Decomposition` é `hx` (atalho). Para células retangulares, `hy` é diferente e precisa ser passado explicitamente ao downscaler.
  - `test_downscaler_obstacle.py` importava de `mprm_core` (símbolo legado), não de `mprm.downscaler`.
- **O que ainda NÃO existe:**
  - Integração ponta a ponta von Kármán + quadtree (Task 9).
  - Equivalência sequencial ↔ MPI sob quadtree (Task 10).
