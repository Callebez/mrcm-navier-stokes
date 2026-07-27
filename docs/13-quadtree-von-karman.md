# Quadtree-MPrM para o problema de von Kármán — Resultados

> **Branch:** `feature/quadtree-mprm` (13 tarefas, 74 testes)
> **Data:** 2026-07-27

## Arquitetura

O Método da Projeção Multiescala (MPrM) foi estendido com uma camada de
decomposição de domínio adaptativa (`Decomposition`) que isola a topologia do
restante do solver. A decomposição cartesiana uniforme tornou-se um caso
particular, permitindo testar a generalização por equivalência.

### Componentes novos

| Arquivo | Responsabilidade |
|---------|-----------------|
| `mprm/decomposition.py` | `Leaf`, `GammaEdge`, `Decomposition` (uniform + quadtree) |
| `mprm/triplets.py` | Separação matriz/RHS das triplas de A_Γ |
| `mprm/benchmarks/quadtree_vk.py` | Estudo comparativo quadtree vs cartesiano |

### Componentes modificados

| Arquivo | Mudança principal |
|---------|------------------|
| `mprm/projector.py` | Consome `Decomposition`, A_Γ offline, Robin por folha, traços por segmento |
| `mprm/downscaler.py` | Cache LU por geometria, defeito retornado, folha sólida |
| `mprm/flow.py` | `mprm_projection_step` com `leaf_boundary_fluxes`, `run_flow` com `decomposition` |

## Decisões de design

1. **h uniforme global:** A malha fina MAC é uniforme. Cada folha recebe
   `nloc_i = H_i/h` células. Simplifica a média da Eq. 21 (face a face).

2. **Balanceamento 2:1:** `mrcm.quadtree.QuadTree.balance_2to1()`, já
   validado por `Skeleton.from_quadtree`.

3. **Quadtree diádico:** Toda fronteira de folha coincide com fronteira de
   célula MAC. `nloc` par em todos os níveis.

4. **Duas famílias de alinhamento corpo/árvore:**
   - (a): corpo interior a uma folha travada (`N_by` ímpar, `H0 > D`)
   - (b): corpo como bloco 2×2 exato (`N_by` par, refino até a superfície)

## Resultados numéricos

### Smoke test — família (a), Ly=6D, max_level=1

| Métrica | Valor |
|---------|-------|
| N_sub | 28 folhas (base 12 + refinadas) |
| n_gamma_edges | 35 |
| H_min / H_max | 1.0 / 2.0 |
| max\|div u\| no fluido | < 1e-12 |
| Tempo offline | ~3s |
| Tempo online por passo | ~20ms |

### Família (b) — corpo como união de folhas

| Métrica | Valor |
|---------|-------|
| N_sub | 40 folhas (base 8 + refinadas) |
| Corpo | 4 folhas de nível 3 |
| n_gamma_edges | 64 |
| H_min / H_max | 0.25 / 2.0 |

## Correções de bugs encontrados

1. **Robin global vs local (Achado 6):** O β do solver local usava L global;
   corrigido para `robin_L_xy_for(Hx, Hy, nlocx, nlocy)` por folha.

2. **Folha travada subdividida pelo balanço:** Cap de nível gradado impede
   que `balance_2to1` subdivida a folha que contém o corpo na família (a).

3. **hash() salgado em MPI:** `assert_consistent_across_ranks` usa
   `zlib.crc32` (mesma abordagem de `mrcm/solver.py:339`).

4. **Folha 100% sólida divide por zero:** Guarda explícita no downscaler
   para `nfluid = 0`.

## Testes

- **74 testes** passando (19 baseline + 55 novos em 13 tarefas)
- Cobertura: decomposição uniforme ≡ quadtree uniforme, nó pendurado,
  folha sólida, cavidade regressão, von Kármán smoke test,
  benchmark schema, MPI equivalence (slow, deselectado na suíte rápida)

## Referências

- Lages, C. — *The Multiscale Projection Method for Incompressible
  Navier-Stokes* (tese de doutorado)
- Guiraldello et al. — *The Multiscale Robin Coupled Method* (MRCM)
- de Berg et al. — *Computational Geometry*, §14.3 (balanceamento 2:1)
- Breuer et al. (2000) — valores de referência do von Kármán
