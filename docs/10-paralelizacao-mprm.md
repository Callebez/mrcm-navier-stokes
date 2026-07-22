# Oportunidades de paralelização no MPrM

> **Branch:** `mprm-von-karman` | `mprm_core.py`
> **Data:** 2026-07-22

---

## 1. Estado atual da paralelização

### 1.1 O que já é paralelo (MPI via `comm`)

| Componente | Mecanismo | Eficiência |
|------------|-----------|------------|
| **Subdomínios MRCM** | Round-robin: `isub % size == rank` | ✅ Boa (subdomínios independentes) |
| **LocalSolver** (parte particular) | Cada rank resolve seus subdomínios | ✅ Boa |
| **Reconstrução** (`reconstruct_subdomain`) | Paralela por subdomínio | ✅ Boa |
| **Traços de fluxo** | `allgather` dos arrays de interface | ✅ Boa (arrays pequenos) |

### 1.2 Gargalos sequenciais atuais

| Gargalo | Localização | Impacto |
|---------|-------------|---------|
| 🔴 **Montagem global** | `project()` — `gather` → rank 0 → `bcast` | **Alto** — rank 0 resolve sistema global sozinho |
| 🔴 **Downscaler** | `mprm_projection_step` — laço `for isub` sequencial | **Alto** — patches independentes, mas sequenciais |
| 🟡 **Transporte MAC** | `mac_transport` — NumPy full-grid | **Médio** — leve, mas cresce com $N^2$ |
| 🟡 **Pressão diagnóstica** | `_diagnostic_pressure` — Poisson global scipy | **Médio** — chamado a cada passo com `obstacle_metrics` |
| 🟢 **Forças** | `obstacle_forces` — NumPy no volume de controle | **Baixo** — $O(1)$ células |

---

## 2. Análise detalhada de cada gargalo

### 2.1 🔴 Montagem global do sistema de interface (maior impacto)

**Localização:** `MPrMProjector.project()`, linhas 635–648

```python
# 2. Triplets locais -> sistema global de interface no rank 0.
my_trip = {isub: compute_subdomain_triplets(...) for isub in self.my_subs}
gathered = comm.gather(my_trip, root=0)
if self.rank == 0:
    trip = {}
    for d in gathered:
        trip.update(d)
    _, _, X = self.assembler.assemble_from_triplets(trip)  # ← resolve sistema global
else:
    X = None
X = comm.bcast(X, root=0)
```

**Problema:** Todo o sistema global de interface (matriz densa $N_\Gamma \times N_\Gamma$) é montado e resolvido **apenas no rank 0**. Os demais ranks ficam ociosos.

**Tamanho do sistema:** $N_\Gamma = N_{\text{dom}} \times (N_{\text{dom}}-1) \times 2$ interfaces, cada uma com $(k_U+1) + (k_P+1)$ DOFs.

| $N_{\text{dom}}$ | Interfaces | DOFs/interface ($U_{H,1},P_{H,3}$) | Sistema global |
|-------------------|------------|--------------------------------------|----------------|
| 4 | 24 | $2 + 4 = 6$ | $144 \times 144$ |
| 8 | 112 | 6 | $672 \times 672$ |
| 16 | 480 | 6 | $2880 \times 2880$ |

Para $N_{\text{dom}} \ge 8$, o sistema já tem milhares de DOFs e a solução sequencial domina o tempo total.

**Oportunidade:** Usar um solver **paralelo** para o sistema de interface (PETSc, MUMPS, SuperLU_DIST). O `mrcm` já tem suporte a `executor_kind="mpi"` no `MRCMConfig` — mas o `run_flow` atual hardcoded `executor_kind="sequential"`:

```python
cfg = MRCMConfig(..., executor_kind="sequential")  # ← mprm_core.py:1004
```

**Solução proposta:**

```python
# Em run_flow, permitir executor paralelo:
cfg = MRCMConfig(..., executor_kind="mpi" if comm.size > 1 else "sequential")
```

⚠️ Isso requer verificar se o `GlobalAssembler` e o pipeline MPI do `mrcm` funcionam com o `DarcyProblem` generalizado (com `permeability` heterogênea e `boundary` não trivial).

---

### 2.2 🔴 Downscaler sequencial (2º maior impacto)

**Localização:** `mprm_projection_step()`, laço sobre `projector.my_subs`

```python
my_patch = {}
for isub in projector.my_subs:          # ← sequencial!
    ...
    vt1, vt2 = downscaler.solve_patch(
        divw[ox:ox + n, oy:oy + m], fW, fE, fS, fN, solid_loc=solid_loc)
    my_patch[isub] = (vt1, vt2)
```

**Problema:** Cada subdomínio resolve um problema de Neumann independente (matriz $n_{\text{loc}}^2 \times n_{\text{loc}}^2$). Com $N_{\text{dom}}=4$ e $n_{\text{loc}}=16$, são 16 patches de $256 \times 256$ cada. **Todos são independentes**, mas resolvidos sequencialmente.

**Oportunidade:** Paralelizar com `concurrent.futures` (ThreadPool ou ProcessPool) ou simplesmente distribuir entre ranks MPI (cada rank já tem `my_subs`).

**Solução proposta (MPI, mais natural):**

O downscaler já está dentro de `mprm_projection_step`, que é chamado coletivamente. Cada rank já possui seu subconjunto `projector.my_subs`. O laço já é paralelo **entre ranks** (cada rank processa seus subdomínios), mas **dentro de cada rank** é sequencial.

Para `N_{\text{dom}}=4` com 4 ranks MPI: cada rank tem 4 subdomínios. O laço interno processa 4 patches sequencialmente. Com `ThreadPool`, poderíamos processar os 4 patches em paralelo dentro de cada rank:

```python
from concurrent.futures import ThreadPoolExecutor

def _solve_one_patch(args):
    isub, divw_patch, fW, fE, fS, fN, solid_loc, downscaler = args
    return isub, downscaler.solve_patch(divw_patch, fW, fE, fS, fN, solid_loc=solid_loc)

with ThreadPoolExecutor(max_workers=4) as ex:
    futures = []
    for isub in projector.my_subs:
        ...
        futures.append(ex.submit(_solve_one_patch, (isub, ...)))
    for f in futures:
        isub, (vt1, vt2) = f.result()
        my_patch[isub] = (vt1, vt2)
```

**Ganho estimado:** ~2–3× dentro de cada rank (limitado pelo GIL do Python para operações NumPy/scipy, mas `spla.splu().solve()` libera o GIL).

---

### 2.3 🟡 Transporte MAC

**Localização:** `mac_transport()`

**Problema:** Opera na malha global $(N_x+1) \times N_y + N_x \times (N_y+1)$ com estêncil de 5 pontos. Totalmente sequencial.

**Oportunidade:** Para malhas grandes ($N > 200$), o transporte pode ser decomposto em blocos com ghost cells. Como o estêncil é local (vizinhança de 1 célula), a paralelização é trivial com sobreposição de 1 célula.

**Ganho estimado:** Moderado (o transporte é $\sim 10\%$ do tempo total para $N \le 100$).

---

### 2.4 🟡 Pressão diagnóstica

**Localização:** `_diagnostic_pressure()`

**Problema:** Reconstrói e resolve um sistema de Poisson global (matriz esparsa $N_x N_y \times N_x N_y$) a cada passo de tempo quando `obstacle_metrics=True`.

**Oportunidades:**

1. **Reutilizar fatoração:** A matriz do Poisson físico **não muda** durante a simulação (geometria e BCs fixas). Atualmente, `_diagnostic_pressure` reconstrói e refatora a matriz a **cada chamada**.

```python
# Atual: reconstrói e refatora a cada passo
A = sp.csr_matrix(...)
lu = spla.splu(A.tocsc())    # ← refatoração cara a cada passo!
p = lu.solve(b)
```

**Solução:** Mover a construção e fatoração para um cache (similar ao `MACDownscaler._masked_lu`):

```python
# Proposto: cache da fatoração
class DiagnosticPressureSolver:
    def __init__(self, problem):
        self.lu = ...  # constrói e fatora uma vez
    def solve(self, u, v):
        source = compute_source(u, v)  # só o RHS muda
        return self.lu.solve(source)
```

2. **Subamostragem:** Para métricas como $C_D$ e Strouhal, não é necessário calcular a pressão a **cada** passo. Amostrar a cada 5–10 passos reduziria o custo em 5–10×.

3. **Reutilizar a pressão da projeção:** Em vez de resolver um Poisson separado, usar a pressão do passo de projeção (disponível em `projector.last_pT`). Isso eliminaria completamente o custo da pressão diagnóstica.

**Solução proposta (item 3, mais impactante):**

```python
# Em run_flow, usar pressão do projetor em vez de _diagnostic_pressure:
if method == "mprm":
    # Passa pressão do projetor para obstacle_forces
    p_sub = projector.last_pT  # pressão DG0 por subdomínio
    # Reconstrói pressão global (allgather dos patches)
    ...
```

---

## 3. Tabela de impacto

| Oportunidade | Dificuldade | Ganho estimado | Código afetado |
|-------------|-------------|----------------|----------------|
| **Solver global paralelo** | Média (mudar `executor_kind`) | 2–4× com 4 ranks | `run_flow`: `MRCMConfig` |
| **Downscaler com ThreadPool** | Baixa | 1.5–2× por rank | `mprm_projection_step` |
| **Cache da fatoração da pressão** | Baixa | 5–10× no custo de diagnóstico | `_diagnostic_pressure` |
| **Reutilizar pressão da projeção** | Média | Elimina Poisson extra | `run_flow` + `obstacle_forces` |
| **Subamostragem de métricas** | Trivial | 5–10× no custo de diagnóstico | `run_flow`: `obstacle_metrics_every` |
| **Transporte paralelo** | Média | 1.2–1.5× | `mac_transport` |

---

## 4. Quick wins (implementação imediata)

### 4.1 Cache da fatoração da pressão diagnóstica

**Custo:** ~20 linhas. **Ganho:** 5–10× no tempo de diagnóstico.

A fatoração LU do sistema de Poisson domina o tempo de `_diagnostic_pressure`. Como a matriz não muda (geometria e BCs fixas), a fatoração deveria ser feita **uma vez** e reutilizada.

### 4.2 Subamostragem de métricas (`obstacle_metrics_every`)

**Custo:** ~5 linhas. **Ganho:** $N_{\text{every}} \times$ no tempo de diagnóstico.

Adicionar um parâmetro `obstacle_metrics_every=1` ao `run_flow`:

```python
if obstacle_metrics and problem.solid is not None and step % obstacle_metrics_every == 0:
    p = _diagnostic_pressure(u, v, problem, dt)
    cd, cl = obstacle_forces(u, v, p, problem, U_ref, D)
    cl_hist.append((dt * (step + 1), cd, cl))
```

### 4.3 Ativar `executor_kind="mpi"` quando `comm.size > 1`

**Custo:** ~3 linhas. **Ganho:** 2–4× com 4 ranks (se o pipeline MPI do mrcm for compatível).

```python
cfg = MRCMConfig(...,
    executor_kind="mpi" if comm.size > 1 else "sequential")
```

---

## 5. Perfil de tempo típico (Ndom=4, nloc=16, T=50, degreeU=3, degreeP=3)

Medido em 1 core (sequencial), simulação de 118s com 500 passos:

| Componente | Tempo estimado | % do total |
|------------|---------------|------------|
| Transporte MAC | ~8s | 7% |
| Projeção MRCM (local solves) | ~35s | 30% |
| Projeção MRCM (global assembly + solve) | ~15s | 13% |
| Downscaling (patches Neumann) | ~25s | 21% |
| Pressão diagnóstica (Poisson global) | ~30s | 25% |
| Forças + I/O + outros | ~5s | 4% |

> Com as otimizações propostas (cache da pressão + downscaler ThreadPool + subamostragem), o tempo total poderia cair para **~60s** (ganho de ~2×) sem alterar a física.

---

## 6. Caminho para paralelização híbrida (MPI + threads)

```
┌─────────────────────────────────────────────────────┐
│                  run_flow (time loop)                │
├─────────────────────────────────────────────────────┤
│  1. mac_transport (full grid)                       │
│     └→ oportunidade: decomposição por blocos MPI    │
├─────────────────────────────────────────────────────┤
│  2. MPrMProjector.project(divw)                     │
│     ├→ local solves: já paralelo (MPI por subdom)   │
│     ├→ global assembly: rank 0 ← GARGALO            │
│     │   └→ oportunidade: solver paralelo (PETSc)    │
│     └→ reconstruction: já paralelo (MPI)            │
├─────────────────────────────────────────────────────┤
│  3. Downscaler (solve_patch por subdom)             │
│     └→ oportunidade: ThreadPool intra-rank          │
├─────────────────────────────────────────────────────┤
│  4. Métricas (obstacle_forces, strouhal)            │
│     └→ oportunidade: cache da fatoração + subamostra│
└─────────────────────────────────────────────────────┘
```
