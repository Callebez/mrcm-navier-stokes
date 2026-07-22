# Design — Otimização, Paralelização e Modularização do MPrM

**Data:** 2026-07-22
**Módulo alvo:** `~/mrcm/NavierStokes/` (MPrM — método de projeção multiescala)
**Diagnóstico base:** `docs/07-diagnostico-von-karman.md`, `docs/10-paralelizacao-mprm.md`
**Branch:** `mprm-von-karman` → novo branch `mprm-opt-parallel`

---

## 1. Objetivo

Transformar o protótipo funcional do MPrM em um **pacote modular, otimizado e
paralelo**, corrigindo déficits de acurácia física identificados no benchmark de
von Kármán e eliminando gargalos de desempenho que impedem malhas de alta
resolução ($D/h \ge 15$, $N \ge 240$).

Três eixos de trabalho:

1. **Correção física** — alinhar o setup e as métricas do benchmark de von
   Kármán ao artigo de Lages (§3.5, Tabela 3), recuperando $C_D$, $C_L$ e $St$
   dentro das faixas de referência.
2. **Paralelização** — eliminar os 4 gargalos sequenciais identificados no
   perfil de tempo, habilitando simulações com $N \ge 200$ em tempo viável.
3. **Modularização** — separar `mprm_core.py` (~1400 linhas) em submódulos com
   responsabilidades claras, cada um com seu suite de testes.

## 2. Contexto — déficits atuais

### 2.1 Déficits de acurácia (doc 07)

| # | Problema | Severidade | Localização |
|---|----------|------------|-------------|
| A | `Strouhal` é artefato do bin da FFT ($\Delta f = 0.033$) | 🔴 | `strouhal()` |
| B | `obstacle_forces` omite termo viscoso e transiente | 🔴 | `obstacle_forces()` |
| C | Setup diverge do artigo (perfil, paredes, geometria, $U_{ref}$) | 🔴 | `von_karman_problem`, defaults |
| D | Malha anisotrópica ($h_x/h_y=2$), obstáculo com 3 células | 🟠 | `run_von_karman` ($N_x=N_y$) |
| E | $C_L$ reportado como rms vs amplitude do artigo | 🟠 | `test_von_karman.py` |
| G | Pressão diagnóstica re-fatorizada a cada passo | 🟡 | `_diagnostic_pressure` |
| H | CFL usa só $h_x$, ignora $h_y$ | 🟡 | `run_flow` |

### 2.2 Gargalos de desempenho (doc 10)

| # | Gargalo | % tempo | Já paralelo? | Solução |
|---|---------|---------|---------------|---------|
| 1 | Pressão diagnóstica (Poisson global) | 25% | ❌ | Cache LU + subamostragem |
| 2 | Downscaler (patches Neumann) | 21% | ⚠️ MPI-rank, sequencial intra | ThreadPool intra-rank |
| 3 | Montagem global (rank 0) | 13% | ❌ | `executor_kind="mpi"` |
| 4 | Transporte MAC (full grid) | 7% | ❌ | Blocos com ghost cells |

## 3. Arquitetura alvo — pacote modular

```
mprm/
├── __init__.py              # exports públicos
├── problem.py               # SideBC, FlowProblem, construtores, máscaras
├── transport.py             # mac_transport, mac_wstar, mac_divergence
├── projector.py             # MPrMProjector, FineProjector (antigo)
├── downscaler.py            # MACDownscaler
├── flow.py                  # run_flow, run_cavity, run_von_karman, mprm_projection_step
├── forces.py                # obstacle_forces, strouhal, _diagnostic_pressure, ChannelFinePoisson
├── parallel.py              # utilitários de paralelismo (ThreadPool, MPI helpers)
├── viz.py                   # make_frame_saver, make_gif, plot_shedding
└── benchmarks/
    ├── hh.py                # hh_benchmark_mac, olshanskii_u_np, hh_p_np
    └── cavity.py            # (se extraído de flow.py)
```

### 3.1 Princípios de design

- **D1 — Separação NumPy/Firedrake**: submódulos puro-NumPy (`problem.py`,
  `transport.py`, `forces.py`) importáveis **sem** Firedrake. Submódulos com
  Firedrake (`projector.py`) mantêm imports locais.
- **D2 — Imutabilidade de APIs públicas**: `mac_wstar(u,v,hx,hy,dt,nu,lid)`,
  `run_cavity(...)`, `mprm_projection_step(...)`, `hh_benchmark_mac(...)`
  preservam assinatura e semântica.
- **D3 — Cada submódulo tem seu `tests/test_<modulo>.py`**: cobertura mínima
  de 80% por módulo, testes rápidos (< 5s cada para NumPy, < 30s para Firedrake).
- **D4 — Configuração centralizada**: `MRCMConfig` + `FlowProblem` são as
  únicas estruturas de configuração; sem dispersão de parâmetros soltos.
- **D5 — Paralelismo transparente**: `run_flow` aceita `comm` e `num_threads`;
  internamente decide entre MPI, ThreadPool ou sequencial conforme recursos.

## 4. Decisões de design

### 4.1 Correção das forças (P1 do diagnóstico)

**Opção escolhida: P1-B0 + P1-B1 progressivo.**

- **Fase 1 (B0):** Completar o balanço de volume de controle existente com os
  termos viscoso ($\nu(\nabla u + \nabla u^\top)\cdot n$) e transiente
  ($\frac{d}{dt}\int_{VC} u\,dV$). Isso resolve ~80% do déficit de $C_D$ com
  esforço baixo.
- **Fase 2 (B1):** Integral de superfície diretamente nas faces do obstáculo
  via `solid_faces`, conforme Eq. 43 do artigo. Implementação definitiva, requer
  gradientes one-sided na parede.

**Validação:** teste com escoamento de Stokes analítico (Couette/Poiseuille)
onde a força viscosa é conhecida exatamente.

### 4.2 Setup físico do benchmark (P2 + P3)

- **Defaults do artigo:** `inflow="parabolic"` com normalização por máximo
  (`inflow_norm="max"`), `walls="slip"` (free-slip).
- **Geometria:** `Ndomx ≠ Ndomy` para obter $h_x = h_y$ e centralizar o
  obstáculo longe de interfaces. Exemplo canônico: $L_x=16, L_y=8$, $Ndom_x=10,
  Ndom_y=5, nloc=30$ → $h=0.053$, $D/h=19$, obstáculo centrado em $(4,4)$.

### 4.3 Estimador de Strouhal (P4)

- **Zero-padding** ($pad=8$) na FFT para refinar bins de frequência.
- **Interpolação parabólica** de 3 pontos no pico para resolução sub-bin.
- **Tempo longo:** $T \ge 180 \, D/U_\infty$ (20+ períodos de shedding).
- **Saída:** reportar $St$ com 3 casas decimais + incerteza $\pm 1/(2T)$.

### 4.4 Estratégia de paralelização

| Nível | Mecanismo | O que paraleliza |
|-------|-----------|-----------------|
| **MPI** (já existe) | `comm` no `MPrMProjector` | Subdomínios MRCM, LocalSolver, reconstrução |
| **MPI** (a ativar) | `executor_kind="mpi"` | Montagem e solução do sistema global de interface |
| **Threads** (novo) | `ThreadPoolExecutor` no downscaler | Patches de Neumann independentes por subdomínio |
| **Vetorização** (novo) | Substituir laços Python em `obstacle_forces` por operações vetorizadas NumPy | Cálculo de forças |
| **Cache** (novo) | `DiagnosticPressureSolver` com LU cacheada | Pressão diagnóstica |

### 4.5 Cache de fatorações

Toda matriz que não muda durante a simulação deve ser fatorada **uma vez**:

| Matriz | Local | Estratégia |
|--------|-------|------------|
| Poisson do downscaler (sem sólido) | `MACDownscaler.__init__` | ✅ já cacheado |
| Poisson do downscaler (com sólido) | `MACDownscaler._masked_lu` | ✅ já cacheado |
| Poisson da pressão diagnóstica | `_diagnostic_pressure` | ❌ **refatora a cada passo** |
| Poisson do `ChannelFinePoisson` | `ChannelFinePoisson.__init__` | ✅ já cacheado |

## 5. Suite de testes

### 5.1 Testes existentes (a preservar)

```
tests/test_flow_problem.py        # 5 testes (NumPy)
tests/test_transport.py            # 4 testes (NumPy)
tests/test_downscaler_obstacle.py  # 2 testes (SciPy)
tests/test_projector.py            # 2 testes (Firedrake)
tests/test_forces_strouhal.py      # 3 testes (NumPy/SciPy)
tests/test_run_regression.py       # 2 testes (Firedrake)
tests/test_von_karman.py           # 2 modos (smoke + CLI)
```

### 5.2 Testes novos

| Arquivo | Testa | Tipo |
|---------|-------|------|
| `tests/test_forces_stokes.py` | `obstacle_forces` com solução analítica de Stokes | ✅ novo |
| `tests/test_strouhal_fft.py` | `strouhal` com sinal sintético, verifica resolução sub-bin | ✅ novo |
| `tests/test_diagnostic_pressure.py` | `DiagnosticPressureSolver` cacheia e reusa LU | ✅ novo |
| `tests/test_parallel_downscaler.py` | Downscaler com ThreadPool produz mesmo resultado que sequencial | ✅ novo |
| `tests/test_mprm_package.py` | `import mprm` não puxa Firedrake; submódulos corretos | ✅ novo |

### 5.3 Testes de regressão de benchmark

| Teste | Alvo | Tolerância |
|-------|------|------------|
| `test_von_karman_accuracy` | $C_D \in [1.28, 1.38]$, $St \in [0.118, 0.126]$ | ±5% |
| `test_cavity_regression` | `divmax < 1e-10`, `err_u < 0.3` (HH) | mantido |
| `test_mac_wstar_bit_identical` | `mac_wstar` ≡ estêncil legado | `atol=1e-14` |

## 6. Métricas de sucesso

| Métrica | Antes | Depois (alvo) |
|---------|-------|---------------|
| $C_D$ (von Kármán, Re=100) | 1.19 (−11%) | 1.28–1.38 (±3% do artigo) |
| $C_L$ (amplitude) | 0.13 (rms) | 0.22–0.32 (amplitude) |
| $St$ (von Kármán) | 0.133 (bin FFT) | 0.121 ± 0.005 |
| Tempo (64², T=50, $U_{H,3}/P_{H,3}$) | 118s | < 60s (1 core) / < 25s (4 cores) |
| Tempo (192², T=180, $U_{H,3}/P_{H,3}$) | inviável (>1h) | < 10min (4 cores) |
| Linhas de `mprm_core.py` | ~1400 | < 300 (só imports e re-exports) |
| Cobertura de testes | ~60% | > 80% |
| `import mprm` sem Firedrake | ✅ (já é) | ✅ (mantido) |

## 7. Riscos

| Risco | Mitigação |
|-------|-----------|
| `executor_kind="mpi"` quebrar com `DarcyProblem` generalizado | Testar isoladamente com `test_projector_mpi.py` antes de integrar |
| ThreadPool + Firedrake = conflito (PETSc não é thread-safe) | ThreadPool **apenas** no downscaler (scipy, sem Firedrake) |
| Refatoração quebrar APIs públicas | Testes de regressão como gate de commit |
| Malha $N \ge 200$ consumir memória excessiva | Usar `scipy.sparse` + `splu` (já é); monitorar com `memory_profiler` |
