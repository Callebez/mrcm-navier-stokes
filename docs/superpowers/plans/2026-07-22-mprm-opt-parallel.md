# MPrM — Otimização, Paralelização e Modularização — Plano de Implementação

> **Spec:** `docs/superpowers/specs/2026-07-22-mprm-opt-parallel-design.md`
> **Branch alvo:** `mprm-opt-parallel` (a partir de `mprm-von-karman`)
> **Diagnósticos base:** `docs/07-diagnostico-von-karman.md`, `docs/10-paralelizacao-mprm.md`

**Goal:** Corrigir déficits de acurácia física do benchmark de von Kármán,
eliminar gargalos de paralelização, e modularizar `mprm_core.py` (~1400 linhas)
em um pacote `mprm/` com submódulos testáveis independentemente.

**Tech Stack:** Python 3.12, NumPy, SciPy, Firedrake (via `mrcm`), mpi4py, pytest 9.

**Global Constraints:**
- Ambiente: `source ~/venv-firedrake/bin/activate`; todos os comandos de `~/mrcm/NavierStokes`.
- Retrocompatibilidade: `mac_wstar`, `run_cavity`, `hh_benchmark_mac`, `mprm_projection_step` preservados.
- `import mprm` NÃO deve puxar Firedrake.
- `mixed_order=0` mantido (ver `docs/09-dg0-projecao.md`).

---

## Estrutura de arquivos (alvo)

```
mprm/
├── __init__.py
├── problem.py               # SideBC, FlowProblem, square_cylinder, cavity_problem, von_karman_problem
├── transport.py             # mac_transport, mac_wstar, mac_divergence, solid_faces
├── projector.py             # MPrMProjector, _impermeable_problem, _darcy_boundary_from_problem
├── downscaler.py            # MACDownscaler
├── flow.py                  # run_flow, run_cavity, run_von_karman, mprm_projection_step
├── forces.py                # ChannelFinePoisson, obstacle_forces, strouhal, DiagnosticPressureSolver
├── parallel.py              # ThreadPool downscaler, MPI helpers
├── viz.py                   # make_frame_saver, make_gif
└── benchmarks/
    ├── __init__.py
    ├── hh.py                # hh_benchmark_mac, olshanskii_u_np, hh_p_np, hh_build_w
    └── cavity.py            # (se extraído)
tests/
├── conftest.py
├── test_problem.py
├── test_transport.py
├── test_projector.py
├── test_downscaler.py
├── test_forces_strouhal.py
├── test_forces_stokes.py         # NOVO
├── test_strouhal_fft.py          # NOVO
├── test_diagnostic_pressure.py   # NOVO
├── test_parallel.py              # NOVO
├── test_run_regression.py
├── test_von_karman.py
├── test_alpha_hh.py
├── test_alpha_cavity.py
├── test_convergence_hh.py
├── common.py
└── params.py
mprm_core.py                 # → vira wrapper de re-export (deprecated)
```

---

## Task 0: Setup do branch e infraestrutura de testes

**Files:**
- Git: criar branch `mprm-opt-parallel` a partir de `mprm-von-karman`
- Create: `mprm/__init__.py` (inicialmente vazio ou re-export do `mprm_core`)
- Create: `tests/test_mprm_package.py`

**Interfaces:**
- `mprm/__init__.py` re-exporta todas as APIs públicas atuais de `mprm_core`
- `test_mprm_package` verifica que `import mprm` funciona e não puxa Firedrake

- [ ] **Step 0.1: Criar branch**

```bash
cd ~/mrcm/NavierStokes && git checkout mprm-von-karman && \
git checkout -b mprm-opt-parallel
```

- [ ] **Step 0.2: Criar estrutura de diretórios**

```bash
mkdir -p mprm/benchmarks
touch mprm/__init__.py mprm/benchmarks/__init__.py
```

- [ ] **Step 0.3: Criar `tests/test_mprm_package.py`**

```python
"""Testa que o pacote mprm importa corretamente e sem firedrake."""
import sys

def test_import_mprm_no_firedrake():
    import mprm
    assert "firedrake" not in sys.modules

def test_public_api_present():
    from mprm import (SideBC, FlowProblem, cavity_problem, von_karman_problem,
                      mac_transport, mac_wstar, mac_divergence, solid_faces,
                      MPrMProjector, MACDownscaler, mprm_projection_step,
                      run_flow, run_cavity, run_von_karman)
```

- [ ] **Step 0.4: Rodar suite atual para baseline verde**

```bash
pytest tests/ -q --ignore=tests/test_von_karman.py::test_von_karman_quick -k "not von_karman_quick"
```

---

## Task 1: Correção das forças — termo viscoso + transiente (P1-B0)

**Files:**
- Modify: `mprm/forces.py` (extrair de `mprm_core.py`)
- Create: `tests/test_forces_stokes.py`

**Diagnóstico:** `docs/07-diagnostico-von-karman.md` §B — `obstacle_forces` omite
os termos viscoso ($\nu(\nabla u + \nabla u^\top)\cdot n$) e transiente
($\frac{d}{dt}\int_{VC} u\,dV$).

**Interfaces:**
- `obstacle_forces(u, v, p, problem, U_ref, D, u_prev=None, v_prev=None, dt=None)`
  — assinatura estendida com campos anteriores e dt para termo transiente.
- `run_flow` passa `u_prev, v_prev, dt` quando `obstacle_metrics=True`.

- [ ] **Step 1.1: Escrever teste de referência analítica**

Criar `tests/test_forces_stokes.py` com escoamento de Couette onde a força
viscosa na placa é conhecida: $\tau = \mu U / H$. Validar que `obstacle_forces`
recupera o valor analítico com < 1% de erro.

- [ ] **Step 1.2: Implementar termos viscoso e transiente**

Adicionar ao `obstacle_forces`:
- Gradientes centrados via `np.gradient` nos centros das células
- Tensor viscoso $\sigma_{ij} = \nu(\partial_j u_i + \partial_i u_j)$
- Termo transiente: $\frac{1}{\Delta t}(\int_{VC} u^{n+1} - \int_{VC} u^n)$
- Sinais: tração é $-\sigma \cdot n$ (subtraído do fluxo convectivo+pressão)

- [ ] **Step 1.3: Rodar benchmark para verificar melhora no $C_D$**

```bash
python -c "
from mprm import run_von_karman
r = run_von_karman(Lx=16, Ly=8, D=1, xc=3, yc=3, U_in=1, Re=100,
                   Ndom=4, nloc=12, T=50, degreeU=3, degreeP=3)
cd = r['cl_history'][:,1]; print(f'C_D={cd[len(cd)//2:].mean():.3f}')
"
```
Alvo: $C_D > 1.25$ (subida de ~1.19 pelo termo viscoso).

---

## Task 2: Setup físico conforme artigo (P2)

**Files:**
- Modify: `mprm/problem.py` (`von_karman_problem`)
- Modify: `mprm/flow.py` (`run_von_karman`)

**Diagnóstico:** `docs/07-diagnostico-von-karman.md` §C — 4 divergências do artigo.

**Mudanças:**
1. `inflow="parabolic"` com normalização por máximo (`inflow_norm="max"`)
2. `walls="slip"` (free-slip) como default
3. `U_ref = U_in` (que agora é o máximo do perfil)
4. Novo parâmetro `inflow_norm` em `von_karman_problem`

- [ ] **Step 2.1: Atualizar `von_karman_problem`**

```python
def von_karman_problem(..., inflow="parabolic", walls="slip",
                       inflow_norm="max", outflow="pressure"):
    if inflow == "parabolic" and inflow_norm == "max":
        def profile(y):  # pico = U_in
            return 4.0 * U_in * y * (Ly_c - y) / (Ly_c ** 2)
```

- [ ] **Step 2.2: Atualizar defaults de `run_von_karman`**

`inflow="parabolic"`, `walls="slip"`.

- [ ] **Step 2.3: Teste de unidade do perfil parabólico**

```python
def test_parabolic_profile_max_is_U_in():
    p = von_karman_problem(..., U_in=1.5, inflow="parabolic", inflow_norm="max")
    prof = p.side("left").profile
    assert prof(Ly/2) == pytest.approx(1.5)  # máximo no centro
```

---

## Task 3: Malha isotrópica com $Ndom_x \neq Ndom_y$ (P3)

**Files:**
- Modify: `mprm/flow.py` (`run_flow`, `run_von_karman`)
- Modify: `mprm/projector.py` (se necessário adaptar `MPrMProjector`)

**Diagnóstico:** `docs/07-diagnostico-von-karman.md` §D — malha anisotrópica,
obstáculo deslocado.

**Mudanças:**
- `run_von_karman` aceita `Ndomx, Ndomy` separados (ou `Ndom` com fallback)
- `run_flow` aceita `Ndomx, Ndomy` separados
- $N_x = Ndom_x \times nloc$, $N_y = Ndom_y \times nloc$
- $h_x = L_x/N_x$, $h_y = L_y/N_y$ → com $Ndom_x = 2 Ndom_y$, $h_x = h_y$

- [ ] **Step 3.1: Generalizar `run_flow` para $Ndom_x \neq Ndom_y$**

```python
def run_flow(problem, Ndomx, Ndomy, nloc, T, ...):
    cfg = MRCMConfig(Ndomx=Ndomx, Ndomy=Ndomy, nlocx=nloc, nlocy=nloc, ...)
```

- [ ] **Step 3.2: Geometria canônica centrada**

Para $L_x=16, L_y=8, D=1$: usar $Ndom_x=10, Ndom_y=5, nloc=30$ →
$h=0.053$, $D/h=19$, obstáculo centrado em $(4,4)$ dentro do subdomínio.

- [ ] **Step 3.3: Teste de regressão com cavidade**

Garantir que `run_cavity` (que usa $Ndom_x = Ndom_y = Ndom$) não quebrou.

---

## Task 4: Estimador de Strouhal robusto (P4)

**Files:**
- Modify: `mprm/forces.py` (`strouhal`)
- Create: `tests/test_strouhal_fft.py`

**Diagnóstico:** `docs/07-diagnostico-von-karman.md` §A — St é artefato do bin.

**Mudanças:**
- Zero-padding ($pad=8$) na FFT
- Interpolação parabólica de 3 pontos no pico
- Reportar $St$ com 3 casas + incerteza

- [ ] **Step 4.1: Escrever teste com sinal sintético**

```python
def test_strouhal_resolve_sub_bin():
    U, D, dt = 1.0, 1.0, 0.01
    f_true = 0.121
    t = np.arange(0, 200, dt)
    sig = np.sin(2 * np.pi * f_true * t) + 0.01 * np.random.randn(len(t))
    st = strouhal(sig, dt, U, D, t_skip_frac=0.25)
    assert st == pytest.approx(0.121, abs=0.005)
```

- [ ] **Step 4.2: Implementar zero-padding + interpolação parabólica**

- [ ] **Step 4.3: Ajustar $T$ default para $\ge 180$**

Em `run_von_karman`: `T = max(180.0, 40.0 * D / U_in)` quando `obstacle_metrics=True`.

---

## Task 5: Cache da pressão diagnóstica (P5)

**Files:**
- Create: `mprm/forces.py` — classe `DiagnosticPressureSolver`
- Create: `tests/test_diagnostic_pressure.py`

**Diagnóstico:** `docs/07-diagnostico-von-karman.md` §G — refatoração LU a cada passo.

**Mudanças:**
- Encapsular montagem + fatoração em `DiagnosticPressureSolver.__init__`
- `solve(u, v)` apenas computa fonte e resolve com LU cacheada
- `run_flow` instancia uma vez e reusa

- [ ] **Step 5.1: Extrair `DiagnosticPressureSolver`**

```python
class DiagnosticPressureSolver:
    def __init__(self, problem):
        # monta e fatora A uma vez
        self.lu = ...
    def solve(self, u, v):
        source = self._compute_source(u, v)
        return self.lu.solve(source)
```

- [ ] **Step 5.2: Teste de que fatoração é reutilizada**

Mock ou contador de chamadas a `splu`.

- [ ] **Step 5.3: Integrar em `run_flow`**

```python
if obstacle_metrics:
    diag_solver = DiagnosticPressureSolver(problem)  # uma vez
    for step in ...:
        p = diag_solver.solve(u, v)
```

---

## Task 6: Paralelização do downscaler (ThreadPool)

**Files:**
- Create: `mprm/parallel.py`
- Create: `tests/test_parallel.py`

**Diagnóstico:** `docs/10-paralelizacao-mprm.md` §2.2 — patches independentes,
resolvidos sequencialmente.

**Mudanças:**
- `mprm_projection_step` aceita `num_threads=1`
- Com `num_threads > 1`, usa `ThreadPoolExecutor` para `solve_patch`
- `spla.splu().solve()` libera o GIL → ganho real com threads

- [ ] **Step 6.1: Implementar `parallel_downscale`**

```python
def parallel_downscale(patches, downscaler, num_threads=1):
    """Resolve solve_patch para múltiplos patches em paralelo."""
    if num_threads <= 1:
        return {isub: downscaler.solve_patch(*args) for isub, args in patches.items()}
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        futures = {isub: ex.submit(downscaler.solve_patch, *args)
                   for isub, args in patches.items()}
        return {isub: f.result() for isub, f in futures.items()}
```

- [ ] **Step 6.2: Teste de equivalência sequencial vs paralelo**

```python
def test_parallel_equals_sequential():
    # 4 patches aleatórios → mesmo resultado com 1 e 4 threads
```

---

## Task 7: Ativar `executor_kind="mpi"` no MRCM

**Files:**
- Modify: `mprm/flow.py` (`run_flow`)
- Create: `tests/test_projector_mpi.py`

**Diagnóstico:** `docs/10-paralelizacao-mprm.md` §2.1 — montagem global só no rank 0.

**Mudanças:**
- `run_flow` detecta `comm.size > 1` e usa `executor_kind="mpi"`
- Teste de regressão: `mpiexec -n 2` produz mesmo resultado que sequencial

- [ ] **Step 7.1: Condicionar `executor_kind` ao tamanho do comm**

```python
executor_kind = "mpi" if comm.size > 1 else "sequential"
cfg = MRCMConfig(..., executor_kind=executor_kind)
```

- [ ] **Step 7.2: Teste MPI mínimo**

```bash
mpiexec -n 2 python -c "from mprm import run_cavity; ..."
```

---

## Task 8: Subamostragem de métricas (quick win)

**Files:**
- Modify: `mprm/flow.py` (`run_flow`)

**Diagnóstico:** `docs/10-paralelizacao-mprm.md` §4.2 — pressão calculada todo passo.

**Mudanças:**
- Adicionar `obstacle_metrics_every=1` a `run_flow`
- Só chama `_diagnostic_pressure` + `obstacle_forces` a cada N passos

- [ ] **Step 8.1: Adicionar parâmetro e lógica condicional**

```python
if obstacle_metrics and step % obstacle_metrics_every == 0:
    p = diag_solver.solve(u, v)
    ...
```

---

## Task 9: Modularização — extrair submódulos

**Files:**
- Create: `mprm/problem.py`, `transport.py`, `projector.py`, `downscaler.py`,
  `flow.py`, `forces.py`, `parallel.py`, `viz.py`, `benchmarks/hh.py`
- Modify: `mprm/__init__.py` (re-exports)
- Modify: `mprm_core.py` (vira `from mprm import *`)

**Ordem de extração (minimiza dependências circulares):**

1. `problem.py` (zero dependências internas)
2. `transport.py` (depende de `problem`)
3. `downscaler.py` (depende de `mrcm.config`)
4. `forces.py` (depende de `problem`, `transport`)
5. `projector.py` (depende de `problem`, `mrcm.*`)
6. `parallel.py` (depende de `downscaler`)
7. `flow.py` (depende de todos acima)
8. `viz.py` (depende de `matplotlib`)
9. `benchmarks/hh.py` (depende de `transport`, `projector`, `downscaler`, `flow`)

- [ ] **Step 9.1: Extrair `problem.py`**

Mover: `SideBC`, `FlowProblem`, `square_cylinder`, `cavity_problem`,
`von_karman_problem`, `assert_solid_interior_to_subdomains`,
`_impermeable_problem`, `_darcy_boundary_from_problem`.

- [ ] **Step 9.2: Extrair `transport.py`**

Mover: `solid_faces`, `_tangential_ghost`, `mac_transport`, `_profile_on_faces`,
`_apply_normal_bc_x`, `_apply_normal_bc_y`, `mac_wstar`, `mac_divergence`.

- [ ] **Step 9.3: Extrair `downscaler.py`**

Mover: `MACDownscaler` (com `_build_masked_lu`, `_get_masked`, `solve_patch`).

- [ ] **Step 9.4: Extrair `forces.py`**

Mover: `ChannelFinePoisson`, `_diagnostic_pressure` → `DiagnosticPressureSolver`,
`obstacle_forces`, `strouhal`.

- [ ] **Step 9.5: Extrair `projector.py`**

Mover: `MPrMProjector`.

- [ ] **Step 9.6: Extrair `flow.py`**

Mover: `_apply_external_normals_after_step`, `mprm_projection_step`,
`_seed_inflow`, `_sample_probe`, `run_flow`, `run_cavity`, `run_von_karman`.

- [ ] **Step 9.7: Extrair `parallel.py` e `viz.py`**

- [ ] **Step 9.8: Extrair `benchmarks/hh.py`**

Mover: `A_OLSH`, `olshanskii_u_np`, `hh_p_np`, `hh_gradp_np`, `hh_build_w`,
`hh_benchmark_mac`.

- [ ] **Step 9.9: Atualizar `mprm/__init__.py`**

```python
from mprm.problem import *
from mprm.transport import *
from mprm.downscaler import MACDownscaler
from mprm.projector import MPrMProjector
from mprm.forces import ChannelFinePoisson, obstacle_forces, strouhal, DiagnosticPressureSolver
from mprm.flow import run_flow, run_cavity, run_von_karman, mprm_projection_step
from mprm.benchmarks.hh import hh_benchmark_mac
```

- [ ] **Step 9.10: Atualizar `mprm_core.py`**

```python
# DEPRECATED: use `import mprm` instead
from mprm import *
```

- [ ] **Step 9.11: Atualizar imports nos testes**

```python
# Antes: from mprm_core import ...
# Depois: from mprm import ...
```

---

## Task 10: Métricas de relatório ($C_L$ amplitude, P7)

**Files:**
- Modify: `tests/test_von_karman.py`

**Diagnóstico:** `docs/07-diagnostico-von-karman.md` §E.

**Mudanças:**
- Reportar `Cl_amp = 0.5*(cl[half:].max() - cl[half:].min())` além de `Cl_rms`
- Comparar `Cl_amp` com $C_L$ do artigo (amplitude/pico)

---

## Task 11: Validação final — benchmark completo

**Files:**
- Modify: `tests/test_von_karman.py`

**Configuração canônica:**
```python
r = run_von_karman(Lx=16, Ly=8, D=1, xc=4, yc=4, U_in=1, Re=100,
                   Ndomx=10, Ndomy=5, nloc=30, degreeU=3, degreeP=3,
                   alpha=1.0, T=200, inflow="parabolic", walls="slip",
                   inflow_norm="max")
```

- [ ] **Step 11.1: Rodar benchmark e verificar métricas**

Alvos: $C_D \in [1.28, 1.38]$, $C_L^{amp} \in [0.22, 0.32]$, $St \in [0.118, 0.126]$.

- [ ] **Step 11.2: Gerar GIF de alta resolução**

```bash
python gif.py  # com a nova geometria e parâmetros
```

- [ ] **Step 11.3: Rodar suíte completa de testes**

```bash
pytest tests/ -q
```

---

## Ordem de execução e dependências

```
T0 (setup)
 │
 ├─ T1 (forças) ─────────────────────┐
 ├─ T2 (setup físico) ──────────────┐│
 ├─ T4 (Strouhal) ─────────────────┐││
 ├─ T8 (subamostragem) ───────────┐│││
 │                                 ││││
 ├─ T5 (cache pressão) ───────────┤│││
 ├─ T6 (ThreadPool downscaler) ───┤│││
 ├─ T7 (MPI executor) ────────────┤│││
 │                                 ││││
 ├─ T3 (Ndomx≠Ndomy) ─────────────┤│││
 │                                 ││││
 └─ T9 (modularização) ◄──────────┴┴┴┴┘  (depende de T1–T8)
      │
      ├─ T10 (métricas C_L)
      └─ T11 (validação final)
```

T1–T8 são **independentes entre si** e podem ser implementadas em paralelo por
diferentes workers. T9 (modularização) deve ser a **última** tarefa de código,
pois mexe em todos os arquivos. T10 e T11 são validação.

---

## Notas finais

- **Git:** commits atômicos por task, messages no padrão
  `feat(mprm): ...` / `fix(mprm): ...` / `test(mprm): ...` / `refactor(mprm): ...`.
- **Testes como gate:** cada task só é considerada concluída quando seus testes
  passam E a suíte de regressão permanece verde.
- **Handoff:** ao final, gerar `docs/superpowers/handoffs/2026-07-22-mprm-opt-parallel.md`
  com tabela de métricas antes/depois e perfil de tempo.
