# Design — Atualização do módulo de projeção (MPrM) para a nova API de BCs do
# mrcm + benchmark genérico de von Kármán

**Data:** 2026-07-20
**Módulo alvo:** `~/mrcm/NavierStokes/` (MPrM — método de projeção multiescala)
**Dependência:** pacote `mrcm` (`~/mrcm/mrcm-firedrake`)

---

## 1. Objetivo

Duas entregas acopladas:

1. **Atualizar** `mprm_core.py` para consumir a **generalização de BCs** feita no
   `mrcm` (branch `worktree-mrcm-bc-generalization`), removendo o monkey-patch
   que hoje contorna o bug de no-flow forte em left/right.
2. **Implementar o benchmark de von Kármán** (escoamento em torno de um cilindro
   quadrado, Re = 100 — §3.5 / Tabela 3 do artigo de Lages) **como uma
   capacidade genérica e parametrizada do módulo de projeção**, não como um
   script hard-coded. Inclui um teste em `tests/` no mesmo padrão dos
   `test_alpha_*`.

## 2. Contexto — as atualizações do mrcm

O branch `worktree-mrcm-bc-generalization` (8 commits, 2026-07-20) faz
**fast-forward** limpo sobre `main` (verificado: sem conflitos). Ele entrega:

- `39655c3` — **no-flow forte (`u·n = 0`) em TODOS os 4 lados externos**, não só
  bottom/top. É exatamente o que `mprm_core._build_flux_dirichlet_bcs_allsides`
  hoje replica via monkey-patch. Passa a ser nativo do `mrcm`.
- `5eeaa3f` — **`BoundaryData.flux_strong`**: BC essencial de fluxo normal
  **não nulo** (`u·n = z` imposto forte nos dofs de face RT/BDM), em vez do
  termo fraco/natural (que impunha `p = −z`, a origem do bug antigo).
- `6b12b6f`/`f27f5f5`/`30ead52`/`2d8f67d` — *lifting* de BC inomogênea via
  `action(a, g)` em `LocalSolver` e `fine_reference` (correto para `flux_strong`).
- `2906e08` — `external_bc_kinds` no *fingerprint* do cache de MsBF (subdomínios
  com tipos de BC externos distintos não compartilham entrada de cache).

**Mudança de assinatura relevante:**
`build_flux_dirichlet_bcs(W, internal_flags, problem)` →
`build_flux_dirichlet_bcs(W, mesh, internal_flags, problem)`.

## 3. Decisões (confirmadas com o usuário)

- **D1 — Integração do mrcm:** *merge* `worktree-mrcm-bc-generalization` → `main`
  (fast-forward). O `mprm_core` importa o pacote de `~/mrcm/mrcm-firedrake/mrcm/`,
  que passa a ter a API nova.
- **D2 — Modelo do obstáculo:** **híbrido** — obstáculo imposto de forma exata
  no transporte MAC e no *downscaler* (células sólidas mascaradas, faces sólidas
  como no-flow ⇒ `∇·u = 0` nas células de fluido e `u·n = 0` exato nas faces do
  obstáculo), e como **permeabilidade baixa `K → ε`** no solve grosso do MRCM
  (usa o suporte nativo do mrcm a `K` heterogêneo). Restrição documentada:
  obstáculo interior aos subdomínios (não atravessa interfaces).
- **D3 — Escopo do teste:** teste rápido de regressão + função de benchmark
  executável completa. `tests/test_von_karman.py` no padrão `test_alpha_*`.

## 4. Tratamento de BCs da projeção (o ponto crítico de corretude)

Tratamento clássico de Chorin para contornos abertos. Para o problema elíptico
`δt∇²p = ∇·w*`, `v = δt∇p`, `uⁿ⁺¹ = w* − v`:

| Lado físico        | Pressão (Poisson) | `v·n` na projeção | mrcm `BoundaryData` |
|--------------------|-------------------|-------------------|---------------------|
| parede (no-slip)   | Neumann homogêneo | `0`               | `flux = 0` (forte)  |
| parede (free-slip) | Neumann homogêneo | `0`               | `flux = 0` (forte)  |
| **entrada**        | Neumann homogêneo | `0`               | `flux = 0` (forte)  |
| **face do obstáculo** | Neumann homogêneo | `0`            | via `K → ε` + máscara|
| **saída**          | Dirichlet `p = 0` | `≠ 0` (livre)     | `pressure = 0`      |

**Insight de projeto (simplifica a implementação):** entrada, paredes e
obstáculo são **todos `v·n = 0`** (no-flow) para a projeção *e* para o
*downscaler*. A velocidade de entrada `U_in` vive **apenas em `w*`** (imposta no
transporte); como a projeção preserva `u·n = w*·n` onde `v·n = 0`, a entrada
sobrevive automaticamente a `uⁿ⁺¹ = w* − ṽ` sem precisar de `flux_strong`. A
**única** face com `v·n ≠ 0` é a **saída** (Dirichlet `p = 0`, que também ancora
a pressão e remove o nullspace de Neumann puro que a cavidade tinha).

- `flux_strong` fica cabeado e disponível para qualquer BC de `u·n` não nulo
  genuíno, mas **von Kármán não o exige**. O que desbloqueia von Kármán é
  (a) o no-flow forte em left/right (fix `39655c3`) para a entrada e as paredes
  laterais, e (b) a saída como Dirichlet de pressão (já suportada).

## 5. Arquitetura

### 5.1 Abstração de problema (o que torna von Kármán "não hard-coded")

Uma dataclass declarativa substitui as hipóteses específicas da cavidade hoje
embutidas em `mac_wstar`/`MPrMProjector`/`MACDownscaler`.

```python
@dataclass
class SideBC:
    kind: str            # "wall" | "slip" | "inflow" | "outflow"
    velocity: float | Callable | None = None  # tangencial (wall/slip)
    profile:  float | Callable | None = None   # perfil normal (inflow)

@dataclass
class FlowProblem:
    Nx: int; Ny: int
    Lx: float; Ly: float
    nu: float
    sides: dict[str, SideBC]          # chaves: left, right, top, bottom
    solid: np.ndarray | None = None    # (Nx, Ny) bool — células do obstáculo
    force: tuple[float, float] | None = None
    # helpers derivados: hx, hy, has_outflow, is_impermeable, solid_faces_*, ...
```

Construtores (nunca blocos hard-coded):

```python
def square_cylinder(Nx, Ny, Lx, Ly, D, xc, yc) -> np.ndarray   # máscara bool
def cavity_problem(N, Re, lid_velocity=1.0) -> FlowProblem
def von_karman_problem(Lx, Ly, Nx, Ny, D, xc, yc, U_in, Re,
                       inflow="uniform", walls="wall",
                       outflow="pressure") -> FlowProblem
```

A **cavidade é reexpressa como um `FlowProblem`** (tampa = `SideBC("wall",
velocity=lid_velocity)`; demais lados `wall`; `solid=None`), de modo que
cavidade e von Kármán compartilham um único caminho de código.

### 5.2 Transporte (`mac_wstar`) — generalização

`mac_wstar(u, v, problem, dt)` lê o `FlowProblem` em vez de tampa+paredes fixas:

- fantasmas tangenciais por lado a partir de `SideBC` (a tampa da cavidade vira
  um `wall` com `velocity`);
- faces de **entrada** recebem o perfil prescrito; **saída** por extrapolação de
  gradiente nulo;
- **obstáculo** por mascaramento: velocidade `0` nas células sólidas, `u·n = 0`
  e no-slip (fantasma) nas faces sólidas.

A cavidade reduz-se exatamente ao estêncil de hoje (protegido por assert de
regressão — ver §8).

### 5.3 Projeção (`MPrMProjector`) — BCs + obstáculo

`MPrMProjector(cfg, dt, problem, comm=None, verbose=False)`:

- **BCs externas** montadas a partir de `problem.sides` conforme a tabela §4:
  no-flow (`flux=0`) em entrada/paredes; `pressure=0` na saída. Usa
  `mrcm.local_form.build_flux_dirichlet_bcs(W, mesh, flag_side, problem=darcy)`
  (assinatura nova) — **o monkey-patch é removido**.
- **Obstáculo (coarse):** `permeability` passa a devolver uma `Function` DG0 na
  malha triangular do subdomínio, `= δt` no fluido e `= δt·ε` nas células
  sólidas (reusa o mapa ponto-médio→célula global já existente em `cell_map`);
  `inv_permeability = 1/K`.
- O objeto `DarcyProblem` do mrcm é construído a partir do `FlowProblem`
  (mapeamento de lados → `BoundaryData`, `K` heterogêneo).
- **Extração de fluxo de saída:** além dos fluxos de interface (Eq. 21), o
  projetor extrai o traço de `v·n` nas facetas de **saída** (mesma máquina
  `HDiv Trace` grau 0), para alimentar o *downscaler* dos subdomínios de saída.

### 5.4 Downscaling (`MACDownscaler`) + passo

`MACDownscaler(cfg, dt, problem)`:

- **Obstáculo:** células sólidas excluídas do solve local de Neumann; faces
  sólidas tratadas como no-flow ⇒ `ṽ` exatamente conservativo no fluido e
  `ṽ·n = 0` nas faces do obstáculo. Subdomínios com sólido usam uma matriz FV
  própria (a máscara muda o estêncil); subdomínios sem sólido continuam
  compartilhando a matriz fatorada única.
- **Saída:** faces de saída dos subdomínios de saída recebem o fluxo `v·n`
  reconstruído pelo solve grosso como dado de Neumann (abordagem primária,
  espelha o tratamento das interfaces). *Fallback* se a qualidade da saída for
  ruim: impor Dirichlet `p = 0` na face de saída do solve FV do downscaler
  (exige fatoração separada para a coluna de saída).

`mprm_projection_step(w1, w2, projector, downscaler, problem)` carrega os fluxos
externos (entrada/paredes/obstáculo = 0; saída = reconstruído) para os patches
dos subdomínios de contorno, em vez de assumir paredes impermeáveis em todo o
contorno.

### 5.5 Laço temporal e drivers

```python
def run_flow(problem, cfg_params..., T, dt=None, method="mprm",
             comm=None, verbose=True, log_every=200,
             frame_callback=None, frame_every=0, probes=None) -> dict
def run_cavity(...)      # wrapper: cavity_problem(...) → run_flow(...)
def run_von_karman(Lx=..., Ly=..., D=..., xc=..., yc=..., U_in=..., Re=100.0,
                   Ndom=..., nloc=..., degreeU=..., degreeP=..., alpha=1.0,
                   basis_kind="legendre_P", T=..., dt=None, method="mprm",
                   comm=None, ...) -> dict
```

`run_cavity` e `hh_benchmark_mac` mantêm assinatura pública e comportamento
(compatibilidade com os testes `test_alpha_*` e o notebook). `run_von_karman`
devolve o campo final + série temporal `C_L(t)` (e/ou séries de sondas).

### 5.6 Diagnósticos

```python
def obstacle_forces(u, v, problem, dt, U_ref, D) -> (Cd, Cl)
def strouhal(cl_history, dt, U_ref, D) -> float
```

- **Forças (`C_D`, `C_L`):** via **balanço de momento em volume de controle** ao
  redor do obstáculo (fluxo de momento + pressão nas faces do VC) — robusto na
  malha MAC e **sem** reconstruir uma pressão global. `C_D = 2F_x/(ρU²D)`,
  `C_L = 2F_y/(ρU²D)` (ρ = 1).
- **Strouhal:** `St = f·D/U`, `f` da FFT de `C_L(t)` (ou de uma sonda de
  velocidade na esteira), descartando o transiente inicial.

## 6. Fluxo de dados (por passo, online)

```
uⁿ ──mac_wstar(problem)──▶ w*  (U_in na entrada; 0 em paredes/obstáculo)
w* ──mac_divergence──▶ ∇·w*  (fonte DG0 por célula; sólido ⇒ fonte local)
∇·w* ──MPrMProjector.project──▶ fluxos de interface (Eq. 21) + fluxo de saída
        (coarse com K→ε no sólido, saída Dirichlet p=0)
──MACDownscaler.solve_patch(máscara sólida, fluxos)──▶ ṽ conservativo
uⁿ⁺¹ = w* − ṽ   (u·n=U_in preservado na entrada; u·n=0 no obstáculo)
[diagnósticos: obstacle_forces → C_L(t)]
```

## 7. Testes (`tests/test_von_karman.py`)

Mesmo padrão de `test_alpha_cavity.py` (CLI via `params.py`, JSON+figuras via
`common.py`, MPI-aware).

- **Rápido (`--quick`, regressão/smoke):** malha pequena, poucos passos. Asserts:
  1. `max|div u|` no fluido ≈ 0 (machine precision);
  2. `u·n = 0` exato nas faces do obstáculo;
  3. balanço de massa entrada = saída (`∮ u·n ≈ 0`);
  4. estrutura dos campos (shapes, sem NaN).
- **Completo:** roda o transiente e reporta `C_D`, `C_L`, `St` (alvo `St ≈ 0.121`,
  valor do artigo) + figura de *vortex shedding*. Recomendado sob MPI.
- **Regressão da cavidade** (novo assert num teste existente ou dedicado):
  `run_cavity` padrão continua ≈ 5.3e-4 vs malha fina e `div ~1e-15`, garantindo
  que o refactor de `mac_wstar`/projeção não regrediu o caminho validado.

## 8. Riscos e mitigações

- **R1 — Contraste de `K` (ε) extremo** ill-condiciona os solves locais e infla
  `β = αL/K` nas células sólidas. *Mitigação:* ε moderado (p.ex. 1e-4..1e-6
  sintonizável); se necessário, calcular `β` a partir do `K` do fluido (excluir
  sólido do campo de Robin).
- **R2 — Refactor de `mac_wstar`/projeção toca código validado** (cavidade/HH).
  *Mitigação:* cavidade reexpressa como `FlowProblem` + assert de regressão
  (§7); rodar `test_alpha_cavity --quick` e `hh_benchmark_mac` antes/depois.
- **R3 — Qualidade da saída** (Neumann com fluxo reconstruído vs Dirichlet no
  downscaler). *Mitigação:* começar com Neumann (primária); *fallback* Dirichlet
  documentado em §5.4 se `C_D/C_L` degradarem.
- **R4 — Obstáculo atravessando interfaces** gera fluxo de interface
  inconsistente. *Mitigação:* restrição "obstáculo interior aos subdomínios"
  documentada e verificada por assert no construtor `von_karman_problem`.

## 9. Critérios de aceitação

1. `worktree-mrcm-bc-generalization` mergeado em `main`; monkey-patch removido de
   `mprm_core.py`; `import mprm_core` funciona sem o patch.
2. `test_alpha_cavity --quick` e `hh_benchmark_mac` produzem os mesmos números de
   antes (regressão verde).
3. `run_von_karman(...)` roda (sequencial e MPI) e o teste rápido passa (div≈0 no
   fluido, `u·n=0` no obstáculo, massa balanceada).
4. Execução completa reproduz *vortex shedding* e `St ≈ 0.121` (±~10%) com
   espaços enriquecidos (`P_{H,3}`), consistente com a Tabela 3 do artigo.
5. Nenhum parâmetro de von Kármán hard-coded: domínio, obstáculo, Re, perfil de
   entrada, decomposição e espaços são argumentos.

## 10. Fora de escopo (YAGNI)

- Cilindro **circular** / geometria não alinhada à malha (o artigo usa quadrado).
- Degrau descendente (BFS) e demais benchmarks da Tabela 2.
- Viscosidade variável / Brinkman "de verdade" (rotas A/B do `docs/04`).
- Robin adaptativo (K homogêneo no fluido ⇒ irrelevante aqui).
- `git init` do diretório `NavierStokes/` (decisão separada do usuário).
