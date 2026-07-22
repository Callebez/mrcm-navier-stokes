# MPrM — Nova API de BCs do mrcm + benchmark genérico de von Kármán — Plano de Implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Atualizar o módulo de projeção MPrM para a API de BCs generalizada do `mrcm` e implementar o benchmark de von Kármán (cilindro quadrado, Re=100) como capacidade parametrizada, com teste rápido + benchmark completo.

**Architecture:** `mprm_core.py` ganha uma abstração declarativa de problema (`FlowProblem`/`SideBC`) que substitui as hipóteses fixas da cavidade no transporte MAC, no projetor MRCM e no downscaler. Todas as generalizações são **retrocompatíveis** (`problem=None` ⇒ comportamento impermeável atual), preservando a cavidade/HH validadas, o notebook e `run_mprm.py`. O obstáculo é híbrido: exato no transporte/downscaler (máscara) + `K→ε` no solve grosso.

**Tech Stack:** Python 3.12, NumPy, SciPy (sparse LU), Firedrake (via pacote `mrcm`), mpi4py, pytest 9, matplotlib.

## Global Constraints

- **Ambiente:** `source ~/venv-firedrake/bin/activate` (tem firedrake, mpi4py 4.1.1, pytest 9.0.3). Todos os comandos rodam de `~/mrcm/NavierStokes`.
- **`import mprm_core` NÃO deve puxar firedrake** (hoje não puxa — mantenha os `from firedrake import ...` locais, dentro de métodos). Testes de componentes NumPy devem rodar sem firedrake.
- **Retrocompatibilidade obrigatória:** assinaturas públicas existentes não podem quebrar — `mac_wstar(u,v,hx,hy,dt,nu,lid_velocity=1.0)`, `MPrMProjector(cfg,dt,comm=...)`, `MACDownscaler(cfg,dt)`, `mprm_projection_step(w1,w2,projector,downscaler)`, `run_cavity(...)`, `hh_benchmark_mac(...)`. Callers a preservar: `run_mprm.py`, `run_mprm_mpi.py`, `tests/test_alpha_*.py`, `tests/test_convergence_hh.py`, `mprm_notebook.ipynb`.
- **Convenção de sinal (não mexer):** com `K=δt`, `f=∇·w*`, o `uT` reconstruído do mrcm JÁ é `v=+δt∇p`; nenhuma correção de sinal. Só a pressão do mrcm tem sinal trocado (visualização).
- **`mixed_order=0`** (fonte DG0 por célula) — asserção existente mantida.
- **Base `legendre_P`** obrigatória com graus assimétricos (preserva π₀).
- **Restrição do obstáculo:** interior aos subdomínios (não atravessa interfaces) — verificada por assert.
- **von Kármán:** nada hard-coded — domínio, obstáculo, Re, perfil de entrada, decomposição e espaços são argumentos.
- **Referência teórica:** artigo de Lages (`docs/01-resumo-mprm.md`), tabela de BCs de Chorin no spec `docs/superpowers/specs/2026-07-20-mprm-von-karman-design.md` §4.

---

## Estrutura de arquivos

- `~/mrcm/mrcm-firedrake/` — merge `worktree-mrcm-bc-generalization` → `main` (Task 0).
- `mprm_core.py` — MODIFICAR: remover monkey-patch; adicionar `SideBC`, `FlowProblem`, `square_cylinder`, `cavity_problem`, `von_karman_problem` (T1); `mac_transport` + wrapper `mac_wstar` (T2); `MACDownscaler.solve_patch(...solid_loc)` + máscaras (T3); `MPrMProjector` com `problem` + K heterogêneo + BCs nativas + fluxo de saída (T4); `mprm_projection_step`/`run_flow`/`run_cavity`/`run_von_karman` (T5); `ChannelFinePoisson`/`obstacle_forces`/`strouhal` (T6).
- `tests/conftest.py` — CRIAR: sys.path para pytest (T1).
- `tests/test_flow_problem.py` — CRIAR: unidade de `FlowProblem`/máscara/construtores (T1).
- `tests/test_transport.py` — CRIAR: caracterização da cavidade + inflow/obstáculo (T2).
- `tests/test_downscaler_obstacle.py` — CRIAR: máscara no downscaler (T3).
- `tests/test_forces_strouhal.py` — CRIAR: forças/Strouhal analíticos (T6).
- `tests/test_von_karman.py` — CRIAR: smoke pytest (rápido) + estudo CLI completo (T7).

---

### Task 0: Merge da generalização de BCs do mrcm

**Files:**
- Modify (git): `~/mrcm/mrcm-firedrake` branch `main`

**Interfaces:**
- Produces: pacote `mrcm` com `BoundaryData.flux_strong`, `build_flux_dirichlet_bcs(W, mesh, internal_flags, problem)` (assinatura nova, no-flow forte nos 4 lados), `local_form.external_bc_kinds`.

- [ ] **Step 1: Confirmar fast-forward e estado limpo**

Run:
```bash
cd ~/mrcm/mrcm-firedrake && git status --short && \
git merge-base --is-ancestor main worktree-mrcm-bc-generalization && echo "FF-OK"
```
Expected: sem saída de `git status` (árvore limpa) e `FF-OK`.

- [ ] **Step 2: Merge fast-forward na main**

Run:
```bash
cd ~/mrcm/mrcm-firedrake && git checkout main && \
git merge --ff-only worktree-mrcm-bc-generalization && git log --oneline -1
```
Expected: `main` avança para `2d8f67d ... apply the same inhomogeneous-BC lifting fix ...`. NÃO apagar o worktree (está locked).

- [ ] **Step 3: Verificar a API nova disponível no pacote importado**

Run:
```bash
cd ~/mrcm/NavierStokes && source ~/venv-firedrake/bin/activate && \
python -c "import inspect, sys; sys.path.insert(0, '$HOME/mrcm/mrcm-firedrake'); \
from mrcm.problem import BoundaryData; from mrcm.local_form import build_flux_dirichlet_bcs, external_bc_kinds; \
assert 'flux_strong' in BoundaryData.__dataclass_fields__; \
assert list(inspect.signature(build_flux_dirichlet_bcs).parameters)[:2]==['W','mesh']; \
print('mrcm BC API OK')"
```
Expected: `mrcm BC API OK`.

- [ ] **Step 4: Rodar os testes de BC do mrcm (regressão do merge)**

Run:
```bash
cd ~/mrcm/mrcm-firedrake && source ~/venv-firedrake/bin/activate && \
python -m pytest mrcm/tests/test_problem.py mrcm/tests/test_local_solver.py mrcm/tests/test_fine_reference.py -q
```
Expected: todos passam (a branch já traz esses testes). Se algum falhar, PARAR e reportar — o merge não deve prosseguir com testes vermelhos.

- [ ] **Step 5: Commit (nenhum — o merge JÁ é o commit)**

O merge fast-forward não cria commit novo; `main` apenas aponta para `2d8f67d`. Nada a commitar.

---

### Task 1: Abstração `FlowProblem` + construtores + conftest de testes

**Files:**
- Modify: `mprm_core.py` (adicionar bloco de abstração após os imports, antes do transporte)
- Create: `tests/conftest.py`
- Create: `tests/test_flow_problem.py`

**Interfaces:**
- Produces:
  - `SideBC(kind: str, velocity=None, profile=None)` — `kind ∈ {"wall","slip","inflow","outflow"}`.
  - `FlowProblem(Nx,Ny,Lx,Ly,nu,sides:dict,solid=None,force=None)` com props `.hx`, `.hy`, `.has_outflow`, `.is_impermeable`, e método `.side(name)->SideBC`.
  - `square_cylinder(Nx,Ny,Lx,Ly,D,xc,yc) -> np.ndarray[bool] (Nx,Ny)`.
  - `cavity_problem(N,Re,lid_velocity=1.0) -> FlowProblem`.
  - `von_karman_problem(Lx,Ly,Nx,Ny,D,xc,yc,U_in,Re,inflow="uniform",walls="wall",outflow="pressure") -> FlowProblem`.

- [ ] **Step 1: Criar `tests/conftest.py` (path para pytest)**

Create `tests/conftest.py`:
```python
"""Insere os paths necessários para o pytest importar mprm_core, common,
params e o pacote mrcm — independente do rootdir. Sem isto, `from common
import ...` e `import mprm_core` falham sob pytest (o import direto por
`python tests/test_x.py` continua funcionando via script-dir)."""
import os
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(TESTS_DIR)
MRCM_REPO = os.path.expanduser("~/mrcm/mrcm-firedrake")
for _p in (TESTS_DIR, ROOT_DIR, MRCM_REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)
```

- [ ] **Step 2: Escrever o teste que falha**

Create `tests/test_flow_problem.py`:
```python
"""Testes de unidade da abstração de problema (puro NumPy, sem firedrake)."""
import numpy as np
import pytest

from mprm_core import (SideBC, FlowProblem, square_cylinder,
                       cavity_problem, von_karman_problem)


def test_sidebc_valida_kind():
    with pytest.raises(ValueError):
        SideBC("banana")
    with pytest.raises(ValueError):
        SideBC("inflow")  # inflow exige profile


def test_square_cylinder_mask_shape_e_centro():
    Nx, Ny, Lx, Ly, D = 40, 20, 4.0, 2.0, 0.5
    xc, yc = 1.0, 1.0
    mask = square_cylinder(Nx, Ny, Lx, Ly, D, xc, yc)
    assert mask.shape == (Nx, Ny)
    assert mask.dtype == bool
    # célula do centro é sólida; um canto distante é fluido
    hx, hy = Lx / Nx, Ly / Ny
    ic, jc = int(xc / hx), int(yc / hy)
    assert mask[ic, jc]
    assert not mask[0, 0]
    # área ~ D*D (dentro de 1 célula de tolerância por lado)
    area = mask.sum() * hx * hy
    assert abs(area - D * D) < 2 * max(hx, hy) * D


def test_flowproblem_props():
    p = cavity_problem(N=12, Re=100.0, lid_velocity=1.0)
    assert (p.Nx, p.Ny) == (12, 12)
    assert p.hx == pytest.approx(1.0 / 12)
    assert p.is_impermeable
    assert not p.has_outflow
    assert p.side("top").kind == "wall"
    assert p.side("top").velocity == 1.0
    assert p.nu == pytest.approx(1.0 / 100.0)
    assert p.solid is None


def test_von_karman_problem_estrutura():
    p = von_karman_problem(Lx=4.0, Ly=2.0, Nx=40, Ny=20, D=0.5,
                           xc=1.0, yc=1.0, U_in=1.0, Re=100.0)
    assert p.side("left").kind == "inflow"
    assert p.side("right").kind == "outflow"
    assert p.side("top").kind == "wall"
    assert p.has_outflow
    assert not p.is_impermeable
    assert p.solid is not None and p.solid.any()
    assert p.nu == pytest.approx(1.0 * 0.5 / 100.0)  # nu = U_in*D/Re


def test_von_karman_inflow_uniform_profile():
    p = von_karman_problem(Lx=4.0, Ly=2.0, Nx=40, Ny=20, D=0.5,
                           xc=1.0, yc=1.0, U_in=1.5, Re=100.0,
                           inflow="uniform")
    prof = p.side("left").profile
    # perfil uniforme: função (ou constante) devolve U_in em qualquer y
    val = prof(1.0) if callable(prof) else prof
    assert val == pytest.approx(1.5)
```

- [ ] **Step 3: Rodar para confirmar a falha**

Run: `cd ~/mrcm/NavierStokes && source ~/venv-firedrake/bin/activate && pytest tests/test_flow_problem.py -q`
Expected: FALHA com `ImportError: cannot import name 'SideBC'`.

- [ ] **Step 4: Implementar a abstração em `mprm_core.py`**

Em `mprm_core.py`, **remover** o bloco do monkey-patch (linhas ~49–93: o comentário de cabeçalho da seção "Correção de BCs", `_build_flux_dirichlet_bcs_allsides`, `apply_bc_patch` e a chamada `apply_bc_patch()`). Inserir, no lugar, a seção de abstração:

```python
# ===========================================================================
# Abstração de problema (contorno + obstáculo) — compartilhada pelo transporte
# MAC, pelo projetor MRCM e pelo downscaler. A cavidade é um caso particular.
# ===========================================================================

from dataclasses import dataclass          # noqa: E402
from typing import Callable, Optional, Union  # noqa: E402

_SIDE_KINDS = ("wall", "slip", "inflow", "outflow")


@dataclass
class SideBC:
    """Condição de um lado externo do domínio MAC.

    kind:
      "wall"    — u·n = 0, no-slip tangencial (velocity = veloc. da parede);
      "slip"    — u·n = 0, free-slip (gradiente tangencial nulo);
      "inflow"  — u·n = profile (perfil normal prescrito), tangencial = velocity;
      "outflow" — contorno aberto (Dirichlet p=0 na projeção; grad. nulo no
                  transporte).
    velocity: velocidade tangencial da parede (float ou callable(coord)->float).
    profile:  perfil de u·n na entrada (float ou callable(coord)->float).
    """
    kind: str
    velocity: Optional[Union[float, Callable]] = None
    profile: Optional[Union[float, Callable]] = None

    def __post_init__(self):
        if self.kind not in _SIDE_KINDS:
            raise ValueError(f"SideBC.kind inválido: {self.kind!r} "
                             f"(use um de {_SIDE_KINDS})")
        if self.kind == "inflow" and self.profile is None:
            raise ValueError("SideBC(kind='inflow') exige 'profile'.")

    def tangential(self, coord=0.0) -> float:
        """Velocidade tangencial da parede (0 se não especificada)."""
        v = self.velocity
        if v is None:
            return 0.0
        return float(v(coord)) if callable(v) else float(v)


@dataclass
class FlowProblem:
    """Descrição declarativa do problema de escoamento na malha MAC.

    Nx,Ny células; domínio [0,Lx]×[0,Ly]; viscosidade nu; `sides` mapeia
    'left'/'right'/'top'/'bottom' -> SideBC; `solid` é uma máscara (Nx,Ny)
    booleana de células do obstáculo (None = sem obstáculo); `force` é uma
    força de corpo (fx,fy) opcional.
    """
    Nx: int
    Ny: int
    Lx: float
    Ly: float
    nu: float
    sides: dict
    solid: Optional[np.ndarray] = None
    force: Optional[tuple] = None

    def __post_init__(self):
        for s in ("left", "right", "top", "bottom"):
            if s not in self.sides:
                raise ValueError(f"FlowProblem.sides sem o lado '{s}'.")
            if not isinstance(self.sides[s], SideBC):
                raise TypeError(f"sides['{s}'] deve ser SideBC.")
        if self.solid is not None:
            self.solid = np.asarray(self.solid, dtype=bool)
            if self.solid.shape != (self.Nx, self.Ny):
                raise ValueError(
                    f"solid.shape {self.solid.shape} != (Nx,Ny) "
                    f"({self.Nx},{self.Ny}).")

    def side(self, name: str) -> SideBC:
        return self.sides[name]

    @property
    def hx(self) -> float:
        return self.Lx / self.Nx

    @property
    def hy(self) -> float:
        return self.Ly / self.Ny

    @property
    def has_outflow(self) -> bool:
        return any(bc.kind == "outflow" for bc in self.sides.values())

    @property
    def is_impermeable(self) -> bool:
        return (self.solid is None and
                all(bc.kind in ("wall", "slip")
                    for bc in self.sides.values()))


def square_cylinder(Nx, Ny, Lx, Ly, D, xc, yc) -> np.ndarray:
    """Máscara booleana (Nx,Ny) de um cilindro QUADRADO de lado D centrado
    em (xc,yc): célula sólida se seu centro está dentro do quadrado."""
    hx, hy = Lx / Nx, Ly / Ny
    xcell = (np.arange(Nx) + 0.5) * hx
    ycell = (np.arange(Ny) + 0.5) * hy
    X, Y = np.meshgrid(xcell, ycell, indexing="ij")
    return (np.abs(X - xc) <= D / 2.0) & (np.abs(Y - yc) <= D / 2.0)


def cavity_problem(N, Re, lid_velocity=1.0) -> FlowProblem:
    """Cavidade quadrada unitária com tampa deslizante (Lref=1; nu=U/Re)."""
    sides = {
        "top": SideBC("wall", velocity=lid_velocity),
        "bottom": SideBC("wall", velocity=0.0),
        "left": SideBC("wall", velocity=0.0),
        "right": SideBC("wall", velocity=0.0),
    }
    return FlowProblem(Nx=N, Ny=N, Lx=1.0, Ly=1.0,
                       nu=lid_velocity / Re, sides=sides)


def von_karman_problem(Lx, Ly, Nx, Ny, D, xc, yc, U_in, Re,
                       inflow="uniform", walls="wall",
                       outflow="pressure") -> FlowProblem:
    """Escoamento em torno de um cilindro quadrado (von Kármán). Re = U_in*D/nu.

    inflow: "uniform" (u=U_in) ou "parabolic" (perfil de canal, média U_in).
    walls:  "wall" (no-slip) ou "slip" (free-slip) para top/bottom.
    outflow: "pressure" (Dirichlet p=0). Demais valores reservados p/ extensão.
    """
    if outflow != "pressure":
        raise ValueError("Apenas outflow='pressure' é suportado.")
    nu = U_in * D / Re
    if inflow == "uniform":
        profile = float(U_in)
    elif inflow == "parabolic":
        Ly_c = float(Ly)
        def profile(y):                       # média = U_in
            return 6.0 * U_in * y * (Ly_c - y) / (Ly_c ** 2)
    else:
        raise ValueError(f"inflow inválido: {inflow!r}")
    sides = {
        "left": SideBC("inflow", profile=profile),
        "right": SideBC("outflow"),
        "top": SideBC(walls),
        "bottom": SideBC(walls),
    }
    solid = square_cylinder(Nx, Ny, Lx, Ly, D, xc, yc)
    return FlowProblem(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, nu=nu,
                       sides=sides, solid=solid)
```

- [ ] **Step 5: Rodar o teste até passar**

Run: `pytest tests/test_flow_problem.py -q`
Expected: PASS (5 passed).

- [ ] **Step 6: Confirmar que `import mprm_core` ainda não puxa firedrake e que os testes existentes ainda importam**

Run:
```bash
python -c "import sys; sys.path.insert(0,'.'); import mprm_core; print('firedrake carregado =', 'firedrake' in sys.modules)"
```
Expected: `firedrake carregado = False`.

- [ ] **Step 7: Commit**

```bash
cd ~/mrcm/NavierStokes && git init -q 2>/dev/null; \
git add tests/conftest.py tests/test_flow_problem.py mprm_core.py && \
git commit -q -m "feat(mprm): FlowProblem/SideBC + construtores; remove monkey-patch de BCs" || echo "(sem repo git — pular commit)"
```
Nota: `~/mrcm/NavierStokes` pode não ser repo git. Se `git init` não for desejado, **pular o commit** (o usuário decide o versionamento). Não versionar `results_alpha/`, `frames_mprm/`, GIFs/PNGs.

---

### Task 2: Transporte MAC generalizado (`mac_transport`) + wrapper `mac_wstar`

**Files:**
- Modify: `mprm_core.py` (substituir `mac_wstar` por `mac_transport` + wrapper)
- Create: `tests/test_transport.py`

**Interfaces:**
- Consumes: `FlowProblem`, `SideBC`, `cavity_problem` (Task 1).
- Produces:
  - `mac_transport(u, v, problem: FlowProblem, dt) -> (w1, w2)` — w* geral.
  - `mac_wstar(u, v, hx, hy, dt, nu, lid_velocity=1.0) -> (w1, w2)` — wrapper retrocompatível (constrói cavidade e chama `mac_transport`).
  - `solid_faces(solid) -> (fx, fy)` — máscaras de face sólida: `fx (Nx+1,Ny)`, `fy (Nx,Ny+1)`, True se qualquer célula adjacente é sólida.

- [ ] **Step 1: Escrever o teste de caracterização (falha)**

Create `tests/test_transport.py`:
```python
"""Transporte MAC generalizado: caracterização da cavidade (bit-idêntico ao
estêncil antigo) + comportamento de entrada/obstáculo. Puro NumPy."""
import numpy as np
import pytest

from mprm_core import mac_transport, mac_wstar, cavity_problem, \
    von_karman_problem, solid_faces


def _wstar_legacy(u, v, hx, hy, dt, nu, lid_velocity=1.0):
    """Cópia FIEL do estêncil da cavidade pré-refactor (referência dourada)."""
    Nx = u.shape[0] - 1
    Ny = v.shape[1] - 1
    ug = np.empty((Nx + 1, Ny + 2)); ug[:, 1:-1] = u
    ug[:, 0] = -u[:, 0]; ug[:, -1] = 2.0 * lid_velocity - u[:, -1]
    vg = np.empty((Nx + 2, Ny + 1)); vg[1:-1, :] = v
    vg[0, :] = -v[0, :]; vg[-1, :] = -v[-1, :]
    w1 = u.copy(); w2 = v.copy()
    ue = 0.5 * (u[1:, :] + u[:-1, :])
    Fuu = np.where(ue >= 0.0, u[:-1, :], u[1:, :]) * ue
    dudx_conv = (Fuu[1:, :] - Fuu[:-1, :]) / hx
    v_at_corner = 0.5 * (vg[1:-2, :] + vg[2:-1, :])
    u_upw_y = np.where(v_at_corner >= 0.0, ug[1:-1, :-1], ug[1:-1, 1:])
    Fuv = u_upw_y * v_at_corner
    duvdy = (Fuv[:, 1:] - Fuv[:, :-1]) / hy
    lap_u = ((ug[2:, 1:-1] - 2.0 * ug[1:-1, 1:-1] + ug[:-2, 1:-1]) / hx**2
             + (ug[1:-1, 2:] - 2.0 * ug[1:-1, 1:-1] + ug[1:-1, :-2]) / hy**2)
    w1[1:-1, :] = u[1:-1, :] + dt * (-dudx_conv - duvdy + nu * lap_u)
    vc = 0.5 * (v[:, 1:] + v[:, :-1])
    Fvv = np.where(vc >= 0.0, v[:, :-1], v[:, 1:]) * vc
    dvdy_conv = (Fvv[:, 1:] - Fvv[:, :-1]) / hy
    u_at_corner = 0.5 * (ug[:, 1:-2] + ug[:, 2:-1])
    v_upw_x = np.where(u_at_corner >= 0.0, vg[:-1, 1:-1], vg[1:, 1:-1])
    Fvu = v_upw_x * u_at_corner
    duvdx = (Fvu[1:, :] - Fvu[:-1, :]) / hx
    lap_v = ((vg[2:, 1:-1] - 2.0 * vg[1:-1, 1:-1] + vg[:-2, 1:-1]) / hx**2
             + (vg[1:-1, 2:] - 2.0 * vg[1:-1, 1:-1] + vg[1:-1, :-2]) / hy**2)
    w2[:, 1:-1] = v[:, 1:-1] + dt * (-duvdx - dvdy_conv + nu * lap_v)
    return w1, w2


def test_cavity_transporte_bit_identico_ao_legado():
    rng = np.random.default_rng(0)
    N = 16
    u = rng.standard_normal((N + 1, N)); u[0, :] = u[-1, :] = 0.0
    v = rng.standard_normal((N, N + 1)); v[:, 0] = v[:, -1] = 0.0
    hx = hy = 1.0 / N; dt = 1e-3; nu = 1e-2; lid = 1.3
    w1a, w2a = _wstar_legacy(u, v, hx, hy, dt, nu, lid)
    prob = cavity_problem(N, Re=lid / nu, lid_velocity=lid)
    w1b, w2b = mac_transport(u, v, prob, dt)
    assert np.allclose(w1a, w1b, atol=1e-14, rtol=0)
    assert np.allclose(w2a, w2b, atol=1e-14, rtol=0)


def test_wrapper_mac_wstar_igual_a_transport():
    rng = np.random.default_rng(1)
    N = 8
    u = rng.standard_normal((N + 1, N)); u[0, :] = u[-1, :] = 0.0
    v = rng.standard_normal((N, N + 1)); v[:, 0] = v[:, -1] = 0.0
    w1a, w2a = mac_wstar(u, v, 1.0 / N, 1.0 / N, 1e-3, 1e-2, 1.0)
    w1b, w2b = _wstar_legacy(u, v, 1.0 / N, 1.0 / N, 1e-3, 1e-2, 1.0)
    assert np.allclose(w1a, w1b, atol=1e-14)
    assert np.allclose(w2a, w2b, atol=1e-14)


def test_inflow_carrega_perfil_na_face_esquerda():
    prob = von_karman_problem(Lx=4.0, Ly=2.0, Nx=40, Ny=20, D=0.5,
                              xc=1.0, yc=1.0, U_in=1.0, Re=100.0)
    u = np.zeros((41, 20)); v = np.zeros((40, 21))
    w1, w2 = mac_transport(u, v, prob, dt=1e-3)
    # face de entrada (x=0) recebe U_in=1 (exceto onde há sólido — aqui não há)
    assert np.allclose(w1[0, :], 1.0)


def test_obstaculo_zera_faces_solidas():
    prob = von_karman_problem(Lx=4.0, Ly=2.0, Nx=40, Ny=20, D=0.5,
                              xc=1.0, yc=1.0, U_in=1.0, Re=100.0)
    rng = np.random.default_rng(2)
    u = rng.standard_normal((41, 20)); v = rng.standard_normal((40, 21))
    w1, w2 = mac_transport(u, v, prob, dt=1e-3)
    fx, fy = solid_faces(prob.solid)
    assert np.all(w1[fx] == 0.0)
    assert np.all(w2[fy] == 0.0)
```

- [ ] **Step 2: Rodar para confirmar a falha**

Run: `pytest tests/test_transport.py -q`
Expected: FALHA com `ImportError: cannot import name 'mac_transport'`.

- [ ] **Step 3: Implementar `mac_transport`, `solid_faces` e o wrapper**

Em `mprm_core.py`, substituir a função `mac_wstar` inteira (linhas ~108–173) por:

```python
def solid_faces(solid):
    """Máscaras de face MAC tocadas por célula sólida.

    fx (Nx+1, Ny): face-u sólida se célula à esquerda OU à direita é sólida.
    fy (Nx, Ny+1): face-v sólida se célula abaixo OU acima é sólida.
    """
    Nx, Ny = solid.shape
    fx = np.zeros((Nx + 1, Ny), dtype=bool)
    fx[:-1, :] |= solid
    fx[1:, :] |= solid
    fy = np.zeros((Nx, Ny + 1), dtype=bool)
    fy[:, :-1] |= solid
    fy[:, 1:] |= solid
    return fx, fy


def _tangential_ghost(interior, side_bc: SideBC):
    """Célula-fantasma tangencial de um lado: no-slip (wall/inflow) =>
    2*V_t - interior; free-slip/aberto (slip/outflow) => interior."""
    if side_bc.kind in ("wall", "inflow"):
        return 2.0 * side_bc.tangential() - interior
    return interior          # slip, outflow: gradiente tangencial nulo


def mac_transport(u, v, problem: "FlowProblem", dt):
    """Velocidade tentativa w* = uⁿ + δt(−∇·(uu) + ν∇²u + f) na malha MAC,
    para um `FlowProblem` arbitrário (contorno + obstáculo).

    Convecção upwind (donor-cell) + difusão central (§2.6 do artigo). A
    cavidade é o caso `cavity_problem(...)` e reproduz o estêncil antigo
    bit-a-bit. Entrada/paredes/obstáculo tratados via fantasmas + máscara
    de faces sólidas.
    """
    Nx = u.shape[0] - 1
    Ny = v.shape[1] - 1
    hx, hy, nu = problem.hx, problem.hy, problem.nu
    S = problem.sides

    # ---- fantasmas tangenciais (u em j: bottom/top; v em i: left/right) ----
    ug = np.empty((Nx + 1, Ny + 2)); ug[:, 1:-1] = u
    ug[:, 0] = _tangential_ghost(u[:, 0], S["bottom"])
    ug[:, -1] = _tangential_ghost(u[:, -1], S["top"])
    vg = np.empty((Nx + 2, Ny + 1)); vg[1:-1, :] = v
    vg[0, :] = _tangential_ghost(v[0, :], S["left"])
    vg[-1, :] = _tangential_ghost(v[-1, :], S["right"])

    w1 = u.copy(); w2 = v.copy()

    # ---- componente x (faces internas i=1..Nx-1) ----
    ue = 0.5 * (u[1:, :] + u[:-1, :])
    Fuu = np.where(ue >= 0.0, u[:-1, :], u[1:, :]) * ue
    dudx_conv = (Fuu[1:, :] - Fuu[:-1, :]) / hx
    v_at_corner = 0.5 * (vg[1:-2, :] + vg[2:-1, :])
    u_upw_y = np.where(v_at_corner >= 0.0, ug[1:-1, :-1], ug[1:-1, 1:])
    Fuv = u_upw_y * v_at_corner
    duvdy = (Fuv[:, 1:] - Fuv[:, :-1]) / hy
    lap_u = ((ug[2:, 1:-1] - 2.0 * ug[1:-1, 1:-1] + ug[:-2, 1:-1]) / hx**2
             + (ug[1:-1, 2:] - 2.0 * ug[1:-1, 1:-1] + ug[1:-1, :-2]) / hy**2)
    fbx = 0.0 if problem.force is None else problem.force[0]
    w1[1:-1, :] = u[1:-1, :] + dt * (-dudx_conv - duvdy + nu * lap_u + fbx)

    # ---- componente y (faces internas j=1..Ny-1) ----
    vc = 0.5 * (v[:, 1:] + v[:, :-1])
    Fvv = np.where(vc >= 0.0, v[:, :-1], v[:, 1:]) * vc
    dvdy_conv = (Fvv[:, 1:] - Fvv[:, :-1]) / hy
    u_at_corner = 0.5 * (ug[:, 1:-2] + ug[:, 2:-1])
    v_upw_x = np.where(u_at_corner >= 0.0, vg[:-1, 1:-1], vg[1:, 1:-1])
    Fvu = v_upw_x * u_at_corner
    duvdx = (Fvu[1:, :] - Fvu[:-1, :]) / hx
    lap_v = ((vg[2:, 1:-1] - 2.0 * vg[1:-1, 1:-1] + vg[:-2, 1:-1]) / hx**2
             + (vg[1:-1, 2:] - 2.0 * vg[1:-1, 1:-1] + vg[1:-1, :-2]) / hy**2)
    fby = 0.0 if problem.force is None else problem.force[1]
    w2[:, 1:-1] = v[:, 1:-1] + dt * (-duvdx - dvdy_conv + nu * lap_v + fby)

    # ---- faces normais de contorno por lado ----
    _apply_normal_bc_x(w1, S["left"], hy, Ny, side="left")
    _apply_normal_bc_x(w1, S["right"], hy, Ny, side="right")
    _apply_normal_bc_y(w2, S["bottom"], hx, Nx, side="bottom")
    _apply_normal_bc_y(w2, S["top"], hx, Nx, side="top")

    # ---- obstáculo: zera faces tocadas por célula sólida (u=0 no sólido) ----
    if problem.solid is not None:
        fx, fy = solid_faces(problem.solid)
        w1[fx] = 0.0
        w2[fy] = 0.0

    return w1, w2


def _profile_on_faces(profile, coords):
    """Avalia um perfil de entrada (float ou callable) nos centros de face."""
    if callable(profile):
        return np.array([profile(c) for c in coords])
    return np.full(len(coords), float(profile))


def _apply_normal_bc_x(w1, side_bc: SideBC, hy, Ny, side):
    """u·n nas faces x=0 (side='left') ou x=Lx (side='right')."""
    idx = 0 if side == "left" else -1
    ycen = (np.arange(Ny) + 0.5) * hy
    if side_bc.kind in ("wall", "slip"):
        w1[idx, :] = 0.0
    elif side_bc.kind == "inflow":
        w1[idx, :] = _profile_on_faces(side_bc.profile, ycen)
    elif side_bc.kind == "outflow":
        w1[idx, :] = w1[1 if side == "left" else -2, :]  # gradiente nulo


def _apply_normal_bc_y(w2, side_bc: SideBC, hx, Nx, side):
    """u·n nas faces y=0 (side='bottom') ou y=Ly (side='top')."""
    idx = 0 if side == "bottom" else -1
    xcen = (np.arange(Nx) + 0.5) * hx
    if side_bc.kind in ("wall", "slip"):
        w2[:, idx] = 0.0
    elif side_bc.kind == "inflow":
        w2[:, idx] = _profile_on_faces(side_bc.profile, xcen)
    elif side_bc.kind == "outflow":
        w2[:, idx] = w2[:, 1 if side == "bottom" else -2]


def mac_wstar(u, v, hx, hy, dt, nu, lid_velocity=1.0):
    """RETROCOMPATÍVEL — cavidade. Constrói o FlowProblem da cavidade e
    delega a `mac_transport`. Mantido para `run_mprm.py`, notebook etc."""
    Nx = u.shape[0] - 1
    Ny = v.shape[1] - 1
    sides = {
        "top": SideBC("wall", velocity=lid_velocity),
        "bottom": SideBC("wall", velocity=0.0),
        "left": SideBC("wall", velocity=0.0),
        "right": SideBC("wall", velocity=0.0),
    }
    prob = FlowProblem(Nx=Nx, Ny=Ny, Lx=Nx * hx, Ly=Ny * hy,
                       nu=nu, sides=sides)
    return mac_transport(u, v, prob, dt)
```

- [ ] **Step 4: Rodar os testes até passar**

Run: `pytest tests/test_transport.py -q`
Expected: PASS (4 passed). O teste de caracterização garante que a cavidade não mudou.

- [ ] **Step 5: Commit**

```bash
git add mprm_core.py tests/test_transport.py && \
git commit -q -m "feat(mprm): mac_transport geral (entrada/saída/obstáculo) + wrapper mac_wstar" || echo "(sem repo git)"
```

---

### Task 3: Downscaler ciente de obstáculo (`solve_patch(..., solid_loc)`)

**Files:**
- Modify: `mprm_core.py` (`MACDownscaler`)
- Create: `tests/test_downscaler_obstacle.py`

**Interfaces:**
- Consumes: `MACDownscaler(cfg, dt)` existente.
- Produces:
  - `MACDownscaler.solve_patch(divw_loc, fW, fE, fS, fN, solid_loc=None)` — se `solid_loc` (n,m bool) dado, exclui células sólidas (p=0 desacoplado), trata faces sólidas como no-flow; fatoração cacheada por padrão de `solid_loc`. `solid_loc=None` reproduz o comportamento atual.
  - Devolve `(vt1, vt2)` com `vt·n=0` nas faces sólidas.

- [ ] **Step 1: Escrever o teste que falha**

Create `tests/test_downscaler_obstacle.py`:
```python
"""Downscaler com máscara de obstáculo (puro scipy/NumPy)."""
import numpy as np
import pytest

from mrcm.config import MRCMConfig
from mprm_core import MACDownscaler


def _cfg(n=8):
    return MRCMConfig(Ndomx=1, Ndomy=1, nlocx=n, nlocy=n,
                      degreeU=1, degreeP=1, alpha=1.0,
                      mixed_family="BDM", mixed_order=0,
                      executor_kind="sequential")


def test_sem_solido_reproduz_comportamento_atual():
    cfg = _cfg(8); dt = 0.5
    ds = MACDownscaler(cfg, dt)
    rng = np.random.default_rng(0)
    divw = rng.standard_normal((8, 8)); divw -= divw.mean()
    z = np.zeros(8)
    vt1, vt2 = ds.solve_patch(divw, z, z, z, z)          # sem solid_loc
    # conservação: div(vt) == divw (Neumann puro, RHS balanceado)
    dv = (vt1[1:, :] - vt1[:-1, :]) / ds.hx + (vt2[:, 1:] - vt2[:, :-1]) / ds.hy
    assert np.allclose(dv, divw, atol=1e-9)


def test_solido_zera_faces_e_conserva_no_fluido():
    cfg = _cfg(8); dt = 0.5
    ds = MACDownscaler(cfg, dt)
    solid = np.zeros((8, 8), dtype=bool)
    solid[3:5, 3:5] = True                                # bloco 2x2 interno
    rng = np.random.default_rng(1)
    divw = rng.standard_normal((8, 8))
    divw[solid] = 0.0                                     # fonte nula no sólido
    z = np.zeros(8)
    vt1, vt2 = ds.solve_patch(divw, z, z, z, z, solid_loc=solid)
    # faces sólidas: vt·n = 0
    fx = np.zeros((9, 8), bool); fx[:-1, :] |= solid; fx[1:, :] |= solid
    fy = np.zeros((8, 9), bool); fy[:, :-1] |= solid; fy[:, 1:] |= solid
    assert np.all(vt1[fx] == 0.0)
    assert np.all(vt2[fy] == 0.0)
    # conservação nas células de FLUIDO: div(vt) == divw
    dv = (vt1[1:, :] - vt1[:-1, :]) / ds.hx + (vt2[:, 1:] - vt2[:, :-1]) / ds.hy
    assert np.allclose(dv[~solid], divw[~solid], atol=1e-9)
```

- [ ] **Step 2: Rodar para confirmar a falha**

Run: `pytest tests/test_downscaler_obstacle.py -q`
Expected: FALHA (`solve_patch() got an unexpected keyword argument 'solid_loc'`).

- [ ] **Step 3: Implementar o mascaramento no `MACDownscaler`**

Em `mprm_core.py`, no `MACDownscaler`: guardar `n,m,hx,hy` (já guardados), manter a matriz base fatorada em `self.lu` e adicionar um cache de fatorações mascaradas. Substituir `solve_patch` e adicionar helpers:

No `__init__`, após `self.lu = spla.splu(A.tocsc())`, adicionar:
```python
        self._A_base = A.tocsc()          # sem sólido (referência p/ máscara)
        self._masked_lu = {}              # solid_loc.tobytes() -> splu
```

Adicionar métodos:
```python
    def _build_masked_lu(self, solid_loc):
        """Fatoração FV para um padrão de sólido: células sólidas viram
        p=0 (linha identidade, desacopladas); faces sólidas somem do estêncil
        (no-flow). Cacheada por padrão."""
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        n, m = self.cfg.nlocx, self.cfg.nlocy
        dt, hx, hy = self.dt, self.hx, self.hy
        solid = solid_loc.reshape(n, m)

        def lid(i, j):
            return i + j * n

        rows, cols, vals = [], [], []
        for j in range(m):
            for i in range(n):
                c = lid(i, j)
                if solid[i, j]:
                    rows.append(c); cols.append(c); vals.append(1.0)
                    continue
                diag = 0.0
                for (di, dj, d, ln) in ((1, 0, hx, hy), (-1, 0, hx, hy),
                                        (0, 1, hy, hx), (0, -1, hy, hx)):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < n and 0 <= jj < m and not solid[ii, jj]:
                        w = dt * ln / d
                        rows.append(c); cols.append(lid(ii, jj))
                        vals.append(w); diag -= w
                rows.append(c); cols.append(c); vals.append(diag)
        A = sp.csr_matrix((vals, (rows, cols)), shape=(n * m, n * m)).tolil()
        # pino: primeira célula de FLUIDO
        pin = int(np.argmin(solid.flatten(order="F")))
        A[pin, :] = 0.0; A[pin, pin] = 1.0
        return spla.splu(A.tocsc()), pin

    def _get_masked(self, solid_loc):
        key = solid_loc.tobytes()
        if key not in self._masked_lu:
            self._masked_lu[key] = self._build_masked_lu(solid_loc)
        return self._masked_lu[key]
```

Substituir `solve_patch` por:
```python
    def solve_patch(self, divw_loc, fW, fE, fS, fN, solid_loc=None):
        """Resolve o Neumann local (Eq. 22) e devolve ṽ (MAC).

        Se `solid_loc` (n,m bool) for dado, células sólidas são excluídas
        (p=0) e faces sólidas ficam com ṽ·n=0.
        """
        cfg, dt = self.cfg, self.dt
        n, m = cfg.nlocx, cfg.nlocy
        hx, hy = self.hx, self.hy

        rhs = (divw_loc * hx * hy).copy()
        rhs[0, :] += fW * hy
        rhs[-1, :] -= fE * hy
        rhs[:, 0] += fS * hx
        rhs[:, -1] -= fN * hx

        if solid_loc is None:
            b = rhs.flatten(order="F")
            defect = b.sum()
            self.last_defect = max(self.last_defect,
                                   abs(defect) / (hx * hy * n * m))
            b -= defect / b.size
            b[0] = 0.0
            p = self.lu.solve(b).reshape((n, m), order="F")
            solid = np.zeros((n, m), dtype=bool)
        else:
            solid = solid_loc.reshape(n, m)
            lu, pin = self._get_masked(solid)
            rhs[solid] = 0.0                       # fonte nula no sólido
            b = rhs.flatten(order="F")
            fluid = ~solid.flatten(order="F")
            nfluid = int(fluid.sum())
            defect = b[fluid].sum()
            self.last_defect = max(self.last_defect,
                                   abs(defect) / (hx * hy * nfluid))
            b[fluid] -= defect / nfluid            # compatibiliza só no fluido
            b[~fluid] = 0.0
            b[pin] = 0.0
            p = lu.solve(b).reshape((n, m), order="F")
            p[solid] = 0.0

        vt1 = np.empty((n + 1, m))
        vt1[1:-1, :] = dt * (p[1:, :] - p[:-1, :]) / hx
        vt1[0, :], vt1[-1, :] = fW, fE
        vt2 = np.empty((n, m + 1))
        vt2[:, 1:-1] = dt * (p[:, 1:] - p[:, :-1]) / hy
        vt2[:, 0], vt2[:, -1] = fS, fN

        if solid_loc is not None:
            fx = np.zeros((n + 1, m), bool); fx[:-1, :] |= solid; fx[1:, :] |= solid
            fy = np.zeros((n, m + 1), bool); fy[:, :-1] |= solid; fy[:, 1:] |= solid
            vt1[fx] = 0.0
            vt2[fy] = 0.0
        return vt1, vt2
```

- [ ] **Step 4: Rodar os testes até passar**

Run: `pytest tests/test_downscaler_obstacle.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add mprm_core.py tests/test_downscaler_obstacle.py && \
git commit -q -m "feat(mprm): downscaler ciente de obstáculo (máscara sólida em solve_patch)" || echo "(sem repo git)"
```

---

### Task 4: `MPrMProjector` generalizado (BCs por lado, K heterogêneo, fluxo de saída)

**Files:**
- Modify: `mprm_core.py` (`MPrMProjector.__init__` e `.project`)
- Test (regressão via existente): `tests/test_alpha_hh.py` (deve continuar passando)

**Interfaces:**
- Consumes: `FlowProblem` (T1), pacote `mrcm` com API nova (T0).
- Produces:
  - `MPrMProjector(cfg, dt, problem=None, comm=None, verbose=False)` — `problem=None` ⇒ impermeável (comportamento atual, all-no-flow). Com `FlowProblem`: BCs por lado (§4), `K=δt` no fluido e `K=δt*eps_solid` no sólido, extração do traço de fluxo de saída.
  - `self.problem` (o FlowProblem efetivo), `self.eps_solid` (default 1e-6).
  - `project(divw)` continua devolvendo `fluxes` (interfaces) e passa a **também** preencher `self.last_boundary_fluxes` — dict `("out", ix, iy) -> array` com `v·n` reconstruído nas faces de saída de subdomínios de borda.
  - `assert_solid_interior_to_subdomains(solid, cfg)` (módulo) — valida a restrição do obstáculo.

- [ ] **Step 1: Teste de regressão — projetor impermeável inalterado**

Adicionar a `tests/test_downscaler_obstacle.py` NÃO; criar um teste leve de projetor em novo arquivo `tests/test_projector.py` (usa firedrake):
```python
"""Projetor MRCM: regressão do caminho impermeável + validação do
assert de obstáculo interior. (usa firedrake — mais lento)"""
import numpy as np
import pytest

from mrcm.config import MRCMConfig
from mprm_core import (MPrMProjector, assert_solid_interior_to_subdomains,
                       von_karman_problem)


def _cfg(Ndom=2, nloc=8, dU=1, dP=1):
    return MRCMConfig(Ndomx=Ndom, Ndomy=Ndom, nlocx=nloc, nlocy=nloc,
                      degreeU=dU, degreeP=dP, alpha=1.0,
                      basis_kind="legendre_P",
                      mixed_family="BDM", mixed_order=0,
                      executor_kind="sequential")


def test_projector_impermeavel_problem_none_roda():
    cfg = _cfg()
    proj = MPrMProjector(cfg, dt=1.0)          # problem=None -> impermeável
    assert proj.problem.is_impermeable
    divw = np.zeros((cfg.Nglobx, cfg.Ngloby))
    fluxes = proj.project(divw)
    # fonte nula -> todos os fluxos de interface ~ 0
    for arr in fluxes.values():
        assert np.allclose(arr, 0.0, atol=1e-8)


def test_assert_obstaculo_interior():
    cfg = _cfg(Ndom=2, nloc=8)   # subdomínios de 8 células; interface em x=8
    # obstáculo que ATRAVESSA a interface central deve falhar
    solid = np.zeros((16, 16), dtype=bool)
    solid[7:10, 6:10] = True     # cruza a coluna 8 (interface)
    with pytest.raises(ValueError):
        assert_solid_interior_to_subdomains(solid, cfg)
    # obstáculo contido no subdomínio inferior-esquerdo: ok
    solid2 = np.zeros((16, 16), dtype=bool)
    solid2[2:5, 2:5] = True
    assert_solid_interior_to_subdomains(solid2, cfg)  # não levanta
```

- [ ] **Step 2: Rodar para confirmar a falha**

Run: `pytest tests/test_projector.py -q`
Expected: FALHA (`cannot import name 'assert_solid_interior_to_subdomains'`).

- [ ] **Step 3: Implementar a generalização do projetor**

Em `mprm_core.py`, adicionar a função de validação (perto dos construtores de problema):
```python
def assert_solid_interior_to_subdomains(solid, cfg):
    """Garante que o obstáculo não atravessa interfaces de subdomínio:
    nenhuma face de interface (colunas i*nlocx, linhas j*nlocy) pode separar
    fluido de sólido de forma inconsistente — exigimos que cada célula sólida
    e suas vizinhas na direção normal à interface estejam no mesmo subdomínio.
    Critério prático: nenhuma interface interna corta o bounding-box do sólido.
    """
    if solid is None or not solid.any():
        return
    nx, ny = cfg.nlocx, cfg.nlocy
    xs = np.where(solid.any(axis=1))[0]
    ys = np.where(solid.any(axis=0))[0]
    imin, imax = xs.min(), xs.max() + 1        # [imin, imax) em células
    jmin, jmax = ys.min(), ys.max() + 1
    for i in range(1, cfg.Ndomx):
        if imin < i * nx < imax:
            raise ValueError(
                f"Obstáculo atravessa a interface vertical x-cel={i*nx} "
                f"(bbox x=[{imin},{imax})). Alinhe o obstáculo ao interior "
                f"de um subdomínio (spec §5.3, R4).")
    for j in range(1, cfg.Ndomy):
        if jmin < j * ny < jmax:
            raise ValueError(
                f"Obstáculo atravessa a interface horizontal y-cel={j*ny} "
                f"(bbox y=[{jmin},{jmax})).")
```

No `MPrMProjector.__init__`, mudar a assinatura e a construção do `DarcyProblem`. Substituir o cabeçalho e o bloco do problema (linhas ~201–247) por:

```python
    def __init__(self, cfg: MRCMConfig, dt: float, problem=None,
                 comm=None, verbose=False, eps_solid=1e-6):
        from firedrake import (Constant, FunctionSpace, TestFunction,
                               TestFunctions, Function, FacetNormal,
                               VectorFunctionSpace, SpatialCoordinate,
                               assemble, dot)
        from mrcm.topology import SIDE_TO_MARKER

        if comm is None:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
        self.comm = comm
        self.rank, self.size = comm.rank, comm.size
        self.cfg = cfg
        self.dt = float(dt)
        self.verbose = verbose
        self.eps_solid = float(eps_solid)

        # problem=None => cavidade impermeável (comportamento retrocompatível)
        if problem is None:
            problem = _impermeable_problem(cfg)
        self.problem = problem
        if problem.solid is not None:
            assert_solid_interior_to_subdomains(problem.solid, cfg)

        self.skel = Skeleton.cartesian(cfg.Ndomx, cfg.Ndomy)
        self.dofm = DOFManager(self.skel.NGamma, cfg.ndofsU, cfg.ndofsP)
        self.my_subs = list(range(self.rank, cfg.Nsubd, self.size))

        dt_c = self.dt
        eps = self.eps_solid
        solid = problem.solid
        hx_f = cfg.Hx / cfg.nlocx
        hy_f = cfg.Hy / cfg.nlocy
        self.hx_f, self.hy_f = hx_f, hy_f

        def _K(mesh):
            from firedrake import (Constant, Function, FunctionSpace,
                                   VectorFunctionSpace, SpatialCoordinate)
            if solid is None:
                return Constant(dt_c)
            Kf = FunctionSpace(mesh, "DG", 0)
            # subdomínio tem malha em coordenadas GLOBAIS (como o cell_map
            # existente): floor(x/hx_f) dá o índice de célula global.
            xc = Function(VectorFunctionSpace(mesh, "DG", 0)).interpolate(
                SpatialCoordinate(mesh))
            xcd = xc.dat.data_ro
            ic = np.clip(np.floor(xcd[:, 0] / hx_f).astype(int),
                         0, cfg.Nglobx - 1)
            jc = np.clip(np.floor(xcd[:, 1] / hy_f).astype(int),
                         0, cfg.Ngloby - 1)
            k = Function(Kf)
            k.dat.data[:] = np.where(solid[ic, jc], dt_c * eps, dt_c)
            return k

        def _invK(mesh):
            from firedrake import Constant, Function
            if solid is None:
                return Constant(1.0 / dt_c)
            K = _K(mesh)
            ik = Function(K.function_space())
            ik.dat.data[:] = 1.0 / K.dat.data_ro
            return ik

        def _zero(x, y):
            from firedrake import Constant
            return Constant(0.0)

        self.problem_darcy = DarcyProblem(
            Lx=cfg.Lx, Ly=cfg.Ly,
            permeability=_K, inv_permeability=_invK,
            source=_zero,
            boundary=_darcy_boundary_from_problem(problem),
        )
```

Depois, no restante do `__init__`, **trocar todas as referências** `self.problem` que apontavam ao `DarcyProblem` por `self.problem_darcy`:
- na task dict: `"problem": self.problem_darcy`;
- na chamada de BCs (linha ~288), trocar o monkey-patch pela API nativa:
  ```python
  from mrcm.local_form import build_flux_dirichlet_bcs
  bcs = build_flux_dirichlet_bcs(W, mesh, flag_side, problem=self.problem_darcy)
  ```
- em `beta_expr`/`build_local_form`, usar `_invK(mesh)` (já usa).

Adicionar, ainda no `__init__`, a montagem dos operadores de traço de **saída** (após o laço que monta `self.trace` das interfaces): para cada subdomínio na coluna de saída, montar o traço `HDiv Trace` na faceta externa do lado `outflow`. Inserir dentro do laço `for isub in self.my_subs:`, ao lado do bloco de traço de interfaces:
```python
            # traço de fluxo nas faces de SAÍDA (lado outflow externo)
            self.out_trace = getattr(self, "out_trace", {})
            for s_name, marker in (("left", 1), ("right", 2),
                                   ("bottom", 3), ("top", 4)):
                if problem.side(s_name).kind != "outflow":
                    continue
                flag_idx = {"bottom": 0, "right": 1, "top": 2, "left": 3}[s_name]
                if flag_side[flag_idx]:      # só bordas externas reais
                    continue
                ln = assemble(tau * ds_m(marker)).dat.data_ro.copy()
                idx = np.nonzero(np.abs(ln) > 1e-14)[0]
                if idx.size == 0:
                    continue
                cm = (assemble(SpatialCoordinate(mesh)[1] * tau * ds_m(marker))
                      .dat.data_ro[idx] / ln[idx]) if s_name in ("left", "right") \
                    else (assemble(SpatialCoordinate(mesh)[0] * tau * ds_m(marker))
                          .dat.data_ro[idx] / ln[idx])
                order = np.argsort(cm)
                sign = +1.0 if s_name in ("right", "top") else -1.0
                self.out_trace[(isub, s_name)] = {
                    "form": dot(uh, n_facet) * tau * ds_m(marker),
                    "idx": idx, "len": ln[idx], "order": order, "sign": sign}
```

Em `project(...)`, após o laço de reconstrução que preenche `my_traces`, extrair também os fluxos de saída e publicá-los. Antes do `return fluxes`, inserir:
```python
        # fluxos de saída (v·n reconstruído nas faces externas outflow)
        my_out = {}
        for isub in self.my_subs:
            self.uh[isub].assign(self.last_uT[isub])
            for (jsub, s_name), tr in getattr(self, "out_trace", {}).items():
                if jsub != isub:
                    continue
                vals = assemble(tr["form"]).dat.data_ro[tr["idx"]]
                my_out[(isub, s_name)] = tr["sign"] * (vals / tr["len"])[tr["order"]]
        self.last_boundary_fluxes = {}
        for d in comm.allgather(my_out):
            self.last_boundary_fluxes.update(d)
```

Adicionar os dois helpers de módulo (perto de `assert_solid_interior_to_subdomains`):
```python
def _impermeable_problem(cfg) -> "FlowProblem":
    """FlowProblem impermeável (4 paredes, sem obstáculo) coerente com o cfg —
    o comportamento retrocompatível do projetor/downscaler."""
    sides = {s: SideBC("wall", velocity=0.0)
             for s in ("left", "right", "top", "bottom")}
    return FlowProblem(Nx=cfg.Nglobx, Ny=cfg.Ngloby,
                       Lx=cfg.Lx, Ly=cfg.Ly, nu=0.0, sides=sides)


def _darcy_boundary_from_problem(problem) -> dict:
    """Mapeia os lados do FlowProblem para BoundaryData do Darcy da projeção
    (spec §4): entrada/paredes -> flux=0 (no-flow forte); saída -> pressure=0."""
    def _zero(x, y):
        from firedrake import Constant
        return Constant(0.0)
    bnd = {}
    for s, bc in problem.sides.items():
        if bc.kind == "outflow":
            bnd_val = BoundaryData(pressure=_zero)
        else:                       # wall, slip, inflow -> v·n = 0
            bnd_val = BoundaryData(flux=_zero)
        bnd[s] = bnd_val
    return bnd
```

- [ ] **Step 4: Rodar o teste do projetor até passar**

Run: `pytest tests/test_projector.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Regressão do caminho impermeável (HH inalterado)**

Run:
```bash
python -c "import sys; sys.path.insert(0,'.'); sys.path.insert(0,'$HOME/mrcm/mrcm-firedrake'); \
from mprm_core import hh_benchmark_mac; r=hh_benchmark_mac(Ndom=3, nloc=16, degreeU=1, degreeP=1); \
print('err_u=%.4e divmax=%.1e defect=%.1e' % (r['err_u'], r['divmax'], r['defect'])); \
assert r['divmax'] < 1e-8 and r['err_u'] < 0.3"
```
Expected: `err_u ~ 2.3e-1` (grau 1), `divmax < 1e-8` — mesmos números de antes do refactor (o caminho `problem=None` é idêntico).

- [ ] **Step 6: Commit**

```bash
git add mprm_core.py tests/test_projector.py && \
git commit -q -m "feat(mprm): projetor com BCs por lado, K heterogêneo (obstáculo) e fluxo de saída; remove patch" || echo "(sem repo git)"
```

---

### Task 5: `mprm_projection_step` + `run_flow` + `run_cavity` (wrapper) + `run_von_karman`

**Files:**
- Modify: `mprm_core.py` (`mprm_projection_step`, `run_cavity`; adicionar `run_flow`, `run_von_karman`)
- Test: regressão da cavidade

**Interfaces:**
- Consumes: `mac_transport`, `MPrMProjector(...,problem)`, `MACDownscaler.solve_patch(...,solid_loc)`, `FineProjector`.
- Produces:
  - `mprm_projection_step(w1, w2, projector, downscaler)` — assinatura inalterada; lê `projector.problem` (obstáculo + fluxos de saída).
  - `run_flow(problem, Ndom, nloc, T, dt=None, method="mprm", degreeU=1, degreeP=1, alpha=1.0, basis_kind="legendre_P", eps_solid=1e-6, comm=None, verbose=True, log_every=200, frame_callback=None, frame_every=0, probes=None, obstacle_metrics=False, U_ref=1.0, D=1.0) -> dict` (inclui `cl_history`/`probe_history` quando há obstáculo/probes).
  - `run_cavity(...)` — assinatura e retorno inalterados (wrapper).
  - `run_von_karman(Lx, Ly, D, xc, yc, U_in=1.0, Re=100.0, Ndom=None, nloc=None, degreeU=1, degreeP=3, alpha=1.0, basis_kind="legendre_P", T=None, dt=None, method="mprm", inflow="uniform", walls="wall", eps_solid=1e-6, comm=None, verbose=True, log_every=200, probes=None, frame_callback=None, frame_every=0) -> dict` — `Lx,Ly,D,xc,yc` obrigatórios; `Ndom,nloc` obrigatórios (validados por assert).

- [ ] **Step 1: Escrever teste de regressão da cavidade (end-to-end)**

Create `tests/test_run_regression.py`:
```python
"""Regressão end-to-end: run_cavity --quick e run_von_karman smoke."""
import numpy as np
import pytest

from mprm_core import run_cavity, run_von_karman
from mprm_core import mac_divergence


def test_run_cavity_quick_conserva_massa():
    r = run_cavity(Re=100.0, Ndom=3, nloc=8, T=0.2, degreeU=1, degreeP=1,
                   basis_kind="legendre_P", method="mprm", verbose=False)
    assert r["divmax"] < 1e-10           # div ~ machine precision (cavidade)
    assert np.isfinite(r["u"]).all()


def test_run_cavity_fine_vs_mprm_proximos():
    common = dict(Re=100.0, Ndom=3, nloc=8, T=0.2, verbose=False)
    a = run_cavity(method="mprm", degreeU=1, degreeP=1,
                   basis_kind="legendre_P", **common)
    b = run_cavity(method="fine", **common)
    num = np.sum((a["u"] - b["u"])**2) + np.sum((a["v"] - b["v"])**2)
    den = np.sum(b["u"]**2) + np.sum(b["v"]**2)
    assert np.sqrt(num / den) < 5e-2     # concordância grosseira em T curto
```

- [ ] **Step 2: Rodar (a cavidade já deve funcionar; von Kármán ainda não)**

Run: `pytest tests/test_run_regression.py -q`
Expected: os testes de cavidade PASSAM (o caminho já existe); se falharem, é regressão do refactor — corrigir antes de seguir.

- [ ] **Step 3: Generalizar `mprm_projection_step` e adicionar `run_flow`/`run_von_karman`; refatorar `run_cavity`**

Substituir `mprm_projection_step` (linhas ~508–549) por:
```python
def mprm_projection_step(w1, w2, projector: "MPrMProjector",
                         downscaler: "MACDownscaler"):
    """Passos 2-4 do MPrM. Lê `projector.problem` (obstáculo + saída).
    Assinatura inalterada — retrocompatível."""
    cfg = projector.cfg
    comm = projector.comm
    problem = projector.problem
    n, m = cfg.nlocx, cfg.nlocy
    hx, hy = downscaler.hx, downscaler.hy
    solid = problem.solid

    divw = mac_divergence(w1, w2, hx, hy)
    if solid is not None:
        divw = divw.copy(); divw[solid] = 0.0
    fluxes = projector.project(divw)
    out_fluxes = getattr(projector, "last_boundary_fluxes", {})

    u = w1.copy(); v = w2.copy()
    zeros_m, zeros_n = np.zeros(m), np.zeros(n)
    my_patch = {}
    for isub in projector.my_subs:
        ix, iy = isub % cfg.Ndomx, isub // cfg.Ndomx
        ox, oy = ix * n, iy * m
        fW = fluxes.get(("v", ix, iy), zeros_m)
        fE = fluxes.get(("v", ix + 1, iy), zeros_m)
        fS = fluxes.get(("h", ix, iy), zeros_n)
        fN = fluxes.get(("h", ix, iy + 1), zeros_n)
        # fluxos externos de SAÍDA (v·n reconstruído nas faces outflow);
        # entrada/paredes/obstáculo permanecem 0.
        if (isub, "right") in out_fluxes:
            fE = out_fluxes[(isub, "right")]
        if (isub, "left") in out_fluxes:
            fW = out_fluxes[(isub, "left")]
        if (isub, "top") in out_fluxes:
            fN = out_fluxes[(isub, "top")]
        if (isub, "bottom") in out_fluxes:
            fS = out_fluxes[(isub, "bottom")]
        solid_loc = None
        if solid is not None:
            sl = solid[ox:ox + n, oy:oy + m]
            solid_loc = sl if sl.any() else None
        vt1, vt2 = downscaler.solve_patch(
            divw[ox:ox + n, oy:oy + m], fW, fE, fS, fN, solid_loc=solid_loc)
        my_patch[isub] = (vt1, vt2)

    for d in comm.allgather(my_patch):
        for isub, (vt1, vt2) in d.items():
            ix, iy = isub % cfg.Ndomx, isub // cfg.Ndomx
            ox, oy = ix * n, iy * m
            u[ox:ox + n + 1, oy:oy + m] = w1[ox:ox + n + 1, oy:oy + m] - vt1
            v[ox:ox + n, oy:oy + m + 1] = w2[ox:ox + n, oy:oy + m + 1] - vt2

    # contorno externo por lado (impermeável => 0; saída/entrada preservam w*)
    _apply_external_normals_after_step(u, v, problem)
    if solid is not None:
        fx, fy = solid_faces(solid)
        u[fx] = 0.0; v[fy] = 0.0
    return u, v


def _apply_external_normals_after_step(u, v, problem):
    """Após uⁿ⁺¹=w*−ṽ: paredes/slip impõem u·n=0; entrada e saída preservam
    o que veio de w*−ṽ (entrada: ṽ·n=0 ⇒ u·n=perfil; saída: corrigido)."""
    S = problem.sides
    if S["left"].kind in ("wall", "slip"):
        u[0, :] = 0.0
    if S["right"].kind in ("wall", "slip"):
        u[-1, :] = 0.0
    if S["bottom"].kind in ("wall", "slip"):
        v[:, 0] = 0.0
    if S["top"].kind in ("wall", "slip"):
        v[:, -1] = 0.0
```

Adicionar `run_flow` e refatorar `run_cavity` como wrapper. Substituir `run_cavity` (linhas ~600–671) por:
```python
def run_flow(problem, Ndom, nloc, T, dt=None, method="mprm",
             degreeU=1, degreeP=1, alpha=1.0, basis_kind="legendre_P",
             eps_solid=1e-6, comm=None, verbose=True, log_every=200,
             frame_callback=None, frame_every=0, probes=None,
             obstacle_metrics=False, U_ref=1.0, D=1.0):
    """Laço temporal genérico do MPrM para um FlowProblem. Cavidade e von
    Kármán compartilham este caminho. Devolve dict(u,v,dt,nsteps,divmax,
    defect,t_wall[,cl_history,probe_history])."""
    import time
    if comm is None:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD

    Nx, Ny = problem.Nx, problem.Ny
    hx, hy = problem.hx, problem.hy
    if dt is None:
        umax = max(U_ref, 1e-12)
        nu = max(problem.nu, 1e-12)
        dt = 0.4 * min(hx / umax, hx * hx / (4.0 * nu))
    nsteps = int(np.ceil(T / dt))

    if method == "mprm":
        cfg = MRCMConfig(Ndomx=Ndom, Ndomy=Ndom, nlocx=nloc, nlocy=nloc,
                         degreeU=degreeU, degreeP=degreeP, alpha=alpha,
                         basis_kind=basis_kind, Lx=problem.Lx, Ly=problem.Ly,
                         mixed_family="BDM", mixed_order=0,
                         executor_kind="sequential")
        projector = MPrMProjector(cfg, dt, problem=problem, comm=comm,
                                  eps_solid=eps_solid)
        downscaler = MACDownscaler(cfg, dt)
    else:
        fine = ChannelFinePoisson(problem, dt)
        projector = None

    u = np.zeros((Nx + 1, Ny)); v = np.zeros((Nx, Ny + 1))
    # inicializa a entrada em w* (campo inicial = perfil de entrada nas faces)
    w1_0, w2_0 = mac_transport(u, v, problem, dt)
    u = _seed_inflow(u, w1_0, problem)

    cl_hist = []
    probe_hist = [] if probes else None
    t0 = time.perf_counter()
    for step in range(nsteps):
        w1, w2 = mac_transport(u, v, problem, dt)
        if method == "mprm":
            u, v = mprm_projection_step(w1, w2, projector, downscaler)
        else:
            u, v = fine.step(w1, w2)

        if comm.rank == 0:
            if obstacle_metrics and problem.solid is not None:
                p = _diagnostic_pressure(u, v, problem, dt)
                cd, cl = obstacle_forces(u, v, p, problem, U_ref, D)
                cl_hist.append((dt * (step + 1), cd, cl))
            if probes:
                probe_hist.append(
                    [dt * (step + 1)] + [_sample_probe(u, v, problem, px, py)
                                         for (px, py) in probes])
        if frame_every > 0 and frame_callback is not None and comm.rank == 0:
            if step == 0 or (step + 1) % frame_every == 0:
                frame_callback(step, dt * (step + 1), u, v, hx, hy)
        if verbose and comm.rank == 0 and (step + 1) % log_every == 0:
            dv = np.abs(mac_divergence(u, v, hx, hy))
            dv = dv[~problem.solid].max() if problem.solid is not None else dv.max()
            print(f"  passo {step+1}/{nsteps}  t={dt*(step+1):.3f}  "
                  f"max|div u| = {dv:.2e}", flush=True)
    t_wall = time.perf_counter() - t0

    dvall = np.abs(mac_divergence(u, v, hx, hy))
    divmax = float(dvall[~problem.solid].max() if problem.solid is not None
                   else dvall.max())
    out = {"u": u, "v": v, "dt": dt, "nsteps": nsteps, "N": Nx,
           "t_wall": t_wall, "divmax": divmax,
           "defect": (downscaler.last_defect if method == "mprm" else 0.0)}
    if cl_hist:
        out["cl_history"] = np.array(cl_hist)     # colunas: t, Cd, Cl
    if probe_hist is not None:
        out["probe_history"] = np.array(probe_hist)
    return out


def _seed_inflow(u, w1_0, problem):
    """Semeia a face de entrada no campo inicial (evita transiente de partida
    com entrada nula)."""
    if problem.side("left").kind == "inflow":
        u = u.copy(); u[0, :] = w1_0[0, :]
    return u


def _sample_probe(u, v, problem, px, py):
    """v-velocidade interpolada no ponto (px,py) — sonda de esteira p/ Strouhal."""
    hx, hy = problem.hx, problem.hy
    j = min(int(py / hy), problem.Ny - 1)
    i = min(int(px / hx), problem.Nx - 1)
    return 0.5 * (v[i, j] + v[i, j + 1])


def run_cavity(Re=100.0, Ndom=3, nloc=16, T=8.0, dt=None, lid_velocity=1.0,
               degreeU=1, degreeP=1, alpha=1.0, method="mprm",
               basis_kind="nodal_Q", comm=None, verbose=True, log_every=200,
               frame_callback=None, frame_every=0):
    """Cavidade com tampa deslizante (wrapper sobre run_flow). Assinatura e
    retorno preservados para compatibilidade com testes/notebook."""
    N = Ndom * nloc
    problem = cavity_problem(N, Re, lid_velocity=lid_velocity)
    if dt is None:
        hx = 1.0 / N
        dt = 0.4 * min(hx / lid_velocity, hx * hx / (4.0 * (lid_velocity / Re)))
    return run_flow(problem, Ndom, nloc, T, dt=dt, method=method,
                    degreeU=degreeU, degreeP=degreeP, alpha=alpha,
                    basis_kind=basis_kind, comm=comm, verbose=verbose,
                    log_every=log_every, frame_callback=frame_callback,
                    frame_every=frame_every, U_ref=lid_velocity)


def run_von_karman(Lx, Ly, D, xc, yc, U_in=1.0, Re=100.0,
                   Ndom=None, nloc=None, degreeU=1, degreeP=3, alpha=1.0,
                   basis_kind="legendre_P", T=None, dt=None, method="mprm",
                   inflow="uniform", walls="wall", eps_solid=1e-6,
                   comm=None, verbose=True, log_every=200,
                   probes=None, frame_callback=None, frame_every=0):
    """Escoamento em torno de um cilindro quadrado (von Kármán, artigo §3.5).

    Malha global Nx=Ndom_x*nloc, Ny=Ndom_y*nloc — aqui usamos Ndom (por
    direção) e nloc iguais; ajuste `nloc`/`Ndom` para caber o domínio
    retangular (Nx/Ny ≈ Lx/Ly). Espaços default (degreeU=1,degreeP=3)
    seguem a recomendação do artigo p/ escoamento transiente (pressão rica).
    Devolve o dict de run_flow + cl_history (t,Cd,Cl) e probe_history.
    """
    assert Ndom is not None and nloc is not None, "Ndom e nloc obrigatórios"
    Nx = Ny = Ndom * nloc
    # ajusta a razão de células ao domínio (mantém subdomínios quadrados em H)
    problem = von_karman_problem(Lx, Ly, Nx, Ny, D, xc, yc, U_in, Re,
                                 inflow=inflow, walls=walls)
    if T is None:
        T = 40.0 * D / U_in
    if probes is None:
        probes = [(xc + 3.0 * D, yc)]      # sonda de esteira default
    return run_flow(problem, Ndom, nloc, T, dt=dt, method=method,
                    degreeU=degreeU, degreeP=degreeP, alpha=alpha,
                    basis_kind=basis_kind, eps_solid=eps_solid, comm=comm,
                    verbose=verbose, log_every=log_every, probes=probes,
                    obstacle_metrics=True, U_ref=U_in, D=D,
                    frame_callback=frame_callback, frame_every=frame_every)
```

Nota de implementação: `run_flow` **referencia** `ChannelFinePoisson`, `obstacle_forces`, `_diagnostic_pressure` (Task 6), mas só os **chama** nos caminhos `method="fine"` ou `obstacle_metrics=True` — que os testes de cavidade deste task (`method="mprm"`, sem obstáculo) NÃO exercitam. Como Python resolve nomes em tempo de chamada, `run_flow` importa/define sem erro mesmo antes do Task 6. **Nenhum stub é necessário**; basta implementar o Task 6 antes de rodar von Kármán (Task 7). O helper `_apply_external_normals_after_step` é definido AQUI (Task 5) e reutilizado pelo `ChannelFinePoisson` no Task 6.

- [ ] **Step 4: Rodar a regressão da cavidade (deve continuar verde)**

Run: `pytest tests/test_run_regression.py::test_run_cavity_quick_conserva_massa tests/test_run_regression.py::test_run_cavity_fine_vs_mprm_proximos -q`
Expected: PASS. Se `run_cavity` regrediu, corrigir antes de seguir.

- [ ] **Step 5: Verificar callers legados (run_mprm, testes HH)**

Run:
```bash
pytest tests/test_alpha_hh.py -q 2>&1 | tail -5 || true
python -c "import sys; sys.path.insert(0,'.'); sys.path.insert(0,'$HOME/mrcm/mrcm-firedrake'); \
import mprm_core; from mprm_core import run_cavity; r=run_cavity(Ndom=2,nloc=8,T=0.1,verbose=False); \
print('run_cavity legado OK, divmax=%.1e' % r['divmax'])"
```
Expected: import de `run_mprm`/testes HH sem erro de assinatura; `run_cavity legado OK`.

- [ ] **Step 6: Commit**

```bash
git add mprm_core.py tests/test_run_regression.py && \
git commit -q -m "feat(mprm): run_flow genérico + run_von_karman + step ciente de obstáculo/saída; run_cavity vira wrapper" || echo "(sem repo git)"
```

---

### Task 6: Referência fina masc. + forças + Strouhal (`ChannelFinePoisson`, `obstacle_forces`, `strouhal`)

**Files:**
- Modify: `mprm_core.py` (adicionar as 3 peças + `_diagnostic_pressure`)
- Create: `tests/test_forces_strouhal.py`

**Interfaces:**
- Produces:
  - `class ChannelFinePoisson(problem, dt)` com `.step(w1,w2)->(u,v)` e `.solve(w1,w2)->(u,v,p)` — Poisson MAC global mascarado (entrada/paredes Neumann 0, saída Dirichlet p=0, obstáculo excluído).
  - `_diagnostic_pressure(u, v, problem, dt) -> p (Nx,Ny)` — pressão física do passo (resolve o Poisson de projeção da velocidade atual).
  - `obstacle_forces(u, v, p, problem, U_ref, D) -> (Cd, Cl)` — balanço de momento em volume de controle no bounding-box do obstáculo.
  - `strouhal(signal, dt, U_ref, D, t_skip_frac=0.5) -> float` — FFT do sinal (descarta transiente).

- [ ] **Step 1: Escrever os testes que falham**

Create `tests/test_forces_strouhal.py`:
```python
"""Forças (balanço de momento) e Strouhal (FFT) — validação analítica."""
import numpy as np
import pytest

from mprm_core import (ChannelFinePoisson, obstacle_forces, strouhal,
                       von_karman_problem, mac_divergence)


def test_strouhal_recupera_frequencia_conhecida():
    U, D = 1.0, 1.0
    f_true = 0.121 * U / D              # St alvo do artigo
    dt = 0.01
    t = np.arange(0, 200, dt)
    sig = np.sin(2 * np.pi * f_true * t) + 0.05 * np.random.default_rng(0).standard_normal(t.size)
    st = strouhal(sig, dt, U, D, t_skip_frac=0.25)
    assert st == pytest.approx(0.121, abs=0.01)


def test_channel_fine_poisson_projeta_divergencia_zero():
    prob = von_karman_problem(Lx=8.0, Ly=4.0, Nx=32, Ny=16, D=1.0,
                              xc=2.0, yc=2.0, U_in=1.0, Re=100.0)
    fp = ChannelFinePoisson(prob, dt=0.05)
    rng = np.random.default_rng(1)
    w1 = rng.standard_normal((33, 16)); w2 = rng.standard_normal((32, 17))
    w1[0, :] = 1.0                                    # entrada
    u, v = fp.step(w1, w2)
    dv = np.abs(mac_divergence(u, v, prob.hx, prob.hy))
    assert dv[~prob.solid].max() < 1e-8               # div~0 no fluido
    # u·n=0 nas faces sólidas
    fx = np.zeros((33, 16), bool); fx[:-1, :] |= prob.solid; fx[1:, :] |= prob.solid
    assert np.all(u[fx] == 0.0)


def test_obstacle_forces_campo_uniforme_forca_nula():
    prob = von_karman_problem(Lx=8.0, Ly=4.0, Nx=32, Ny=16, D=1.0,
                              xc=2.0, yc=2.0, U_in=1.0, Re=100.0)
    # campo uniforme fora do sólido, pressão constante: forças ~ 0
    u = np.ones((33, 16)); v = np.zeros((32, 17)); p = np.zeros((32, 16))
    fx = np.zeros((33, 16), bool); fx[:-1, :] |= prob.solid; fx[1:, :] |= prob.solid
    fy = np.zeros((32, 17), bool); fy[:, :-1] |= prob.solid; fy[:, 1:] |= prob.solid
    u[fx] = 0.0; v[fy] = 0.0
    cd, cl = obstacle_forces(u, v, p, prob, U_ref=1.0, D=1.0)
    assert abs(cl) < 1e-9
    assert np.isfinite(cd)
```

- [ ] **Step 2: Rodar para confirmar a falha**

Run: `pytest tests/test_forces_strouhal.py -q`
Expected: FALHA (`cannot import name 'ChannelFinePoisson'`).

- [ ] **Step 3: Implementar as três peças**

Adicionar ao `mprm_core.py`:
```python
# ===========================================================================
# Referência fina mascarada + diagnósticos (von Kármán)
# ===========================================================================

class ChannelFinePoisson:
    """Poisson MAC global mascarado para o passo de projeção de referência.

    Entrada/paredes/slip: Neumann homogêneo (v·n=0). Saída: Dirichlet p=0.
    Obstáculo: células sólidas excluídas (p=0), faces sólidas no-flow.
    Serve (a) como referência fina de von Kármán e (b) como fonte de pressão
    física para as forças.
    """

    def __init__(self, problem, dt):
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        self.problem = problem
        self.dt = float(dt)
        Nx, Ny = problem.Nx, problem.Ny
        hx, hy = problem.hx, problem.hy
        self.Nx, self.Ny, self.hx, self.hy = Nx, Ny, hx, hy
        solid = problem.solid if problem.solid is not None \
            else np.zeros((Nx, Ny), bool)
        self.solid = solid
        has_out = {s: problem.side(s).kind == "outflow"
                   for s in ("left", "right", "bottom", "top")}
        self.has_out = has_out

        def lid(i, j):
            return i + j * Nx

        rows, cols, vals = [], [], []
        for j in range(Ny):
            for i in range(Nx):
                c = lid(i, j)
                if solid[i, j]:
                    rows.append(c); cols.append(c); vals.append(1.0); continue
                diag = 0.0
                # vizinhos internos (fluido)
                for (di, dj, d, ln) in ((1, 0, hx, hy), (-1, 0, hx, hy),
                                        (0, 1, hy, hx), (0, -1, hy, hx)):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < Nx and 0 <= jj < Ny and not solid[ii, jj]:
                        w = dt * ln / d
                        rows.append(c); cols.append(lid(ii, jj))
                        vals.append(w); diag -= w
                # faces de SAÍDA (Dirichlet p=0): fantasma p=0 a meia célula
                for (side, at_edge, d, ln) in (
                        ("left", i == 0, hx, hy), ("right", i == Nx - 1, hx, hy),
                        ("bottom", j == 0, hy, hx), ("top", j == Ny - 1, hy, hx)):
                    if at_edge and has_out[side]:
                        diag -= dt * ln / (0.5 * d)     # p_ghost=0 (Dirichlet)
                rows.append(c); cols.append(c); vals.append(diag)
        A = sp.csr_matrix((vals, (rows, cols)), shape=(Nx * Ny, Nx * Ny)).tolil()
        self._pure_neumann = not any(has_out.values())
        if self._pure_neumann:
            pin = int(np.argmin(solid.flatten(order="F")))
            A[pin, :] = 0.0; A[pin, pin] = 1.0
            self._pin = pin
        self.lu = spla.splu(A.tocsc())

    def solve(self, w1, w2):
        dt, hx, hy = self.dt, self.hx, self.hy
        Nx, Ny, solid = self.Nx, self.Ny, self.solid
        divw = mac_divergence(w1, w2, hx, hy)
        divw = divw.copy(); divw[solid] = 0.0
        b = (divw * hx * hy).flatten(order="F")
        b[solid.flatten(order="F")] = 0.0
        if self._pure_neumann:
            fluid = ~solid.flatten(order="F")
            b[fluid] -= b[fluid].sum() / fluid.sum()
            b[self._pin] = 0.0
        p = self.lu.solve(b).reshape((Nx, Ny), order="F")
        p[solid] = 0.0
        u = w1.copy(); v = w2.copy()
        u[1:-1, :] -= dt * (p[1:, :] - p[:-1, :]) / hx
        v[:, 1:-1] -= dt * (p[:, 1:] - p[:, :-1]) / hy
        # saída: gradiente de p do lado da célula de borda; entrada/parede v·n
        _apply_external_normals_after_step(u, v, self.problem)
        fx, fy = solid_faces(solid)
        u[fx] = 0.0; v[fy] = 0.0
        return u, v, p

    def step(self, w1, w2):
        u, v, _ = self.solve(w1, w2)
        return u, v


def _diagnostic_pressure(u, v, problem, dt):
    """Pressão física do instante: projeta a velocidade atual (já ~solenoidal)
    num Poisson mascarado e devolve p (= potencial/δt·δt ... aqui p tal que
    δt∇²p=∇·w*). Usa ChannelFinePoisson sobre (u,v) como w*."""
    fp = ChannelFinePoisson(problem, dt)
    _, _, p = fp.solve(u, v)
    return p


def obstacle_forces(u, v, p, problem, U_ref, D):
    """C_D, C_L por balanço de momento num volume de controle retangular que
    envolve o obstáculo (uma célula de folga). Fluxo convectivo + pressão +
    viscoso nas 4 faces do VC. ρ=1; C = 2F/(ρ U² D)."""
    solid = problem.solid
    if solid is None or not solid.any():
        return 0.0, 0.0
    hx, hy, nu = problem.hx, problem.hy, problem.nu
    Nx, Ny = problem.Nx, problem.Ny
    xs = np.where(solid.any(axis=1))[0]; ys = np.where(solid.any(axis=0))[0]
    i0, i1 = max(xs.min() - 1, 0), min(xs.max() + 1, Nx - 1)   # células do VC
    j0, j1 = max(ys.min() - 1, 0), min(ys.max() + 1, Ny - 1)

    # velocidade no centro das células
    uc = 0.5 * (u[:-1, :] + u[1:, :])
    vc = 0.5 * (v[:, :-1] + v[:, 1:])

    Fx = 0.0; Fy = 0.0
    # faces verticais (esquerda i0, direita i1) — normal ±x
    for i, s in ((i0, -1.0), (i1 + 1, +1.0)):     # i = face index (u-face)
        ii = min(max(i, 0), Nx)
        un = u[ii, j0:j1 + 1]                      # u·n (x) na face
        uc_f = un
        vc_f = 0.5 * (vc[min(ii, Nx - 1), j0:j1 + 1] + vc[max(ii - 1, 0), j0:j1 + 1])
        pf = 0.5 * (p[min(ii, Nx - 1), j0:j1 + 1] + p[max(ii - 1, 0), j0:j1 + 1])
        # dp/dt absorvido no p; força no corpo = -(fluxo conv + pressão - visc)
        Fx += -s * np.sum((uc_f * un + pf) * hy)
        Fy += -s * np.sum((vc_f * un) * hy)
    # faces horizontais (baixo j0, cima j1) — normal ±y
    for j, s in ((j0, -1.0), (j1 + 1, +1.0)):
        jj = min(max(j, 0), Ny)
        vn = v[i0:i1 + 1, jj]
        vc_f = vn
        uc_f = 0.5 * (uc[i0:i1 + 1, min(jj, Ny - 1)] + uc[i0:i1 + 1, max(jj - 1, 0)])
        pf = 0.5 * (p[i0:i1 + 1, min(jj, Ny - 1)] + p[i0:i1 + 1, max(jj - 1, 0)])
        Fx += -s * np.sum((uc_f * vn) * hx)
        Fy += -s * np.sum((vc_f * vn + pf) * hx)

    q = 0.5 * U_ref * U_ref * D                    # pressão dinâmica * D (ρ=1)
    return Fx / q, Fy / q


def strouhal(signal, dt, U_ref, D, t_skip_frac=0.5):
    """St = f·D/U da frequência dominante do sinal (após descartar a fração
    inicial de transiente)."""
    sig = np.asarray(signal, float)
    n0 = int(len(sig) * t_skip_frac)
    s = sig[n0:]
    s = s - s.mean()
    if s.size < 8:
        return float("nan")
    freqs = np.fft.rfftfreq(s.size, d=dt)
    amp = np.abs(np.fft.rfft(s * np.hanning(s.size)))
    amp[0] = 0.0
    f_dom = freqs[int(np.argmax(amp))]
    return float(f_dom * D / U_ref)
```

- [ ] **Step 4: Rodar os testes até passar**

Run: `pytest tests/test_forces_strouhal.py -q`
Expected: PASS (3 passed). Ajuste a tolerância do `test_strouhal` só se a resolução de frequência (`1/(T*U)`) exigir — não afrouxe além de `abs=0.02`.

- [ ] **Step 5: Commit**

```bash
git add mprm_core.py tests/test_forces_strouhal.py && \
git commit -q -m "feat(mprm): ChannelFinePoisson mascarado + obstacle_forces (VC) + strouhal (FFT)" || echo "(sem repo git)"
```

---

### Task 7: `tests/test_von_karman.py` — smoke rápido + estudo completo

**Files:**
- Create: `tests/test_von_karman.py`

**Interfaces:**
- Consumes: `run_von_karman`, `strouhal`, `von_karman_problem`, `mac_divergence`, `solid_faces` (T1–T6); `common.py`, `params.py`.
- Produces: função pytest `test_von_karman_quick()` (smoke) + `main(argv)` CLI (estudo completo → JSON + figuras em `results_alpha/`).

- [ ] **Step 1: Escrever o smoke test + esqueleto do estudo**

Create `tests/test_von_karman.py`:
```python
#!/usr/bin/env python3
"""tests/test_von_karman.py — von Kármán (cilindro quadrado, Re=100).

Benchmark da Tabela 3 do artigo do MPrM (Lages, §3.5), agora possível com a
API de BCs generalizada do mrcm (no-flow forte nos 4 lados + saída Dirichlet
de pressão). Duas facetas:

  * `test_von_karman_quick()` — smoke pytest: malha pequena, poucos passos.
    Verifica div≈0 no fluido, u·n=0 no obstáculo e balanço de massa
    entrada≈saída.
  * `main()` — estudo completo (CLI): transiente longo, mede C_D/C_L/St e
    salva figura de vortex shedding.

Uso:
  pytest tests/test_von_karman.py::test_von_karman_quick -q     # smoke
  python tests/test_von_karman.py                               # estudo
  python tests/test_von_karman.py --quick
  mpiexec -n 3 python tests/test_von_karman.py                  # paralelo
"""
from __future__ import annotations
import time
import numpy as np

from common import (get_comm, apply_plot_style, banner, make_meta,
                    save_json, save_figure, COLORS, INK, DEFAULT_OUTDIR)
from params import CommonParams, parse_args

from mprm_core import (run_von_karman, strouhal, von_karman_problem,
                       mac_divergence, solid_faces)

# Geometria default do benchmark (canal com cilindro quadrado).
DEFAULT_GEOM = dict(Lx=16.0, Ly=8.0, D=1.0, xc=4.0, yc=4.0, U_in=1.0)


def _mass_balance(u, v, problem):
    """∮ u·n no contorno externo (entrada − saída); ~0 em incompressível."""
    hy, hx = problem.hy, problem.hx
    inflow = np.sum(u[0, :]) * hy
    outflow = np.sum(u[-1, :]) * hy
    top = np.sum(v[:, -1]) * hx
    bot = np.sum(v[:, 0]) * hx
    return inflow - outflow + bot - top


def test_von_karman_quick():
    """Smoke: estrutura + conservação (rápido, sequencial)."""
    comm = get_comm()
    r = run_von_karman(Lx=8.0, Ly=4.0, D=1.0, xc=2.0, yc=2.0, U_in=1.0,
                       Re=100.0, Ndom=2, nloc=8, degreeU=1, degreeP=3,
                       T=0.3, comm=comm, verbose=False)
    if comm.rank != 0:
        return
    u, v = r["u"], r["v"]
    prob = von_karman_problem(Lx=8.0, Ly=4.0, Nx=16, Ny=16, D=1.0,
                              xc=2.0, yc=2.0, U_in=1.0, Re=100.0)
    # 1. div ~ 0 no fluido
    dv = np.abs(mac_divergence(u, v, prob.hx, prob.hy))
    assert dv[~prob.solid].max() < 1e-8
    # 2. u·n = 0 nas faces do obstáculo
    fx, fy = solid_faces(prob.solid)
    assert np.all(u[fx] == 0.0) and np.all(v[fy] == 0.0)
    # 3. balanço de massa entrada≈saída
    assert abs(_mass_balance(u, v, prob)) < 1e-6
    assert np.isfinite(u).all() and np.isfinite(v).all()


def run_study(comm, geom, Re, Ndom, nloc, degreeU, degreeP, alpha, T, dt,
              verbose=True):
    """Roda o transiente completo e extrai C_D médio, C_L rms e Strouhal."""
    if verbose:
        banner(f"von Kármán Re={Re} — cilindro quadrado D={geom['D']} "
               f"({Ndom*nloc}² células, {Ndom}×{Ndom} subd., "
               f"U_H,{degreeU}/P_H,{degreeP})", comm)
    t0 = time.perf_counter()
    r = run_von_karman(**geom, Re=Re, Ndom=Ndom, nloc=nloc, degreeU=degreeU,
                       degreeP=degreeP, alpha=alpha, T=T, dt=dt, comm=comm,
                       verbose=verbose)
    res = {"meta": make_meta("von_karman", comm, Re=Re, Ndom=Ndom, nloc=nloc,
                             degreeU=degreeU, degreeP=degreeP, alpha=alpha,
                             T=T, dt=r["dt"], nsteps=r["nsteps"], **geom)}
    if comm.rank == 0:
        clh = r.get("cl_history")
        if clh is not None and len(clh) > 8:
            t, cd, cl = clh[:, 0], clh[:, 1], clh[:, 2]
            half = len(t) // 2                 # descarta transiente
            st = strouhal(cl, r["dt"], geom["U_in"], geom["D"])
            res["metrics"] = {
                "Cd_mean": float(cd[half:].mean()),
                "Cl_rms": float(np.sqrt((cl[half:] ** 2).mean())),
                "St": float(st)}
            res["cl_history"] = clh.tolist()
        res["divmax"] = r["divmax"]; res["wall_s"] = time.perf_counter() - t0
        res["u"], res["v"] = r["u"], r["v"]      # p/ a figura (removido no JSON)
    return res


def plot_shedding(res, geom, outdir):
    plt = apply_plot_style()
    u, v = res.pop("u"), res.pop("v")
    Nx = u.shape[0] - 1; Ny = v.shape[1] - 1
    hx = geom["Lx"] / Nx; hy = geom["Ly"] / Ny
    xc = (np.arange(Nx) + 0.5) * hx; yc = (np.arange(Ny) + 0.5) * hy
    uc = 0.5 * (u[:-1, :] + u[1:, :]); vc = 0.5 * (v[:, :-1] + v[:, 1:])
    vort = ((v[1:, :] - v[:-1, :]) / hx)[:, :-1] - ((u[:, 1:] - u[:, :-1]) / hy)[:-1, :]
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.pcolormesh(xc[:-1], yc[:-1], vort.T, shading="auto",
                       cmap="RdBu_r", vmin=-2, vmax=2)
    ax.add_patch(plt.Rectangle((geom["xc"] - geom["D"] / 2,
                                geom["yc"] - geom["D"] / 2),
                               geom["D"], geom["D"], color="k"))
    ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
    m = res.get("metrics", {})
    ax.set_title(f"von Kármán Re={res['meta']['Re']:.0f} — vorticidade  "
                 f"(St={m.get('St', float('nan')):.3f}, "
                 f"C_D={m.get('Cd_mean', float('nan')):.3f})")
    plt.colorbar(im, ax=ax, label=r"$\omega_z$", shrink=0.7)
    return save_figure(fig, outdir, "von_karman_shedding.png")


def main(argv=None):
    defaults = CommonParams(Ndom=4, nloc=12, degreeU=1, degreeP=3, Re=100.0,
                            T=60.0, outdir=DEFAULT_OUTDIR)
    args = parse_args("von Kármán (cilindro quadrado) — MPrM", defaults=defaults,
                      argv=argv)
    if args.quick:
        Ndom, nloc, T = 2, 8, 1.0
    else:
        Ndom, nloc, T = args.Ndom, args.nloc, args.T
    comm = get_comm()
    res = run_study(comm, DEFAULT_GEOM, args.Re, Ndom, nloc, args.degreeU,
                    args.degreeP, args.alpha, T, args.dt, verbose=args.verbose)
    if comm.rank == 0:
        outdir = args.outdir
        fig = plot_shedding(res, DEFAULT_GEOM, outdir)
        save_json(res, outdir, "von_karman.json")
        m = res.get("metrics", {})
        print(f"\nC_D={m.get('Cd_mean', float('nan')):.4f}  "
              f"C_L,rms={m.get('Cl_rms', float('nan')):.4f}  "
              f"St={m.get('St', float('nan')):.4f}  (alvo St≈0.121)")
        print(f"figura: {fig}")
    return res


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Rodar o smoke test (deve passar após T1–T6)**

Run: `pytest tests/test_von_karman.py::test_von_karman_quick -q`
Expected: PASS. Se falhar em `div`, revisar a máscara do downscaler (T3); se falhar em `u·n=0`, revisar `_apply_external_normals_after_step`/`solid_faces` (T2/T5).

- [ ] **Step 3: Rodar o estudo `--quick` (fumaça do pipeline completo, com métricas)**

Run: `python tests/test_von_karman.py --quick`
Expected: imprime `C_D=... C_L,rms=... St=...` (valores possivelmente ruidosos em T=1), gera `results_alpha/von_karman.json` e `results_alpha/von_karman_shedding.png` sem erro.

- [ ] **Step 4: Rodar a suíte inteira de unidade (regressão global)**

Run:
```bash
cd ~/mrcm/NavierStokes && source ~/venv-firedrake/bin/activate && \
pytest tests/test_flow_problem.py tests/test_transport.py \
  tests/test_downscaler_obstacle.py tests/test_projector.py \
  tests/test_forces_strouhal.py tests/test_run_regression.py \
  tests/test_von_karman.py::test_von_karman_quick -q
```
Expected: tudo verde.

- [ ] **Step 5: (opcional, longo) Validação quantitativa completa sob MPI**

Run: `mpiexec -n 3 python tests/test_von_karman.py`
Expected: shedding visível na figura; `St ≈ 0.121` (±~10%) e `C_D` na faixa ~1.4–1.6 típica de cilindro quadrado Re=100. Registrar os valores no handoff. Se St divergir muito, checar: (a) `eps_solid` (R1), (b) tratamento de saída (R3 — testar Dirichlet no downscaler), (c) resolução/`T` (janela de FFT).

- [ ] **Step 6: Commit**

```bash
git add tests/test_von_karman.py && \
git commit -q -m "test(mprm): von Kármán — smoke pytest + estudo CLI (C_D/C_L/St + shedding)" || echo "(sem repo git)"
```

---

## Notas finais de execução

- **Ordem de dependência:** implemente em ordem do documento — T0 → T1 → T2 → T3 → T4 → T5 → T6 → T7. Os testes de cada task passam nessa ordem: os testes de cavidade do T5 (`method="mprm"`, sem obstáculo) não invocam os símbolos do T6, que só entram no caminho de von Kármán (T7). O único acoplamento é o helper `_apply_external_normals_after_step`, definido no T5 e reutilizado pelo `ChannelFinePoisson` do T6 (T6 vem depois — sem problema).
- **Regressões a vigiar sempre:** `hh_benchmark_mac` (err_u≈2.3e-1 grau 1), `run_cavity` (div~1e-15, mprm×fine~5e-4 em T longo), `tests/test_alpha_*` importam sem erro. O caminho `problem=None` é o guarda-chuva de compatibilidade.
- **`git init`:** se `~/mrcm/NavierStokes` não for repo, os `git commit` acima falham silenciosamente (`|| echo`). Decidir com o usuário se versiona o diretório e o que ignorar (`results_alpha/`, `frames_mprm/`, `*.gif`, `*.png`, `*.ipynb_checkpoints`).
- **Handoff:** ao fim, escrever `~/mrcm/mrcm-firedrake/handoffs/2026-07-20-mprm-von-karman.md` (merge da branch de BCs + capacidade de von Kármán + valores de C_D/C_L/St medidos).
```
