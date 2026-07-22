# Guia de Execução — Benchmark von Kármán (MPrM)

> **Branch:** `mprm-von-karman` | **Repositório:** `~/mrcm/NavierStokes`
> **Ambiente:** `source ~/venv-firedrake/bin/activate`
> **Artigo de referência:** Lages, "The multiscale projection method for the solution of incompressible Navier-Stokes equations", §3.5 / Tabela 3.

---

## 1. Execução rápida (smoke test + estudo CLI)

### 1.1 Smoke test (pytest)

```bash
cd ~/mrcm/NavierStokes && source ~/venv-firedrake/bin/activate

# Apenas o smoke test (malha pequena, T curto, ~8s)
pytest tests/test_von_karman.py::test_von_karman_quick -q

# Suíte completa de regressão
pytest tests/test_flow_problem.py tests/test_transport.py \
  tests/test_downscaler_obstacle.py tests/test_projector.py \
  tests/test_forces_strouhal.py \
  tests/test_run_regression.py::test_run_cavity_quick_conserva_massa \
  tests/test_von_karman.py::test_von_karman_quick -q
```

### 1.2 Estudo CLI (completo, com métricas e figura)

```bash
# Configuração padrão: Ndom=4, nloc=12, T=60, grauU=1, grauP=3, alpha=1, Re=100
python tests/test_von_karman.py

# Versão rápida (smoke, T=1)
python tests/test_von_karman.py --quick

# Paralelo (MPI, 3 ranks)
mpiexec -n 3 python tests/test_von_karman.py
```

### 1.3 Saída gerada

| Arquivo | Descrição |
|---------|-----------|
| `results_alpha/von_karman.json` | Séries temporais $C_D(t)$, $C_L(t)$ + métricas |
| `results_alpha/von_karman_shedding.png` | Campo de vorticidade no instante final |

---

## 2. Parâmetros disponíveis

### 2.1 Linha de comando (`python tests/test_von_karman.py [flags]`)

Todos os parâmetros abaixo são aceitos como flags `--nome valor`:

#### Domínio e decomposição

| Flag | Default | Descrição |
|------|---------|-----------|
| `--Lx` | `16.0` | Comprimento do canal |
| `--Ly` | `8.0` | Altura do canal |
| `--Ndom` | `4` | Subdomínios por direção ($N_H$) |
| `--nloc` | `12` | Células finas por subdomínio (malha total = $Ndom \times nloc$) |

> ⚠️ **Restrição**: o obstáculo deve ficar **totalmente dentro de um subdomínio** (não pode cruzar interfaces). A geometria default $(x_c,y_c)=(3,3)$ com $D=1$ funciona para $Ndom \ge 4$ com $L_x=16$. Para $Ndom=2$, use `--quick` (ajusta automaticamente).

#### Espaços de interface e Robin

| Flag | Default | Descrição |
|------|---------|-----------|
| `--degreeU` | `1` | Grau polinomial do espaço de fluxo $U_H$ ($k_U =$ grau $+ 1$ DOFs por interface) |
| `--degreeP` | `3` | Grau polinomial do espaço de pressão $P_H$ |
| `--basis-kind` | `legendre_P` | Família da base: `legendre_P`, `monomial_P`, `nodal_Q` |
| `--alpha` | `1.0` | Parâmetro de Robin ($\beta = \alpha \cdot L / K$) |

> 📌 **Recomendação do artigo**: para escoamentos transientes (von Kármán), use pressão rica ($P_{H,3}$). A base `legendre_P` é hierárquica e preserva $\pi_0$ (constante) mesmo com graus assimétricos — `nodal_Q` **não** preserva quando `degreeU ≠ degreeP`.

#### Fluido e tempo

| Flag | Default | Descrição |
|------|---------|-----------|
| `--Re` | `100.0` | Número de Reynolds ($Re = U_\infty D / \nu$) |
| `--T` | `60.0` | Tempo final de integração (em unidades $D/U_\infty$) |
| `--dt` | `None` | Passo de tempo (automático por CFL se `None`) |

#### Execução e saída

| Flag | Default | Descrição |
|------|---------|-----------|
| `--outdir` | `results_alpha` | Diretório de saída para JSON e figuras |
| `--quick` | `False` | Versão reduzida ($Ndom=2, T=1$) |
| `--quiet` | `False` | Suprime prints por passo |

### 2.2 API Python (`run_von_karman`)

```python
from mprm_core import run_von_karman

result = run_von_karman(
    # Geometria (obrigatórios)
    Lx=16.0, Ly=8.0,           # dimensões do canal
    D=1.0,                       # lado do cilindro quadrado
    xc=3.0, yc=3.0,             # centro do obstáculo

    # Escoamento
    U_in=1.0,                    # velocidade de entrada
    Re=100.0,                    # número de Reynolds

    # Decomposição (obrigatórios)
    Ndom=4,                      # subdomínios por direção
    nloc=12,                     # células finas por subdomínio

    # Espaços de interface
    degreeU=1,                   # grau do espaço de fluxo U_H
    degreeP=3,                   # grau do espaço de pressão P_H
    alpha=1.0,                   # parâmetro de Robin
    basis_kind="legendre_P",     # família da base

    # Tempo
    T=60.0,                      # tempo final
    dt=None,                     # passo de tempo (None = CFL automático)

    # Opções de contorno
    inflow="uniform",            # "uniform" ou "parabolic"
    walls="wall",                # "wall" (no-slip) ou "slip" (free-slip)

    # Diagnósticos
    eps_solid=1e-6,              # fator K no sólido (K_solid = eps*K_fluid)
    probes=[(6.0, 3.0)],         # lista de (x,y) para sondas de velocidade
    frame_callback=None,         # callable(step, t, u, v, hx, hy) p/ frames
    frame_every=0,               # intervalo em passos (0 = desabilitado)

    # Execução
    comm=None,                   # comunicador MPI (None = COMM_WORLD)
    verbose=True,
    log_every=200,
)
```

O dicionário retornado contém:
- `u`, `v` — campos de velocidade MAC
- `dt`, `nsteps` — passo de tempo e número de passos
- `divmax` — divergência máxima
- `cl_history` — array `(nsteps, 3)` com colunas `(t, C_D, C_L)`
- `probe_history` — array com séries das sondas

---

## 3. Gerando animações (GIF)

### 3.1 Script rápido para GIF

```bash
cd ~/mrcm/NavierStokes && source ~/venv-firedrake/bin/activate

python << 'PYEOF'
import numpy as np, os, glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mprm_core import run_von_karman, make_gif

# ---- CONFIGURAÇÃO DA ANIMAÇÃO ----
OUTDIR     = "frames_von_karman"      # diretório dos frames
GIF_OUT    = "results_alpha/von_karman.gif"
FPS        = 10                       # quadros por segundo no GIF
FRAME_EVERY = 15                      # 1 frame a cada N passos

# ---- PARÂMETROS DA SIMULAÇÃO ----
Ndom, nloc    = 4, 12                 # decomposição
degreeU, degreeP = 1, 3              # espaços de interface
alpha         = 1.0                   # Robin
Re            = 100.0                 # Reynolds
T             = 40.0                  # tempo final (unidades D/U)

os.makedirs(OUTDIR, exist_ok=True)
for f in glob.glob(f"{OUTDIR}/frame_*.png"):
    os.remove(f)

frame_idx = [0]
def frame_callback(step, t, u, v, hx, hy):
    """Salva vorticidade a cada frame."""
    Nx, Ny = u.shape[0]-1, v.shape[1]-1
    dvdx = (v[1:,:] - v[:-1,:]) / hx
    dudy = (u[:,1:] - u[:,:-1]) / hy
    vort = dvdx[:, 1:-1] - dudy[1:-1, :]

    fig, ax = plt.subplots(figsize=(10, 5))
    xc = (np.arange(1, Nx) + 0.5) * hx
    yc = (np.arange(1, Ny) + 0.5) * hy
    ax.pcolormesh(xc, yc, vort.T, shading="auto",
                  cmap="RdBu_r", vmin=-2, vmax=2)
    ax.add_patch(plt.Rectangle((2.5, 2.5), 1.0, 1.0, color="k"))
    ax.set_aspect("equal")
    ax.set_title(f"von Kármán  Re={Re}  t={t:.1f}")
    plt.colorbar(ax.collections[0], ax=ax, label=r"$\omega_z$", shrink=0.7)
    fname = os.path.join(OUTDIR, f"frame_{frame_idx[0]:05d}.png")
    fig.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close(fig)
    frame_idx[0] += 1

print(f"Simulando: Ndom={Ndom}, nloc={nloc}, U_H,{degreeU}/P_H,{degreeP}, "
      f"α={alpha}, Re={Re}, T={T}")
r = run_von_karman(Lx=16.0, Ly=8.0, D=1.0, xc=3.0, yc=3.0, U_in=1.0, Re=Re,
                   Ndom=Ndom, nloc=nloc, degreeU=degreeU, degreeP=degreeP,
                   alpha=alpha, T=T, comm=None, verbose=True, log_every=500,
                   frame_callback=frame_callback, frame_every=FRAME_EVERY)

frames = sorted(glob.glob(f"{OUTDIR}/frame_*.png"))
print(f"\n{len(frames)} frames capturados. Gerando GIF...")
gif_path = make_gif(OUTDIR, GIF_OUT, fps=FPS)
print(f"✓ GIF: {gif_path}")
PYEOF
```

### 3.2 Parâmetros que afetam a animação

| Parâmetro | Efeito |
|-----------|--------|
| `FRAME_EVERY` | Controla a densidade temporal (menor = mais frames, GIF maior) |
| `FPS` | Velocidade de reprodução (8–15 recomendado) |
| `vmin`/`vmax` no `pcolormesh` | Escala de cor da vorticidade ($\pm 2$ default) |
| `T` | Tempo total simulado (maior = mais períodos de shedding) |

---

## 4. Grades de parâmetros (estudos paramétricos)

### 4.1 Varredura de $\alpha$ (parâmetro de Robin)

```bash
# Estudo do efeito de alpha no Strouhal e C_D
for alpha in 1e-6 1e-3 1e-1 1 10 100; do
    echo "=== α = $alpha ==="
    python tests/test_von_karman.py --alpha $alpha --T 30 --nloc 8 --Ndom 4 --quiet
done
```

Resultados esperados (artigo §8.4): melhores resultados com $\alpha = 10^{-6}$ e pressão rica ($P_{H,3}$) para fluxos transientes.

### 4.2 Varredura dos espaços de interface

```bash
# Efeito do enriquecimento dos espaços U_H e P_H
for dU in 0 1 2; do
    for dP in 0 1 2 3; do
        echo "=== U_H,$dU / P_H,$dP ==="
        python tests/test_von_karman.py --degreeU $dU --degreeP $dP \
            --T 30 --nloc 8 --Ndom 4 --quiet
    done
done
```

> Artigo: melhor caso com $P_{H,3}, U_{H,2..3}$, erro de $C_D \sim 10^{-3}$.

### 4.3 Variação do número de Reynolds

```bash
for Re in 60 80 100 120 150 200; do
    echo "=== Re = $Re ==="
    python tests/test_von_karman.py --Re $Re --T $((Re/2)) --nloc 8 --Ndom 4 --quiet
done
```

> 💡 Ajuste `--T` proporcionalmente: quanto menor Re, mais tempo para desenvolver shedding. Use $T \approx 0.5 \times Re \times D/U_\infty$ como guia.

### 4.4 Refinamento de malha

```bash
# Malha fina: Ndom=4, nloc=16 → 64² células (~4 min)
python tests/test_von_karman.py --Ndom 4 --nloc 16 --T 50

# Malha grossa (rápida): Ndom=3, nloc=8 → 24² células (~30s)
# ATENÇÃO: ajuste xc,yc para não cruzar interfaces com Ndom=3!
python -c "
from mprm_core import run_von_karman
r = run_von_karman(Lx=16, Ly=8, D=1, xc=2.5, yc=2.5, U_in=1, Re=100,
                   Ndom=3, nloc=8, T=20, verbose=True)
print('St=', r.get('cl_history',[])[-1] if r.get('cl_history') is not None else 'N/A')
"
```

---

## 5. Parâmetros avançados (API Python)

### 5.1 Base dos espaços de interface (`basis_kind`)

| Valor | Descrição | Preserva $\pi_0$ com graus assimétricos? |
|-------|-----------|------------------------------------------|
| `legendre_P` | Legendre hierárquica (recomendada) | ✅ Sim |
| `monomial_P` | Monômios hierárquicos | ✅ Sim |
| `nodal_Q` | Lagrange nodal (grau único) | ❌ Não |

> 📌 **Achado D3/π₀**: `nodal_Q` trunca a base de Lagrange ao grau máximo comum e **perde a constante** quando `degreeU ≠ degreeP`, quebrando o balanço de massa (defeito $\sim 10^{-3}$). Use `legendre_P` sempre que usar graus assimétricos.

### 5.2 Condições de contorno (`inflow`, `walls`)

```python
# Entrada parabólica (perfil de Poiseuille com média U_in)
run_von_karman(..., inflow="parabolic")

# Paredes com deslizamento (free-slip) em vez de no-slip
run_von_karman(..., walls="slip")
```

### 5.3 Permeabilidade do sólido (`eps_solid`)

Controla $K_{\text{sólido}} = \epsilon \cdot K_{\text{fluido}}$ no projetor MRCM. Valores menores tornam o sólido "mais impermeável", mas podem causar mal-condicionamento.

```python
run_von_karman(..., eps_solid=1e-8)   # sólido mais rígido (padrão: 1e-6)
```

### 5.4 Método de projeção (`method`)

```python
# Referência de malha fina (sem decomposição MRCM)
run_von_karman(..., method="fine")

# MPrM com projeção multiescala (default)
run_von_karman(..., method="mprm")
```

---

## 6. Tabela de custo computacional (referência)

| Ndom | nloc | Células | Tempo (T=50) | Uso típico |
|------|------|---------|-------------|------------|
| 2 | 8 | 16² | ~5s | Smoke test |
| 3 | 12 | 36² | ~30s | Desenvolvimento |
| 4 | 12 | 48² | ~90s | Estudo CLI padrão |
| 4 | 16 | 64² | ~100s | Benchmark |
| 5 | 16 | 80² | ~3min | Alta resolução |

> Medido em 1 core, `degreeU=1, degreeP=3`, Re=100. Tempos podem variar.

---

## 7. Resultados esperados (referência do artigo)

| Métrica | Artigo Lages (§3.5) | Literatura (Re=100) |
|---------|---------------------|---------------------|
| $C_D$ médio | $\sim 1.51$–$1.61$ | $1.4$–$1.6$ |
| $C_{L,\text{rms}}$ | — | $0.2$–$0.3$ |
| Strouhal | $\approx 0.121$ | $0.121$–$0.150$ |
| $\max\|\nabla\cdot u\|$ | — | $\sim 10^{-14}$ |

> Valores obtidos com esta implementação (Ndom=4, nloc=12, $U_{H,1}/P_{H,3}$, $\alpha=1$, T=60):
> $C_D \approx 1.20$, $C_{L,\text{rms}} \approx 0.128$, $St \approx 0.133$.
> A diferença no $C_D$ decorre da malha relativamente grossa e da posição deslocada do obstáculo ($x_c=3.0$ para caber no subdomínio).

---

## 8. Solução de problemas

| Problema | Causa provável | Solução |
|----------|---------------|---------|
| `Obstáculo atravessa a interface` | Geometria não cabe no subdomínio | Ajuste `xc`, `yc` ou aumente `Ndom` |
| $C_D$ negativo | Pressão diagnóstica incorreta | Verifique `_diagnostic_pressure` |
| `St=nan` | $T$ muito curto ou `cl_history` < 8 pontos | Aumente `--T` |
| GIF não gerado | `imageio` ou `Pillow` não instalado | `pip install imageio` |
| Erro de memória com MPI | Subdomínios insuficientes por rank | Aumente `Ndom` ou reduza ranks |
