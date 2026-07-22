# Diagnóstico Crítico — Benchmark von Kármán (MPrM)

> **Branch:** `mprm-von-karman` · **Data:** 2026-07-22
> **Artigo de referência:** C. Lages, *"The multiscale projection method for the
> solution of incompressible Navier–Stokes equations"*, §3.5 (Channel flow past a
> square cylinder), Tabelas 2–3, Eqs. 41–44.
> **Benchmark base:** Breuer, Bernsdorf, Zeiser, Durst (2000), *"Accurate
> computations of the laminar flow past a square cylinder based on two different
> methods: lattice-Boltzmann and finite-volume"*, IJHFF 21:186–196
> (canal com *blockage* $D/H = 1/8 \Rightarrow L_y = 8D$, cilindro centrado).

---

## 1. Resumo executivo

O núcleo MRCM está saudável: a divergência do campo final é $4.5\times10^{-14}$
(conservação de massa exata). **O problema não está no projetor**, e sim (a) na
extração dos coeficientes de força/Strouhal e (b) no *setup* físico do benchmark,
que diverge do artigo em praticamente todos os pontos. A concordância aparente do
$C_D$ (1.19 vs 1.33) é, em boa medida, cancelamento de erros de sinais opostos.

### Quadro comparativo

| Métrica | Camila (fine mesh) | Camila (MPrM, melhor) | Implementação atual |
|---|---|---|---|
| $C_D$ | **1.33** | 1.33–1.35 (erro $\sim10^{-3}$) | **1.192** (−10 %) |
| $C_L$ | **0.263** | 0.25–0.31 | **0.130** (rms) |
| $St$ | **0.121** | 0.121–0.126 | **0.1333** (bin de FFT) |
| $\max\lvert\nabla\!\cdot u\rvert$ | — | — | $4.5\times10^{-14}$ ✅ |

*Run analisado (`results_alpha/von_karman.json`):* Re=100, 48² células,
$U_{H,3}/P_{H,3}$, $\alpha=1$, T=60, dt=0.1333, 450 passos, **inflow uniforme +
paredes no-slip + obstáculo em (3,3)** (defaults de `run_von_karman`).

### Setup do artigo vs. implementação

| Aspecto | Artigo §3.5 | Implementação (default) | Severidade |
|---|---|---|---|
| Perfil de entrada | parabólico, **máximo = 1** | `inflow="uniform"` | 🔴 |
| Normalização $\lVert u_\infty\rVert$ | **máximo** do inlet | média (`U_ref=U_in`) | 🔴 |
| Paredes topo/fundo | **free-slip** | `walls="wall"` (no-slip) | 🔴 |
| Posição do obstáculo | centrado ($y_c = L_y/2$) | $y_c=3$ (deslocado $1D$) | 🔴 |
| Força no corpo | $\oint_{\Gamma_s}\sigma\,n\,d\Gamma$, $\sigma=-pI+\mu(\nabla u+\nabla u^\top)$ | balanço de VC **sem termo viscoso nem transiente** | 🔴 |
| Resolução do obstáculo | 30×30 células/subdomínio | **3 células** em $x$ | 🟠 |
| Malha | subdomínios quadrados ($h_x=h_y$) | anisotrópica 2:1 ($h_x=0.33$, $h_y=0.17$) | 🟠 |
| Estatística de $C_L$ | amplitude/pico | rms | 🟠 |
| Estimador de $St$ | — | `rfft` puro, $\Delta f = 0.033$ | 🔴 |

---

## 2. Achados detalhados

### 🔴 A — `St = 0.1333` é artefato da resolução de FFT, não física

**Local:** [`strouhal()`](../mprm_core.py#L1361), uso em
[`test_von_karman.py:95`](../tests/test_von_karman.py#L95).

Com 450 amostras, `t_skip_frac=0.5` (→ 225 usadas) e $dt=0.1333$, a resolução em
frequência é
$$\Delta f = \frac{1}{N_\text{usado}\cdot dt} = \frac{1}{225 \cdot 0.1333}
= \frac{1}{30} \approx 0.0333.$$
Como $D=U=1$, os *bins* de $St$ acessíveis são $\{0.10, \mathbf{0.1333}, 0.1667,
\dots\}$. O valor físico ($\approx0.121$) cai **entre** dois bins e encaixa no mais
próximo. O `0.13333333` exato no JSON confirma: é $4/30$.

**Impacto:** não se distingue $0.121$ de $0.133$ (diferença $0.012 <$ um bin
$0.033$). Imprimir `St={st:.4f}` é precisão espúria; qualquer varredura de $St$
(guia §4.3) mede o grid da FFT, não a física.

---

### 🔴 B — `obstacle_forces` **não implementa o termo viscoso** (docstring mente)

**Local:** [`obstacle_forces()`](../mprm_core.py#L1319).

O docstring diz *"Fluxo convectivo + pressão + viscoso nas 4 faces"*; lê
`nu = problem.nu` na [linha 1326](../mprm_core.py#L1326) e **nunca usa `nu`**. As
somas ([1345–1355](../mprm_core.py#L1345)) só têm convecção + pressão. Além disso,
o balanço de volume de controle **omite o termo transiente**
$\frac{d}{dt}\int_{VC} u\,dV$ (o comentário *"dp/dt absorvido no p"* é
hand-waving).

O artigo integra o tensor completo $\sigma = -pI + \mu(\nabla u + \nabla u^\top)$
(Eq. 43). A Re=100 a parcela viscosa do arrasto num corpo rombudo é significativa;
sua omissão **subestima sistematicamente $C_D$** — e $1.33\times(1-\text{fração
viscosa})$ cai justamente na casa de 1.19. Parte relevante do déficit é este bug,
**independente da malha**. O termo transiente omitido anula-se na média (ok para
$\bar C_D$) mas **distorce a amplitude instantânea de $C_L$** — exatamente a
quantidade reportada baixa.

---

### 🔴 C — *Setup* físico diverge do artigo em quatro eixos simultâneos

**Local:** defaults de [`von_karman_problem()`](../mprm_core.py#L167) e
[`run_von_karman()`](../mprm_core.py#L1085); geometria em
[`test_von_karman.py:35`](../tests/test_von_karman.py#L35).

1. **Perfil de entrada.** Default `inflow="uniform"`. Mesmo `inflow="parabolic"`
   ([linha 183](../mprm_core.py#L183)) normaliza para **média = U_in**, enquanto o
   artigo fixa **máximo = 1**. Parabólico com média 1 tem pico 1.5 → Re efetivo
   50 % maior. Como $C_D, C_L, St$ e Re dependem de $\lVert u_\infty\rVert^2$, há
   inconsistência de normalização direta.
2. **Paredes.** Default no-slip; o artigo usa **free-slip**. No-slip cria
   camadas-limite nas paredes do canal, aumentando o *blockage* efetivo e o
   arrasto.
3. **Obstáculo deslocado.** $y_c=3$ em $L_y=8$ → $1D$ fora do centro. Quebra a
   simetria topo/fundo, enviesa $C_L$ e distorce o *shedding*. É consequência da
   restrição [`assert_solid_interior_to_subdomains`](../mprm_core.py#L198): com
   $N_{dom,y}=4$ há interface exatamente em $y=4$ (o centro).
4. **Normalização de $U_{ref}$.** `run_von_karman` passa `U_ref=U_in`
   ([linha 1112](../mprm_core.py#L1112)); o artigo normaliza por $\max$ do inlet.

O $C_D\approx1.19$ perto de 1.33 sob **todos** esses desvios é cancelamento de
erros (bug B puxa para baixo; no-slip/blockage puxa para cima).

---

### 🟠 D — Malha grosseira e anisotrópica; obstáculo com 3 células

**Local:** [`run_von_karman:1100`](../mprm_core.py#L1100),
[`run_flow`](../mprm_core.py#L985).

$N_x=N_y=48$, $L_x=16$, $L_y=8$ → $h_x=0.333$, $h_y=0.167$ (anisotropia 2:1). O
quadrado $D=1$ é resolvido por **3 células em $x$ × 6 em $y$**; o artigo usa
**30×30 por subdomínio**. O comentário *"mantém subdomínios quadrados em H"*
([linha 1101](../mprm_core.py#L1101)) é **falso**: cada subdomínio é 4×2 físico. A
API força `Nx=Ny=Ndom*nloc` ([linha 1100](../mprm_core.py#L1100)), então **não há
como obter células quadradas num domínio não-quadrado** — limitação estrutural.

---

### 🟠 E — `C_L,rms` comparado contra o `C_L` (amplitude) do artigo

**Local:** [`test_von_karman.py:98`](../tests/test_von_karman.py#L98).

Reporta-se `Cl_rms`, mas a Tabela do artigo traz $C_L$ (amplitude/pico). Se 0.263
for amplitude, o rms equivalente é $0.263/\sqrt2\approx0.186$ — ainda acima de
0.13, mas parte da "discrepância" é **definicional**. Comparar estatísticas
diferentes mascara o quanto o erro é físico vs. de definição.

---

### 🟡 F — Pressão diagnóstica: segundo solve, inconsistente com o momento avançado

**Local:** [`_diagnostic_pressure`](../mprm_core.py#L1204).

Resolve um Poisson **separado** $\nabla^2 p = -\nabla\!\cdot((u\cdot\nabla)u)$ com
Neumann homogêneo nas paredes e $p=0$ imposto no sólido. Mas (i) o BC correto na
parede é $\partial p/\partial n \neq 0$ e (ii) o campo resultante não é
garantidamente o do balanço de momento discreto que o MPrM realmente integrou. O
artigo calcula forças a partir do **$p$ e $u$ efetivos**. O caminho limpo é
reaproveitar a pressão do próprio projetor (`ChannelFinePoisson.solve` já devolve
`p`; o `MPrMProjector` resolve um Darcy cuja pressão é recuperável).

---

### 🟡 G — `_diagnostic_pressure` re-fatoriza a mesma matriz 450 vezes

**Local:** [`_diagnostic_pressure:1313`](../mprm_core.py#L1313).

A matriz $A$ é **constante no tempo** (geometria fixa), mas o `splu(A)` está
**dentro da função, a cada passo** → 450 fatorações LU idênticas.
`ChannelFinePoisson` fatora uma vez no `__init__` e reusa; o diagnóstico deveria
fazer o mesmo (fatorar uma vez, trocar só o RHS). Ganho de ordem de grandeza.

---

### 🟡 H — `dt` cego à anisotropia

**Local:** [`run_flow:1003`](../mprm_core.py#L1003).

`dt = 0.4*min(hx/umax, hx²/(4ν))` usa só `hx`. Com $h_y=0.167$ e $v\sim O(1)$ no
*shedding*, o CFL em $y$ é $\approx0.8$ (alto para upwind de 1ª ordem). Usar
`min(hx,hy)`.

---

## 3. Propostas de código

As mudanças abaixo são independentes; a ordem em §4 prioriza impacto/custo.

### P1 — Força no corpo consistente com o artigo (corrige B)

Duas opções. **Recomendada: B1** (fiel ao artigo). A **B0** conserta o VC atual
com menos reescrita.

#### B0 — Completar o balanço de volume de controle (viscoso + transiente)

O balanço para força no corpo (ρ=1), com $V$ = fluido entre $\Gamma_s$ e a
superfície externa $S_o$ do VC:
$$
F_i = -\underbrace{\frac{d}{dt}\!\int_V u_i\,dV}_{\text{transiente}}
      + \oint_{S_o}\!\Big[-u_i(u\cdot n) - p\,n_i
      + \nu(\partial_j u_i + \partial_i u_j)\,n_j\Big]dS .
$$
A parte convectiva+pressão já está certa. Falta, em cada face do VC:

- **Face vertical** (normal $s\hat x$): somar a $F_x$ o termo
  $s\,\nu\,(2\,\partial_x u)\,h_y$ e a $F_y$ o termo
  $s\,\nu\,(\partial_x v + \partial_y u)\,h_y$.
- **Face horizontal** (normal $s\hat y$): somar a $F_x$ o termo
  $s\,\nu\,(\partial_y u + \partial_x v)\,h_x$ e a $F_y$ o termo
  $s\,\nu\,(2\,\partial_y v)\,h_x$.
- **Transiente**: manter $u,v$ do passo anterior no VC e subtrair
  $\frac{1}{dt}\big(\int_V u_i^{n+1} - \int_V u_i^{n}\big)$ (soma nas células do VC
  × $h_x h_y$). `run_flow` já tem o campo anterior disponível.

```python
def obstacle_forces(u, v, p, problem, U_ref, D, u_prev=None, v_prev=None, dt=None):
    """C_D, C_L por balanço de momento num VC retangular. Inclui fluxo
    convectivo + pressão + VISCOSO nas 4 faces e (se u_prev/dt dados) o termo
    transiente d/dt ∫_VC u dV. σ = -pI + ν(∇u+∇uᵀ); ρ=1; C = 2F/(ρ U² D)."""
    solid = problem.solid
    if solid is None or not solid.any():
        return 0.0, 0.0
    hx, hy, nu = problem.hx, problem.hy, problem.nu
    Nx, Ny = problem.Nx, problem.Ny
    xs = np.where(solid.any(axis=1))[0]; ys = np.where(solid.any(axis=0))[0]
    i0, i1 = max(xs.min() - 1, 0), min(xs.max() + 1, Nx - 1)
    j0, j1 = max(ys.min() - 1, 0), min(ys.max() + 1, Ny - 1)

    uc = 0.5 * (u[:-1, :] + u[1:, :])           # (Nx, Ny) centrado
    vc = 0.5 * (v[:, :-1] + v[:, 1:])
    # gradientes centrados (para o tensor viscoso nas faces do VC)
    dudx = np.gradient(uc, hx, axis=0); dudy = np.gradient(uc, hy, axis=1)
    dvdx = np.gradient(vc, hx, axis=0); dvdy = np.gradient(vc, hy, axis=1)

    Fx = Fy = 0.0
    for i, s in ((i0, -1.0), (i1 + 1, +1.0)):           # faces verticais
        ii = min(max(i, 0), Nx); ic = min(max(i - (s > 0), 0), Nx - 1)
        un = u[ii, j0:j1 + 1]
        vc_f = 0.5 * (vc[min(ii, Nx-1), j0:j1+1] + vc[max(ii-1, 0), j0:j1+1])
        pf = 0.5 * (p[min(ii, Nx-1), j0:j1+1] + p[max(ii-1, 0), j0:j1+1])
        visc_xx = 2.0 * nu * dudx[ic, j0:j1+1]
        visc_yx = nu * (dvdx[ic, j0:j1+1] + dudy[ic, j0:j1+1])
        Fx += -s * np.sum((un * un + pf - visc_xx) * hy)
        Fy += -s * np.sum((vc_f * un      - visc_yx) * hy)
    for j, s in ((j0, -1.0), (j1 + 1, +1.0)):           # faces horizontais
        jj = min(max(j, 0), Ny); jc = min(max(j - (s > 0), 0), Ny - 1)
        vn = v[i0:i1 + 1, jj]
        uc_f = 0.5 * (uc[i0:i1+1, min(jj, Ny-1)] + uc[i0:i1+1, max(jj-1, 0)])
        pf = 0.5 * (p[i0:i1+1, min(jj, Ny-1)] + p[i0:i1+1, max(jj-1, 0)])
        visc_xy = nu * (dudy[i0:i1+1, jc] + dvdx[i0:i1+1, jc])
        visc_yy = 2.0 * nu * dvdy[i0:i1+1, jc]
        Fx += -s * np.sum((uc_f * vn      - visc_xy) * hx)
        Fy += -s * np.sum((vn * vn + pf   - visc_yy) * hx)

    if u_prev is not None and dt is not None:            # termo transiente
        upc = 0.5 * (u_prev[:-1, :] + u_prev[1:, :]); vpc = 0.5 * (v_prev[:, :-1] + v_prev[:, 1:])
        dVol = hx * hy
        Fx -= (uc[i0:i1+1, j0:j1+1].sum() - upc[i0:i1+1, j0:j1+1].sum()) * dVol / dt
        Fy -= (vc[i0:i1+1, j0:j1+1].sum() - vpc[i0:i1+1, j0:j1+1].sum()) * dVol / dt

    q = 0.5 * U_ref * U_ref * D
    return Fx / q, Fy / q
```

> Sinais: o termo viscoso entra como $\sigma\cdot n$ (traction), portanto
> $-\nu(\dots)$ do lado do fluxo (subtraído do convectivo+pressão). Validar com o
> teste de regressão sugerido em §4.

#### B1 — Integral de superfície do tensor no obstáculo (fiel à Eq. 43)

Mais próximo do artigo: somar $\sigma\cdot n$ diretamente nas faces MAC entre
sólido e fluido (as máscaras de [`solid_faces`](../mprm_core.py#L262) já as dão).
Para um quadrado alinhado à malha, arrasto vem das faces frontal/traseira
(pressão + $\sigma_{xx}$) e sustentação das faces superior/inferior. Requer
derivadas one-sided na parede (o fluido está só de um lado). Recomendado como
implementação definitiva; ver checklist §4.

---

### P2 — *Setup* do escoamento igual ao artigo (corrige C)

```python
def von_karman_problem(Lx, Ly, Nx, Ny, D, xc, yc, U_in, Re,
                       inflow="parabolic", walls="slip",     # <- defaults do artigo
                       inflow_norm="max",                    # "max" (artigo) | "mean"
                       outflow="pressure") -> FlowProblem:
    nu = U_in * D / Re
    if inflow == "uniform":
        profile = float(U_in)
    elif inflow == "parabolic":
        Ly_c = float(Ly)
        if inflow_norm == "max":      # pico = U_in  (‖u_∞‖ = U_in, como no artigo)
            def profile(y):
                return 4.0 * U_in * y * (Ly_c - y) / (Ly_c ** 2)
        else:                          # média = U_in  (comportamento anterior)
            def profile(y):
                return 6.0 * U_in * y * (Ly_c - y) / (Ly_c ** 2)
    ...
```

- `run_von_karman`: expor `inflow`, `walls`, `inflow_norm` e manter `U_ref=U_in`
  (que agora coincide com $\max$ do inlet quando `inflow_norm="max"`).
- **Re** definido com a mesma velocidade característica ($U_{in}=$ pico).

---

### P3 — Malha isotrópica e obstáculo centrado (corrige C.3 e D)

Permitir subdomínios independentes por direção para (a) $h_x=h_y$ e (b) nenhuma
interface no centro do canal.

```python
def run_von_karman(Lx, Ly, D, xc, yc, U_in=1.0, Re=100.0,
                   Ndomx=None, Ndomy=None, nloc=None, ...):
    Nx = Ndomx * nloc
    Ny = Ndomy * nloc
    ...   # run_flow com Ndomx, Ndomy separados
```

**Exemplo que satisfaz tudo** (células quadradas + obstáculo centrado):
`Ndomx=10, Ndomy=5, nloc=n` ⇒ $h_x = L_x/(10n) = h_y = L_y/(5n) = 2/n$; interfaces
em $y \in \{1.6, 3.2, 4.8, 6.4\}$, e o centro $y_c=4$ com $D=1$ cai em
$[3.5,4.5]\subset[3.2,4.8]$ (interior de um subdomínio). Para 15–20 células por
lado do obstáculo, use $n\ge30$ ($h=2/30\approx0.067$, $D/h=15$).

> Requer generalizar `run_flow`/`MRCMConfig` para `Ndomx≠Ndomy` (hoje força
> `Ndomx=Ndomy=Ndom`, [linha 1007](../mprm_core.py#L1007)).

---

### P4 — Estimador de Strouhal robusto (corrige A)

```python
def strouhal(signal, dt, U_ref, D, t_skip_frac=0.5, pad=8):
    """St por pico da FFT com zero-padding + interpolação parabólica de 3 pontos."""
    sig = np.asarray(signal, float)
    s = sig[int(len(sig) * t_skip_frac):]
    if s.size < 16:
        return float("nan")
    s = (s - s.mean()) * np.hanning(s.size)
    n = int(2 ** np.ceil(np.log2(s.size * pad)))        # zero-pad p/ refinar bins
    amp = np.abs(np.fft.rfft(s, n=n)); amp[0] = 0.0
    freqs = np.fft.rfftfreq(n, d=dt)
    k = int(np.argmax(amp))
    if 0 < k < len(amp) - 1:                             # interpolação parabólica
        a, b, c = amp[k-1], amp[k], amp[k+1]
        k = k + 0.5 * (a - c) / (a - 2*b + c + 1e-30)
    f_dom = k * (freqs[1] - freqs[0])
    return float(f_dom * D / U_ref)
```

E **T longo**: para 20–30 períodos de *shedding* a $St\approx0.12$
($T_{period}=D/(St\,U)\approx8.3$), usar `T ≥ 180`. Reportar `St` com no máximo
3 casas.

---

### P5 — Pressão diagnóstica fatorada uma vez (corrige G) ou reaproveitada (F)

Mínimo: encapsular a montagem/fatoração da matriz do Poisson diagnóstico numa
classe (como `ChannelFinePoisson`) e só trocar o RHS por passo. Ideal: extrair a
pressão do próprio `MPrMProjector` (variável dual do Darcy) e passá-la a
`obstacle_forces`, eliminando o solve separado.

---

### P6 — `dt` ciente da anisotropia (corrige H)

```python
h = min(hx, hy)
dt = 0.4 * min(h / umax, h * h / (4.0 * nu))
```

### P7 — `C_L` reportado como amplitude (corrige E)

Reportar `Cl_amp = 0.5*(cl[half:].max() - cl[half:].min())` **e** `Cl_rms`, e
comparar `Cl_amp` contra o $C_L$ do artigo.

---

## 4. Plano de implementação priorizado

| # | Ação | Corrige | Esforço | Impacto |
|---|---|---|---|---|
| 1 | **P1‑B0**: viscoso + transiente em `obstacle_forces` | B | baixo | 🔴 $C_D$/$C_L$ |
| 2 | **P2**: parabólico max=1 + free-slip + `U_ref`=máx | C | baixo | 🔴 problema correto |
| 3 | **P4**: estimador de $St$ + $T\ge180$ | A | baixo | 🔴 $St$ confiável |
| 4 | **P7 + E**: reportar `Cl_amp` | E | trivial | 🟠 comparação justa |
| 5 | **P3**: `Ndomx≠Ndomy`, células quadradas, obstáculo centrado | C.3, D | médio | 🟠 acurácia/simetria |
| 6 | **P5**: cache/reuso da pressão | F, G | médio | 🟡 custo/consistência |
| 7 | **P6**: `dt` com `min(hx,hy)` | H | trivial | 🟡 estabilidade |
| 8 | **P1‑B1**: integral de superfície (Eq. 43) | B | alto | 🟢 fidelidade |

### Validação (TDD sugerido)

1. **Teste de força — patch de Stokes analítico:** campo com solução conhecida
   (ex.: escoamento de Couette/Poiseuille sobre placa) onde a força viscosa é
   analítica → fixa o sinal/magnitude do termo viscoso de `obstacle_forces`.
2. **Teste de $St$ — sinal sintético:** senoide de frequência conhecida
   ($St=0.121$) amostrada como no run → `strouhal` deve recuperar $0.121\pm0.002$.
3. **Regressão de benchmark:** após P1–P4, alvo
   $C_D\in[1.28,1.38]$, $C_L^{amp}\in[0.22,0.32]$, $St\in[0.118,0.126]$
   (Tabela 3 do artigo), com malha $D/h\ge15$.

---

## 5. Metodologia da Camila e adequação

O artigo (§3.5) segue o benchmark de Breuer et al. (2000) — canal com *blockage*
$1/8$, cilindro **centrado**, entrada **parabólica** ($\max=1$), paredes
**free-slip**, saída *outflow*, força pela **integral do tensor de Cauchy**
$\sigma = -pI + \mu(\nabla u+\nabla u^\top)$ sobre o obstáculo (Eqs. 43–44) e
$Re, St$ com $\lVert u_\infty\rVert=\max$ do inlet (Eqs. 41–42). A pressão usada
nas forças é a **do próprio método** (projeção), não um Poisson auxiliar.

As adequações P1–P5 alinham a implementação a esse protocolo. Depois delas, o
benchmark comparável é o da **Tabela 3** (varredura de $\alpha$ e espaços de
interface), com fine mesh $C_D=1.33$, $C_L=0.263$, $St=0.121$ e MPrM atingindo
erro relativo $\sim10^{-3}$ para $\alpha=10^{-6}$ com $P_{H,3}$.

---

## 6. Referências

- C. Lages, *The multiscale projection method for the solution of incompressible
  Navier–Stokes equations*, §3.5, Tabelas 2–3, Eqs. 41–44.
- M. Breuer, J. Bernsdorf, T. Zeiser, F. Durst, *Accurate computations of the
  laminar flow past a square cylinder…*, Int. J. Heat Fluid Flow 21 (2000)
  186–196.
- Guia de execução: [`docs/von-karman-guia-execucao.md`](von-karman-guia-execucao.md).
