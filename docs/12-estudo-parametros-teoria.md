# Estudo Teórico dos Parâmetros do MPrM — Análise da Ausência de Vortex Shedding no Benchmark de von Kármán

**Data:** 2026-07-22 · **Atualizado:** 2026-07-22 (resultados experimentais)  
**Contexto:** Simulação MPrM do escoamento em torno de cilindro quadrado ($Re=100$)  
**Problema ativo:** Coeficiente de sustentação $C_L \approx 0$ — ausência de emissão alternada de vórtices (von Kármán vortex street)  
**Referências:** `~/Dissertação/Refs/markdown/`

---

## ⭐ Resultados Experimentais (2026-07-22)

Testes sistemáticos das hipóteses com $Re=100$, perfil parabólico, paredes free-slip, $T=60$, 4 ranks MPI:

| Experimento | $D/h$ | Esquema | Método | $C_D$ | $C_L$ (amp) | $St$ | $\max\|\nabla\cdot\mathbf{u}\|$ | Tempo |
|---|---|---|---|---|---|---|---|---|---|
| **H1** `nloc=30` | **18.75** | superbee | MRCM | **$1.31 \pm 0.02$** | **$\mathbf{0.113}$** ✅ | $0.48$ | $8.4 \times 10^{-14}$ | 763s |
| H2 `nloc=20` | 12.5 | superbee | MRCM | $1.23 \pm 0.006$ | $\mathbf{0}$ ❌ | — | $2.7 \times 10^{-14}$ | 401s |
| H5 `fine` | 12.5 | superbee | Poisson global | $1.20 \pm 0.005$ | $\mathbf{0}$ ❌ | — | $0.12$ ⚠️ | 25s |
| Ref (TVD T80) | 12.5 | minmod | MRCM | $1.68$ | $\mathbf{0}$ ❌ | — | $7.5 \times 10^{-14}$ | 586s |
| Ref (upwind) | 12.5 | upwind 1ª | MRCM | $1.15$–$1.58$ | $\mathbf{0}$ ❌ | — | $\sim 10^{-14}$ | 400–765s |

**Conclusão experimental:** 
- ✅ **H1 (resolução) é a causa dominante.** Com $D/h = 18.75$ ($\approx 1.9$ células na camada limite), o shedding emerge ($C_L = 0.11$). Com $D/h = 12.5$ ($\approx 1.25$ células), $C_L = 0$ em TODAS as configurações testadas.
- ✅ **H5 descartada:** O método `fine` (Poisson global sem MRCM) produz o mesmo $C_L = 0$ que o MRCM — a decomposição multiescala **não** é a causa.
- ✅ **H2 parcial:** Superbee sozinho (sem refinar malha) não basta para disparar shedding.
- ⚠️ **$St = 0.48$** está ~3–4× acima do esperado ($0.12$–$0.15$). Possível harmônico ou ruído na FFT com amplitude baixa ($C_L = 0.11$). Requer $T$ mais longo ou filtragem.
- ⚠️ **$C_L = 0.11$** é ~2× menor que a literatura ($0.22$–$0.32$). Malha ainda marginal ou necessidade de $T > 60$ para shedding plenamente desenvolvido.

---

## 1. Formulação do Problema

### 1.1 Equações de Navier-Stokes Incompressível

As equações governantes, na forma adimensionalizada por $\rho$, são [Lages, §2.1; Quarteroni et al. 2000]:

$$
\begin{aligned}
\frac{\partial \mathbf{u}}{\partial t} + \nabla \cdot (\mathbf{u}\mathbf{u}) &= -\nabla p + \nu \nabla^2 \mathbf{u} + \mathbf{f} \quad &&\text{(1) — momento} \\
\nabla \cdot \mathbf{u} &= 0 \quad &&\text{(2) — continuidade}
\end{aligned}
$$

onde $\nu = \mu/\rho = U_\infty D / Re$ é a viscosidade cinemática. Para o benchmark de von Kármán: $U_\infty = 1$, $D = 1$, $Re = 100$ ⇒ $\nu = 0.01$.

### 1.2 Método de Projeção de Chorin (Não-Incremental)

O MPrM usa o método de projeção clássico de Chorin [Chorin 1968; Quarteroni et al. 2000, §5] com avanço explícito de 1ª ordem no tempo [Lages, §2.1, Eqs. 8–10]:

1. **Velocidade tentativa** (transporte explícito, sem $\nabla p$):
   $$\mathbf{w}^* = \mathbf{u}^n - \delta t \, \nabla \cdot (\mathbf{u}^n \mathbf{u}^n) + \delta t \, \nu \nabla^2 \mathbf{u}^n + \delta t \, \mathbf{f}^n \quad \text{(8)}$$

2. **Projeção** (decomposição de Helmholtz-Hodge):
   $$\delta t \, \nabla^2 p^{n+1} = \nabla \cdot \mathbf{w}^* \;\;\text{em}\;\; \Omega, \quad \delta t \, \nabla p^{n+1} \cdot \mathbf{n} = \mathbf{w}^* \cdot \mathbf{n} \;\;\text{em}\;\; \partial\Omega \quad \text{(9)}$$
   $$\mathbf{v}^{n+1} = \delta t \, \nabla p^{n+1}$$

3. **Correção**:
   $$\mathbf{u}^{n+1} = \mathbf{w}^* - \mathbf{v}^{n+1} \quad \text{(10)}$$

### 1.3 Decomposição Multiescala (MRCM)

A etapa de projeção (passo 2) é um problema de Poisson com condição de Neumann — estruturalmente idêntico a um problema de Darcy com $K = \delta t \cdot I$ constante. O MRCM [Guiraldello et al. 2018; Guiraldello 2019] resolve esse problema via decomposição de domínio não-sobreposta com acoplamento de Robin nas interfaces [Lages, §2.3–2.5]:

- **Parâmetro de Robin $\alpha = \beta/H$**: controla o balanço entre continuidade de fluxo ($\alpha \to 0$, limite MMMFEM) e continuidade de pressão ($\alpha \to \infty$, limite MHM) nas interfaces do esqueleto $\Gamma$.
- **Funções de base multiescala (MsBF)**: resolvem problemas locais de Robin por subdomínio; $4(k_U + k_P)$ MsBF por subdomínio interno em 2D.
- **Sistema global de interface**: $A\mathbf{x} = \mathbf{b}$ esparso, com incógnitas $U_H$ (fluxo) e $P_H$ (pressão) na escala grossa $H$.
- **Downscaling** [Lages, Eqs. 21–23]: restaura balanço de massa na escala fina $h$ via problemas locais de Neumann com fluxo médio de interface — este passo é $O(h^2)$ e recupera superconvergência [docs/06-convergencia-hh.md].

---

## 2. Parâmetros do Método e sua Influência

### 2.1 Parâmetro de Robin $\alpha$

Controla o acoplamento fraco entre subdomínios [Lages, §3.1; Guiraldello et al. 2018, §3]:

| $\alpha$ | Limite | Comportamento |
|----------|--------|---------------|
| $\alpha \to 0$ | MMMFEM | Continuidade fraca de pressão; fluxo contínuo forte |
| $\alpha \approx 0.1$–$1.0$ | MRCM ótimo | Melhor balanço erro vs estabilidade |
| $\alpha \to \infty$ | MHM | Continuidade fraca de fluxo; pressão contínua forte |

**Evidência experimental** [docs/05-estudo-alpha.md]:
- HH modificado: ótimo em $\alpha = 0.1$ para todos os graus $k = 1..4$ (1.3–2.3× melhor que $\alpha = 1$)
- Cavidade $Re = 100$: erro decresce monotonicamente com $\alpha$, mínimo em $\alpha = 10^4$ ($3.26 \times 10^{-3}$)
- **Para von Kármán:** usa-se $\alpha = 1.0$ (valor canônico do artigo, Tabela 3)

**Impacto no shedding:** O parâmetro $\alpha$ afeta o erro de decomposição espacial, não a física do escoamento. Com $\max|\nabla \cdot \mathbf{u}| \sim 10^{-14}$ (reportado em todas as simulações), o erro de projeção é desprezível — **$\alpha$ não é a causa da ausência de shedding.**

### 2.2 Graus dos Espaços de Interface ($k_U$, $k_P$)

Controlam a riqueza da representação multiescala [Lages, §3.1, Tabela 1; Guiraldello et al. 2018, §4]:

| $k_U$ | $k_P$ | Interpretação |
|-------|-------|---------------|
| 1 | 1 | Fluxo linear, pressão constante (mínimo viável) |
| 2 | 2 | Fluxo quadrático, pressão linear |
| 3 | 3 | Fluxo cúbico, pressão quadrática **(usado em von Kármán)** |

Com $k_U = k_P = 3$ (nossos `degreeU=3, degreeP=3`), o erro de decomposição é $O(H^4)$ na escala grossa — o erro espacial do MRCM é ordens de grandeza menor que o erro do transporte MAC. **Os graus não limitam a física.**

> **Nota:** O estudo de convergência Hh [docs/06-convergencia-hh.md] mostra que, com graus simétricos e $\alpha = 1$, o MPrM atinge $O(H^{1.99})$ (superconvergência via downscaling), não $O(H)$ como o MRCM cru.

### 2.3 Resolução da Malha ($N_x \times N_y$, $h$, $n_{loc}$, $N_{dom}$)

A malha global é $N_x = N_{dom,x} \cdot n_{loc}$, $N_y = N_{dom,y} \cdot n_{loc}$. Para von Kármán:

| Configuração | $N_x \times N_y$ | $h$ | $D/h$ (células/diâmetro) |
|---|---|---|---|
| Atual (`Ndomx=10,Ndomy=5,nloc=20`) | $200 \times 100$ | $0.08$ | $12.5$ |
| Artigo (Lages, Tabela 3) | $320 \times 160$ | $0.05$ | $20$ |
| Literatura típica (Re=100) | — | — | $20$–$40$ |

**Resolução $D/h \approx 12.5$ é marginal para $Re = 100$.** A camada limite laminar tem espessura $\delta \sim D / \sqrt{Re} = 0.1D$. Com $h = 0.08$, temos apenas $\approx 1.25$ células na camada limite — insuficiente para resolver o gradiente de cisalhamento que dispara a instabilidade de Kelvin-Helmholtz responsável pelo shedding.

**Recomendação:** $n_{loc} \ge 30$ ($D/h = 18.75$) ou $n_{loc} \ge 40$ ($D/h = 25$) para garantir $\ge 2$–$3$ células na camada limite.

### 2.4 Passo de Tempo e Condição CFL

#### 2.4.1 Derivação da Condição CFL

O transporte explícito na malha MAC impõe restrições de estabilidade [Quarteroni et al. 2000, §2–3; Lages, §2.6]. Para o esquema upwind de 1ª ordem com difusão centrada, a condição CFL combinada (convecção + difusão) é:

$$
\delta t \le \min\left( C_{adv} \frac{h_{min}}{U_{max}}, \; C_{diff} \frac{h_{min}^2}{2\nu} \right)
$$

onde:
- $h_{min} = \min(h_x, h_y)$ — menor espaçamento da malha
- $U_{max}$ — velocidade máxima característica
- $\nu$ — viscosidade cinemática
- $C_{adv} \le 1$ (upwind 1ª ordem) ou $C_{adv} \le 0.5$ (TVD 2ª ordem)
- $C_{diff} \le 1$ (explícito, centrado)

**Nossa implementação** (`mprm/flow.py:run_flow`):
```python
dt = 0.4 * min(h_min / U_ref, h_min * h_min / (4.0 * nu))
```
Isso corresponde a $C_{adv} = 0.4$, $C_{diff} = 0.1$ — **conservador e estável.**

#### 2.4.2 Números de CFL para o Caso von Kármán

| Parâmetro | Valor | Fórmula |
|-----------|-------|---------|
| $h_{min}$ | $0.08$ | $L_x / N_x = 16/200$ |
| $U_{max}$ | $\approx 1.5$ | Velocidade máxima na esteira |
| $\nu$ | $0.01$ | $U_\infty D / Re$ |
| $\delta t_{adv}$ | $0.0213$ | $0.4 \cdot 0.08 / 1.5$ |
| $\delta t_{diff}$ | $0.064$ | $0.4 \cdot 0.08^2 / (4 \cdot 0.01)$ |
| $\delta t$ **usado** | $0.0213$ | $\min(\delta t_{adv}, \delta t_{diff})$ |
| $CFL_{adv}$ efetivo | $\approx 0.4$ | $U_{max} \delta t / h$ |
| $CFL_{diff}$ efetivo | $\approx 0.033$ | $2\nu \delta t / h^2$ |

O passo de tempo é dominado pela restrição convectiva ($CFL_{adv} = 0.4$), não pela difusiva ($CFL_{diff} \ll 1$). **O passo de tempo é adequado; não há violação de CFL.**

#### 2.4.3 Número de Passos e Tempo Físico

| $T$ | $\delta t$ | Passos | Períodos de shedding ($St \approx 0.13$) |
|-----|-----------|--------|------------------------------------------|
| $30$ | $0.032$ | $938$ | $\approx 3.9$ |
| $60$ | $0.032$ | $1875$ | $\approx 7.8$ |
| $80$ | $0.032$ | $2500$ | $\approx 10.4$ |

Mesmo com $T = 80$ (10+ períodos esperados), $C_L \approx 0$ persiste — indicando que **o problema não é transiente insuficiente, mas sim que o shedding nunca se inicia.**

### 2.5 Ordem da Convecção (Esquema Upwind vs TVD)

A dissipação numérica do esquema de advecção é crítica para escoamentos vorticais [Quarteroni et al. 2000, §7; Dokken et al. 2020, §4]:

| Esquema | Ordem | Dissipação numérica efetiva | $C_D$ observado | Estabilidade |
|---------|-------|-----------------------------|-----------------|--------------|
| Upwind 1ª ordem (donor-cell) | $O(h)$ | $\nu_{num} \approx \frac{1}{2}|u|h(1 - CFL)$ | $1.44$–$1.58$ | ✅ Estável |
| TVD 2ª ordem (minmod) | $O(h^2)$ | $\nu_{num} \ll$ upwind | $1.73$ | ✅ Estável |
| TVD 2ª ordem (superbee) | $O(h^2)$ | Mínima entre TVD | — | ✅ Estável |
| Centrado 2ª ordem (sem limiter) | $O(h^2)$ | $\approx 0$ | NaN (instável) | ❌ Overflow |

**A dissipação numérica do upwind de 1ª ordem age como uma viscosidade artificial $\nu_{num} \propto h$.** Com $h = 0.08$ e $U \approx 1$, temos $\nu_{num} \approx 0.04$ — **4× maior que a viscosidade física $\nu = 0.01$.** Isso suprime flutuações de pequena escala (vórtices) e estabiliza artificialmente o escoamento, prevenindo o shedding.

A mudança para TVD 2ª ordem reduziu a dissipação (evidência: $C_D$ subiu de $1.44$ para $1.73$), mas mesmo com minmod/superbee, $C_L \approx 0$ persiste. **A dissipação numérica residual do limitador TVD ainda pode ser excessiva para $D/h = 12.5$,** ou o problema primário é a resolução insuficiente da camada limite (ver §2.3).

### 2.6 Condições de Contorno

O tratamento de BCs no MPrM segue a prescrição clássica de Chorin [Quarteroni et al. 2000, §5; Lages, §2.2]:

| Fronteira | Física ($\mathbf{u}$) | Projeção ($\mathbf{v} = \delta t \nabla p$) | Pressão ($p$) |
|-----------|----------------------|---------------------------------------------|---------------|
| Entrada (`inflow`) | $\mathbf{u} \cdot \mathbf{n} = U_{in}(y)$, $\mathbf{u} \cdot \boldsymbol{\tau} = 0$ | $\mathbf{v} \cdot \mathbf{n} = 0$ (Neumann homogêneo) | $\partial p/\partial n = 0$ |
| Paredes (`wall`) | $\mathbf{u} = 0$ (no-slip) | $\mathbf{v} \cdot \mathbf{n} = 0$ | $\partial p/\partial n = 0$ |
| Paredes (`slip`) | $\mathbf{u} \cdot \mathbf{n} = 0$, $\partial u_\tau/\partial n = 0$ | $\mathbf{v} \cdot \mathbf{n} = 0$ | $\partial p/\partial n = 0$ |
| Saída (`outflow`) | Livre | $\mathbf{v} \cdot \mathbf{n} \neq 0$ (livre) | $p = 0$ (Dirichlet) |
| Obstáculo | $\mathbf{u} = 0$ (no-slip) | $\mathbf{v} \cdot \mathbf{n} = 0$ (via $K \to \varepsilon$) | $\partial p/\partial n = 0$ |

**Verificação de corretude:** O divergente $\max|\nabla \cdot \mathbf{u}| \sim 10^{-14}$ em todas as simulações confirma que o balanço de massa é exato e as BCs da projeção estão corretas. **As BCs não são a causa do problema.**

### 2.7 Perfil de Entrada e Perturbação Inicial

A condição inicial e o perfil de entrada afetam o transiente que dispara a instabilidade:

| Configuração | Perfil | Perturbação |
|---|---|---|
| Atual | Uniforme ($U_{in} = 1$) ou parabólico ($U_{max} = 1$) | Nenhuma (campo inicial $= 0$ + perfil de entrada) |
| Literatura | Uniforme ou parabólico | Ruído ou assimetria inicial |

Em simulações numéricas de escoamentos com simetria geométrica (domínio retangular, obstáculo centrado), o shedding de von Kármán é uma **instabilidade que requer uma perturbação assimétrica para ser disparada.** Em simulações de alta precisão (como as nossas, com $\max|\nabla \cdot \mathbf{u}| \sim 10^{-14}$), o ruído numérico de arredondamento pode ser insuficiente para quebrar a simetria dentro do tempo de simulação.

**Possíveis soluções:**
1. Adicionar perturbação explícita no campo inicial: $\mathbf{u}(t=0) = \mathbf{u}_0 + \epsilon \cdot \boldsymbol{\eta}(\mathbf{x})$ com $\epsilon \sim 10^{-2} U_\infty$
2. Deslocar ligeiramente o obstáculo da linha de centro: $y_c = L_y/2 + \delta y$, $\delta y \sim 0.01D$–$0.05D$
3. Usar perfil de entrada ligeiramente assimétrico
4. Aguardar tempo muito longo ($T > 200$) para que erros de arredondamento se acumulem

---

## 3. Diagnóstico: Por que $C_L = 0$?

### 3.1 Hipóteses e Evidências

| # | Hipótese | Evidência a favor | Evidência contra | Probabilidade |
|---|----------|-------------------|------------------|---------------|
| H1 | **Resolução insuficiente** ($D/h = 12.5$) | $\delta_{BL} \approx 0.1D$, $h/\delta_{BL} \approx 0.8$ — menos de 1 célula na camada limite | Simulações com upwind reportam $C_D \approx 1.44$–$1.58$ (fisicamente plausível) | **ALTA** |
| H2 | **Dissipação numérica** do upwind/TVD | $C_D$ caiu de $1.73$ (TVD) para $1.44$ (upwind) — dissipação suprime flutuações | TVD 2ª ordem com superbee é minimamente dissipativo; $C_D = 1.73$ sugere que a física convectiva está presente | **MÉDIA** |
| H3 | **Simetria numérica exata** (sem perturbação) | $\max|\nabla \cdot \mathbf{u}| \sim 10^{-14}$ — máquina de precisão dupla sem ruído assimétrico significativo | Escoamentos Re=100 em malhas grosseiras tipicamente desenvolvem shedding mesmo sem perturbação explícita | **MÉDIA** |
| H4 | **Tempo de simulação insuficiente** | $T = 80$ cobre $\approx 10$ períodos esperados — suficiente para shedding estabelecido | O transiente inicial até o início do shedding pode levar $> 50$ unidades de tempo convectivo em alguns casos | **BAIXA** |
| H5 | **Erro na implementação das forças** | Foram identificados e corrigidos bugs anteriores (ausência de termos viscosos e transientes nas forças) | $C_D = 1.73$ é fisicamente razoável; $C_L = 0$ com $C_D \neq 0$ sugere escoamento simétrico, não bug de medição | **BAIXA** |
| H6 | **BCs incorretas** na saída ou obstáculo | $\max|\nabla \cdot \mathbf{u}| \sim 10^{-14}$ — conservação de massa exata | — | **DESPREZÍVEL** |

### 3.2 Diagnóstico Principal

**A causa mais provável é uma combinação de H1 + H2 + H3:**

1. **Resolução marginal** ($D/h = 12.5$) → a camada limite não é adequadamente resolvida, o que suaviza os gradientes de cisalhamento que alimentam a instabilidade de Kelvin-Helmholtz.

2. **Dissipação numérica residual** — mesmo com TVD 2ª ordem, o limitador (minmod ou superbee) introduz difusão numérica local onde detecta gradientes elevados (justamente na camada limite e na região de separação). Com apenas ~1 célula na camada limite, o limitador atua em TODA a região de cisalhamento, efetivamente suprimindo a instabilidade.

3. **Simetria numérica** — sem perturbação explícita, o escoamento permanece simétrico em relação à linha de centro $y = L_y/2$. Em precisão dupla, o ruído de arredondamento ($\approx 10^{-16}$) é pequeno demais para disparar a instabilidade dentro de $T = 80$.

### 3.3 Referencial Teórico

O fenômeno de von Kármán vortex street é uma **instabilidade global do escoamento base simétrico**. A teoria de estabilidade hidrodinâmica [Drazin & Reid 2004] estabelece que:

- O escoamento base (simétrico, estacionário) torna-se **instável** para $Re > Re_{cr} \approx 47$ (cilindro circular) / $Re_{cr} \approx 50$–$60$ (cilindro quadrado).
- A instabilidade se manifesta como um **modo de Hopf** (oscilação auto-sustentada com frequência característica $St \approx 0.12$–$0.16$ para $Re = 100$).
- Em simulações numéricas, o crescimento da perturbação a partir do ruído de arredondamento segue $\|\delta \mathbf{u}\| \sim e^{\sigma t}$ onde $\sigma = \Re(\lambda)$ é a taxa de crescimento do modo mais instável.

Com $\sigma \approx 0.05$–$0.1$ (estimado para $Re = 100$), o tempo para amplificar uma perturbação de $10^{-16}$ para $O(1)$ é:

$$
t_{crit} \sim \frac{\ln(10^{16})}{\sigma} \approx \frac{36.8}{0.05\text{–}0.1} \approx 368\text{–}736
$$

Isso excede em muito nossos $T = 80$. **Se o escoamento base é perfeitamente simétrico (como nossa discretização o produz em precisão dupla), a instabilidade jamais se desenvolve em tempo finito de simulação.**

---

## 4. Plano de Ação Corretiva

### 4.1 Resultado dos Testes (2026-07-22)

| Hipótese | Status | Evidência |
|----------|--------|-----------|
| **H1 — Resolução insuficiente** | ✅ **CONFIRMADA** | $D/h = 18.75$ produz $C_L = 0.11$; $D/h = 12.5$ produz $C_L = 0$ em todas as configs |
| H2 — Dissipação numérica | ⚠️ Parcial | Superbee ajuda ($C_D$ sobe) mas não basta sem refinar malha |
| H3 — Simetria numérica | ⚠️ Indeterminada | Offset $y_c = 4.02$ aplicado em todos os testes; efeito isolado não medido |
| H5 — Erro do MRCM | ✅ **DESCARTADA** | `method="fine"` produz mesmo $C_L = 0$ que MRCM |

### 4.2 Ações Recomendadas (Atualizado)

| Prioridade | Ação | Impacto Esperado |
|---|---|---|
| 🔴 **P1** | Refinar para $n_{loc} \ge 30$ ($D/h \ge 18.75$) | **Já confirmado:** shedding aparece |
| 🔴 **P1** | Rodar $T \ge 100$ com $n_{loc} = 30$ | Estatísticas de shedding mais limpas, $St$ correto |
| 🟡 **P2** | Refinar para $n_{loc} = 40$ ($D/h = 25$) | $C_L$ deve convergir para $\approx 0.25$ |
| 🟡 **P2** | Corrigir `ChannelFinePoisson` — $\max\|\nabla\cdot\mathbf{u}\| = 0.12$ | Referência fina confiável para benchmark |
| 🟢 **P3** | Adicionar perturbação explícita no campo inicial | Reduz transiente, permite $T$ menor |
| 🟢 **P3** | Investigar $St = 0.48$ (harmônico? FFT com poucos períodos?) | Validar contra literatura |

### 4.3 Simulação de Validação Recomendada

```python
run_von_karman(Lx=16, Ly=8, D=1, xc=4, yc=4, U_in=1, Re=100,
               Ndomx=10, Ndomy=5, nloc=30,         # 300×150, D/h=18.75
               degreeU=3, degreeP=3,
               inflow='parabolic', walls='slip', inflow_norm='max',
               alpha=1.0, T=100,                     # ~12 períodos de shedding
               limiter='superbee',
               obstacle_metrics_every=1)             # Alta resolução temporal p/ FFT
```

---

## 5. Referências

1. **Lages, C.** *The Multiscale Projection Method for the Solution of Incompressible Navier-Stokes Equations.* Artigo principal do MPrM. `Refs/markdown/The_multiscale_projection_method_for_the_solution_of_incompressible_Navier_Stokes_equations.md`

2. **Guiraldello, R.T.** *Multiscale Methods for Oil Reservoir Simulation.* Tese de Doutorado, ICMC-USP, 2019. Define o MRCM e sua formulação variacional. `Refs/markdown/RafaelTrevisanutoGuiraldello_revisada.md`

3. **Guiraldello, R.T.; Ausas, R.F.; Sousa, F.S.; Pereira, F.; Buscaglia, G.C.** *The Multiscale Robin Coupled Method for flows in porous media.* J. Comput. Phys. 355:1–21, 2018. Artigo canônico do MRCM. `Refs/markdown/MRCM.md`

4. **Quarteroni, A.; Saleri, F.; Veneziani, A.** *Factorization methods for the numerical approximation of Navier–Stokes equations.* Comput. Methods Appl. Mech. Engrg. 188:505–526, 2000. Formulação algébrica dos métodos de projeção e condição CFL. `Refs/markdown/Factorization methods for the numerical approximation of Navier–Stokes equations.md`

5. **Dokken, J.S.; Johansson, A.; Massing, A.; Funke, S.W.** *A multimesh finite element method for the Navier–Stokes equations based on projection methods.* Comput. Methods Appl. Mech. Engrg. 368:113129, 2020. Esquema de projeção incremental, referência para Turek-Schäfer (DFG 2D-3). `Refs/markdown/A multimesh finite element method for the Navier–Stokes equations based on projection methods.md`

6. **Abdulle, A.; Budáč, O.** *An adaptive finite element heterogeneous multiscale method for Stokes flow in porous media.* SIAM Multiscale Model. Simul. 13(1):256–290, 2015. FE-HMM — via de homogeneização multiescala. `Refs/markdown/An adaptive finite element heterogeneous multiscale.md`

7. **Araya, R.; Barrenechea, G.R.; Valentin, F.** *Stabilized finite element methods based on multiscale enrichment for the Stokes problem.* SIAM J. Numer. Anal. 44(1):322–348, 2006. Enriquecimento multiescala para Stokes — precursor do MHM. `Refs/markdown/STABILIZED FINITE ELEMENT METHODS BASED ON MULTISCALE ENRICHMENT FOR THE STOKES PROBLEM.md`

8. **Baiges, J.; Codina, R.; Idelsohn, S.** *A domain decomposition strategy for reduced order models. Application to the incompressible Navier–Stokes equations.* Comput. Methods Appl. Mech. Engrg. 267:23–42, 2013. ROM + DD para Navier-Stokes. `Refs/markdown/A domain decomposition strategy for reduced order models. Application to the incompressible Navier–Stokes equations.md`

9. **Chorin, A.J.** *Numerical solution of the Navier-Stokes equations.* Math. Comp. 22:745–762, 1968. Método de projeção original.

10. **Drazin, P.G.; Reid, W.H.** *Hydrodynamic Stability.* 2nd ed., Cambridge University Press, 2004. Teoria de estabilidade hidrodinâmica — instabilidade de Hopf e vortex shedding.

---

## Apêndice A: Resumo dos Parâmetros e Seus Valores

| Parâmetro | Símbolo | Valor Atual | Faixa Recomendada | Impacto no Shedding |
|-----------|---------|-------------|-------------------|---------------------|
| Número de Reynolds | $Re$ | $100$ | $50$–$200$ | Fixo (benchmark) |
| Resolução (células/$D$) | $D/h$ | $12.5$ | $\ge 20$ | **CRÍTICO** |
| Passo de tempo (CFL) | $\delta t$ | $0.032$ (CFL=0.4) | CFL $\le 0.5$ | Estável, adequado |
| Ordem da convecção | — | 2 (TVD minmod) | 2 (TVD superbee) | **ALTO** |
| Parâmetro de Robin | $\alpha$ | $1.0$ | $0.1$–$10$ | Baixo (erro $<10^{-14}$) |
| Graus de interface | $k_U, k_P$ | $3, 3$ | $1$–$4$ | Baixo |
| Subdomínios | $N_{dom,x} \times N_{dom,y}$ | $10 \times 5$ | $n_{loc} \ge 20$ por subdom | Indireto (via $h$) |
| Perfil de entrada | — | Uniforme/Parabólico | Parabólico | Baixo |
| Perturbação inicial | $\epsilon$ | $0$ | $10^{-2}$ | **CRÍTICO** |
| Tempo de simulação | $T$ | $80$ | $\ge 60$ | Adequado |

## Apêndice B: Fórmulas Relevantes

### B.1 Números Adimensionais

| Número | Definição | Valor (von Kármán) |
|--------|-----------|---------------------|
| Reynolds | $Re = U_\infty D / \nu$ | $100$ |
| Strouhal | $St = f D / U_\infty$ | $0.12$–$0.16$ (esperado) |
| CFL convectivo | $CFL_u = U \delta t / h$ | $\le 0.4$ |
| CFL difusivo | $CFL_\nu = \nu \delta t / h^2$ | $\le 0.1$ |
| Células na camada limite | $N_{BL} = \delta / h \approx D/(h\sqrt{Re})$ | $\approx 1.25$ |

### B.2 Estimativa da Dissipação Numérica (Upwind 1ª Ordem)

Para o esquema donor-cell na malha MAC, a viscosidade numérica efetiva é [Hirsch 2007]:

$$
\nu_{num} = \frac{1}{2} |u| h (1 - CFL_u) + O(h^2)
$$

Com $|u| \approx 1$, $h = 0.08$, $CFL_u = 0.4$:
$$\nu_{num} \approx \frac{1}{2} \cdot 1 \cdot 0.08 \cdot 0.6 = 0.024$$

Comparando com $\nu = 0.01$: $\nu_{eff} = \nu + \nu_{num} \approx 0.034$ → **$Re_{eff} \approx 29$** (3.4× menor que o nominal!).

Com TVD 2ª ordem (minmod/superbee), $\nu_{num}$ é reduzido em $\approx 1$–$2$ ordens de grandeza em regiões suaves, mas permanece $O(h^2)$ localmente onde o limitador atua (gradientes elevados).
