# Plano de Validação — Benchmark von Kármán (MPrM)

> **Branch:** `mprm-opt-parallel` | **Data:** 2026-07-22
> **Objetivo:** Validar quantitativamente o MPrM contra a literatura e diagnosticar déficits remanescentes.

---

## 1. Referências de benchmark

### 1.1 Artigo de Lages (2024) — referência primária

**Fonte:** *"The multiscale projection method for the solution of incompressible Navier-Stokes equations"*, §3.5, Tabelas 2–3.

Setup do artigo:
- Domínio: $L_x \times L_y$ (não especificado explicitamente, mas inferred: canal alongado)
- Obstáculo: quadrado de lado $D$, centrado
- Perfil de entrada: **parabólico**, normalizado pelo **máximo** ($u_{\max} = 1$)
- Paredes: **free-slip** (deslizamento)
- $Re = u_{\max} D / \nu = 100$
- Malha fina de referência: **não decomposta** (Poisson global)
- MRCM: subdomínios com 30×30 células cada (inferido de $N_H=3$, malha 90²)
- Espaços de interface: $P_{H,k}$ (pressão), $U_{H,k}$ (fluxo), $k = 0,1,2,3$
- $\alpha \in \{10^{-6}, 1, 10^6\}$

**Valores de referência (Tabela 2, $\alpha=1$):**

| Configuração | $C_D$ | Erro $C_D$ | $C_L$ (amplitude) | Erro $C_L$ | $St$ |
|---|---|---|---|---|---|
| **Fine mesh** (referência) | **1.33** | — | **0.263** | — | **0.121** |
| $P_{H,2}, U_{H,2}$ | 1.35 | $1.1\times10^{-2}$ | 0.267 | $1.3\times10^{-2}$ | 0.121 |
| $P_{H,3}, U_{H,3}$ | 1.33 | $9.6\times10^{-4}$ | 0.264 | $3.1\times10^{-3}$ | 0.121 |
| $P_{H,3}, U_{H,2}$ | 1.33 | $1.9\times10^{-3}$ | 0.263 | $1.9\times10^{-3}$ | 0.121 |

**Valores de referência (Tabela 3, variação de $\alpha$):**

| $\alpha$ | Espaços | $C_D$ | $C_L$ | $St$ |
|----------|---------|-------|-------|------|
| $10^{-6}$ | $P_{H,3}, U_{H,2}$ | 1.33 | 0.263 | 0.121 |
| $1$ | $P_{H,3}, U_{H,2}$ | 1.35 | 0.267 | 0.121 |
| $10^{6}$ | $P_{H,3}, U_{H,2}$ | — (não reportado) | — | — |

> **Conclusão do artigo:** melhores resultados com $\alpha = 10^{-6}$ e pressão rica ($P_{H,3}$). O enriquecimento da pressão de interface domina a qualidade em escoamentos transientes.

### 1.2 Breuer et al. (2000) — benchmark internacional

**Fonte:** *"Accurate computations of the laminar flow past a square cylinder based on two different methods: lattice-Boltzmann and finite-volume"*, IJHFF 21:186–196.

Setup:
- Canal com bloqueio $D/H = 1/8$ (altura $L_y = 8D$)
- Cilindro quadrado centrado
- Perfil parabólico na entrada (máximo = 1)
- $Re = 100$

| Método | $C_D$ | $St$ |
|--------|-------|------|
| Lattice-Boltzmann | 1.38–1.47 | 0.130–0.145 |
| Finite-volume | 1.38–1.47 | 0.130–0.145 |

> Nota: $C_L$ não é reportado neste benchmark.

### 1.3 Domain Decomposition ROM (2020) — setup idêntico

**Fonte:** *"A domain decomposition strategy for reduced order models. Application to the incompressible Navier–Stokes equations"*, §7.1.

Setup (exatamente igual ao nosso):
- Domínio: $16 \times 8$
- Cilindro de diâmetro 1 centrado em $(4,4)$
- Inflow horizontal = 1, viscosidade $\mu = 1$ → $Re = 100$
- **Sem valores quantitativos de $C_D$, $C_L$, $St$ reportados** (artigo foca em comparação FOM vs ROM, não em validação de benchmark)

### 1.4 Dokken et al. (2020) — multimesh projection

**Fonte:** *"A multimesh finite element method for the Navier-Stokes equations based on projection methods"*, CMAME 2020.

Setup:
- Benchmark **Turek-Schäfer** (DFG 2D-3): cilindro **circular** (não quadrado)
- $Re = 100$, domínio $2.2 \times 0.41$, cilindro $D = 0.1$
- Valores de referência: $C_D \in [2.95, 3.23]$, $C_L \in [0.48, 0.99]$, $St \in [0.29, 0.31]$
- ⚠️ Geometria diferente — não diretamente comparável

---

## 2. Estado atual da implementação

### 2.1 Melhor resultado obtido (com forças viscosas corrigidas)

| Parâmetro | Valor |
|-----------|-------|
| Setup | Uniform inflow, no-slip walls, $x_c=3, y_c=3$ |
| Malha | $48^2$ células, $Ndom=4, nloc=12$ |
| Espaços | $U_{H,3}, P_{H,3}$ |
| $T$ | $60$ ($D/U_\infty$) |

| Métrica | Nosso | Artigo (fine) | Artigo (MPrM) | Status |
|---------|-------|---------------|---------------|--------|
| $C_D$ | **1.44** | 1.33 | 1.33–1.35 | 🟡 +8% |
| $C_L$ (amp) | **~0.003** | 0.263 | 0.25–0.31 | 🔴 100× menor |
| $St$ | **0.124** | 0.121 | 0.121 | 🟢 +2.5% |
| $\max\|\nabla\cdot u\|$ | **$10^{-14}$** | — | — | 🟢 |

### 2.2 Déficits identificados

| # | Problema | Severidade | Causa raiz |
|---|----------|------------|------------|
| 1 | $C_L \approx 0$ — sem shedding | 🔴 Crítico | $Re$ efetivo baixo (perfil uniforme = $U_{avg}=1$ vs parabólico com $U_{max}=1$ → $U_{avg}=2/3$); obstáculo descentrado quebra simetria |
| 2 | $C_D$ ~8% alto | 🟡 Moderado | No-slip walls → bloqueio efetivo maior; possível over-estimation do termo viscoso |
| 3 | $St$ depende do bin da FFT | 🟡 Resolvido (T4) | Zero-padding implementado |
| 4 | Malha grossa ($D/h \approx 6$) | 🟡 Moderado | $Ndom_x=Ndom_y$ força células não-quadradas ou obstáculo deslocado |
| 5 | Setup não-canônico | 🟡 Moderado | Obstáculo em $(3,3)$, não $(4,4)$; $L_x=16$ pode ser curto |

---

## 3. Plano de validação progressiva

### Fase 1 — Reproduzir setup do artigo (correções de setup)

**Objetivo:** Rodar com o **mesmo** setup físico do artigo (parabólico max=1, free-slip, obstáculo centrado).

**Configuração alvo:**
```python
run_von_karman(Lx=16, Ly=8, D=1, xc=4, yc=4, U_in=1, Re=100,
               Ndomx=10, Ndomy=5, nloc=30,   # 300×150, h=0.053, D/h=19
               degreeU=3, degreeP=3,
               inflow="parabolic", walls="slip", inflow_norm="max",
               alpha=1.0, T=200)
```

**Métricas esperadas:** $C_D \in [1.28, 1.38]$, $C_L \in [0.22, 0.32]$, $St \in [0.118, 0.126]$.

**Custo estimado:** ~15–20 min com 4 ranks MPI.

### Fase 2 — Validação cruzada com fine mesh solver

**Objetivo:** Comparar MPrM vs solver fino global (sem decomposição) no mesmo setup.

```python
# Fine mesh (referência)
r_fine = run_von_karman(..., method="fine")
# MPrM
r_mprm = run_von_karman(..., method="mprm")
```

**Métrica:** $\|C_D^{mprm} - C_D^{fine}\| / C_D^{fine} < 5\%$.

### Fase 3 — Varredura de parâmetros (igual artigo)

**Objetivo:** Reproduzir Tabelas 2–3 do artigo.

| Parâmetro | Valores |
|-----------|---------|
| $\alpha$ | $10^{-6}, 1, 10^6$ |
| $U_H$ | graus 1, 2, 3 |
| $P_H$ | graus 1, 2, 3 |
| $Ndom_x \times Ndom_y$ | $10\times5, 12\times6, 16\times8$ |

### Fase 4 — Comparação com Breuer et al. (2000)

**Objetivo:** Validar contra benchmark internacional independente.

Setup Breuer: $L_x$ maior, $L_y = 8D$, perfil parabólico, $D/h \ge 20$.

---

## 4. Protocolo de medição

### 4.1 Coeficiente de arrasto ($C_D$)

$$
C_D = \frac{2 F_x}{\rho U_\infty^2 D}
$$

Onde $F_x$ é a força horizontal (direção do escoamento) no obstáculo, calculada por balanço de momento no volume de controle (termos convectivo + pressão + viscoso + transiente). $\rho = 1$, $U_\infty = U_{in}$ (pico do perfil parabólico).

**Pós-processamento:** descartar 50% inicial (transiente), reportar média e desvio padrão.

### 4.2 Coeficiente de sustentação ($C_L$)

$$
C_L = \frac{2 F_y}{\rho U_\infty^2 D}
$$

Reportar **amplitude** (pico a pico / 2) para comparar com o artigo, **e** rms para comparar com Breuer.

### 4.3 Número de Strouhal ($St$)

$$
St = \frac{f D}{U_\infty}
$$

$f$ = frequência dominante de $C_L(t)$ via FFT com:
- Zero-padding ($pad=8\times$)
- Interpolação parabólica de 3 pontos no pico
- Janela de Hanning
- Descartar 25–50% inicial (transiente)
- $T \ge 180$ para $\Delta St \le 0.006$

### 4.4 Conservação de massa

$\max |\nabla \cdot u|$ na malha MAC — deve ser $\sim 10^{-14}$ (precisão de máquina).

---

## 5. Tabela comparativa completa

| Fonte | Geometria | $C_D$ | $C_L$ (amp) | $St$ | Notas |
|-------|-----------|-------|-------------|------|-------|
| **Lages (2024) — fine mesh** | Canal, quadrado centrado | 1.33 | 0.263 | 0.121 | Referência primária |
| **Lages (2024) — MPrM** $P_{H,3},U_{H,3}$ | Canal, 3×3 subd, 90² | 1.33 | 0.264 | 0.121 | Melhor caso |
| **Breuer et al. (2000)** — LBM | $D/H=1/8$, parabólico | 1.38–1.47 | — | 0.130–0.145 | Benchmark clássico |
| **Breuer et al. (2000)** — FVM | $D/H=1/8$, parabólico | 1.38–1.47 | — | 0.130–0.145 | |
| **DD-ROM (2020)** | $16\times8$, $D=1$ em $(4,4)$ | — | — | — | Setup idêntico, sem valores |
| **Dokken (2020)** — multimesh | Cilindro circular, Turek-Schäfer | 2.95–3.23 | 0.48–0.99 | 0.29–0.31 | Geometria diferente |
| **Nosso (atual)** — uniform, no-slip | $16\times8$, $D=1$ em $(3,3)$ | 1.44 | ~0.003 | 0.124 | Pré-validação |
| **Nosso (alvo Fase 1)** | $16\times8$, $D=1$ em $(4,4)$, parabólico, slip | [1.28,1.38] | [0.22,0.32] | [0.118,0.126] | A implementar |

---

## 6. Check-list de implementação

- [ ] **Fase 1:** Rodar benchmark com setup canônico ($Ndom_x=10, Ndom_y=5, nloc=30$, $T \ge 180$)
- [ ] **Fase 1:** Verificar $C_D$, $C_L$, $St$ contra Tabela 2 do artigo
- [ ] **Fase 2:** Rodar `method="fine"` no mesmo setup para referência interna
- [ ] **Fase 3:** Varredura $\alpha \in \{10^{-6}, 1, 10^6\}$ com $P_{H,3}, U_{H,2}$
- [ ] **Fase 3:** Varredura de espaços $(U_{H,k}, P_{H,k})$ para $k=1,2,3$
- [ ] **Fase 4:** Comparar $C_D$, $St$ com Breuer et al. (2000)
- [ ] **Documentação:** Atualizar `docs/07-diagnostico-von-karman.md` com resultados finais
- [ ] **Handoff:** Gerar `handoffs/2026-07-22-mprm-validation.md` com tabelas e figuras

---

## 7. Riscos e mitigações

| Risco | Probabilidade | Mitigação |
|-------|--------------|-----------|
| $C_L$ continuar ~0 após correções | Média | Verificar simetria do setup; testar com perturbação artificial no campo inicial |
| Tempo de simulação excessivo ($T=200$, $300\times150$) | Alta | Usar 4+ ranks MPI; reduzir $T$ para 100 e verificar tendência |
| Divergir do artigo por diferenças não documentadas (ex.: $L_x$ exato) | Média | Testar variações de $L_x$ (16, 20, 24); comparar com fine mesh interno |
| Memória insuficiente para $300\times150$ | Baixa | Reduzir `nloc` para 20 ($200\times100$, $D/h=13$) |
