# Levantamento Bibliográfico — Métodos de Projeção para Navier-Stokes com Potencial de Acoplamento ao MRCM

**Data:** 2026-07-19
**Objetivo:** Identificar artigos recentes que utilizam métodos de projeção
(Chorin, fractional-step, pressure-correction) para as equações de
Navier-Stokes, com potencial de extensão via acoplamento com o MRCM
(Multiscale Robin Coupled Method).

**Fontes consultadas:** arXiv.org, ScienceDirect (Elsevier)
**Estratégia de busca:** queries por "projection method Navier-Stokes",
"fractional step", "pressure correction", "Helmholtz-Hodge decomposition",
combinados com "finite element", "domain decomposition", "multiscale",
"variable viscosity", "Brinkman", "Darcy".

---

## 1. Artigos Diretamente Relevantes ao MPrM (Projeção + Multiescala)

### 1.1 Projeção com Decomposição de Domínio

| # | Título | Autores | Ano | Fonte | Link |
|---|--------|---------|-----|-------|------|
| 1 | **A multimesh finite element method for the Navier-Stokes equations based on projection methods** | J.S. Dokken, A. Johansson, A. Massing, S.W. Funke | 2020 | CMAME | [arXiv:1912.06392](https://arxiv.org/abs/1912.06392) |
| 2 | **A coarse-grid incremental pressure-projection method for accelerating low Reynolds-number incompressible flow simulations** | A. Kashefi | 2018 | arXiv | [arXiv:1812.00185](https://arxiv.org/abs/1812.00185) |
| 3 | **Shear rate projection schemes for non-Newtonian fluids** | J. Deteix, D. Yakoubi | 2019 | CMAME | [arXiv:1902.05643](https://arxiv.org/abs/1902.05643) |

**Destaques:**
- **Dokken et al. (2020):** Usa múltiplas malhas não-conformes com condições
  de interface de Nitsche — conceito análogo ao MRCM (diferentes escalas h e H).
  A equação de Poisson da pressão é resolvida em uma malha diferente da
  velocidade. **Alta relevância:** o MRCM poderia substituir o acoplamento de
  Nitsche pela formulação de Robin.
- **Kashefi (2018):** *Coarse Grid Projection (CGP)* — resolve a equação de
  Poisson em malha mais grossa que a velocidade. Estrutura idêntica ao MPrM!
  Porém usa projeção incremental (não-Chorin) e multigrid, não DD multiescala.
  **Diretamente comparável ao MPrM.**

### 1.2 Convergência e Análise de Métodos de Projeção

| # | Título | Autores | Ano | Fonte | Link |
|---|--------|---------|-----|-------|------|
| 4 | **A convergence proof for a finite element discretization of Chorin's projection method** | F. Weber | 2025 | arXiv | [arXiv:2508.13416](https://arxiv.org/abs/2508.13416) |
| 5 | **On convergence of Chorin's projection method to a Leray-Hopf weak solution** | H. Kuroki, K. Soga | 2019 | arXiv | [arXiv:1809.04383](https://arxiv.org/abs/1809.04383) |
| 6 | **Stability analysis of pressure correction schemes for the Navier-Stokes equations with traction boundary conditions** | S. Lee, A.J. Salgado | 2016 | CMAME | [arXiv:1602.02072](https://arxiv.org/abs/1602.02072) |

**Destaques:**
- **Weber (2025):** Prova de convergência para elementos finitos do método de
  Chorin. Relevante para fundamentação teórica do MPrM com BDM₁–DG₀.
- **Lee & Salgado (2016):** Análise de estabilidade para *pressure correction
  schemes* com condições de contorno de tração (Neumann). **Alta relevância**
  pois o MPrM usa Neumann puro no problema de Poisson.

### 1.3 Estabilização e High-Order

| # | Título | Autores | Ano | Fonte | Link |
|---|--------|---------|-----|-------|------|
| 7 | **Projection-based stabilization for high-order incompressible flow solvers** | A. Blanco-Casares, V. Kumar, D. Mira, O. Lehmkuhl | 2025 | arXiv | [arXiv:2509.04352](https://arxiv.org/abs/2509.04352) |
| 8 | **High-order matrix-free incompressible flow solvers with GPU acceleration and low-order refined preconditioners** | M. Franco, J.S. Camier, J. Andrej, W. Pazner | 2020 | Comp. Fluids | [arXiv:1910.03032](https://arxiv.org/abs/1910.03032) |
| 9 | **A Stable and High-Order Accurate Discontinuous Galerkin Based Splitting Method for the Incompressible Navier-Stokes Equations** | M. Piatkowski, S. Müthing, P. Bastian | 2017 | JCP | [arXiv:1612.00657](https://arxiv.org/abs/1612.00657) |

---

## 2. Extensões para Novas Equações — Acoplamento com MRCM

### 2.1 Escoamentos com Viscosidade Variável / Não-Newtonianos

> **Oportunidade:** O MRCM lida naturalmente com $K(\mathbf{x})$ heterogêneo.
> Estender para $\nu(\mathbf{x}, \dot{\gamma})$ (viscosidade variável) no passo
> de projeção é uma generalização direta.

| # | Título | Autores | Ano | Fonte | Link |
|---|--------|---------|-----|-------|------|
| 10 | **Higher-Order Discontinuous Galerkin Splitting Schemes for Fluids with Variable Viscosity** | R. Schussnig, N. Fehn, D.R.Q. Pacheco, M. Kronbichler | 2025 | CMAME | [arXiv:2506.14424](https://arxiv.org/abs/2506.14424) |
| 11 | **Shear rate projection schemes for non-Newtonian fluids** | J. Deteix, D. Yakoubi | 2019 | CMAME | [arXiv:1902.05643](https://arxiv.org/abs/1902.05643) |

**Destaques:**
- **Schussnig et al. (2025):** DG de alta ordem com *operator splitting* para
  fluidos com viscosidade variável. A decomposição do operador de difusão
  (com $\nu(\mathbf{x})$ variável) gera um problema elíptico com coeficiente
  variável — o MRCM é ideal para resolvê-lo em paralelo.
- **Deteix & Yakoubi (2019):** Projeção da taxa de cisalhamento para fluidos
  não-Newtonianos. A equação de projeção envolve um operador elíptico com
  coeficiente dependente da viscosidade local → cenário perfeito para MRCM
  com $\beta_i = \alpha L / \nu_i$.

### 2.2 Escoamentos com Densidade Variável / Compressibilidade Artificial

| # | Título | Autores | Ano | Fonte | Link |
|---|--------|---------|-----|-------|------|
| 12 | **Artificial compressibility method for the incompressible Navier-Stokes equations with variable density** | L. Cappanera, S. Giordano | 2025 | arXiv | [arXiv:2504.15151](https://arxiv.org/abs/2504.15151) |

### 2.3 Escoamentos Bifásicos / Interface (Cahn-Hilliard-Navier-Stokes)

| # | Título | Autores | Ano | Fonte | Link |
|---|--------|---------|-----|-------|------|
| 13 | **Modeling and simulations of high-density two-phase flows using projection-based Cahn-Hilliard Navier-Stokes equations** | A. Rabeh, M.A. Khanwale, J.J. Lee, B. Ganapathysubramanian | 2024 | arXiv | [arXiv:2406.17933](https://arxiv.org/abs/2406.17933) |
| 14 | **Phase-Field Modeling of Two-Phase Flows: A Projection-Based Cahn-Hilliard-Navier-Stokes Framework** | S. Manna, C.M. Megaridis, S.K.R.S. Sankaranarayanan | 2025 | arXiv | [arXiv:2509.00082](https://arxiv.org/abs/2509.00082) |
| 15 | **Mass Conservative and Energy Stable Finite Difference Methods for the Quasi-incompressible Navier-Stokes-Cahn-Hilliard system** | Z. Guo, P. Lin, S. Wise, J. Lowengrub | 2017 | CMAME | [arXiv:1703.06606](https://arxiv.org/abs/1703.06606) |

**Destaques:**
- **Rabeh et al. (2024) e Manna et al. (2025):** Framework de projeção para
  Navier-Stokes-Cahn-Hilliard. A cada passo de tempo, resolve-se uma equação
  de Poisson para pressão + potencial químico. O MRCM poderia acelerar ambas
  as equações elípticas.

### 2.4 Navier-Stokes em Superfícies

| # | Título | Autores | Ano | Fonte | Link |
|---|--------|---------|-----|-------|------|
| 16 | **Releasing the pressure: High-order surface flow discretizations via discrete Helmholtz-Hodge decompositions** | T. Brüers, C. Lehrenfeld, T. van Beeck, M. Wardetzky | 2026 | arXiv | [arXiv:2603.27714](https://arxiv.org/abs/2603.27714) |
| 17 | **Solving the incompressible surface Navier-Stokes equation by surface finite elements** | S. Reuther, A. Voigt | 2017 | Phys. Fluids | [arXiv:1709.02803](https://arxiv.org/abs/1709.02803) |

### 2.5 Modelos de Ordem Reduzida (ROM) com Projeção

| # | Título | Autores | Ano | Fonte | Link |
|---|--------|---------|-----|-------|------|
| 18 | **No pressure? Energy-consistent ROMs for the incompressible Navier-Stokes equations with time-dependent boundary conditions** | H. Rosenberger, B. Sanderse | 2022 | arXiv | [arXiv:2212.12036](https://arxiv.org/abs/2212.12036) |
| 19 | **A segregated reduced order model of a pressure-based solver for turbulent compressible flows** | M. Zancanaro, V. Nkana Ngan, G. Stabile, G. Rozza | 2024 | arXiv | [arXiv:2205.09396](https://arxiv.org/abs/2205.09396) |

---

## 3. Artigos Clássicos de Referência (já conhecidos, para contextualização)

| # | Título | Autores | Ano | Periódico |
|---|--------|---------|-----|-----------|
| C1 | Numerical solution of the Navier-Stokes equations | A.J. Chorin | 1968 | Math. Comp. 22:745–762 |
| C2 | On the convergence of discrete approximations to the Navier-Stokes equations | A.J. Chorin | 1969 | Math. Comp. 23:341–353 |
| C3 | An overview of projection methods for incompressible flows | J.L. Guermond, P. Minev, J. Shen | 2006 | CMAME 195:6011–6045 |
| C4 | A second-order projection method for the incompressible Navier-Stokes equations | J. Kim, P. Moin | 1985 | JCP 59:308–323 |
| C5 | Accurate projection methods for the incompressible Navier-Stokes equations | D.L. Brown, R. Cortez, M.L. Minion | 2001 | JCP 168:464–499 |
| C6 | A numerical method for the incompressible Navier-Stokes equations based on an approximate projection | A.S. Almgren, J.B. Bell, W.G. Szymczak | 1996 | SIAM JSC 17:358–369 |
| C7 | High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method | U. Ghia, K.N. Ghia, C.T. Shin | 1982 | JCP 48:387–411 |

---

## 4. Rotas de Implementação com MRCM (Análise)

### Rota A — Viscosidade Variável (curto prazo, baixo risco)

**Referências:** Schussnig et al. (2025), Deteix & Yakoubi (2019)

**Ideia:** Generalizar o MPrM para fluidos com $\nu = \nu(\mathbf{x})$ (ou
$\nu = \nu(\dot{\gamma})$ para não-Newtonianos). No passo de projeção:
$$\nabla\cdot(\delta t\,\nu\nabla p^{n+1}) = \nabla\cdot\mathbf{w}^*$$

Este é um problema de Darcy com $K = \delta t\,\nu(\mathbf{x})$ **heterogêneo**.
O MRCM foi projetado exatamente para este caso — o parâmetro de Robin
$\beta_i = \alpha L / K_i$ varia por subdomínio/célula, e as 5 variantes
(`harmonic`, `discontinuous`, `adaptive`, etc.) tornam-se relevantes.

**Impacto:** Valida o MRCM em seu regime de heterogeneidade, conectando os
dois mundos (Navier-Stokes + meios porosos).

### Rota B — Acoplamento Stokes-Darcy/Brinkman (médio prazo)

**Referências:** Araya et al. (2013, 2017) — já conhecidos; Dokken et al. (2020)

**Ideia:** Usar o MRCM para o acoplamento Stokes-Darcy em meios parcialmente
porosos: Stokes ($\nu\nabla^2\mathbf{u} - \nabla p = \mathbf{f}$) em canais
livres, Darcy ($\mathbf{u} = -K\nabla p$) no meio poroso, com condição de
interface de Beavers-Joseph-Saffman. O MRCM já foi estendido para Stokes e
Brinkman (Araya et al. 2017) — o MPrM pode ser a ponte para o problema
transiente completo.

### Rota C — Escoamentos Bifásicos com Campo de Fase (longo prazo)

**Referências:** Rabeh et al. (2024), Guo et al. (2017)

**Ideia:** Acoplar Navier-Stokes + Cahn-Hilliard via método de projeção.
A cada passo, duas equações elípticas: Poisson da pressão + Helmholtz do
potencial químico. Ambas admitem decomposição MRCM. O sistema de interface
global acoplaria pressão e potencial químico.

---

## 5. Solicitação de Artigos

### Artigos para baixar (arXiv — acesso aberto)

```bash
# Diretório de destino
mkdir -p ~/Dissertação/Refs/projection_methods
cd ~/Dissertação/Refs/projection_methods

# Artigos prioritários (potencial de implementação imediata)
wget https://arxiv.org/pdf/1912.06392.pdf -O Dokken2020_MultiMesh_NS_Projection.pdf
wget https://arxiv.org/pdf/1812.00185.pdf -O Kashefi2018_CoarseGrid_Projection.pdf
wget https://arxiv.org/pdf/2506.14424.pdf -O Schussnig2025_DG_VariableViscosity.pdf
wget https://arxiv.org/pdf/1902.05643.pdf -O Deteix2019_ShearRate_Projection.pdf
wget https://arxiv.org/pdf/2508.13416.pdf -O Weber2025_Convergence_Chorin.pdf
wget https://arxiv.org/pdf/1602.02072.pdf -O Lee2016_PressureCorrection_Traction.pdf

# Artigos secundários (referência e contexto)
wget https://arxiv.org/pdf/2509.04352.pdf -O BlancoCasares2025_Stabilization.pdf
wget https://arxiv.org/pdf/2504.15151.pdf -O Cappanera2025_ArtificialCompressibility.pdf
wget https://arxiv.org/pdf/2406.17933.pdf -O Rabeh2024_CHNS_Projection.pdf
wget https://arxiv.org/pdf/2212.12036.pdf -O Rosenberger2022_ROM_NS.pdf
wget https://arxiv.org/pdf/2603.27714.pdf -O Bruers2026_Surface_HH_Decomposition.pdf
wget https://arxiv.org/pdf/1703.06606.pdf -O Guo2017_CHNS_Projection.pdf
wget https://arxiv.org/pdf/1910.03032.pdf -O Franco2020_HighOrder_GPU_NS.pdf
```

### Artigos para acessar via Elsevier/Springer (acesso institucional)

| # | Título | Periódico | DOI |
|---|--------|-----------|-----|
| E1 | Guermond, Minev, Shen — *An overview of projection methods for incompressible flows* (2006) | CMAME 195 | [10.1016/j.cma.2005.10.010](https://doi.org/10.1016/j.cma.2005.10.010) |
| E2 | Kim & Moin — *A second-order projection method...* (1985) | JCP 59 | [10.1016/0021-9991(85)90148-2](https://doi.org/10.1016/0021-9991(85)90148-2) |
| E3 | Brown, Cortez, Minion — *Accurate projection methods...* (2001) | JCP 168 | [10.1006/jcph.2001.6715](https://doi.org/10.1006/jcph.2001.6715) |
| E4 | Almgren, Bell, Szymczak — *A numerical method... approximate projection* (1996) | SIAM JSC 17 | [10.1137/0917023](https://doi.org/10.1137/0917023) |

---

## 6. Próximos Passos Sugeridos

1. **Baixar e ler** os 6 artigos prioritários da seção 5 (wget acima).
2. **Rota A (viscosidade variável):** Modificar `mprm_core.py` para aceitar
   $\nu(\mathbf{x})$ no `DarcyProblem` ($K = \delta t\,\nu(\mathbf{x})$),
   testar com o benchmark de cavidade estratificada (duas camadas de fluido).
3. **Rota A (Robin adaptativo):** Ativar `robin_variant="harmonic"` ou
   `"adaptive"` e comparar erros com $\alpha=1$ constante.
4. **Rota B (Brinkman):** Implementar o passo de projeção com termo de
   Brinkman: $\mathbf{v} - \delta t\,\nu\nabla^2\mathbf{v} = \delta t\nabla p$.
   Requer modificação do problema local do MRCM.
5. **Benchmark Coarse Grid Projection:** Comparar MPrM com Kashefi (2018)
   como baseline de performance.
