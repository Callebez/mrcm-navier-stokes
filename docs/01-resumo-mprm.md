# Resumo — The Multiscale Projection Method for the Solution of Incompressible Navier-Stokes Equations

**Autora:** Camila Lages
**Fonte:** `~/Dissertação/Refs/The_multiscale_projection_method_for_the_solution_of_incompressible_Navier_Stokes_equations.pdf`
(conversão: `~/Dissertação/Refs/markdown/…​.md`)

---

## 1. Ideia central

O artigo propõe o **MPrM (Multiscale Projection Method)**: usar o **MRCM**
(Multiscale Robin Coupled Method, Guiraldello et al. 2018) para resolver a
**equação de Poisson da pressão** que surge no **método de projeção de Chorin**
para as equações de Navier-Stokes incompressíveis.

A motivação é que, em métodos de projeção, o custo dominante de cada passo de
tempo é a solução do sistema linear elíptico global da pressão. Como esse
problema tem exatamente a estrutura de um problema de Darcy (com
"permeabilidade" constante `δt`), toda a maquinaria de decomposição de domínio
multiescala do MRCM — problemas locais independentes, funções de base
multiescala (MsBF), sistema global de interface pequeno — se aplica
diretamente, produzindo um algoritmo **embaraçosamente paralelo** sem
comunicação entre subdomínios nas etapas locais.

Ponto-chave de projeto: diferentemente de decomposições de domínio completas
para Stokes/Brinkman (e.g. Araya et al. 2017), que exigem impor continuidade
adicional de tensões tangenciais nas interfaces, o MPrM aplica a decomposição
**apenas ao problema elíptico da projeção**, mantendo a simplicidade da
formulação multiescala original para Darcy.

## 2. O método de projeção (Chorin, não-incremental)

Navier-Stokes incompressível (densidade absorvida: `p = p̃/ρ`, `ν = μ/ρ`):

```
∂u/∂t + ∇·(uu) = −∇p + ν∇²u + f          (1)
∇·u = 0                                    (2)
```

Baseado na decomposição de Helmholtz-Hodge (HH): todo campo `w` se decompõe de
forma única e ortogonal em `w = u + v`, com `∇·u = 0`, `u·n = 0` em `∂Ω` e
`v = ∇p` irrotacional. Discretização temporal explícita de 1ª ordem:

1. **Velocidade tentativa** (momento explícito, sem gradiente de pressão):
   `w* = uⁿ − δt ∇·(uⁿuⁿ) + δt ν ∇²uⁿ + δt fⁿ`                    (8)
2. **Projeção** — resolver o problema elíptico
   `δt ∇²pⁿ⁺¹ = ∇·w*` em Ω, com `δt ∇pⁿ⁺¹·n = w*·n` em ∂Ω;
   então `vⁿ⁺¹ = δt ∇pⁿ⁺¹`                                          (9)
3. **Correção**: `uⁿ⁺¹ = w* − vⁿ⁺¹`                                 (10)

O passo 2, reescrito em **forma mista** (Eq. 13), é o problema que o MRCM
resolve:

```
vⁿ⁺¹ = δt ∇pⁿ⁺¹      em Ω
∇·vⁿ⁺¹ = ∇·w*        em Ω
vⁿ⁺¹·n = w*·n        em ∂Ω     (Neumann puro ⇒ pressão definida a menos de constante)
```

É um problema de Darcy com `K = δt·I` constante (a menos do sinal na lei
constitutiva: Darcy usa `u = −K∇p`; aqui `v = +δt∇p`).

## 3. MRCM aplicado à projeção

- Ω dividido em N subdomínios não sobrepostos Ωᵢ; escala fina `h`, escala
  grossa `H` (tamanho das interfaces do esqueleto Γ).
- **Problemas locais** (Eq. 15) com condição de **Robin** nas interfaces:
  `−βᵢ vᵢ·nᵢ + pᵢ = −βᵢ (V_H n̂·nᵢ) + P_H`, onde `α(x) = β(x)/H` é o
  parâmetro algorítmico do método.
- **Condições de compatibilidade** (Eq. 16) impostas fracamente na escala H
  contra os espaços de interface `U_H` (fluxo) e `P_H` (pressão), subespaços de
  funções lineares por partes nas arestas finas do esqueleto (Eq. 14).
- **Limites**: `α → 0` recupera MMMFEM; `α → ∞` recupera MHM; escolhas
  adequadas recuperam o MuMM. O MRCM é o framework unificador.
- **MsBF** (Eqs. 17–19): solução local = parte homogênea (combinação linear de
  funções de base multiescala, uma por função de base de interface) + parte
  particular (fonte `∇·w*` e condições de contorno externas, Robin homogênea).
  Com `k_U` e `k_P` graus de liberdade por interface, são `4(k_U + k_P)` MsBF
  por subdomínio interno em 2D.
- **Sistema global de interface** `Ax = b` (Eq. 16 + expansões (20)): incógnitas
  `x = [U₁…U_{N_U}, P₁…P_{N_P}]ᵀ`; matriz esparsa se as bases de interface têm
  suporte em uma única interface Γᵢⱼ.

## 4. Regularização da velocidade (downscaling)

Como a compatibilidade é imposta só na escala H, a solução do MRCM é
descontínua na escala h através das interfaces. Para recuperar um campo de
velocidade **conservativo na escala fina** (necessário para avançar no tempo):

1. **Média dos fluxos** nas interfaces (Eq. 21): `Ṽᵢⱼ = ½(vᵢ|Γᵢⱼ + vⱼ|Γᵢⱼ)`.
2. **Problemas locais de Neumann** (Eq. 22) por subdomínio, usando os fluxos
   médios como condição de contorno (pressão fixada num ponto para remover a
   indeterminação) — restauram o balanço de massa fino.
3. **Atualização por subdomínio** (Eq. 23): `uᵢⁿ⁺¹ = wᵢ* − ṽᵢ`.

O `uⁿ⁺¹` resultante tem divergência nula (Eq. 38) e fluxo normal contínuo.
A pressão multiescala permanece descontínua — mas isso não afeta a evolução
temporal, pois só o gradiente regularizado `ṽ` entra na atualização.

## 5. Discretização espacial

- **Volumes finitos em malha cartesiana deslocada (staggered/MAC)**: pressão no
  centro das células, componentes normais de velocidade nas faces — equivalente
  ao elemento de **Raviart-Thomas de ordem mais baixa (RT₀)** para os problemas
  elípticos locais.
- `w*` explícito: **upwind** no termo convectivo, **diferenças centrais** na
  difusão, regra do ponto médio nas integrais de face (Prosperetti & Tryggvason).
- Os problemas locais podem ser resolvidos por qualquer método que incorpore
  condição de Robin (FEM, FD, FV).

## 6. Algoritmo (Algoritmo 1 do artigo)

**Offline** (uma vez):
- Para cada Ωᵢ (em paralelo): computar e armazenar todas as MsBF (Eq. 18);
  montar e fatorar (e.g. LU) as matrizes locais Aᵢ.
- Montar a matriz global de interface A (Eq. 16).

**Online** (a cada passo de tempo):
1. Computar `w*` a partir de `uⁿ` (Eq. 8).
2. Para cada Ωᵢ (em paralelo): resolver a parte particular (Eq. 19).
3. Montar o lado direito `b` do sistema de interface; resolver `Ax = b`.
4. Reconstruir `v` nas interfaces; calcular fluxos médios (Eq. 21).
5. Para cada Ωᵢ (em paralelo): resolver o downscaling (Eq. 22) e atualizar
   `uⁿ⁺¹` (Eq. 23).

## 7. Estimativa de custo (§2.8)

Com malha fina `N_x × N_y` e decomposição `N_Hx × N_Hy`, custo relativo por
passo de tempo vs. solver fino global (matrizes banda, fatorações offline):

- **Paralelo** (Eq. 36): `RCᵖ = 2/N + 4(k_U + k_P)/(N_hx N_hy) < 1` — vantagem
  clara; existe um *sweet spot* de decomposição (para 360², `N_H = 12` dá 2–6%
  do custo do sistema fino global).
- **Sequencial** (Eq. 37): `RCˢ ≥ 2` — o MPrM sequencial **não** compensa; o
  método é intrinsecamente voltado a execução paralela.

## 8. Resultados numéricos

### 8.1 Decomposição de Helmholtz-Hodge modificada (benchmark de Olshanskii)
- Solução analítica (Eq. 40), malha 90×90, decomposição 3×3, α = 1.
- Erro relativo da velocidade cai de ~100% (espaços lineares `P_{H,1}, U_{H,1}`)
  para 0,4% (`P_{H,4}, U_{H,4}`) — Tabela 1. Não vale enriquecer só pressão ou
  só fluxo: o ganho é limitado pelo espaço mais pobre.
- `∇·u_h ≈ 0` em precisão de máquina; o erro se concentra na parte
  **irrotacional** (`∇×v_h ≠ 0` ao longo das interfaces, decaindo com o
  enriquecimento) — consistente com a análise das Eqs. (38)–(39).
- Erros menores para α grande neste teste (priorizar conservação de fluxo);
  aplicações sucessivas da decomposição HH modificada são **estáveis** (sem
  crescimento de erro em centenas de iterações).

### 8.2 Cavidade com tampa deslizante (lid-driven cavity)
- Re ∈ {10, 100, 400, 1000}; comparação com solução de malha fina e Ghia et al.
- Com apenas espaços lineares `(P_{H,1}, U_{H,1})` e α = 1:
  `‖err‖_L² ∈ [6×10⁻³, 3×10⁻²]`. **A solução de Navier-Stokes é muito menos
  sensível ao erro da decomposição HH do que o teste da HH pura** — espaços
  pobres bastam.

### 8.3 Degrau descendente (backward-facing step)
- Erro vs. α tem mínimo em α ≈ 1 (consistente com Guiraldello et al. 2018).
- Três estratégias de custo comparadas (refinar malha fina; enriquecer U_H com
  decomposição fixa; refinar a decomposição com espaços fixos): enriquecer o
  espaço de fluxo com decomposição intermediária é a mais competitiva para
  erros ~10⁻².

### 8.4 Escoamento ao redor de cilindro quadrado (von Kármán, Re = 100)
- Fluxo transiente; métricas: C_D, C_L, Strouhal (St ≈ 0,121 reproduzido).
- Necessário enriquecer **ambos** os espaços (melhor caso: erro de C_D ~10⁻³
  com `P_{H,3}, U_{H,2..3}`, α = 1).
- Varredura α ∈ {10⁻⁶, 1, 10⁶}: melhores resultados com α = 10⁻⁶ e pressão
  rica (`P_{H,3}`) — em fluxos transientes a pressão de interface domina a
  qualidade, mesmo com downscaling do fluxo.

## 9. Mensagens para a implementação

1. A projeção é um **Darcy com K = δt constante e homogêneo** — caso trivial
   para o solver MRCM (sem heterogeneidade).
2. Contorno global é **Neumann puro** (`v·n = w*·n`): pressão definida a menos
   de constante; a compatibilidade `∮ w*·n = ∫ ∇·w*` deve ser satisfeita.
3. O termo fonte muda a cada passo de tempo (`∇·w*`), mas o **operador não** —
   todo o offline (MsBF, fatorações, matriz de interface) é reutilizado.
4. O **downscaling** (média de fluxos + solves de Neumann locais) é obrigatório
   para conservação fina antes de avançar no tempo.
5. Recomendações práticas: α = 1 como default; espaços lineares suficientes
   para cavidade/estacionários; enriquecer pressão para fluxos transientes.
