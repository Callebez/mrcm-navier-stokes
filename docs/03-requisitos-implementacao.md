# Requisitos — Implementação do MPrM com o módulo `mrcm`

**Objetivo:** implementar o Método de Projeção Multiescala (MPrM) para as
equações de Navier-Stokes incompressíveis em um **Jupyter notebook**, usando o
módulo `mrcm` (repositório `~/mrcm/mrcm-firedrake`) como solver do passo de
projeção, com suporte a computação paralela e células markdown explicativas
baseadas nas referências (sem citar a dissertação em andamento).

---

## R1 — Requisitos funcionais

- **R1.1** Resolver as equações de Navier-Stokes incompressíveis 2D
  transientes pelo método de projeção de Chorin (não-incremental, explícito no
  tempo), conforme §2.1 do artigo do MPrM.
- **R1.2** O passo de projeção (Poisson da pressão em forma mista, Eq. 13 do
  artigo) deve ser resolvido pelo MRCM através do módulo `mrcm`, com
  "permeabilidade" `K = δt·I` constante.
- **R1.3** Estrutura offline/online do Algoritmo 1: MsBF, fatorações locais e
  matriz de interface montadas **uma única vez**; a cada passo de tempo apenas
  parte particular + RHS de interface + downscaling são recomputados.
- **R1.4** Downscaling (regularização da velocidade): média dos fluxos nas
  interfaces (Eq. 21) + problemas locais de Neumann (Eq. 22) + atualização
  `uⁿ⁺¹ = w* − ṽ` (Eq. 23), garantindo `∇·uⁿ⁺¹ = 0` na escala fina.
- **R1.5** Benchmark de validação mínimo: (a) decomposição de Helmholtz-Hodge
  modificada com a solução analítica de Olshanskii (Eq. 40, Tabela 1 do
  artigo); (b) cavidade com tampa deslizante (lid-driven cavity) com
  comparação qualitativa/quantitativa a uma solução de malha fina sem
  decomposição (e perfis de Ghia et al. quando aplicável).
- **R1.6** Parâmetros expostos e variáveis: α (Robin), `k_U`/`k_P` (graus dos
  espaços de interface), decomposição `N_Hx × N_Hy`, malha fina, δt, ν, Re.
- **R1.7** Notebook com narrativa em markdown: cada célula de código precedida
  de explicação (equações do artigo, decisões de implementação, resultado
  esperado), citando as referências publicadas (Lages; Guiraldello et al.
  2018; Chorin 1968; Ghia et al. 1982; etc.), **não** a dissertação.
- **R1.8** Execução paralela: o caminho MRCM deve poder rodar com MPI
  (subdomínios distribuídos entre ranks), mantendo também um caminho
  sequencial para depuração dentro do notebook.

## R2 — Requisitos não funcionais

- **R2.1** Ambiente: `~/venv-firedrake` (Firedrake + PETSc + mpi4py).
- **R2.2** Código do notebook autocontido no diretório `~/mrcm/NavierStokes/`,
  importando o pacote `mrcm` de `~/mrcm/mrcm-firedrake` sem modificá-lo (ou
  com modificações mínimas e documentadas).
- **R2.3** Reprodutibilidade: parâmetros no topo, seeds fixas se houver
  aleatoriedade, células executáveis em ordem.

---

## Q — Perguntas de implementação em aberto

*Escritas antes da leitura da documentação do `mrcm-firedrake`; respondidas na
seção A abaixo após a leitura (Vault/, README, código).*

- **Q1 (Neumann puro / singularidade).** O `MRCMSolver` aceita problema com
  condição de Neumann em **todo** o contorno externo (`v·n = w*·n`)? Como ele
  remove a indeterminação da pressão (nullspace, pino de pressão, média
  nula)? O `DarcyProblem`/`BoundaryData` suportam dado de Neumann não nulo?
- **Q2 (fonte definida numericamente).** O termo fonte do MRCM pode ser uma
  `Function` do Firedrake calculada em tempo de execução (`∇·w*` muda a cada
  passo), em vez de expressão analítica (UFL/manufactured)? Em qual espaço a
  fonte deve viver (DG0?)?
- **Q3 (reuso offline/online).** O pipeline do `MRCMSolver` (fases 0–5) permite
  refazer **apenas** a parte online (RHS local particular + RHS de interface +
  reconstrução) com MsBF/fatorações/matriz de interface congeladas? Existe API
  para isso ou o `run()` recomputa tudo? Que refatoração mínima seria
  necessária?
- **Q4 (downscaling).** O repositório já implementa média de fluxo nas
  interfaces + re-solve local de Neumann (downscaling à la Guiraldello et al.
  2020)? Em qual módulo (`reconstructor.py`?)? Se não, implementar no notebook
  usando os solvers locais existentes.
- **Q5 (discretização/malha).** Quais elementos o `mrcm` usa (RT em triângulos?
  RTCF em quadriláteros? BDM?) e qual o análogo mais fiel ao FV staggered/RT₀
  do artigo? Malha estruturada `UnitSquareMesh`/`RectangleMesh`? O parâmetro
  `mixed_order` controla o quê?
- **Q6 (sinal/convenção).** Qual a convenção constitutiva do `mrcm`
  (`u = −K∇p`?) e como mapear `v = +δt∇p` (trocar sinal da pressão ou da
  fonte)? Como o artigo define fluxo positivo nas condições de Robin?
- **Q7 (espaços de interface).** Como escolher `k_U`/`k_P` e o tipo de base
  (nodal, monomial, Legendre) no `mrcm`? O que corresponde a `(P_{H,i},
  U_{H,j})` do artigo (projeções de polinômios de grau i+1/j+1)?
- **Q8 (α de Robin).** Como fixar α constante (= 1) no `robin.py`? Qual das 5
  variantes usar? α adaptativo (Rocha 2020) é relevante aqui (K homogêneo)?
- **Q9 (paralelismo em notebook).** Como o `--executor mpi` funciona
  (mpi4py? subdomínio por rank?) e qual a melhor forma de acioná-lo a partir
  de um notebook: célula `%%writefile` + `mpiexec` via subprocess, ou
  ipyparallel? A malha do Firedrake é distribuída ou replicada?
- **Q10 (advecção-difusão explícita).** O passo `w*` (Eq. 8) fica fora do
  `mrcm`. Implementar em Firedrake (projeções L²/UFL em espaço vetorial DG/CG)
  ou em arrays estruturados NumPy (FV staggered fiel ao artigo)? Qual
  interopera melhor com o formato de entrada/saída do `mrcm`?
- **Q11 (extração de campos).** Como obter do `mrcm` os campos reconstruídos
  por subdomínio (`v`, `p`) e os traços de fluxo nas interfaces, para (a) o
  downscaling e (b) a atualização `uⁿ⁺¹ = w* − ṽ`?
- **Q12 (CFL/estabilidade).** Sendo o transporte explícito, qual restrição de
  δt adotar (CFL convectivo + difusivo)? Como δt entra em `K = δt` sem
  reconstruir o offline quando δt é fixo?

---

## A — Respostas (preenchidas após leitura da documentação do `mrcm-firedrake`)

### Q1 — Neumann puro / singularidade
**Resposta:** O `mrcm` suporta condição de Neumann (fluxo) em todos os lados
via `BoundaryData(flux=...)`. A indeterminação da pressão é tratada
externamente ao `mrcm`: no downscaling MAC (`MACDownscaler`), a matriz FV de
Neumann puro tem o modo constante removido pinando a pressão na célula 0
(`A[0,:]=0; A[0,0]=1`). O RHS é compatibilizado distribuindo o defeito de
balanço uniformemente (`b -= defect/b.size`).

O patch `_build_flux_dirichlet_bcs_allsides` em `mprm_core.py` garante que o
no-flow forte (`u·n = 0`) seja imposto em TODOS os lados externos (bottom,
right, top, left), corrigindo o comportamento padrão do `mrcm` que só pinava
bottom/top. Isso é necessário para o problema de projeção com contorno
impermeável.

**Duas ressalvas verificadas experimentalmente:**
1. Com contorno impermeável nos 4 lados, o sistema de interface `A_Γ` fica
   **quase-singular** (nullspace do modo constante de `P_H`; cond ≈ 10¹²–10¹³).
   O aviso do UMFPACK "(almost) singular matrix" nas saídas é **esperado e
   inofensivo**: a componente no nullspace só desloca a pressão de interface
   por uma constante e não afeta a velocidade (verificado: erro de velocidade
   1e-3 e conservação de massa em precisão de máquina).
2. **Fluxo de Neumann NÃO nulo não é suportado corretamente pelo `mrcm`**: o
   caminho fraco de `solve_particular` impõe, na formulação mista dual, a
   condição natural `p = −z` (pressão), não `u·n = z`. Só o fluxo
   identicamente nulo (pino forte) funciona — exatamente o caso do problema
   de projeção com paredes impermeáveis (`w*·n = 0`). Ver o nó de bug no
   Vault do mrcm (`2026-07-19-noflow-strong-bc-left-right-missing`).

### Q2 — Fonte definida numericamente
**Resposta:** Sim. A fonte é uma `Function` do Firedrake no espaço DG₀
(`Q = FunctionSpace(mesh, "DG", 0)`) cujo vetor de dados é atualizado a cada
passo de tempo com `self.src_fn[isub].dat.data[:] = divw[ic, jc]`. O mapa
`cell_map` converte índices globais de célula MAC para DOFs DG₀ locais. O
termo fonte entra na forma linear `L_forms[isub] = -f_i * q_test * dx_m`.

### Q3 — Reuso offline/online
**Resposta:** O pipeline `mrcm` original (6 fases via `MRCMSolver.run()`) é
monolítico. Para o MPrM, foi necessário refatorar para o padrão
offline/online. A classe `MPrMProjector` faz isso:
- **Offline** (`__init__`): `build_subdomain_workspace` (MsBF + caches B1/B2),
  `LocalSolver` com fatoração LU persistente, `GlobalAssembler` no rank 0.
- **Online** (`project(divw)`): atualiza fonte DG₀, resolve parte particular
  via `LocalSolver.solve()`, monta triplets → sistema de interface, resolve
  `Ax = b`, reconstrói com `reconstruct_subdomain`, extrai traços de fluxo.

### Q4 — Downscaling
**Resposta:** O `mrcm` original não implementa downscaling. A classe
`MACDownscaler` em `mprm_core.py` implementa as Eqs. 21–23 do artigo:
- Matriz FV 5 pontos (`K = δt`) montada e fatorada (LU) uma única vez.
- `solve_patch()`: resolve o problema de Neumann local (Eq. 22) usando os
  fluxos médios como condição de contorno, devolvendo `ṽ` na malha MAC fina.
- O defeito de compatibilidade (`last_defect`) é monitorado.

### Q5 — Discretização / malha
**Resposta:** O `mrcm` usa pares `H(div)×L²` estáveis:
- `BDM₁–DG₀` (`mixed_family="BDM"`, `mixed_order=0`): BDM de ordem 1 para
  velocidade, DG de ordem 0 para pressão. Equivalente ao RT₀ do artigo em
  malhas cartesianas (ambos têm 1 DOF de fluxo por face).
- `RT₁–DG₀` (`mixed_family="RT"`, `mixed_order=0`): alternativa.
- Malha: `RectangleMesh` do Firedrake com `diagonal="left"`, triangulação
  uniforme. A malha MAC (FV staggered) do artigo é usada para o transporte
  explícito e downscaling; o MRCM usa a malha triangular do Firedrake.

### Q6 — Sinal / convenção
**Resposta:** O `mrcm` usa a convenção de Darcy **u_D = −K∇p_D** com
**∇·u_D = f**. Passando `K = δt` e `f = ∇·w*`, a unicidade do problema
elíptico dá **p_D = −p** (a pressão do MRCM é a pressão física da projeção
com sinal trocado, a menos de constante) e, portanto,
**u_D = −δt∇p_D = +δt∇p = v** — ou seja, a velocidade reconstruída `uT` do
`mrcm` **já é exatamente o campo v da projeção; nenhuma correção de sinal é
aplicada (nem necessária) em ponto algum do pipeline**. Os traços de fluxo
extraídos de `uT` entram diretamente como `V̄` no downscaling, e o
`MACDownscaler` trabalha na convenção física (`ṽ = +δt∇p̃`, matriz de
`δt·Δp̃ = ∇·w*`). A única inversão de sinal existente é na **pressão**
retornada pelo MRCM (`last_pT` = −p física), relevante apenas para
visualização. A convenção de fluxo positivo nas condições de Robin segue a
normal exterior `n̂ⁱ` de cada subdomínio.

### Q7 — Espaços de interface
**Resposta:** Controlados por `degreeU` e `degreeP` no `MRCMConfig`:
- `ndofsU = degreeU + 1`, `ndofsP = degreeP + 1` graus de liberdade por interface.
- `basis_kind`: `"nodal_Q"` (Lagrange equiespaçado), `"monomial_P"`, `"legendre_P"`.
- Correspondência com o artigo: `(P_{H,i}, U_{H,j})` ↔ `degreeP = i`,
  `degreeU = j` — cada espaço tem `k = grau + 1` DOFs por interface
  (polinômios de grau ≤ i por aresta).
- Escolhas do artigo para Navier-Stokes: cavidade e degrau usam espaços
  **lineares** `(P_{H,1}, U_{H,1})` → `degreeU = degreeP = 1` (a solução de
  NS é pouco sensível ao erro da decomposição HH); o escoamento transiente
  (von Kármán) exigiu enriquecer a **pressão** (`P_{H,3}`, Tabelas 2–3).
  Verificado nesta implementação (N=48, 3×3): erro vs malha fina 5.3e-4 com
  grau 3 e 22.6% no benchmark HH puro com grau 1 — consistente com a
  Tabela 1 do artigo.

### Q8 — α de Robin
**Resposta:** Usa-se `ConstantRobin` (`robin_variant="constant"`):
`β = α·L` com `L = min(Hx, Hy)`. Como `K = δt` é homogêneo e constante no
tempo, todas as 5 variantes são equivalentes (invariante I1). O α adaptativo
(Rocha 2020) não é relevante aqui pois não há heterogeneidade. `alpha=1.0` é
o valor recomendado. O parâmetro `robin_length_scale="H"` (convenção 2019)
torna α independente de h.

### Q9 — Paralelismo em notebook
**Resposta:** O `mrcm` suporta 3 modos: `sequential`, `process_pool`, `mpi`.
- **No notebook:** usa-se `executor_kind="sequential"` (sem MPI) para
  depuração. A malha do Firedrake é replicada (cada subdomínio tem sua
  própria malha local). O `MPrMProjector` aceita um comunicador `comm`; com
  `MPI.COMM_WORLD` de tamanho 1, degenera para sequencial.
- **Com MPI:** `mpiexec -n N python script.py`. Subdomínios distribuídos
  round-robin (`range(rank, Nsubd, size)`). `allgather` para trocar fluxos de
  interface e patches do downscaling.
- **Alternativa para notebook:** `%%writefile` + `!mpiexec` via subprocess.

### Q10 — Advecção-difusão explícita (w*)
**Resposta:** Implementado em **arrays NumPy** (FV staggered/MAC) — não usa
Firedrake. Esta escolha é fiel ao artigo (§2.6, Eqs. 24–29) e evita
overhead de projeções UFL. A função `mac_wstar()` opera diretamente sobre
arrays `(Nx+1, Ny)` e `(Nx, Ny+1)`. A comunicação com o MRCM ocorre via
`mac_divergence()` → array NumPy → atualização das Functions DG₀.

### Q11 — Extração de campos
**Resposta:** `reconstruct_subdomain()` devolve `(pT, uT)` — Functions
Firedrake para pressão e velocidade reconstruídas por subdomínio. Os traços
de fluxo nas interfaces são extraídos via operadores `HDiv Trace` (grau 0,
1 DOF por faceta = momento ∫u·n), montados em `self.trace` durante o
offline. O fluxo médio por face fina é `sign * (vals/len)[order]`.

### Q12 — CFL / estabilidade
**Resposta:** δt fixo, escolhido como:
`dt = 0.4 * min(h / U_max, h² / (4ν))` — o mínimo entre CFL convectivo e
difusivo. Para Re ≤ 1000 e malhas moderadas, δt ≈ 0.001–0.01. Como `K = δt`
é constante, o estágio offline (MsBF, fatorações, matriz de interface) **não**
precisa ser reconstruído quando δt é fixo. Se δt variar (CFL adaptativo), o
offline precisaria ser refeito (K muda).
