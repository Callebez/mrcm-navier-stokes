# MPrM sobre decomposição quadtree — von Kármán — Design Spec

> **Data:** 2026-07-25
> **Repositório:** `~/mrcm/NavierStokes` (pacote `mprm`)
> **Branch:** `feature/quadtree-mprm`, com merge ao fim de cada fase
> **Dependência:** `~/mrcm/mrcm-firedrake` (pacote `mrcm`) — sem alterações
>   necessárias, ver §4.6
> **Objetivo:** resolver o passo de projeção do MPrM sobre uma decomposição de
> domínio adaptativa (quadtree) e medir, no benchmark de von Kármán, o que a
> adaptatividade da escala grossa entrega frente à decomposição cartesiana
> uniforme.

---

## 1. Contexto

O MPrM (Lages, §2) resolve Navier–Stokes incompressível por projeção de Chorin,
delegando ao MRCM o problema elíptico da pressão, que tem a forma de um Darcy
com permeabilidade constante `K = δt`. A implementação atual em `mprm/` usa uma
decomposição **cartesiana uniforme** `Ndomx × Ndomy`: todo subdomínio tem o mesmo
tamanho `H` e o mesmo número `nloc` de células.

Num escoamento com obstáculo, essa uniformidade é desperdício em quase todo o
domínio e insuficiência local. A qualidade da solução multiescala depende de
quão bem os espaços de interface `U_H`, `P_H` representam o traço do fluxo
sobre Γ; longe do corpo esse traço é suave e alguns modos bastam, enquanto na
vizinhança do cilindro e na esteira ele varia na escala de `D`. Reduzir `H`
localmente é a via que a teoria do MRCM oferece para atacar esse erro sem tocar
na escala fina.

O `mrcm-firedrake` já implementa a maquinaria de decomposição não uniforme:
`QuadTree` (balanceamento 2:1), `Skeleton.from_quadtree` (nós pendurados como
segmentos independentes de Γ), tarefas de Fase 1 com `geom`/`incidences`, e
caminho MPI validado. Este trabalho **consome** essa maquinaria a partir do
MPrM; não a reimplementa.

---

## 2. Revisão da implementação atual do método de projeção

Levantamento sobre `mprm/projector.py`, `mprm/downscaler.py`, `mprm/flow.py`,
`mprm/transport.py`.

### 2.1 O que está correto

O núcleo é fiel ao artigo. Transporte explícito na malha MAC produz `w*`
(Eq. 8); a projeção é montada como Darcy misto com `K = δt` e resolvida pelo
MRCM (Eqs. 13–20); os traços de fluxo nas interfaces são mediados (Eq. 21) e
alimentam problemas de Neumann locais na malha MAC (Eq. 22), cuja solução
corrige `u = w* − ṽ` (Eq. 10). O defeito de compatibilidade é medido e
reportado. A conservação de massa fecha em `~1e-14`. As fatorações locais são
persistentes (`LocalSolver` construído uma vez no offline).

### 2.2 Achado 1 — sistema global de interface remontado e refatorado a cada passo

`MPrMProjector.project()` chama `compute_subdomain_triplets` por subdomínio e
`assemble_from_triplets` no rank 0, **a cada passo de tempo**. Em
`global_assembly.py`, as triplas `(rows, cols, vals)` dependem exclusivamente de
`homogeneous`, `face_mass` e `cofunc_*` — todos calculados no offline e
constantes no tempo, porque `K = δt` e a geometria não mudam. Só `b_vals =
−⟨c_i, ū⟩` depende do passo, através da parte particular. `assemble_from_triplets`
ainda refaz COO→CSC, `sum_duplicates` e uma **fatoração LU nova**.

Custo redundante por passo: `O(N_B²)` produtos internos por subdomínio, a
montagem esparsa global e a fatoração. Sobre 10³ passos, domina o laço temporal.
O gather por passo também transporta `nnz` floats onde bastariam `n_Γ`.

### 2.3 Achado 2 — traços de fluxo por `assemble` de forma

A extração dos traços usa `assemble(dot(uh, n) * tau * ds(marker))` por lado e
por passo. É a aplicação de um operador linear fixo a `uT`; pré-montar reduz o
overhead de Python/PETSc no caminho quente.

### 2.4 Achado 3 — comunicação por `allgather` de dicionários

`mprm_projection_step` replica o campo global em todos os ranks via
`comm.allgather(my_patch)` com dicionários de arrays NumPy (serialização por
pickle) a cada passo. Um `Allgatherv` sobre buffer contíguo é a versão barata.

### 2.5 Achado 4 — erro de splitting

O esquema é Chorin não incremental: erro de splitting `O(δt)` e camada-limite
numérica na pressão. Isso limita o teto de precisão de `C_L` e `St`
independentemente da malha. Variantes incremental/rotacional mudam apenas o RHS
da projeção, não a estrutura MRCM. **Fora do escopo** deste spec, registrado
como trabalho futuro.

### 2.6 Achado 5 — β do Robin inconsistente sob decomposição não uniforme

`MPrMProjector.__init__` constrói **um** objeto Robin com `cfg.robin_L_xy()`,
que usa `cfg.Hx = Lx/Ndomx` global, e o usa para o `beta_expr` do solver local.
Já `build_subdomain_workspace` constrói o seu próprio Robin com
`cfg.robin_L_xy_for(Hx, Hy, nlocx, nlocy)` — o `H` **daquela** folha. No caso
uniforme os dois coincidem; sob quadtree divergem, e o β do solver local deixaria
de ser o β das funções de base multiescala. É um bug latente que a Fase 3 corrige.

### 2.7 Achado 6 — três representações do sólido

O obstáculo aparece como permeabilidade `δt·ε` no Darcy, como máscara de células
no downscaler e como faces zeradas no transporte. É consistente hoje, mas o
acoplamento com `assert_solid_interior_to_subdomains` (obstáculo não pode cruzar
interface) é justamente a restrição que o refinamento tensiona — tratada
explicitamente na §4.3.

---

## 3. Decisões de design

| # | Decisão | Justificativa |
|---|---------|---------------|
| D1 | Quadtree adapta **apenas a escala grossa H**; `h` é uniforme global | Transporte MAC, cálculo de forças e CFL ficam intactos; o esforço concentra-se na projeção, que é onde o MRCM age |
| D2 | `nloc_i = H_i / h` por folha (não `nloc` fixo) | Consequência direta de D1: folha maior, mais células |
| D3 | Refinamento **geométrico estático**, concentrado nas proximidades do obstáculo | `K = δt` é constante — não há contraste de permeabilidade para guiar; a árvore é construída uma vez e o offline roda uma vez |
| D4 | Interfaces **não atravessam** o obstáculo | Preserva `assert_solid_interior_to_subdomains` e evita folhas parcial ou totalmente sólidas, que exigem tratamento próprio (§4.5). Não tem relação com o cálculo de forças: `obstacle_forces` opera sobre os arrays MAC globais e independe da decomposição |
| D5 | Pré-fatoração de `A_Γ` **antes** do quadtree | Sem isso o tempo de parede do estudo mede remontagem, não decomposição |
| D6 | Obstáculo **centrado** (`yc = Ly/2`) | Corrige o desvio 🔴 do diagnóstico e alinha com Lages §3.5 e Breuer et al. |
| D7 | Duas famílias de alinhamento corpo↔árvore, comparadas no estudo: **(a) corpo interior a uma folha** (D4 estrito, grade base ímpar em `y`) e **(b) corpo = bloco 2×2 de folhas** (grade base par em `y`, refino alcançando a superfície) | O ganho de levar o refino até a parede é uma pergunta empírica, não uma premissa; e o custo de (b) — folhas 100 % sólidas, condicionamento de `A_Γ` — precisa ser medido, não assumido |

---

## 4. Arquitetura

### 4.1 Módulos

```
mprm/decomposition.py   [NOVO]  Decomposition, Leaf, GammaEdge,
                                plan_quadtree_geometry, build_obstacle_tree
mprm/projector.py       [MOD]   offline de A_Γ + consumo de Decomposition
mprm/downscaler.py      [MOD]   cache de LU por (n, m, padrão de sólido)
mprm/flow.py            [MOD]   passo de projeção itera folhas, não (ix, iy)
mprm/benchmarks/quadtree_vk.py  [NOVO]  estudo comparativo (funções puras)
tests/test_decomposition.py     [NOVO]
tests/test_quadtree_projector.py[NOVO]
tests/test_quadtree_von_karman.py [NOVO]  CLI + I/O + figuras
```

A separação segue o padrão já estabelecido em `mprm/benchmarks/convergence.py`:
módulo de benchmark sem I/O, script de teste com CLI, JSON e figuras.

### 4.2 A camada `Decomposition`

Único ponto do MPrM que conhece a topologia da decomposição. O caminho
cartesiano passa a ser um caso particular construído pela mesma classe, o que
torna o teste de equivalência "árvore uniforme ≡ cartesiano" a garantia central
de não regressão.

```python
@dataclass(frozen=True)
class Leaf:
    isub: int
    x0, y0, x1, y1: float      # bbox física
    nlocx, nlocy: int          # células locais = (x1-x0)/h, (y1-y0)/h
    i0, j0: int                # offset em células MAC globais
    level: int

@dataclass(frozen=True)
class GammaEdge:
    ell: int                   # índice da aresta em Γ
    a: tuple[int, int, int]    # (isub, side, seg) de um lado
    b: tuple[int, int, int]    # (isub, side, seg) do outro
    nfaces: int                # faces finas do segmento (igual nos dois lados)
    bbox: tuple[float, ...]    # extensão física do segmento

class Decomposition:
    skeleton: Skeleton
    leaves: list[Leaf]
    gamma: list[GammaEdge]
    h: float                   # espaçamento fino global (hx = hy = h)
    Nx: int; Ny: int           # malha MAC global

    @classmethod
    def uniform(cls, Ndomx, Ndomy, nloc, Lx, Ly) -> "Decomposition": ...
    @classmethod
    def from_quadtree(cls, qt, h, Lx, Ly) -> "Decomposition": ...
    def leaf(self, isub) -> Leaf: ...
    def task(self, isub, cfg, NunkU, problem) -> dict:
        """Tarefa de Fase 1 do mrcm (geom/incidences no caminho quadtree,
        edge_l no caminho uniforme)."""
```

`nfaces` ser igual nos dois lados de toda aresta de Γ é consequência de `h`
uniforme (D1) — é o que torna a média da Eq. 21 uma operação face a face mesmo
na presença de nó pendurado, sem qualquer operador de restrição/prolongação.

### 4.2.1 Composição dos fluxos de contorno de uma folha

O downscaler precisa de quatro vetores (`fS, fE, fN, fW`) por folha, mas sob
quadtree **um lado pode ser composto por dois segmentos** com vizinhos
diferentes. A indexação cartesiana atual (`fluxes[("v", ix, iy)]`) não expressa
isso. A montagem passa por:

```python
def leaf_boundary_fluxes(decomp, flux_avg, isub):
    """Vetores de fluxo dos 4 lados da folha, montados a partir dos
    segmentos de Γ que a tocam.

    Convenção de lados do mrcm: 0=bottom, 1=right, 2=top, 3=left.
    Lados horizontais (0, 2) têm nlocx entradas e indexam por x;
    verticais (1, 3) têm nlocy e indexam por y.
    """
    leaf = decomp.leaf(isub)
    f = {0: np.zeros(leaf.nlocx), 1: np.zeros(leaf.nlocy),
         2: np.zeros(leaf.nlocx), 3: np.zeros(leaf.nlocy)}
    for side, seg, flux, bbox in decomp.segments_of(isub, flux_avg):
        horiz = side in (0, 2)
        origin = leaf.x0 if horiz else leaf.y0
        start  = bbox[0]  if horiz else bbox[1]
        off = int(round((start - origin) / decomp.h))
        f[side][off:off + flux.size] = flux
    return f[0], f[1], f[2], f[3]
```

Faces externas sem aresta de Γ permanecem zeradas (o que é a condição correta
para parede, deslizamento e entrada); lados de saída são sobrescritos pelos
traços reconstruídos, como já ocorre hoje. `off` ser inteiro exato é garantido
pelo alinhamento de segmentos verificado na construção (§7): com `h` uniforme,
o segmento de um nó pendurado cobre exatamente metade das faces finas do lado
grosso, de modo que a composição é concatenação, sem interpolação.

### 4.3 Geometria: restrições e como são resolvidas

O quadtree é diádico: toda fronteira de folha cai em múltiplo de `H_0/2^ℓ`. Esta
seção trata da **família (a)** de D7 (corpo interior a uma folha); a família (b)
está na §4.3.1. Daí:

1. **Na família (a), obstáculo centrado exige grade base ímpar em `y`.** Se
   `N_by` é par, `y = Ly/2` é fronteira em todo nível e nenhum subdomínio contém
   o corpo. Com `N_by` ímpar, o corpo fica no centro da célula base central. (Na
   família (b) a paridade se inverte: é justamente por `N_by` ser par que o
   centro do canal coincide com um vértice da grade.)
2. **A folha do obstáculo não pode ser subdividida.** Estando o corpo no centro
   dessa célula, subdividi-la colocaria uma interface exatamente sobre ele.
   Logo `H_obs = H_0 = Ly/N_by`, e `H_0 > D` é obrigatório.
3. **O refinamento cresce a partir do corpo, não sobre ele.** Pelo balanço 2:1,
   as vizinhas imediatas ficam em `nível 1` (`H_0/2`), o anel seguinte em
   `nível 2` (`H_0/4`), e assim por diante. Como `H_obs = H_0`, quanto menor
   `H_0` melhor a resolução de interface junto ao corpo — ao custo de mais
   folhas na grade base. `N_by` é portanto um parâmetro do estudo:
   `N_by = 3` dá `H_0 = 2,67D` com 30 folhas base (próximo do baseline atual de
   50 subdomínios); `N_by = 5` dá `H_0 = 1,6D` com ~80 folhas base e uma folha
   do corpo bem mais justa. A configuração de referência abaixo usa `N_by = 3`;
   `N_by = 5` é a variante secundária.
4. **`nloc` par em todo nível.** `check_segment_alignment` (mrcm/topology.py)
   exige que a fronteira de um segmento pendurado caia numa gridline da malha
   local; `nloc` ímpar contra nó pendurado desloca o suporte discreto de `h/2` e
   quebra a continuidade fraca de fluxo em `O(h)` sem erro visível.
5. **`h` único global divide todo `H_ℓ`.**

`plan_quadtree_geometry(Lx, Ly, D, N_by, max_level, nloc_min)` resolve essas
restrições, devolve `(h, nloc_por_nível, Nx, Ny, xc, yc)` e falha com mensagem
explícita quando são incompatíveis.

**Configuração de referência da família (a)** (satisfaz 1–5):

| Grandeza | Valor |
|---|---|
| `Ly` | `8D` (blockage `D/H = 1/8`, Breuer et al.) |
| `N_by` | `3` (ímpar) → `H_0 = 8D/3 ≈ 2,67D` |
| `Lx` | `10 · H_0 ≈ 26,67D` |
| `(xc, yc)` | `(6,67D, 4D)` — centro da célula base `(2,1)`; 6,67D a montante, 20D a jusante |
| níveis | `0, 1, 2` → `H ∈ {2,67D; 1,33D; 0,67D}` |
| `nloc` | `(32, 16, 8)` — todos pares |
| `h` | `H_0/32 = D/12` |
| malha global | `320 × 96` |
| folhas base | `30` |

### 4.3.1 Família (b): corpo como bloco 2×2 de folhas

Na família (a) a superfície do corpo fica no **interior** de um subdomínio: não
há aresta de Γ perto dela, e o refino só cresce a partir da folha do corpo. Para
que o refino alcance a parede, a superfície precisa **coincidir com fronteiras de
folha** — ou seja, o obstáculo tem de ser união exata de folhas.

**Impossibilidade de "corpo = uma folha" com corpo centrado.** Se o obstáculo é
exatamente uma folha de nível `ℓ ≥ 1`, então `D = H_ℓ` e seu centro está em
`(k + ½)·H_ℓ`. Centrado no canal: `Ly/2 = (k + ½)·D`, logo `Ly/D = 2k+1` é
**ímpar**. Mas a construção diádica dá `Ly = N_by · 2^ℓ · D`, **par** para
`ℓ ≥ 1`. Com `Ly = 8D` e corpo centrado, essa configuração não existe.

**A saída é `D = 2·H_ℓ`**: o centro do corpo cai num *vértice* da grade diádica,
que é exatamente onde o centro do canal está quando `N_by` é **par**. O corpo
passa a ser um bloco `2×2` de folhas de nível `ℓ`, e as vizinhas podem descer a
`ℓ+1` — encostando na superfície.

| Grandeza | família (a) | família (b) |
|---|---|---|
| `N_by` | 3 (ímpar) | 2 (par) |
| `H_0` | `8D/3 ≈ 2,67D` | `4D` |
| corpo | interior a 1 folha de `2,67D` | união exata de 4 folhas de `D/2` |
| `H` junto à parede | `1,33D` | `D/4` |
| `Lx`, `(xc, yc)` | `26,67D`, `(6,67D, 4D)` | `24D`, `(6D, 4D)` |
| níveis | 0–2 | 0–4 |
| `nloc` | `(32, 16, 8)` | `(64, 32, 16, 8, 4)` |
| `h` | `D/12` | `D/16` |
| malha global | `320 × 96` | `384 × 128` |

Restrição adicional em (b): `nloc` da folha mais fina é 4, o que fica no limite
da condição de solvabilidade que relaciona o espaço de interface à malha fina
(Arbogast–Pencheva–Wheeler–Yotov). `plan_quadtree_geometry` passa a exigir
`nloc_min ≥ max(degreeU, degreeP)`.

### 4.4 Critério de refinamento

`build_obstacle_tree(Lx, Ly, N_bx, N_by, obstacle_bbox, max_level, radii,
wake_length=0.0)`:

- `d(folha)` = distância euclidiana da bbox da folha à bbox do obstáculo,
  normalizada por `D` (zero para folhas que tocam o corpo);
- uma folha de nível `ℓ` é subdividida sse `ℓ < max_level` e
  `d(folha) ≤ radii[ℓ]`, com `radii` estritamente decrescente (ex.:
  `radii = (6D, 3D)` para `max_level = 2`) — a monotonicidade é o que garante
  que o anel mais refinado envolve o corpo em vez de aparecer solto no domínio;
- a folha que contém o obstáculo é marcada e **nunca** subdividida;
- `wake_length > 0` estende a região refinada a jusante (default `0`, isto é,
  isotrópico em torno do corpo — o que foi pedido);
- `balance_2to1()` ao final;
- **validação pós-balanço:** reconfirma que o obstáculo continua interior a uma
  única folha. Com raios monotônicos o balanço nunca precisa subdividir a folha
  travada, mas o `raise` explícito é a rede de segurança — `balance_2to1` do
  mrcm não conhece o conceito de folha travada.

### 4.5 Folhas sólidas (necessário apenas na família (b))

Na família (b) o corpo é união exata de folhas, então passam a existir folhas
**100 % sólidas**, e duas interfaces de Γ atravessam o sólido — inteiramente
dentro dele, sem cortar região fluida. Três pontos exigem tratamento, todos
auditados no código atual:

1. **`MACDownscaler.solve_patch`**: com `nfluid = 0`, a normalização do defeito
   `abs(defect)/(hx·hy·nfluid)` divide por zero e produz `nan`. O campo
   resultante sobrevive por acidente — a matriz mascarada degenera em identidade,
   `p = 0`, e as faces sólidas são zeradas depois — mas o caminho precisa de
   guarda explícita: folha sem fluido devolve `ṽ = 0` sem montar sistema.
2. **Solve local de Darcy**: `K = δt·ε` em toda a folha, logo `K⁻¹ ~ 1e6/δt`. As
   variantes de Robin do mrcm já fazem clamp de `K` (`robin_clamp_eps`) e a
   variante `extreme` foi construída para barreira (`robin_extreme_beta`). É
   escolha e validação, não implementação nova.
3. **Condicionamento de `A_Γ`**: arestas interiores ao sólido geram linhas de
   continuidade fraca quase nulas. É o risco principal da família (b) e precisa
   ser **medido** (número de condição estimado, resíduo do solve global), não
   assumido benigno.

O invariante que substitui "obstáculo não cruza interface" é mais preciso e vale
para as duas famílias:

> **Nenhuma folha pode ter região fluida desconexa.**

A razão é estrutural: o problema local do downscaler é Neumann puro; com fluido
em duas componentes a nulidade é 2, e o pino numa única célula remove apenas um
modo — a projeção sairia errada sem erro nenhum. A verificação usa rotulagem de
componentes conexas (`scipy.ndimage.label`) sobre a máscara de sólido de cada
folha, na construção da decomposição.

### 4.6 Estado verificado das dependências no `mrcm-firedrake`

A `RevisaoQuadtree.md` do `mrcm-firedrake` lista problemas do quadtree, mas é
**anterior** às correções documentadas em
`handoffs/2026-07-25-quadtree-session-review.md`. Verificado no código em
2026-07-25, antes de planejar as fases:

| Dependência | Estado | Evidência |
|---|---|---|
| `nloc` por folha chega à malha local | **Já implementado** | `solver.py:124-127` preenche `geom["nlocx"]/["nlocy"]` a partir de `skeleton.sub_nloc`; `local_solver.py:954-955` lê `geom.get("nlocx", cfg.nlocx)`. `Decomposition.task()` só precisa preencher os mesmos campos |
| `sub_nloc` | Shape `(Nsubd, 2)` = `(nlocx, nlocy)` | `topology.py:213-218`; populado hoje só por `CellRefinement` (`cell_refinement.py:345`) |
| Validação de balanço 2:1 em `from_quadtree` | **Já implementado** | `topology.py:324-332`, `ValueError` explícito |
| `_estimate_contrast` amostrado / `from_uniform` com razão de aspecto / `basis_per_side` | Corrigidos | handoff 2026-07-25, itens 1.1–1.8. Irrelevantes aqui: o MPrM usa critério geométrico próprio |
| Hash do skeleton entre ranks | **Pendente** | Sem implementação no mrcm; é responsabilidade desta spec (§7) |
| `use_fine_reference` + MPI + quadtree | Irrelevante | O MPrM não usa `MRCMSolver`; a referência fina é `ChannelFinePoisson` (SciPy) |

Consequência prática: **não há pré-requisito bloqueante no `mrcm-firedrake`**. O
que falta é exercitar `sub_nloc` com valores *heterogêneos*, coisa que nunca
aconteceu (o docstring de `Skeleton` afirma que em modo quadtree "every leaf
carries the same cell count"). Daí o teste dedicado em §8.

---

## 5. Fases de implementação

### Fase 1 — Offline do sistema global de interface

Divide `compute_subdomain_triplets` em duas responsabilidades: a parte-matriz
(`rows`, `cols`, `vals`), calculada uma vez no construtor do projetor, e a
parte-RHS (`b_idx`, `b_vals`), calculada por passo. `MPrMProjector.__init__`
monta `A_Γ` e a fatora (LU esparsa via SciPy, ou o caminho iterativo quando
`cfg.global_solver == "iterative"`). `project()` passa a: (i) resolver a parte
particular local, (ii) computar `b` local, (iii) reduzir `b` ao rank 0
(`n_Γ` floats), (iv) resolver com a fatoração pronta, (v) broadcast de `X`.

A divisão fica em `mprm/projector.py`, não no `mrcm-firedrake`, para não alterar
o comportamento do solver de Darcy usado por outros estudos.

**Aceite:** `u`, `v`, `X` idênticos ao caminho atual até `1e-12` em cavidade e
von Kármán; redução mensurável no tempo por passo.

### Fase 2 — `mprm/decomposition.py`

Implementa `Leaf`, `GammaEdge`, `Decomposition.uniform`,
`Decomposition.from_quadtree`, `plan_quadtree_geometry`, `build_obstacle_tree`
(parametrizado pela família de alinhamento (a)/(b) de D7) e a verificação de
conexidade do fluido por folha (§4.5).
Popula `Skeleton.sub_nloc` com `H_i/h` (o campo existe e é lido por
`_build_tasks`/`build_subdomain_workspace`, mas `from_quadtree` não o preenche —
hoje o mrcm assume `nloc` igual em toda folha).

**Aceite:** `Decomposition.uniform(...)` e `Decomposition.from_quadtree(árvore
uniforme)` produzem o mesmo conjunto de folhas, offsets e arestas de Γ.

### Fase 3 — Projetor sobre decomposição não uniforme

- `MPrMProjector` recebe uma `Decomposition` e itera folhas;
- Robin construído **por folha** com `cfg.robin_L_xy_for(Hx_i, Hy_i, nloc_i,
  nloc_i)`, eliminando o Achado 6;
- extração de traços **por segmento**: os dois segmentos de um lado compartilham
  o mesmo marcador `ds`, então as faces finas são filtradas pelo
  `skeleton.edge_bbox` do segmento;
- média da Eq. 21 reescrita como laço sobre `Decomposition.gamma`, usando
  `edge2el`/`side_of_edge` — substitui a indexação cartesiana `("v", i, iy)` /
  `("h", ix, j)` e funciona nos dois modos;
- `cell_map` passa a usar o `h` global da decomposição.

**Aceite:** com árvore uniforme, campos idênticos ao caminho cartesiano até
`1e-12`; com árvore adaptativa, salto fraco de fluxo nas arestas penduradas na
ordem do zero de máquina relativo à escala do fluxo do segmento.

### Fase 4 — Downscaler por folha

`MACDownscaler` passa a cachear fatorações por `(n, m, padrão de sólido)` em vez
de manter uma única matriz. O patch é localizado por `(i0, j0, n, m)` da folha.
Como os níveis são poucos, o número de fatorações distintas é pequeno. Inclui a
guarda de folha 100 % sólida (§4.5, item 1), necessária para a família (b).

`solve_patch` passa a **retornar** o defeito de compatibilidade junto com
`(ṽ₁, ṽ₂)`, em vez de acumulá-lo no atributo `last_defect`. Duas razões: o
atributo atual é um `max` que nunca é reiniciado entre passos (mede o pior
instante de toda a simulação, não do passo), e sob quadtree a normalização
`|defeito|/(h²·n·m)` tem escala diferente por folha — um `max` sobre folhas de
tamanhos distintos mistura grandezas. O chamador agrega e reporta por folha e
por passo.

**Aceite:** `max|∇·u| ≤ 1e-12` no fluido com árvore adaptativa; defeito de
compatibilidade da mesma ordem do caso uniforme.

### Fase 5 — Estudo comparativo

`mprm/benchmarks/quadtree_vk.py` executa, sobre **a mesma geometria e o mesmo
`h`**:

1. cartesiano uniforme (baseline);
2. quadtree adaptativo, família (a), calibrado para igualar `n_Γ` ao baseline;
3. quadtree adaptativo, família (a), calibrado para igualar `N_sub` ao baseline;
4. quadtree adaptativo, **família (b)** — refino alcançando a superfície do
   corpo (§4.3.1), com `nloc_min` e níveis próprios;
5. referência `ChannelFinePoisson` (Poisson global na malha fina) — o "fine mesh
   solver" do artigo.

A configuração (4) roda sobre geometria e `h` próprios (`384×128`, `h = D/16`),
então sua comparação com (1)–(3) é feita contra a **mesma referência de malha
fina recomputada naquela malha**, e o custo é reportado em `n_Γ` e tempo, não em
número de células. A pergunta que ela responde: levar o refino de `H` até a
parede compensa o preço de folhas sólidas e do `A_Γ` que elas produzem?

A calibração de (2) e (3) é feita variando `radii` e `max_level`: a árvore é
construída, `n_Γ` e `N_sub` são contados **antes** de qualquer solve, e os raios
são ajustados por busca até o alvo. Como ambos são inteiros, o alvo exato pode
não existir; nesse caso usa-se o valor mais próximo por baixo e a diferença é
reportada junto com a métrica.

**Restrição de comparabilidade a registrar no resultado:** fixada a malha global
`320×96`, as decomposições cartesianas com células quadradas formam a família
discreta `(10k, 3k)`, isto é, `N_sub ∈ {30, 120, 480, …}`. A comparação primária
é portanto por `n_Γ`; a comparação por `N_sub` usa o `k` mais próximo e reporta
explicitamente a diferença.

**Métricas por configuração:** `C_D` médio, amplitude de `C_L`, `St`,
`‖u − u_ref‖₂/‖u_ref‖₂`, `max|∇·u|`, defeito de compatibilidade, `N_sub`, `n_Γ`,
`min/max H`, tempo de offline e tempo de online separados.

**Formato de saída.** O JSON é o artefato que alimenta as tabelas e figuras da
dissertação, então o schema é fixado aqui:

```json
{
  "geometry": {"family": "a", "Lx": 26.67, "Ly": 8.0, "D": 1.0,
               "h": 0.0833, "Nx": 320, "Ny": 96, "xc": 6.67, "yc": 4.0},
  "flow": {"Re": 100, "U_in": 1.0, "inflow": "parabolic",
           "inflow_norm": "max", "walls": "slip", "dt": 0.0333, "T": 40.0},
  "spaces": {"degreeU": 1, "degreeP": 3, "alpha": 1.0,
             "basis_kind": "legendre_P"},
  "configs": [
    {"label": "cartesian_uniform_10x3",
     "N_sub": 30, "n_gamma_edges": 47, "n_gamma_dofs": 282,
     "H_min": 2.67, "H_max": 2.67, "levels": [0],
     "cond_A_gamma": 1.4e3,
     "t_offline_s": 0.0, "t_online_per_step_s": 0.0,
     "Cd_mean": 0.0, "Cd_std": 0.0, "Cl_amp": 0.0, "St": 0.0,
     "div_max": 0.0, "defect_max": 0.0, "err_u_rel": 0.0}
  ]
}
```

Os valores numéricos acima são o *schema*, não resultados — exceto os campos
estruturais do baseline, que são deriváveis: uma grade `10×3` tem
`9·3 + 10·2 = 47` arestas internas, e com `degreeU=1, degreeP=3`
(`ndofsU = degreeU+1 = 2`, `ndofsP = degreeP+1 = 4`) são `47·6 = 282` graus de
liberdade de interface. `cond_A_gamma` é a métrica que responde ao risco das
folhas sólidas na família (b).

### 5.6 Relação com o plano de validação de von Kármán

`docs/11-plano-validacao-von-karman.md` define quatro fases de validação. A
Fase 5 desta spec **absorve** as duas primeiras (setup do artigo e comparação
contra a referência de malha fina), porque roda exatamente essa configuração
como baseline. A varredura de parâmetros daquele plano (`α ∈ {10⁻⁶, 1, 10⁶}`,
graus `k = 1, 2, 3`) é **ortogonal** à decomposição e deve ser feita depois,
sobre a melhor decomposição identificada — rodá-la agora multiplicaria o custo
por um fator que não informa a pergunta desta spec.

---

## 6. Fluxo de dados de um passo de tempo

```
uⁿ, vⁿ  ──mac_transport──►  w*        (MAC global, h uniforme, inalterado)
                             │
                             ├─► ∇·w* por célula
                             │
        para cada folha i:   ├─► fonte local  →  solve particular (LU offline)
                             │                     │
                             │                     └─► b_i = −⟨c, ū_i⟩
                             │
        rank 0:              ├─► b = Σ b_i  →  X = LU(A_Γ)⁻¹ b     [Fase 1]
                             │
        para cada folha i:   ├─► reconstrução  →  traços por segmento
                             │
        por aresta de Γ:     ├─► média dos dois lados (Eq. 21)     [Fase 3]
                             │
        para cada folha i:   ├─► Neumann local na MAC (Eq. 22)     [Fase 4]
                             │
                             └─► uⁿ⁺¹ = w* − ṽ
```

---

## 7. Validações e tratamento de erro

Falhas devem aparecer na construção, não como resultado numérico plausível:

| Verificação | Onde | Ação |
|---|---|---|
| `H_ℓ / h` inteiro e par em todo nível | `plan_quadtree_geometry` | `ValueError` nomeando o nível |
| `nloc_min ≥ max(degreeU, degreeP)` | `plan_quadtree_geometry` | `ValueError` |
| Família (a): obstáculo interior a uma única folha (antes e depois do balanço) | `build_obstacle_tree` | `ValueError` com a bbox ofensora |
| Família (b): obstáculo é união exata de folhas | `build_obstacle_tree` | `ValueError` com as folhas parcialmente cobertas |
| Nenhuma folha com região fluida desconexa | `Decomposition` (§4.5) | `ValueError` nomeando a folha |
| Balanço 2:1 | `Skeleton.from_quadtree` (já existente) | `ValueError` do mrcm |
| Alinhamento de segmentos | `check_segment_alignment` (já existente) | `ValueError` do mrcm |
| `h` único / `hx == hy` | `Decomposition` | `ValueError` |
| Consistência do skeleton entre ranks MPI | `Decomposition.from_quadtree` | `assert` sobre hash de `el2edge`/`edge2el` |

A última merece destaque: cada rank constrói a árvore independentemente. É
determinístico, mas se divergir os triplets são montados contra numerações de Γ
diferentes e o sistema global sai numericamente errado **sem erro algum** — a
mesma falha que a revisão do quadtree no `mrcm-firedrake` (§2.3 da
`RevisaoQuadtree.md`) identificou.

---

## 8. Testes

**Equivalência (a rede de segurança principal)**
1. `Decomposition.uniform(2,2,...)` ≡ `from_quadtree(árvore uniforme 2×2)`:
   mesmas folhas, offsets e arestas de Γ.
2. Projeção com árvore uniforme ≡ caminho cartesiano atual, campo a campo (`1e-12`).
3. Fase 1 (pré-fatoração) ≡ caminho de remontagem, campo a campo (`1e-12`).

**`nloc` heterogêneo (nunca exercitado — §4.6)**
3b. Com `sub_nloc` variando entre folhas, a malha local de cada folha tem
    exatamente `nlocx_i × nlocy_i` células e cobre a bbox da folha. É o elo que
    nunca rodou com valores diferentes por folha; se falhar, todos os testes de
    corretude abaixo falham por consequência.

**Corretude sob adaptatividade**
4. `max|∇·u|` no fluido `≤ 1e-12` com árvore adaptativa (cavidade e von Kármán).
5. Continuidade fraca de fluxo nas arestas penduradas: salto ponderado
   desprezível frente ao fluxo do segmento — o teste que a revisão do mrcm
   mostrou ser o único capaz de flagrar erro de parametrização de base, e que
   deve rodar com `degreeU ≥ 2` (o modo constante `k=0` é cego a ele).
6. `u·n = 0` nas faces do obstáculo; balanço de massa entrada ≈ saída.

**Geometria**
7. `plan_quadtree_geometry` rejeita `nloc` ímpar, `nloc_min < grau` e, na família
   (a), `N_by` par ou `H_0 ≤ D`.
8. `build_obstacle_tree` mantém o obstáculo interior a uma folha após o balanço
   (família (a)) / como união exata de folhas (família (b)).
9. Folha com fluido desconexo é rejeitada na construção (teste sintético: faixa
   sólida atravessando uma folha de lado a lado).

**Família (b): folhas sólidas**
10. Folha 100 % sólida devolve `ṽ = 0` sem `nan` no defeito.
11. `max|∇·u|` e balanço de massa preservados com o corpo como bloco 2×2.
12. Condicionamento de `A_Γ` registrado (não é critério de passa/falha — é dado
    do estudo).

**Paralelismo**
9. Sequencial ≡ MPI (2 e 4 ranks) em árvore adaptativa, até `1e-10`.

**Regressão**
10. Suíte existente (`test_projector`, `test_transport`, `test_downscaler_obstacle`,
    `test_flow_problem`, `test_von_karman::test_von_karman_quick`,
    `test_run_regression`) passa sem alteração.

---

## 9. Critérios de sucesso

1. von Kármán roda de ponta a ponta sobre decomposição quadtree, com massa
   conservada em `1e-12` e obstáculo centrado.
2. O estudo comparativo produz a tabela das quatro configurações com todas as
   métricas da Fase 5, em JSON e figura.
3. A pré-fatoração reduz mensuravelmente o tempo por passo, com resultado
   numericamente idêntico.
4. O estudo responde, com número, se levar o refino de `H` até a superfície do
   corpo (família (b)) compensa o custo das folhas sólidas.
5. As conclusões são declaradas **com o sinal que os dados mostrarem**. Se a
   decomposição adaptativa não superar a uniforme a `n_Γ` igual, ou se a família
   (b) não superar a (a), isso é o resultado — e é um resultado publicável.

---

## 10. Riscos

| Risco | Mitigação |
|---|---|
| Na família (a) o refino máximo fica a uma folha de distância do corpo (D4 + balanço 2:1), limitando o ganho junto à parede | `H_0` escolhido pouco maior que `D`; a família (b) existe justamente para medir quanto isso custa |
| Na família (b), `A_Γ` degrada por causa das arestas interiores ao sólido (linhas de continuidade fraca quase nulas) | Medir condicionamento e resíduo do solve global; se degradar, testar as variantes `extreme`/`adaptive` de Robin, para as quais esse é o cenário de projeto |
| `Skeleton.sub_nloc` nunca foi exercitado com `nloc` variando por folha (só `CellRefinement` o popula, e a revisão do mrcm aponta que o `k_grid` não chegava à discretização) | Fase 2 valida via teste de equivalência antes de qualquer física |
| Custo do offline cresce: sem cache de MsBF entre folhas (`cacheable=False` em modo quadtree), e a folha de nível 0 tem `nloc = 32` (família (a)) ou `64` (família (b)) | Offline roda uma vez, medido e reportado separado do online. Contingência, se exceder o aceitável: reduzir um nível de refinamento (o que reduz `nloc_0` pela metade) antes de reduzir `nloc_min`, que está travado por `nloc_min ≥ max(degreeU, degreeP)` |
| Desbalanceamento de carga MPI (folhas de custo `nloc_i²` distribuídas round-robin) | Distribuição por peso `nloc_i²`; item de baixo custo, avaliado na Fase 5 |
| Mudança de geometria invalida comparação com resultados anteriores | Aceito explicitamente (D6): a comparação que importa é com Lages e Breuer |

---

## 11. Fora de escopo

- AMR na escala fina (`h` variável, malha MAC composta) — extensão futura, spec próprio.
- Variantes incremental/rotacional do método de projeção (Achado 4).
- Refinamento adaptativo a posteriori ou dentro do laço temporal (D3).
- Obstáculos **não alinhados** à grade diádica (cilindro circular, corpo
  inclinado). Ambas as famílias de D7 dependem do alinhamento do quadrado; um
  corpo arbitrário exige folhas com fluido parcial em geometria qualquer, com a
  análise de conexidade da §4.5 no caminho crítico.
- Achados 2 e 3 da revisão (traço pré-montado, `Allgatherv`) — otimizações
  independentes, sem efeito sobre a validade do estudo.
- Redação da dissertação.

---

## 12. Referências teóricas

Marcação: **[R]** disponível em `~/Dissertação/Refs`; **[F]** a obter.

### 12.1 Método e formulação

| Referência | Uso |
|---|---|
| **[R]** Lages, C. *The multiscale projection method for the solution of incompressible Navier–Stokes equations.* | Base direta: Eqs. 8–10 (projeção), 13 (forma mista), 15–20 (MRCM), 21–23 (média de fluxo e downscaling), §3.5 (benchmark de von Kármán) |
| **[R]** Guiraldello, R. T.; Ausas, R. F.; Sousa, F. S.; Pereira, F.; Buscaglia, G. C. *The Multiscale Robin Coupled Method for flows in porous media* (`MRCM.pdf`) | Formulação do MRCM; condição de Robin; `β = α L / K` (Eq. 65) — o `L` por folha é o ponto do Achado 6 |
| **[R]** Guiraldello et al. *Interface spaces for the Multiscale Robin Coupled Method* (`Interface Spaces - MRCM.pdf`) | Escolha de `U_H`, `P_H` e do grau `k`; base da comparação por `n_Γ` |
| **[R]** Guiraldello, R. T. *Tese* (`RafaelTrevisanutoGuiraldello_revisada.pdf`) | Desenvolvimento completo do MRCM |
| **[R]** Jaramillo, A. *Dissertação* (`Jaramillo_dissertacao.pdf`) | Implementação de referência do MRCM |
| **[F]** Chorin, A. J. (1968); Temam, R. (1969) | Método de projeção original (não incremental), que é o esquema implementado |
| **[R]** Quarteroni, A.; Saleri, F.; Veneziani, A. *Factorization methods for the numerical approximation of Navier–Stokes equations* | Leitura algébrica da projeção como fatoração LU aproximada; fundamenta o Achado 4 |
| **[F]** Guermond, J.-L.; Minev, P.; Shen, J. (2006) *An overview of projection methods for incompressible flows*, CMAME 195 | Taxonomia das variantes e ordens de erro de splitting |
| **[F]** Harlow, F. H.; Welch, J. E. (1965) | Malha MAC deslocada |

### 12.2 Interfaces não conformes e nós pendurados

Núcleo teórico da decomposição quadtree — é aqui que o trabalho se apoia para
justificar segmentos de Γ independentes.

| Referência | Uso |
|---|---|
| **[R]** Arbogast, T.; Cowsar, L. C.; Wheeler, M. F.; Yotov, I. *Mixed finite element methods on nonmatching multiblock grids* | Teoria de interfaces não conformes entre blocos com malhas distintas; condições sobre o espaço de multiplicadores |
| **[R]** Arbogast, T.; Pencheva, G.; Wheeler, M. F.; Yotov, I. *A multiscale mortar mixed finite element method* | Condição de solvabilidade relacionando o espaço de interface à malha fina — o que limita `degreeU`/`degreeP` por segmento |
| **[R]** Bernardi, C.; Maday, Y.; Patera, A. T. *Domain decomposition by the mortar element method* | Formulação mortar; casamento fraco em interfaces não conformes |
| **[R]** Ben Belgacem, F. *The mortar finite element method with Lagrange multipliers* | Análise do método mortar |
| **[R]** Ganis, B.; Yotov, I. *Implementation of a mortar mixed finite element method using a multiscale flux basis* | Estratégia offline/online por base de fluxo — o padrão que a Fase 1 leva às últimas consequências |
| **[R]** Pencheva, G.; Yotov, I. *Implementation of a mortar mixed finite element method* | Detalhes de implementação do acoplamento |

### 12.3 Árvores, adaptatividade e Navier–Stokes

| Referência | Uso |
|---|---|
| **[R]** Olshanskii, M. A.; Terekhov, K. M.; Vassilevski, Y. V. (2013) *An octree-based solver for the incompressible Navier–Stokes equations*, Computers & Fluids 84:231–246 | Referência central de NS sobre malha de árvore graduada; discretização em faces com nós pendurados. **Nota:** a conversão em `Refs/markdown/Olshanskii2013_octree_solver.md` está corrompida (contém texto de outro artigo) — usar o PDF |
| **[R]** *A multimesh finite element method for the Navier–Stokes equations based on projection methods* | Projeção com múltiplas malhas — o parente mais próximo desta construção |
| **[F]** de Berg, M. et al. *Computational Geometry*, §14.3 | Quadtree balanceado 2:1 (algoritmo que o `mrcm-firedrake` implementa) |
| **[F]** Popinet, S. (2003) *Gerris: a tree-based adaptive solver for the incompressible Euler equations*, JCP 190 | Referência canônica de projeção sobre quadtree; contraponto de AMR em `h` |
| **[F]** Losasso, F.; Gibou, F.; Fedkiw, R. (2004) *Simulating water and smoke with an octree data structure* | Projeção com faces penduradas em octree |
| **[F]** Berger, M. J.; Colella, P. (1989) *Local adaptive mesh refinement for shock hydrodynamics* | Fundamento de AMR estruturado; relevante para a extensão futura em `h` |

### 12.4 Elementos finitos mistos e álgebra

| Referência | Uso |
|---|---|
| **[R]** Brezzi, F.; Fortin, M. *Mixed and hybrid finite element methods* | Teoria de problemas de ponto de sela; condição inf-sup |
| **[R]** Boffi, D.; Brezzi, F.; Fortin, M. (`boffi2013`) | Tratamento moderno de elementos mistos |
| **[R]** Arnold, D. N.; Falk, R. S.; Winther, R. (`arnold2006`) | Cálculo exterior de elementos finitos; famílias BDM/RT usadas nos solves locais |
| **[R]** Golub, G. H.; Van Loan, C. F. *Matrix Computations*, 4ª ed. | Fatoração LU esparsa, fill-in e reuso de fatoração — fundamenta a Fase 1 |
| **[R]** *PETSc manual*; **[R]** Rathgeber et al. *Firedrake*; **[R]** Dalcín et al. *mpi4py* | Ferramental da implementação |

### 12.5 Validação

| Referência | Uso |
|---|---|
| **[F]** Breuer, M.; Bernsdorf, J.; Zeiser, T.; Durst, F. (2000) *Accurate computations of the laminar flow past a square cylinder*, IJHFF 21:186–196 | Benchmark internacional: `C_D` e `St` a `Re = 100`, blockage `1/8`, entrada parabólica, paredes free-slip. Define a geometria adotada |
| **[R]** Lages, §3.5, Tabelas 2–3 | Valores de referência do próprio MPrM: `C_D = 1,33`, `C_L = 0,263`, `St = 0,121` (malha fina) e o comportamento em `α` e no grau dos espaços de interface |
| **[R]** Ghia, U.; Ghia, K. N.; Shin, C. T. *High-Re solutions for incompressible flow using the Navier–Stokes equations and a multigrid method* | Cavidade — teste de regressão do caminho generalizado |
| **[R]** Benchmark de Helmholtz–Hodge de Olshanskii (já em `mprm/benchmarks/hh.py`) | Verificação isolada do passo de projeção, independente do transporte |
| **[F]** Sohankar, A.; Norberg, C.; Davidson, L. (1998) | Segunda fonte para cilindro quadrado a `Re` baixo (opcional) |

---

## 13. Trabalho futuro

- AMR em `h` (malha MAC composta, faces penduradas, `dt` pelo menor `h`).
- Projeção incremental/rotacional (Achado 4).
- Refinamento a posteriori guiado pelo salto de fluxo nas interfaces — o
  estimador natural do MRCM, com uma readaptação após o transiente inicial.
- Cache de MsBF entre folhas de mesmo nível (mesmo `Hx, Hy, nloc`): em modo
  quadtree o `mrcm` desliga o cache, mas folhas de mesmo nível são
  geometricamente idênticas a menos de translação.
