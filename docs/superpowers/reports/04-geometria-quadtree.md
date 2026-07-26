# Relatório — Tarefa 4: Geometria do quadtree — `plan_quadtree_geometry` e `build_obstacle_tree`

## 1. Identificação
- **Tarefa:** 4 — Geometria do quadtree (spec §4.3, §4.3.1 e §4.4)
- **Data:** 2026-07-26
- **Commit antes:** `210c7b2` (`git rev-parse --short HEAD` antes de começar, na
  branch `feature/quadtree-mprm`)
- **Commit depois:** `<preenchido no commit de follow-up>`
- **Arquivos criados:**
  - `docs/superpowers/reports/04-geometria-quadtree.md` (este arquivo)
- **Arquivos modificados:**
  - `mprm/decomposition.py` — linhas 169-397 (arquivo passa de 177 para 408
    linhas): bloco novo com `plan_quadtree_geometry` (173-265) e
    `build_obstacle_tree` (267-397), inserido entre `Decomposition.task` e o
    helper privado `_side_bbox` (agora na linha 400). Nada do que a Tarefa 3
    deixou ali foi tocado.
  - `tests/test_decomposition.py` — linhas 80-221 (arquivo passa de 79 para 221
    linhas): bloco novo com as classes `TestPlanGeometry` (87-137) e
    `TestObstacleTree` (139-221), acrescentado ao final. Os 5 testes da
    Tarefa 3 (linhas 15-79) ficaram intactos.

## 2. Estado herdado
Herdado de `docs/superpowers/reports/03-decomposition.md`: `mprm/decomposition.py`
já existia com `Leaf`, `GammaEdge` e `Decomposition` (`uniform`, `segments_of`,
`gamma_edges`, `task`), e `tests/test_decomposition.py` com 5 testes do caminho
cartesiano. Baseline real da suíte na entrada desta tarefa: **33 passed** (não
os 19/21 discutidos no `pre-flight-resolutions.md` item 7 e no §C do plano — a
suíte cresceu nas Tarefas 1-3). Esta tarefa é NumPy puro; não toca Firedrake.

Assinaturas consumidas, copiadas verbatim de `~/mrcm/mrcm-firedrake/mrcm/quadtree.py`
(não modificado — Restrição Global 4):

```python
@dataclass
class QuadTreeNode:
    x0: float; y0: float; x1: float; y1: float
    level: int
    subdomain_id: int | None = None
    children: list[QuadTreeNode] | None = field(default=None, compare=False)
    neighbours: list[list[QuadTreeNode]] = field(default_factory=lambda: [[], [], [], []],
                                                 compare=False)
    @property
    def is_leaf(self) -> bool; Hx; Hy; cx; cy
    def subdivide(self) -> list[QuadTreeNode]      # quadtree.py:122-133

@dataclass
class QuadTree:
    root: QuadTreeNode; max_level: int; Lx: float; Ly: float
    def _invalidate_cache(self) -> None            # quadtree.py:171-173
    def _collect_leaves(self, use_cache: bool = True) -> list[QuadTreeNode]
    @property
    def leaves(self) -> list[QuadTreeNode]         # atribui subdomain_id
    @property
    def Nsubd(self) -> int
    def _build_neighbours(self)                    # quadtree.py:224-278
    def balance_2to1(self) -> None                 # quadtree.py:415-441
```

## 3. Regressão ANTES da execução

Comando exato:

```bash
source ~/venv-firedrake/bin/activate
python -m pytest tests/ -q
```

Saída: `33 passed in 110.40s (0:01:50)`.

| Suíte | Baseline esperado | Obtido antes desta tarefa | Igual? |
|---|---|---|---|
| `pytest tests/ -q` | 33 passed, 0 failed (herdado da Tarefa 3) | 33 passed, 0 failed | sim |

O §C do plano diz `21 passed` e o `pre-flight-resolutions.md` item 7 corrige
para `19 passed`; ambos são números da Tarefa 0 e já foram superados pelas
Tarefas 1-3 (7 + 5 + 2 testes novos). O baseline operante aqui é **33**.

## 4. O que foi feito

`plan_quadtree_geometry` resolve, em aritmética exata sobre potências de 2, a
cadeia de restrições da spec §4.3: `H0 = Ly/N_by`, `H[ℓ] = H0/2^ℓ`,
`h = H[max_level]/nloc_min`, `nloc[ℓ] = H[ℓ]/h`, e valida cada elo — `nloc` par
(um segmento de nó pendurado tem de terminar em fronteira de célula, senão a
base de interface mascarada por célula integra intervalos diferentes dos dois
lados, `mrcm/topology.py:109-160`), `nloc_min ≥ max(degreeU, degreeP)`
(Arbogast et al.), `Lx/h` e `Ly/h` inteiros. As duas famílias de alinhamento
da §4.3.1 viram dois ramos: (a) `N_by` ímpar, corpo interior a uma folha de
nível 0 travada, exige `H0 > D`; (b) `N_by` par, corpo = bloco 2×2 exato de
folhas de nível `body_level` com `D = 2·H[body_level]`.

`build_obstacle_tree` monta a raiz à mão (a grade base `N_bx × N_by` não é
potência de 2, então `QuadTree.from_uniform` não serve — `quadtree.py:304-308`)
e refina em rounds: subdivide a folha de nível ℓ se a distância da sua bbox à
bbox do corpo for `≤ radii[ℓ]`, chamando `balance_2to1` uma vez por round.

**A decisão que o plano não especificava** é o `_cap_de_nivel`
(`mprm/decomposition.py:334-351`), exigido pelo item 1 do
`pre-flight-resolutions.md`: `balance_2to1` não conhece a folha travada e a
subdivide se uma vizinha imediata chegar a nível `travada+2`. O cap grada o
refino em anéis a partir da travada — a distância L∞ até a bbox travada tem de
ser `≥ H_travada·(1 − 2^(1−m))` para o nível `travada+m` ser liberado, que é
exatamente o buffer necessário para caber um anel de cada nível intermediário
(de Berg et al., §14.3). Para `m = 1` isso é a regra do pré-voo ("as vizinhas
imediatas ficam em nível 1"); para `m ≥ 2` é a generalização que mantém a
invariante quando `max_level > 2`. Ver seção 7 para a medição que motivou a
generalização.

## 5. Testes criados nesta tarefa

**Execução 1 (comum a todos os 13):** antes da implementação, o arquivo nem
coleta — o import no topo do bloco novo falha:

```
tests/test_decomposition.py:84: in <module>
    from mprm.decomposition import plan_quadtree_geometry, build_obstacle_tree
E   ImportError: cannot import name 'plan_quadtree_geometry' from 'mprm.decomposition'
ERROR tests/test_decomposition.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 5.01s
```

**Execução 2 (comum a todos os 13):** após a implementação,
`python -m pytest tests/test_decomposition.py -q` → `18 passed in 5.02s`
(5 da Tarefa 3 + 13 novos).

### `test_familia_a_configuracao_de_referencia`
- **Arquivo:** `tests/test_decomposition.py:88`
- **O que valida:** a cadeia aritmética completa da família (a) de referência.
- **Resultado esperado (escrito ANTES de rodar):** `H0=8/3`, `Lx=80/3`,
  `nloc=[32,16,8]`, `h=1/12`, `(Nx,Ny)=(320,96)`, `yc=4`.
- **Valor numérico obtido:** `H0=2.666667`, `Lx=26.666667`, `h=0.0833333`,
  `nloc=[32, 16, 8]`, `(Nx, Ny)=(320, 96)`, `xc=6.666667`, `yc=4.0`,
  `body_level=0`. PASS.

### `test_familia_b_configuracao_de_referencia`
- **Arquivo:** `tests/test_decomposition.py:99`
- **O que valida:** a família (b) e a identificação de `body_level` por
  `D = 2·H[ℓ]`.
- **Resultado esperado (escrito ANTES de rodar):** `H0=4`, `nloc=[64,32,16,8,4]`,
  `h=1/16`, `(Nx,Ny)=(384,128)`, `body_level=3`, `H[3]=0.5`.
- **Valor numérico obtido:** `H0=4.0`, `Lx=24.0`, `h=0.0625`,
  `nloc=[64, 32, 16, 8, 4]`, `(Nx, Ny)=(384, 128)`, `xc=6.0`, `yc=4.0`,
  `body_level=3`, `H[3]=0.5`. PASS.

### `test_familia_a_rejeita_N_by_par`
- **Arquivo:** `tests/test_decomposition.py:111`
- **O que valida:** `N_by` par na família (a) é rejeitado com a palavra "ímpar".
- **Resultado esperado:** `ValueError` casando `ímpar`.
- **Valor numérico obtido:** `N_by=4` → `ValueError: family='a' exige N_by ímpar
  (recebido 4): ...`. PASS.

### `test_familia_b_rejeita_N_by_impar`
- **Arquivo:** `tests/test_decomposition.py:116`
- **O que valida:** `N_by` ímpar na família (b) é rejeitado.
- **Resultado esperado:** `ValueError` casando `par`.
- **Valor numérico obtido:** `N_by=3` → `ValueError: family='b' exige N_by par
  (recebido 3): ...`. PASS.

### `test_rejeita_H0_menor_que_D_na_familia_a`
- **Arquivo:** `tests/test_decomposition.py:121`
- **O que valida:** a folha travada precisa caber o corpo inteiro.
- **Resultado esperado:** `ValueError` casando `H0`.
- **Valor numérico obtido:** `D=4.0` com `H0=8/3=2.6667` → `ValueError:
  family='a' exige H0 > D: H0=2.66667, D=4`. PASS.

### `test_rejeita_nloc_impar`
- **Arquivo:** `tests/test_decomposition.py:126`
- **O que valida:** `nloc` ímpar em algum nível quebra o alinhamento de segmento
  contra nó pendurado.
- **Resultado esperado:** `ValueError` casando `par`.
- **Valor numérico obtido:** `nloc_min=7` → `h=2/21`, `nloc=[28,14,7]` →
  `ValueError: nível 2: nloc=7 é ímpar. ...` PASS.

### `test_rejeita_nloc_min_menor_que_grau`
- **Arquivo:** `tests/test_decomposition.py:132`
- **O que valida:** a condição de Arbogast et al. entre o espaço de interface e
  a malha fina do segmento.
- **Resultado esperado:** `ValueError` casando `degree`.
- **Valor numérico obtido:** `nloc_min=2 < max(degreeU=1, degreeP=3)=3` →
  `ValueError: nloc_min=2 < max(degreeU=1, degreeP=3): ...`. PASS.

### `test_familia_a_obstaculo_interior_a_uma_folha`
- **Arquivo:** `tests/test_decomposition.py:140`
- **O que valida:** a invariante central da família (a) — exatamente 1 folha
  contém o corpo depois de construir e balancear a árvore.
- **Resultado esperado:** `len(conteiners) == 1`.
- **Valor numérico obtido:** `1` folha (nível 0, bbox `[5.3333, 8.0]×[2.6667,
  5.3333]`) contendo o corpo `[6.1667, 7.1667]×[3.5, 4.5]`. **Sem o cap de
  nível este número é `0`** (ver seção 7). PASS.

### `test_familia_b_obstaculo_e_uniao_exata_de_4_folhas`
- **Arquivo:** `tests/test_decomposition.py:150`
- **O que valida:** na família (b) o corpo é a união exata de 4 folhas de nível
  `body_level`.
- **Resultado esperado:** 4 folhas, área somada `= D² = 1.0`.
- **Valor numérico obtido:** `4` folhas de nível 3 (lado 0.5), área
  `4 × 0.25 = 1.0`, tolerância `pytest.approx` (rel. 1e-6). Árvore:
  `Nsubd=348`, níveis `{0:4, 1:8, 2:64, 3:80, 4:192}`. PASS.

### `test_arvore_balanceada_2_para_1`
- **Arquivo:** `tests/test_decomposition.py:162`
- **O que valida:** a árvore da família (a) satisfaz o balanço 2:1 e o limite
  de 2 vizinhos por lado que o `Skeleton` do mrcm assume.
- **Resultado esperado:** `len(neighbours[s]) ≤ 2` e `|Δnível| ≤ 1` em todas as
  folhas e lados.
- **Valor numérico obtido:** máx. vizinhos por lado `= 2`; máx. diferença de
  nível adjacente `= 1`, sobre as 120 folhas × 4 lados. PASS.

### `test_refino_decresce_com_a_distancia_ao_corpo`
- **Arquivo:** `tests/test_decomposition.py:173`
- **O que valida:** o refino é maior perto do corpo do que longe dele.
- **Resultado esperado:** `max(perto) > max(longe)`, com a janela ampliada de
  `2.0` para `3.0` (um `H0`) conforme o pré-voo.
- **Valor numérico obtido:** janela `3.0` → 21 folhas, `max(perto)=2`;
  `|cx−xc| > 12` → 9 folhas, `max(longe)=0`; `2 > 0` ✓. Medição da janela
  original: janela `2.0` → **1 folha**, `max(perto)=0`, ou seja `0 > 0` falso
  (ver seção 7). PASS.

### `test_folha_travada_nao_e_subdividida_pelo_balanco`
- **Arquivo:** `tests/test_decomposition.py:187`
- **O que valida:** fixa os números de referência da família (a) verificados no
  pré-voo, para que uma regressão no cap seja detectada por contagem, não só
  por exceção.
- **Resultado esperado (escrito ANTES de rodar):** `Nsubd == 120` e
  `{0:16, 1:40, 2:64}`.
- **Valor numérico obtido:** `Nsubd=120`, níveis `{0: 16, 1: 40, 2: 64}`
  (exato, sem tolerância). PASS.

### `test_cap_gradado_resiste_a_mais_de_um_anel_refinado`
- **Arquivo:** `tests/test_decomposition.py:202`
- **O que valida:** o cap gradado (e não apenas o anel que encosta na travada)
  preserva a invariante quando `max_level=3` e os raios permitem um segundo
  anel refinado.
- **Resultado esperado (escrito ANTES de rodar):** 1 folha contendo o corpo, de
  nível 0, e balanço 2:1 preservado.
- **Valor numérico obtido:** `max_level=3`, `radii=(8.0, 6.0, 4.0)` →
  `Nsubd=381`, níveis `{0:13, 1:24, 2:120, 3:224}`, 1 folha de nível 0 contendo
  o corpo, máx. `|Δnível|=1`. Com o cap restrito ao anel imediato, o mesmo caso
  dá **0 folhas** contendo o corpo (medido, ver seção 7). PASS.

**Verificação extra (fora da suíte, por script):** `Skeleton.from_quadtree`
aceita as três árvores construídas (família (a) de referência, família (b) de
referência, família (a) com `max_level=3`), devolvendo `is_uniform=False`.
Ainda não virou teste porque a integração `Skeleton ↔ Decomposition` é da
Tarefa 5.

## 6. Regressão DEPOIS da execução

| Suíte | Antes | Depois | Igual? |
|---|---|---|---|
| `pytest tests/ -q` | 33 passed | 46 passed (33 + 13 novos) | sim — nenhum teste mudou de status |
| `pytest tests/test_decomposition.py -q` | 5 passed | 18 passed (5 + 13 novos) | sim |

Saída literal: `46 passed in 127.03s (0:02:07)`. Nenhum teste pré-existente
mudou de status; nenhum arquivo fora de `mprm/decomposition.py` e
`tests/test_decomposition.py` foi tocado.

## 7. Desvios do plano

**Desvio 1 — cap de nível na vizinhança da folha travada (obrigatório).**
Aplicado conforme o item 1 do
`.superpowers/sdd/2026-07-25-mprm-quadtree-von-karman/pre-flight-resolutions.md`.
O `build_obstacle_tree` do plano (Tarefa 4, Step 4) trava a folha que contém o
corpo mas não impede que as vizinhas imediatas sejam refinadas: com os números
literais dos próprios testes do plano (`radii=(6.0, 3.0)`, `max_level=2`,
`N_by=3`, `D=1`), as folhas de nível 1 adjacentes estão a `(H0−D)/2 = 0.8333 ≤
radii[1] = 3.0` do corpo, viram nível 2, e `balance_2to1`
(`mrcm/quadtree.py:415-441`, que não tem noção de folha travada) subdivide a
própria travada. Reproduzido: **0 folhas contendo o corpo**, disparando o
`ValueError` de validação do próprio plano. O argumento de monotonicidade da
spec §4.4 está invertido — monotonicidade garante que o anel mais fino *encosta*
na travada, e é justamente isso que quebra a invariante.

A correção vive só em `mprm/decomposition.py` (`_cap_de_nivel`, linhas 334-351);
`~/mrcm/mrcm-firedrake` não foi tocado.

**Desvio 1b — forma do cap (generalização deliberada da resolução do pré-voo).**
O pré-voo prescreve "uma folha cuja bbox fechada toca a bbox fechada da travada
não pode exceder nível `travada+1`". Implementei a versão gradada: o nível
`travada+m` só é liberado quando a distância L∞ à bbox travada é
`≥ H_travada·(1 − 2^(1−m))`, que para `m=1` é *exatamente* a regra prescrita
(distância 0 ⇒ nível máximo `travada+1`) e para `m ≥ 2` acrescenta o buffer de
gradação. Motivo, medido comparando as duas variantes lado a lado com o mesmo
código:

| caso | cap "toca ⇒ ≤ travada+1" | cap gradado |
|---|---|---|
| referência: `max_level=2`, `radii=(6.0, 3.0)` | `Nsubd=120`, `{0:16, 1:40, 2:64}` | `Nsubd=120`, `{0:16, 1:40, 2:64}` |
| `max_level=3`, `radii=(8.0, 6.0, 4.0)` | `ValueError: ... há 0 folhas contendo o corpo` | `Nsubd=381`, `{0:13, 1:24, 2:120, 3:224}` |

Ou seja: **os números de referência do pré-voo são preservados bit a bit**
(`Nsubd=120`, `{0:16, 1:40, 2:64}`, máx. 2 vizinhos/lado, máx. `|Δnível|=1`,
exatamente 1 folha contendo o corpo, `Skeleton.from_quadtree` aceita), e a
versão gradada ainda sustenta a invariante quando `max_level > 2` — onde a
regra do anel único cai no mesmo modo de falha que ela foi escrita para
corrigir, um anel mais adiante. A validação pós-balanço do plano foi mantida
como guarda alta e passou a ser exercitada por dois testes.

**Desvio 2 — janela do `test_refino_decresce_com_a_distancia_ao_corpo`.**
Também do item 1 do pré-voo: a janela `abs(cx−xc) < 2.0 and abs(cy−yc) < 2.0`
cai exatamente sobre o centro do anel de nível 1 (`Δ = 2.0` exato), e com o cap
sobra só a folha travada dentro dela — `max(perto)=0`, `0 > 0` falso. Ampliada
para `< 3.0` (um `H0 = 2.6667`): `max(perto)=2` sobre 21 folhas contra
`max(longe)=0` sobre 9. Medido nas duas janelas antes de escolher.

**Desvio 3 — um teste a mais que o plano previa (13, não 11).** Além dos 11
testes do brief, acrescentei
`test_folha_travada_nao_e_subdividida_pelo_balanco` (fixa `Nsubd=120` e a
distribuição de níveis, para que uma regressão no cap apareça como contagem
errada e não só como exceção) e
`test_cap_gradado_resiste_a_mais_de_um_anel_refinado` (cobre o Desvio 1b, que é
código sem cobertura no brief). O Step 5 do brief esperava `16 passed` no
arquivo e o Step 6 `42 passed` na suíte; os números reais são `18` e `46`,
porque o baseline herdado é 33 e não 21 (`pre-flight-resolutions.md` itens 6 e
7 mandam reportar o número real).

**Desvio 4 — nome do arquivo de relatório.** O brief (Step 7 e o `git add`) pede
`04-geometria-quadtree.md`; segui o brief.

## 8. Referências consultadas
- `~/mrcm/mrcm-firedrake/mrcm/quadtree.py:76-140` — `QuadTreeNode`: campos
  `x0,y0,x1,y1,level,children,neighbours`, propriedades `cx/cy/Hx/Hy` e
  `subdivide()` (ordem dos filhos SW, SE, NW, NE).
- `~/mrcm/mrcm-firedrake/mrcm/quadtree.py:171-220` — `_invalidate_cache`,
  `_collect_leaves(use_cache)` (ordem row-major por `(y0, x0)`), `leaves`
  (atribui `subdomain_id`), `Nsubd`.
- `~/mrcm/mrcm-firedrake/mrcm/quadtree.py:224-278` — `_build_neighbours`:
  adjacência só por lado compartilhado com sobreposição de intervalo estrita
  (`_intervals_overlap`, linha 475), não por canto. É o que define quais
  folhas o balanço enxerga como vizinhas da travada.
- `~/mrcm/mrcm-firedrake/mrcm/quadtree.py:282-326` — `from_uniform`: exige
  potências de 2 (`_is_power_of_two`, linhas 304-308), daí a raiz montada à mão.
- `~/mrcm/mrcm-firedrake/mrcm/quadtree.py:392-411` — `from_permeability`: o
  padrão de laço "subdivide em rounds → `_invalidate_cache` → `balance_2to1`"
  que `build_obstacle_tree` reproduz.
- `~/mrcm/mrcm-firedrake/mrcm/quadtree.py:415-441` — `balance_2to1`: subdivide
  o vizinho *mais grosso* quando `leaf.level > nbr.level + 1`. É esta linha
  que subdivide a folha travada na versão do plano.
- `~/mrcm/mrcm-firedrake/mrcm/topology.py:109-160` — `check_segment_alignment`:
  a razão de `nloc` ter de ser par.
- `mprm/decomposition.py:1-177` (estado da Tarefa 3) — convenções de lado e de
  malha MAC reaproveitadas no docstring do módulo.
- `.superpowers/sdd/2026-07-25-mprm-quadtree-von-karman/task-4-brief.md` —
  texto integral da tarefa; o código de `plan_quadtree_geometry` e o esqueleto
  de `build_obstacle_tree` vieram daí.
- `.superpowers/sdd/2026-07-25-mprm-quadtree-von-karman/pre-flight-resolutions.md`
  §1 — o bug da folha travada, os números de referência e o ajuste de janela;
  §"Itens menores" e §6/§7 — política de reportar a contagem real de testes.
- `docs/superpowers/plans/2026-07-25-mprm-quadtree-von-karman.md:46-121` — §B,
  formato obrigatório deste relatório.
- `docs/superpowers/reports/03-decomposition.md` — estado herdado.
- de Berg, Cheong, van Kreveld, Overmars, *Computational Geometry: Algorithms
  and Applications*, 3ª ed., §14.3 (quadtrees balanceados) — o argumento do
  buffer de gradação usado em `_cap_de_nivel`.
- Arbogast, Pencheva, Wheeler, Yotov, *A Multiscale Mortar Mixed Finite Element
  Method*, condição que relaciona o espaço de interface à malha fina — origem
  de `nloc_min ≥ max(degreeU, degreeP)`.
  `~/Dissertação/Refs/A MULTISCALE MORTAR MIXED FINITE ELEMENT METHOD.pdf`.

## 9. Estado para a próxima tarefa

**O que existe agora que antes não existia** (verbatim de
`mprm/decomposition.py`):

```python
def plan_quadtree_geometry(*, D, Ly, N_bx, N_by, max_level, nloc_min,
                           family, degreeU=1, degreeP=3):
    """-> dict com as chaves:
       family, D, Lx, Ly, H0, H (list[float], len max_level+1),
       nloc (list[int]), h, Nx, Ny, xc, yc, N_bx, N_by, max_level, body_level
    """

def build_obstacle_tree(geom: dict, radii, wake_length: float = 0.0):
    """-> mrcm.quadtree.QuadTree, já balanceada 2:1 e com neighbours prontos."""
```

**O que a próxima tarefa deve consumir:** a Tarefa 5 (`Skeleton` a partir da
árvore + `Decomposition.from_quadtree`) recebe a `QuadTree` devolvida por
`build_obstacle_tree` e o `dict` de `plan_quadtree_geometry`. `geom["nloc"][ℓ]`
é o número de células finas por lado de uma folha de nível ℓ — a fonte de
`nlocx/nlocy` da `Leaf`; `geom["h"]`, `geom["Nx"]`, `geom["Ny"]` são a malha MAC
global. Os offsets `i0/j0` de cada `Leaf` saem de `round(lf.x0/h)` e
`round(lf.y0/h)`, exatos porque a §3 garante `Lx/h` e `Ly/h` inteiros.

**Armadilhas descobertas:**
- `qt.leaves` (property) atribui `subdomain_id` na ordem row-major por
  `(y0, x0)`; `qt._collect_leaves(use_cache=False)` **não** atribui. Quem
  precisa de IDs estáveis tem de passar pela property.
- `balance_2to1` chama `_build_neighbours` no fim, então `lf.neighbours` está
  quente ao sair de `build_obstacle_tree`. Qualquer `subdivide()` posterior
  exige `_invalidate_cache()` manual — o mrcm não o faz por você
  (`quadtree.py:175-183`).
- `balance_2to1` não tem noção de folha travada. Qualquer código futuro que
  subdivida a árvore depois de `build_obstacle_tree` na família (a) tem de
  reaplicar o cap, ou o corpo deixa de caber numa folha só.
- Na família (a) o refino **não alcança a superfície do corpo**: o cap prende o
  anel imediato em nível 1, e com `radii` monotônicos os níveis mais finos
  ficam a `≥ H0/2` da travada. Isso não é um defeito da implementação — é
  precisamente o que a spec §4.3.1 diz ao introduzir a família (b), e o motivo
  de a família (b) existir. Com `max_level=3` ou `4` e os `radii` de referência
  `(6.0, 3.0, …)`, a árvore da família (a) fica idêntica à de `max_level=2`
  (`Nsubd=120`): aumentar `max_level` sozinho não refina mais nada perto do
  corpo.
- `radii` é indexado por **nível**, não por anel: `radii[ℓ]` é o raio que
  autoriza uma folha de nível ℓ a virar nível ℓ+1. Precisa ter pelo menos
  `max_level` entradas e ser estritamente decrescente.

**O que ainda NÃO existe:** nada consome estas duas funções ainda — não há
`Decomposition.from_quadtree`, nem `Skeleton` construído a partir da árvore
dentro de `mprm/`, nem uso de `wake_length` em qualquer configuração de
referência (o parâmetro está implementado e foi exercitado só por script:
`wake_length=2.0` na família (a) de referência dá `Nsubd=156`,
`{0:12, 1:48, 2:96}`, mas nenhum teste o cobre).
