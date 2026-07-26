# Relatório — Tarefa 3: `Decomposition` — a camada única de topologia (caminho uniforme)

## 1. Identificação
- **Tarefa:** 3 — `Decomposition` — a camada única de topologia (caminho uniforme)
- **Data:** 2026-07-26
- **Commit antes:** `fc3727f` (`git rev-parse --short HEAD` antes de começar, na branch
  `feature/quadtree-mprm`)
- **Commit depois:** `a43b511` (`git log --oneline -1` logo após o commit
  desta tarefa)
- **Arquivos criados:**
  - `mprm/decomposition.py`
  - `tests/test_decomposition.py`
  - `docs/superpowers/reports/03-decomposition.md` (este arquivo)
- **Arquivos modificados:**
  - `mprm/__init__.py` — linha 7 (nova linha na lista de submódulos do
    docstring do pacote) e linhas 36-37 (novo bloco de reexport
    `from mprm.decomposition import Decomposition, Leaf, GammaEdge`, inserido
    logo após o bloco do downscaler, conforme o Step 4 do brief).

## 2. Estado herdado
Herdado de `docs/superpowers/reports/02-prefatoracao.md`: baseline real da
suíte **28 passed** (21 da Tarefa 0/1 + 7 de `tests/test_prefactorization.py`
após o fix pós-revisão registrado na seção 10 daquele relatório). Nenhum
arquivo de `mprm/decomposition.py` existia. Assinaturas consumidas nesta
tarefa, copiadas verbatim do código real (não do brief, que já as citava
corretamente):

```python
# mrcm/topology.py (~/mrcm/mrcm-firedrake, não modificado)
class GammaIncidence(NamedTuple):
    side: int
    seg: int
    ell: int
    bbox: Optional[tuple]

def mark_sides(ix: int, iy: int, Ndomx: int, Ndomy: int):
    """Retorna (flag: bool[4], internal_sides: list[int])."""

class Skeleton:
    @classmethod
    def cartesian(cls, Ndomx: int, Ndomy: int) -> "Skeleton": ...
    @property
    def is_uniform(self) -> bool: ...
    def gamma_incidences(self, sub: int) -> "list[GammaIncidence]": ...
    # campos usados diretamente: .ieGt (Nedges,), .el2edge (Nsubd, 4)

# mrcm/config.py
class MRCMConfig:  # frozen dataclass
    def as_dict(self) -> dict: ...  # asdict(self)
    @property
    def Ndomx(self) -> int; Ndomy: int  # campos, não properties
```

## 3. Regressão ANTES da execução

Comando exato:

```bash
source ~/venv-firedrake/bin/activate
python -m pytest tests/ -q
```

| Suíte | Baseline esperado | Obtido antes desta tarefa | Igual? |
|---|---|---|---|
| `pytest tests/ -q` | 28 passed | `28 passed in 113.81s` | sim |

(Um aviso `WARNING! There are options you set that were not used! ... Option
left: name:-q` aparece ao final — ruído do PETSc/Firedrake sobre argumentos
de linha de comando, não relacionado a testes; já presente antes desta
tarefa e não afeta o resultado.)

## 4. O que foi feito

Transcrito o código do brief (Steps 1-4) quase verbatim — a inspeção real de
`mrcm/topology.py`, `mrcm/config.py`, `mprm/projector.py:72-160` e
`mrcm/solver.py:91-142` (a `_build_tasks` citada como "contrato a imitar")
confirmou que todas as assinaturas citadas no brief (`Skeleton.cartesian`,
`Skeleton.gamma_incidences`, `Skeleton.ieGt`, `Skeleton.el2edge`,
`mark_sides`, `MRCMConfig.as_dict`) existem exatamente como descrito, sem
nenhuma divergência que exigisse adaptação do código proposto.

Criada a classe `Decomposition` em `mprm/decomposition.py`, com:
- `Leaf` — dataclass congelada com geometria física (`x0,y0,x1,y1`),
  resolução local (`nlocx,nlocy`), offsets na malha MAC global (`i0,j0`) e
  `level` (sempre 0 no caminho uniforme). Propriedades `Hx`, `Hy` derivadas.
- `GammaEdge` — dataclass congelada representando uma aresta de Γ como par
  de incidências `(isub, side, seg)`, com `nfaces` e `bbox` do segmento.
- `Decomposition.uniform(Ndomx, Ndomy, nloc, Lx, Ly)` — fábrica que
  reproduz a aritmética `ix = isub % Ndomx, iy = isub // Ndomx` já usada em
  `mprm/projector.py:141` e `mrcm/solver.py:106`, calculando `h = Hx/nloc` e
  validando que `Hy/nloc` coincide (malha MAC quadrada, decisão D1 da spec).
- `segments_of(isub)` — itera `skeleton.gamma_incidences(isub)` e preenche
  `bbox` (via `_side_bbox`, o lado inteiro da folha) quando o mrcm devolve
  `bbox=None` (sempre o caso no caminho uniforme), unificando o formato que
  os consumidores futuros (downscaler, projetor) vão ler.
- `gamma_edges()` — agrupa as incidências por `ell` (índice global de Γ) e
  exige exatamente 2 por aresta, com `nfaces` idêntico dos dois lados —
  ambas as invariantes verificadas pelos testes.
- `task(cfg, isub, NunkU, problem)` — replica bit a bit a construção da
  tarefa de Fase 1 em `mprm/projector.py:140-147` para o caminho uniforme
  (chaves `cfg_dict, isub, NunkU, edge_l, problem`) e a de
  `mrcm/solver.py:114-136` para o caminho não-uniforme (chaves
  `cfg_dict, isub, NunkU, geom, incidences, is_external, cacheable, problem`)
  — este segundo ramo não é exercitado por nenhum teste desta tarefa (não
  há fábrica de árvore quadtree ainda; ela chega na Tarefa 5), mas está
  presente porque a Tarefa 6 vai precisar dele sem reabrir este arquivo.

Nenhuma linha de `mprm/projector.py` foi tocada — a tarefa é aditiva e
independente, exatamente como o plano exige (o projetor só passa a consumir
`Decomposition` na Tarefa 6).

Único ajuste feito por leitura real do código (não uma correção, uma
confirmação): o brief citava `mrcm/topology.py:78-...` para `mark_sides`
sem range final; a função vai de `topology.py:78` a `topology.py:93` — sem
impacto na implementação, só uma checagem de que a referência do brief
apontava para o lugar certo.

## 5. Testes criados nesta tarefa

### `test_uniform_geometria_das_folhas`
- **Arquivo:** `tests/test_decomposition.py:14`
- **O que valida:** para uma grade `2x2, nloc=8, Lx=Ly=1.0`, que `h`, `Nx`,
  `Ny`, a geometria física (`x0,y0,x1,y1`) e os offsets MAC (`i0,j0`) das
  folhas 0 (canto inferior esquerdo) e 3 (canto superior direito) batem com
  a aritmética esperada, e que todas as folhas têm `nlocx=nlocy=8`.
- **Resultado esperado (escrito ANTES de rodar):** `h == pytest.approx(0.5/8)
  == 0.0625`; `(Nx, Ny) == (16, 16)`; folha 0 em `(0,0)-(0.5,0.5)` com
  `(i0,j0)=(0,0)`; folha 3 com `x0=y0=0.5` e `(i0,j0)=(8,8)`.
- **Execução 1 (antes da implementação):**
  ```
  ModuleNotFoundError: No module named 'mprm.decomposition'
  ```
  (mensagem literal, `python -m pytest tests/test_decomposition.py -q`,
  erro de coleta do arquivo inteiro — os 5 testes falham juntos neste
  ponto.)
- **Execução 2 (após a implementação):** `PASSED`.
- **Valor numérico obtido:** `d.h = 0.0625` (== `0.5/8` exato, sem
  arredondamento de ponto flutuante detectável); `(d.Nx, d.Ny) = (16, 16)`;
  `lf0 = (0.0, 0.0, 0.5, 0.5)`, `(i0,j0)=(0,0)`; `lf3.x0=lf3.y0=0.5`,
  `(i0,j0)=(8,8)`.

### `test_uniform_offsets_batem_com_a_aritmetica_do_projetor`
- **Arquivo:** `tests/test_decomposition.py:28`
- **O que valida:** para uma grade assimétrica `3x2, nloc=6`, que
  `(lf.i0, lf.j0) == (ix*nloc, iy*nloc)` para todo `isub`, com
  `ix, iy = isub % Ndomx, isub // Ndomx` — a mesma decodificação usada em
  `mprm/projector.py:141` e `mrcm/solver.py:106`. Este é o teste que amarra
  a `Decomposition` à aritmética exata do projetor hoje (não uma
  aproximação), para que a Tarefa 6 possa trocar uma pela outra com
  confiança.
- **Resultado esperado (escrito ANTES de rodar):** igualdade exata (não
  `approx`) para os 6 subdomínios de uma grade `3x2`.
- **Execução 1 (antes da implementação):**
  ```
  ModuleNotFoundError: No module named 'mprm.decomposition'
  ```
- **Execução 2 (após a implementação):** `PASSED`.
- **Valor numérico obtido:** verificado para os 6 subdomínios
  (`isub=0..5`); por exemplo `isub=4` → `ix=1,iy=1` → `(i0,j0)=(6,6)`,
  igual a `(1*6, 1*6)`. Igualdade exata de inteiros, sem tolerância.

### `test_gamma_edges_contagem_e_pares`
- **Arquivo:** `tests/test_decomposition.py:37`
- **O que valida:** numa grade `3x2, nloc=6`, que `gamma_edges()` devolve
  exatamente 7 arestas (2×2 verticais internas + 3×1 horizontais), que cada
  aresta liga dois subdomínios distintos (`e.a[0] != e.b[0]`), que
  `nfaces == 6` (h uniforme ⇒ mesmo nº de faces dos dois lados), e que a
  contagem de arestas tocando cada subdomínio bate com o nº de lados
  internos que `mark_sides` reporta para aquele subdomínio.
- **Resultado esperado (escrito ANTES de rodar):** `len(edges) == 7`; para
  cada `isub in range(6)`, `n == len(internal)` (onde `internal` vem de
  `mark_sides(ix, iy, 3, 2)`).
- **Execução 1 (antes da implementação):**
  ```
  ModuleNotFoundError: No module named 'mprm.decomposition'
  ```
- **Execução 2 (após a implementação):** `PASSED`.
- **Valor numérico obtido:** `len(edges) = 7`; todas as 7 arestas com
  `nfaces = 6`; contagem por subdomínio confirmada igual a `len(internal)`
  para os 6 subdomínios (por exemplo, o subdomínio central `isub=1`
  — `ix=1,iy=0` numa grade 3×2 — tem 3 lados internos: `left`, `right`,
  `top`; `n=3` bateu com `len(internal)=3`).

### `test_task_uniforme_tem_as_chaves_que_o_mrcm_espera`
- **Arquivo:** `tests/test_decomposition.py:60`
- **O que valida:** que `Decomposition.task(cfg, isub=0, NunkU=10,
  problem="DUMMY")` devolve exatamente as chaves
  `{"cfg_dict", "isub", "NunkU", "edge_l", "problem"}` no caminho uniforme
  (o mesmo formato de `mprm/projector.py:145-147`), com os valores passados
  ecoados e `edge_l` cobrindo exatamente os lados internos de
  `mark_sides(0, 0, 2, 2)`.
- **Resultado esperado (escrito ANTES de rodar):** `set(t) ==
  {"cfg_dict","isub","NunkU","edge_l","problem"}`; `t["isub"]==0`,
  `t["NunkU"]==10`, `t["problem"]=="DUMMY"`; `set(t["edge_l"]) ==
  set(internal)`.
- **Execução 1 (antes da implementação):**
  ```
  ModuleNotFoundError: No module named 'mprm.decomposition'
  ```
- **Execução 2 (após a implementação):** `PASSED`.
- **Valor numérico obtido:** para `isub=0` numa grade `2x2`,
  `internal = [1, 2]` (right, top — já que `ix=iy=0` corta left/bottom);
  `set(t["edge_l"]) == {1, 2}`, igual ao esperado.

### `test_segments_of_uniforme_tem_um_segmento_por_lado_interno`
- **Arquivo:** `tests/test_decomposition.py:69`
- **O que valida:** que `segments_of(0)` devolve um segmento por lado
  interno (mesma contagem que `mark_sides`), com `seg == 0` (uniforme não
  tem nós pendurados), `nfaces == 8` (grade `2x2, nloc=8`: `Hx=Hy=0.5`,
  `h=0.5/8=0.0625`, lado inteiro `0.5/0.0625=8`), e `bbox is not None`
  mesmo no caminho uniforme (onde o mrcm devolve `bbox=None` — a
  `Decomposition` é quem preenche via `_side_bbox`).
- **Resultado esperado (escrito ANTES de rodar):** `len(segs) ==
  len(internal) == 2`; todo segmento com `seg=0`, `nfaces=8`, `bbox`
  preenchida.
- **Execução 1 (antes da implementação):**
  ```
  ModuleNotFoundError: No module named 'mprm.decomposition'
  ```
- **Execução 2 (após a implementação):** `PASSED`.
- **Valor numérico obtido:** `len(segs) = 2`; ambos com `seg=0`,
  `nfaces=8`; `bbox` preenchida (por exemplo, lado `right`:
  `bbox=(0.5, 0.0, 0.5, 0.5)`, o lado inteiro da folha 0).

## 6. Regressão DEPOIS da execução

Comando exato:

```bash
source ~/venv-firedrake/bin/activate
python -m pytest tests/test_decomposition.py -v   # foco
python -m pytest tests/ -q                         # suíte completa
```

Saída resumida:

```
tests/test_decomposition.py::test_uniform_geometria_das_folhas PASSED
tests/test_decomposition.py::test_uniform_offsets_batem_com_a_aritmetica_do_projetor PASSED
tests/test_decomposition.py::test_gamma_edges_contagem_e_pares PASSED
tests/test_decomposition.py::test_task_uniforme_tem_as_chaves_que_o_mrcm_espera PASSED
tests/test_decomposition.py::test_segments_of_uniforme_tem_um_segmento_por_lado_interno PASSED
5 passed in 3.38s
```

```
33 passed in 116.69s (0:01:56)
```

| Suíte | Antes | Depois | Igual? |
|---|---|---|---|
| `pytest tests/ -q` | 28 passed | 33 passed (28 + 5 novos) | sim — sem regressão, +5 testes |

Nenhum teste pré-existente mudou de status. Os 5 testes novos são
exatamente a diferença entre 28 e 33.

## 7. Desvios do plano

1. **Nome do arquivo de relatório:** o brief (Step 7) sugeria
   `docs/superpowers/reports/03-decomposition-uniforme.md`; a instrução
   explícita recebida para esta execução pediu
   `docs/superpowers/reports/03-decomposition.md`. Usado o nome instruído
   explicitamente — conteúdo e formato (§B do plano) idênticos ao que o
   brief pedia.
2. **Contagem de testes esperada no Step 8 do brief ("31 passed"):**
   calculada a partir do "21 passed" antigo de §C do plano (já superado
   pelas Tarefas 0-2, que fixaram o baseline real em 28). O valor real
   obtido é **33 passed** (28 + 5 novos), consistente com
   `docs/superpowers/reports/02-prefatoracao.md` seção 10.4 (`28 passed`
   registrado ali como estado final daquela tarefa).

Nenhum outro desvio: o código de `mprm/decomposition.py` e
`tests/test_decomposition.py` foi transcrito do brief sem alterações — a
inspeção de `mrcm/topology.py`, `mrcm/config.py` e `mprm/projector.py`
confirmou que todas as assinaturas usadas existem exatamente como o brief
descrevia.

## 8. Referências consultadas

- **Código interno:** `mrcm/topology.py:64-93` (~/mrcm/mrcm-firedrake, não
  modificado) — `GammaIncidence`, `mark_sides`.
- **Código interno:** `mrcm/topology.py:220-288` — `Skeleton.cartesian`:
  construção de `el2edge`, `edge2el`, `ieG`/`ieGt`, `NGamma` para a grade
  cartesiana.
- **Código interno:** `mrcm/topology.py:472-527` — `Skeleton.is_uniform`,
  `Skeleton.max_segments_per_side`, `Skeleton.gamma_incidences` (formato
  unificado uniforme/quadtree, `bbox=None` no caminho uniforme).
- **Código interno:** `mrcm/config.py:48-295` — `MRCMConfig` (frozen
  dataclass), campos `Ndomx`, `Ndomy`, `nlocx`, `nlocy`, `degreeU`,
  `degreeP`, `alpha`, `basis_kind`, `mixed_family`, `mixed_order`,
  `executor_kind`, e `as_dict() -> dict` (linha 294-295, `asdict(self)`).
- **Código interno:** `mprm/projector.py:72-160` — construção de
  `Skeleton.cartesian`, `DOFManager`, `my_subs` (linhas 78-80) e da tarefa
  de Fase 1 no caminho uniforme (linhas 140-149: `edge_l` via
  `mark_sides`+`skel.ieGt[skel.el2edge[isub,s]]`, dict `task` com as
  chaves `cfg_dict, isub, NunkU, edge_l, problem`).
- **Código interno:** `mrcm/solver.py:91-142` (`_build_tasks`) — formato
  canônico das tarefas nos dois modos (uniforme e quadtree/skeleton-based),
  usado como contrato para o ramo não-uniforme de `Decomposition.task`.
- **Documento interno:**
  `.superpowers/sdd/2026-07-25-mprm-quadtree-von-karman/task-3-brief.md`
  (íntegra) — código-fonte verbatim de teste e implementação (Steps 1-7).
- **Documento interno:**
  `docs/superpowers/plans/2026-07-25-mprm-quadtree-von-karman.md` §B
  (linhas 46-121, formato deste relatório), §C (linhas 125-156, baseline —
  superado pelo valor real de 21/24/28/33 registrado nas Tarefas 0-3), §D
  (linhas 160-169, estrutura de arquivos: `mprm/decomposition.py` marcado
  para as Tarefas 3, 4, 5).
- **Documento interno:** `docs/superpowers/reports/02-prefatoracao.md` —
  estado herdado, baseline real (28 passed).

## 9. Estado para a próxima tarefa

- **O que existe agora que antes não existia** (assinaturas verbatim,
  copiadas de `mprm/decomposition.py`):

  ```python
  SIDE_BOTTOM, SIDE_RIGHT, SIDE_TOP, SIDE_LEFT = 0, 1, 2, 3

  @dataclass(frozen=True)
  class Leaf:
      isub: int
      x0: float
      y0: float
      x1: float
      y1: float
      nlocx: int
      nlocy: int
      i0: int          # índice da primeira célula MAC em x
      j0: int          # índice da primeira célula MAC em y
      level: int

      @property
      def Hx(self) -> float: ...
      @property
      def Hy(self) -> float: ...

  @dataclass(frozen=True)
  class GammaEdge:
      ell: int
      a: tuple          # (isub, side, seg)
      b: tuple          # (isub, side, seg)
      nfaces: int
      bbox: tuple       # (x0, y0, x1, y1) do segmento

  class Decomposition:
      def __init__(self, skeleton, leaves, h, Lx, Ly): ...
      # atributos: .skeleton, .leaves (list[Leaf]), .h, .Lx, .Ly, .Nx, .Ny, .is_uniform

      @classmethod
      def uniform(cls, Ndomx: int, Ndomy: int, nloc: int,
                  Lx: float, Ly: float) -> "Decomposition": ...

      @property
      def is_uniform(self) -> bool: ...

      def leaf(self, isub: int) -> Leaf: ...

      def segments_of(self, isub: int):
          """-> list[tuple[side:int, seg:int, ell:int, bbox:tuple, nfaces:int]]"""

      def gamma_edges(self) -> list[GammaEdge]: ...

      def task(self, cfg, isub: int, NunkU: int, problem) -> dict:
          """Uniforme: {"cfg_dict","isub","NunkU","edge_l","problem"}.
          Não-uniforme: {"cfg_dict","isub","NunkU","geom","incidences",
                         "is_external","cacheable","problem"}."""
  ```

  Reexportado em `mprm/__init__.py:37` (`from mprm.decomposition import
  Decomposition, Leaf, GammaEdge`).

- **O que a próxima tarefa deve consumir:** a Tarefa 4 (segundo `docs/
  superpowers/plans/2026-07-25-mprm-quadtree-von-karman.md` §D) continua em
  `mprm/decomposition.py` — provavelmente `plan_quadtree_geometry` e
  `build_obstacle_tree`, ainda inexistentes. O quadtree real só entra na
  Tarefa 5; até lá, `Decomposition.uniform(...)` é a única fábrica.
  `gamma_edges()` é memoizado (`self._gamma`, construído sob demanda na
  primeira chamada) — qualquer teste futuro que mude a topologia depois de
  chamar `gamma_edges()` uma vez precisa estar ciente do cache (não há
  invalidação automática; a classe assume topologia imutável após a
  fábrica).

- **Armadilhas descobertas:**
  1. `Decomposition.uniform` valida `Hy/nloc == Hx/nloc` (célula MAC
     quadrada) e levanta `ValueError` se não bater — qualquer teste futuro
     com `Ndomx != Ndomy` precisa escolher `Lx/Ly` proporcionais a
     `Ndomx/Ndomy` (como os testes desta tarefa já fazem: `3x2` com
     `Lx=3.0, Ly=2.0`).
  2. `task()` no ramo uniforme ainda decodifica `ix, iy` a partir de
     `isub % cfg.Ndomx, isub // cfg.Ndomx` — ou seja, mesmo dentro da nova
     classe, o caminho uniforme continua fazendo a aritmética antiga
     internamente. Isso é intencional (o valor da tarefa é a *equivalência*
     comprovada, não eliminar a aritmética do caminho legado ainda) — só a
     Tarefa 6, ao trocar o projetor para consumir `Decomposition`
     diretamente, decide se essa decodificação interna sai também.
  3. `segments_of` e `gamma_edges` dependem de `self.h` para calcular
     `nfaces` a partir do `bbox` (`span / self.h`) — funciona porque a spec
     (decisão D1) fixa `h` uniforme global; uma decomposição com `h`
     variável por folha (fora do escopo desta tarefa e da spec atual)
     quebraria essa conta.

- **O que ainda NÃO existe:** nenhuma fábrica de quadtree
  (`Decomposition.from_quadtree` ou equivalente) — só `Decomposition.uniform`.
  `plan_quadtree_geometry`, `build_obstacle_tree` e `leaf_boundary_fluxes`
  citados em §D do plano para `mprm/decomposition.py` ainda não existem.
  `mprm/projector.py` não foi tocado e continua fazendo a aritmética
  `(ix, iy)` diretamente (a troca é a Tarefa 6). Nenhum teste desta tarefa
  exercitou o ramo não-uniforme de `Decomposition.task` (não há árvore
  quadtree para instanciar `Decomposition` fora do `uniform`, então esse
  ramo está implementado mas sem cobertura de teste direta ainda).
