# Plano de execução — migração do MPrM para RT₀ (RTCF1) em quadriláteros

> **Branch:** `feat/rtcf-quad-mprm` (criada em `~/mrcm/NavierStokes` **e**
> `~/mrcm/mrcm-firedrake` — a migração atravessa os dois repositórios).
> **Data:** 2026-07-31 · **Decisão:** opção B da D-4 do plano do Cap. 4 da dissertação
> (`~/Dissertação/drafts/Plano-Cap4-Implementacao.md`).

---

## 1. Por que migrar

O passo multiescala do MPrM resolve hoje BDM₁ × P₀ em triângulos, mas o seu **único
produto consumido** são os fluxos normais de interface, lidos por
`FunctionSpace(mesh, "HDiv Trace", 0)` (`mprm/projector.py:178`) e divididos pelo
comprimento da face — isto é, **projetados a uma constante por face fina**. A solução de
volume do BDM₁ é descartada; quem avança a simulação é o campo reconstruído pelo downscaler
MAC/FV (`mprm/flow.py:28-86`).

Consequência: os **2 dofs de fluxo normal por face** do BDM₁ são reduzidos a 1 antes de
chegar ao consumidor. A riqueza extra é paga e não é usada.

RTCF1 (o RT₀ de quadriláteros) tem **exatamente 1 dof por aresta** — precisamente o dado que
`HDiv Trace 0` carrega e que uma face MAC comporta. A migração elimina a perda.

**Ganhos adicionais:**

1. Restaura a discretização única de G2018/Lages: as três etapas do passo de tempo passam a
   viver no mesmo arranjo escalonado. Fecha a ponte `sec:PonteFV` do Cap. 2 da dissertação.
2. Remove a ressalva metodológica recorrente "BDM₁ triângulos × FV/RT₀ do artigo", hoje
   registrada em `docs/05-estudo-alpha.md` como uma das explicações para os desacordos com
   Lages. Os resultados passam a ser comparáveis ponto a ponto.
3. O mapa célula-Firedrake ↔ célula-MAC vira **bijeção** (hoje 2 triângulos por célula MAC).
4. **Abre a possibilidade de fechar o desvio π₀** (erratas E35/D1 da dissertação): com
   RTCF1, `F_h(E_h)` é o espaço das funções constantes por face fina, exatamente o alvo da
   projeção das Eqs. 63–64 de G2018. O teste `test_rt_prop2_equivalence`, hoje `xfail`
   estrito, pode passar a `xpass` — o que recuperaria a consistência exata da Prop. 2.
   *Avaliar como tarefa separada (T7), não bloqueante.*

---

## 2. Fatos verificados empiricamente (2026-07-31, `~/venv-firedrake`)

Tudo abaixo foi executado, não inferido.

| Verificação | Resultado |
|---|---|
| `RectangleMesh(..., quadrilateral=True)` | OK |
| `FunctionSpace(m, "RTCF", 1)` | OK — malha 2×2: `dim=12` = **1 dof por aresta** ✅ |
| `FunctionSpace(m, "RT", 1)` em quad | **FALHA** — `ValueError: Cellname 'quadrilateral' invalid for 'Raviart-Thomas'` → é obrigatório dispachar RT → RTCF |
| `FunctionSpace(m, "DG", 0)` em quad | OK (`dim=4`) — **os ~15 sítios com `"DG", 0` fixo sobrevivem sem alteração** |
| `FunctionSpace(m, "DG", k≥1)` em quad | OK — Firedrake **auto-converte para DQ** com `UserWarning`; `dim` e `dofs/cel` corretos (DG1→4/cel, DG2→9/cel). **`interface_space.py:279-282` sobrevive sem alteração** |
| `FunctionSpace(m, "HDiv Trace", 0)` em quad | OK (`dim=12` = 1 por aresta) — a maquinaria de traço de interface sobrevive |
| Marcadores de contorno `ds(1..4)` em quad | OK — comprimento 1,0 em cada lado; **mesma convenção** (1=esq, 2=dir, 3=inf, 4=sup) de `local_solver.py:469` |
| Centroides via `VectorFunctionSpace(m,"DG",0)` | 16 únicos em 16 células → **bijeção** com as células MAC |
| Solve misto RTCF1 × DG0 com termo de Robin em `ds(1)` | **OK**, resolvido com MUMPS; traço `HDiv Trace 0` montado corretamente |

**Leitura:** a migração é substancialmente menor do que a inspeção estática sugeria. Os
pontos de toque reais são poucos.

---

## 3. Tarefas de código

### T1 — `config.py`: declarar a forma da célula  *(mrcm-firedrake)*

Acrescentar `cell_shape: Literal["triangle", "quadrilateral"] = "triangle"` a `MRCMConfig`.
Padrão `"triangle"` preserva **todo** o comportamento atual da Parte I (campanha de Darcy,
SPE10, quadtree) — a migração é opt-in.

Validação na construção: `cell_shape="quadrilateral"` exige `mixed_family="RT"` (BDM não
existe em quadriláteros no Firedrake). Falhar explicitamente, não silenciosamente.

### T2 — `local_form.py`: malha e espaços  *(mrcm-firedrake)*

- `:68` — `RectangleMesh(nx, ny, Hx, Hy, diagonal="left", comm=comm)` →
  ramo `quadrilateral=True` (sem `diagonal`, que é mutuamente exclusivo).
  Manter o cache de topologia DMPlex; a chave passa a incluir `cell_shape`.
- `:79-93` `make_mixed_space` — dispachar por forma da célula:

  | forma | família | V | Q |
  |---|---|---|---|
  | triangle | BDM | `BDM_{order+1}` | `DG_order` |
  | triangle | RT | `RT_{order+1}` | `DG_order` |
  | **quadrilateral** | **RT** | **`RTCF_{order+1}`** | `DG_order` (auto→DQ) |

  Assinatura precisa receber a forma — passar `cell_shape` ou inferir de
  `mesh.ufl_cell().cellname()`. **Inferir da malha é preferível**: elimina a possibilidade
  de config e malha divergirem.

### T3 — `local_solver.py`: centros de célula  *(mrcm-firedrake)*

`:722-754` computa centros assumindo triângulos, com a correção de baricentro por orientação
da diagonal (1/3 vs 2/3). Em quadriláteros o centro é o ponto médio e a lógica **simplifica**;
já verificado que os centroides são únicos por célula. Ramificar por forma da célula e
manter o `assert` de layout de dofs.

### T4 — `fine_reference.py`: malha de referência  *(mrcm-firedrake)*

`:126` e `:728` constroem `RectangleMesh` para a solução fina de referência. Precisam da
mesma forma de célula da decomposição, senão a comparação MPrM × malha fina mistura
discretizações — o erro medido deixaria de ser só o de decomposição. **Gate de integridade:
a referência fina e os subdomínios têm de usar a mesma forma e família.**

### T5 — `spe10.py`: recuperação da grade  *(mrcm-firedrake, baixa prioridade)*

`:371-399` recupera a grade quadrilateral a partir das coordenadas de células triangulares.
Em quadriláteros vira identidade. Só relevante se o SPE10 for rodado em quads; **não
bloqueia o MPrM**, cujo coeficiente é δt homogêneo.

### T6 — `mprm/projector.py`: MPrM  *(NavierStokes)*

- Passar `cell_shape="quadrilateral"`, `mixed_family="RT"` na construção da `MRCMConfig`
  (`mprm/flow.py:152-180`, `:245-260`).
- `:163-174` `cell_map` — hoje mapeia 2 triângulos por célula MAC; com quads vira bijeção.
  Simplificar e **manter o `assert cfg.mixed_order == 0`**.
- **Aplicar a correção da D-1/E48 no mesmo lote** (ver §4): `:149`
  `robin.beta_for_subdomain(isub, mesh)` → passar `_K(mesh)`, com
  `robin_variant="discontinuous"`.

### T7 — π₀ (opcional, avaliar depois dos gates)

Com RTCF1, `F_h(E_h)` = constantes por face fina. Implementar a projeção das Eqs. 63–64 de
G2018 em `interface_space.py` passa a ser direto, e `test_rt_prop2_equivalence` (hoje `xfail`
estrito) deve virar `xpass`. **Não bloqueia a campanha**; se der certo, fecha as erratas
E35/D1 da dissertação e recupera consistência exata.

---

## 4. A correção de β (D-1 / E48) — no mesmo lote

Decidida em 2026-07-31: **manter β = αH/δt**, a convenção adimensional de G2018 Eq. 65 /
G2019 §4.1, dominante na família (Rocha2020 Eq. 12, Jaramillo2022 Eq. 3.1; Lages é a única
exceção).

**Defeito atual:** `mprm/projector.py:149` chama `robin.beta_for_subdomain(isub, mesh)` sem
`K_expr`, e `mrcm/robin.py:170-173` faz *fallback silencioso* para β = αL quando
`K_expr is None`. Com `robin_variant="constant"` (o padrão) o efeito é o mesmo: β = αH,
independente de δt.

**Correção:** `robin_variant="discontinuous"` + passar `_K(mesh)`. Com K = δt constante,
`robin.py:176-177` devolve exatamente αL/δt.

**Correção adicional recomendada:** transformar o *fallback* silencioso em erro explícito —
hoje qualquer variante dependente de K usada sem K degrada sem aviso.

**Efeito sobre os resultados:** α_novo = α_antigo · δt.
- HH/Olshanskii usa δt = 1 → **as convenções coincidem, os α antigos permanecem válidos**.
- Cavidade usa δt ≈ 4,34e-3 → fator ~230. O "platô ótimo em α = 10³–10⁴" vira α′ ≈ 4–43, e a
  conclusão registrada em `docs/05-estudo-alpha.md` ("o mínimo em α ≈ 1 do artigo não
  transfere") **precisa ser reavaliada**: sob a normalização correta o desacordo com a
  literatura encolhe de quatro ordens para menos de uma.

Como a migração já invalida a campanha, este é o momento certo para as duas mudanças.

---

## 5. Gates de verificação — **antes** de gerar qualquer resultado

Histórico do projeto exige rigor aqui: uma campanha anterior foi invalidada por dado
fabricado e por um bug de `gather`. Nenhum número entra em tabela sem passar por estes
gates, e cada gate registra o comando e a saída.

| Gate | O que exige |
|---|---|
| **G0 — suíte atual verde em `cell_shape="triangle"`** | a migração é opt-in; nada da Parte I pode regredir. `pytest` completo nos dois repositórios |
| **G1 — MMS em RTCF1 × DQ0** | verificação por solução manufaturada do par novo, com taxas observadas contra as esperadas da Tabela de convergência do Cap. 2. **Sem isso, nenhum resultado do MPrM tem lastro** |
| **G2 — equivalência entre executores** | sequencial × process pool × MPI devem reproduzir o mesmo `X` e os mesmos erros, dígito a dígito (teste já existente) |
| **G3 — conservação** | `max|div u|` ~1e-14 após downscaling, como no regime atual. É o indicador que detecta quebra do contrato de interface |
| **G4 — referência fina coerente** | referência fina e subdomínios na mesma forma/família (T4). Verificar explicitamente, não presumir |
| **G5 — β conferido numericamente** | imprimir β efetivo e checar contra αH/δt à mão, em um caso pequeno. É o gate que impede a D-1 de regredir silenciosamente |
| **G6 — regressão HH** | com δt = 1 e α equivalentes, o erro do benchmark HH em RTCF1 deve ficar na mesma ordem do de BDM₁ (não idêntico — é outra discretização). Divergência grande = bug, não resultado |

---

## 6. Geração de resultados

### R0 — O que cada estudo de fato valida (ler antes de rotular qualquer tabela)

Levantamento feito em 2026-07-31, verificado no código:

| Estudo | laço temporal | referência |
|---|---|---|
| HH/Olshanskii (R1a, R2, R3) | **NÃO** — `hh_benchmark_mac` (`mprm/benchmarks/hh.py:68-100`) chama `mprm_projection_step` **uma vez**; não há `for step` em `mprm/benchmarks/` | **analítica** (campo de Olshanskii, Eq. 40 de Lages) |
| HH sucessivas (**R2′, novo**) | **sim** (laço de projeções) | **analítica** |
| Cavidade (R1b) | sim | malha fina, mesmo esquema |
| Taylor–Green (**R6, novo**) | **sim** (Navier–Stokes completo) | **analítica** |
| Ghia (R4) | sim | dados numéricos de terceiros |

⚠️ **Consequência de rótulo.** R2 e R3 rodam sobre um benchmark **sem** evolução temporal:
medem a convergência do **operador de projeção**, não do MPrM como solver de Navier–Stokes.
Isso é intencional e correto — o Cap. 3 (`sec:verificacao-hh`) pede justamente isolar a
decomposição "*sem a interferência da evolução temporal*". Mas as tabelas do Cap. 5 **têm de
dizer isso**, sob pena de repetir o erro de rótulo da errata E20.

Sem R2′ e R6, **nada no pacote valida o laço temporal contra solução analítica** — o
cruzamento fica vazio.

---

Sete famílias. Para cada uma: script, saída JSON versionada, e figura.
**Toda tabela declara `cell_shape`, `mixed_family`, `mixed_order`, δt, α (na convenção
nova), (ℓ_U, ℓ_P) e `basis_kind`.**

### R1 — Dependência do erro em α

- **Base:** `tests/test_alpha_hh.py` (HH, δt = 1) e `tests/test_alpha_cavity.py` (cavidade).
- **Atenção:** com a correção de β, o eixo α da cavidade muda de escala. Reportar em α
  adimensional e declarar a conversão em relação à campanha antiga.
- **Faixa:** manter a varredura ampla o suficiente para exibir os dois platôs (MMMFEM em
  α→0, MHM em α→∞) e o mínimo interior, se houver.
- **Pergunta a responder:** o mínimo interior em α ≈ 0,1 do estudo HH sobrevive à mudança de
  discretização? E o desacordo da cavidade com Lages, ainda existe sob α′?

### R2 — Dependência do erro em (ℓ_P, ℓ_U)

- **Base:** `tests/test_convergence_hh_degrees.py` + a matriz de combinações.
- **Atenção:** graus assimétricos exigiam `basis_kind="legendre_P"` (a constante π₀ não
  pertence ao span da base nodal). **Reavaliar sob RTCF1** — se T7 (projeção π₀) for
  implementada, a restrição pode cair. Até lá, manter `legendre_P` e declarar.
- **Formato:** matriz (ℓ_U, ℓ_P), como a Tabela 1 de Lages, o que permite confronto direto.

### R2′ — Decomposições HH sucessivas (**NOVA — obrigatória**)

Réplica da Fig. 7 de Lages, que eu havia omitido. Verbatim
(`The_multiscale_projection_method...md:1128-1145`):

> *"successive applications of the modified HH decomposition over the solution in each
> iteration. More precisely, given u^k and p^k for some iteration k, starting with u⁰ = u and
> p⁰ = p given by (40), we compute the steps: 1. Compute w^{k+1} = u^k + ∇p^k; 2. Solve
> p^{k+1} from ∇²p^{k+1} = ∇·w^{k+1} using the multiscale method; 3. Compute
> u^{k+1} = w^{k+1} − ∇p^{k+1}; 4. Advance k ← k+1"*
>
> *"the successive HH decompositions **do not introduce any growing errors or artifacts that
> can be amplified**. We computed over hundreds of successive decompositions for different
> interface spaces, resulting in constant errors with no indication of attaining a different
> trend."*

**Por que importa:** é o único estudo que combina **laço** com **referência analítica** sem
sair do escopo da decomposição. Testa diretamente a preocupação levantada no Cap. 3
(`sec:regularizacao`): o erro da decomposição realimentado passo a passo poderia acumular. E
dá confronto direto com a Fig. 7 do artigo.

**Custo:** baixo — um laço em torno de `mprm_projection_step`, partindo de u⁰ analítico.
**Reportar:** erro contra o analítico × iteração, para várias combinações de espaço de
interface e α; e `max|div u|` ao longo do laço. Centenas de iterações, como no artigo.

### R3 — Dependência do erro no refinamento Hh

- **Base:** `tests/test_convergence_hh.py`, `test_convergence_H_mode.py`,
  `test_convergence_h_mode.py`, `test_convergence_p_mode.py`.
- ⚠️ **Alerta de integridade:** `hh_convergence_baseline.json` **não contém a família Hh** —
  números anteriores foram invalidados por dado fabricado e por bug de `gather`. **Gerar do
  zero, não reaproveitar baseline.**
- **Reportar ordem observada `p_obs`**, não só o gráfico (pedido M7 do índice de erratas da
  dissertação).

### R4 — Perfis de Ghia

- **Base:** `run_mprm.py:330` `_plot_cavity_profiles` — os 17 pontos das Tabelas I e II de
  Ghia (Re=100) já estão embutidos e foram **conferidos dígito a dígito** contra a fonte.
  A amostragem na malha MAC também está correta.
- **Bloqueante:** **não existe critério de estado estacionário** no código (`grep` por
  `steady|residual|converg` em `flow.py`/`run_mprm.py`: vazio). Os dados de Ghia são da
  solução convergida; roda-se hoje até T fixo (T=16). A Re=100 o tempo viscoso é L²/ν = 100.
  **Implementar critério** (p.ex. `max|uⁿ⁺¹−uⁿ|/δt < 1e-6`) e reportar o tempo atingido.
- **Resolução:** Ghia usa 129² uniforme; a campanha de α usou 48². Subir para ~96²–128².
- **Métrica quantitativa:** interpolar o perfil aos 17 pontos e reportar desvio máximo e RMS
  em tabela. Sem isso é ilustração, não verificação.
- **Comparação a três:** malha fina × Ghia (erro de **discretização**) e MPrM × malha fina
  (erro de **decomposição**). Ambos já computáveis no mesmo script.
- **Cuidados:** usar **Ndom ímpar** (com Ndom=3 as interfaces ficam em 1/3 e 2/3 e x=0,5 é
  interior; com Ndom par a linha de amostragem cai **sobre o esqueleto**); e declarar que a
  discrepância junto à tampa vem da singularidade dos cantos e da resolução (Ghia tabula
  u=1 em y=1; a malha MAC tem u em y=1−h/2).

### R6 — Taylor–Green com paredes de escorregamento (**NOVA — recomendada**)

O único candidato a validar o **laço temporal completo contra solução exata**. Com a função
de corrente ψ = sin(πx)sin(πy),

    u₀ = ( π sin(πx)cos(πy),  −π cos(πx)sin(πy) ),      u(t) = e^{−2π²νt} u₀

- **u·n = 0 nos quatro lados** do quadrado unitário ✅ (verificado à mão).
- **∂u_t/∂n = 0 nas paredes** ✅ — bate com a condição *free-slip* implementada:
  `SideBC(kind="slip")` (`mprm/problem.py:31`) e célula-fantasma = interior
  (`mprm/transport.py:51-57`). O caminho de contorno já existe e é o validado (impermeável).

**O que entrega, e nada mais entrega:** ordem de convergência **temporal**, e validação
conjunta de convecção + difusão + projeção + integração contra solução exata.

⚠️ **A verificar no papel antes de codar** (proposta, não fato estabelecido):
1. que o termo convectivo (u·∇)u é gradiente puro para este ψ — é o resultado clássico do
   Taylor–Green 2D, mas confirmar para esta normalização;
2. que a variante **não incremental** de Chorin (a adotada) não degrada a ordem observada a
   ponto de mascarar o teste — Guermond2006 documenta a limitação de ordem temporal dessa
   variante, e o teste precisa distinguir erro de projeção de erro de passo fracionado.

Se (1) falhar, o caminho alternativo é uma MMS instacionária com termo-fonte, mais flexível
e mais artificial.

### R5 — Figuras de convergência

Camada de plotagem consumindo os JSON de R1–R4. Um estilo único para todas.
Cada figura declara a configuração completa na legenda ou na nota.

---

## 7. Ordem de execução

1. T1 → T2 → T3 (núcleo da migração, `mrcm-firedrake`).
2. **G0** — suíte verde em `triangle`. Se falhar, parar: a migração vazou para a Parte I.
3. **G1** — MMS em RTCF1 × DQ0. Se falhar, parar: não há lastro para nada adiante.
4. T4 (referência fina) → **G4**.
5. T6 (MPrM + correção de β) → **G5**, **G2**, **G3**, **G6**.
6. R4 (critério de estado estacionário) e R2′ (laço de HH sucessivas) — **são código**, não
   só resultado. R6 exige antes a verificação no papel dos dois itens listados.
7. R1 → R2 → R2′ → R3 → R4 → R6 → R5.
8. T7 (π₀) se houver folga; T5 só se o SPE10 for para quads.

---

## 8. Reversão

A migração é **opt-in** por `cell_shape`, cujo padrão é `"triangle"`. Reverter é voltar à
branch anterior (`mprm-opt-parallel` no `NavierStokes`, `feature/spe10-ell-robin` no
`mrcm-firedrake`) ou simplesmente não passar `cell_shape="quadrilateral"`.

Os resultados antigos (`results_alpha/*.json` e afins) ficam **preservados** na branch de
origem e não devem ser sobrescritos: gerar os novos em diretório separado
(p.ex. `results_rtcf/`), para que a comparação BDM₁-triângulos × RTCF1-quads seja possível —
ela própria é um resultado interessante para a dissertação.

---

## 9. Ligações com a dissertação

- `~/Dissertação/drafts/Plano-Cap4-Implementacao.md` — D-1 (β), D-4 (esta migração), §3 (as
  três discretizações), T11 (Ghia).
- `~/Dissertação/Resumos-Obsidian/00 - Índice de erratas e ações de redação.md` — E48
  (β, decidida), E52 (discretizações), E35/D1 (π₀, que T7 pode fechar).
- Concluída a migração, a §4.9 do Cap. 4 deixa de descrever três discretizações e passa a
  descrever **uma**, com a `sec:PonteFV` do Cap. 2 fechando a ponte — o que simplifica o
  capítulo e elimina a E52.
