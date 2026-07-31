# Decomposição por subespaço: onde se aloja o erro

> **Plano:** `docs/superpowers/plans/2026-07-31-decomposicao-por-subespaco.md`
> **Spec:** `docs/superpowers/specs/2026-07-31-defeito-ortogonalidade-design.md`
> **Data:** 2026-07-31
> **Ambiente:** `source ~/venv-firedrake/bin/activate && cd ~/mrcm/NavierStokes`
> **Numeração:** o plano pedia `13-`, já ocupado por `13-plano-migracao-rtcf-quad.md`.

O capítulo do MRCM afirma, analiticamente, que na decomposição multiescala
divergência nula e ortogonalidade se **dissociam**: a regularização recupera a
primeira, a segunda permanece violada, e o erro se aloja inteiramente na parcela
irrotacional (`sec:erro-decomposicao`). Esta é uma afirmação **testável**, e o
benchmark anterior não a testava — ele media um único erro agregado sobre uma
entrada em que ambas as parcelas são não nulas.

Este estudo separa as parcelas e mede cada uma.

---

## 1. A construção

O campo de entrada se decompõe em duas parcelas e o pipeline é linear. Basta,
portanto, anular uma de cada vez:

| Modo | Entrada | Resposta exata | O que mede |
|---|---|---|---|
| **gradient** | $\bm w = \nabla\tilde p$ | $\bm u = \bm 0$ | Vazamento **solenoidal espúrio** |
| **solenoidal** | $\bm w = \bm u$ | $\bm u_h = \bm w$ | Vazamento **irrotacional espúrio** |
| **mixed** | $\bm w = \bm u + \nabla\tilde p$ | $\bm u$ analítico | Erro agregado (histórico) |

No modo `gradient`, **todo** valor não nulo de $\bm u_h$ é erro puro: a resposta
exata é identicamente zero, e não é preciso solução de referência. É a medida
mais limpa possível do defeito.

O potencial permanece $\tilde p = \cos(\pi x)\cos(\pi y)$, cuja derivada normal
se anula em $\partial(0,1)^2$ — condição do caminho impermeável do `mrcm`. Foi
verificado numericamente que as faces externas dos dois campos já valem
exatamente zero, de modo que a imposição de $\bm w\cdot\bhat n = 0$ é uma no-op
e não introduz inconsistência em nenhum dos três modos.

### 1.1 A métrica comparável

`err_u`, o erro relativo à parcela solenoidal exata, **não serve** para comparar
modos: no modo `gradient` seu denominador é zero. A métrica usada é

$$\texttt{leak\_ratio} = \frac{\|\bm u_h - \bm u_{ex}\|_2}{\|\bm w\|_2},$$

a fração da energia da **entrada** que aparece como erro. Está definida nos três
modos e não depende da amplitude.

No modo `mixed`, `err_vs_exact` vale `5.04e-01` enquanto `leak_ratio` vale
`4.43e-02` — 11× menor. A diferença é que $\|\bm w\| \gg \|\bm u_{ex}\|$: a
entrada é dominada pelo gradiente, e normalizar pela parcela solenoidal infla o
número. `leak_ratio` é a métrica honesta.

---

## 2. Por que a varredura de amplitude foi omitida

A tese de Carvalho verifica *pressure-robustness* (§3.4.6) multiplicando a
pressão exata por um parâmetro $m_{pr}$ e conferindo que o erro de velocidade
não cresce. **Essa varredura não foi implementada aqui, e a omissão é
deliberada.**

O pipeline `mprm_projection_step` é **linear** em $\bm w$: `mac_divergence` é um
operador linear; `MPrMProjector.project` resolve um sistema linear; a média de
fluxos e os problemas locais de Neumann do downscaling são lineares; e a
correção $\bm u = \bm w - \tilde{\bm v}$ é linear. Logo
$\mathcal{L}(m\,\bm w) = m\,\mathcal{L}(\bm w)$ exatamente, e a razão
$\|\bm u_h\|/\|\bm w\|$ independe de $m$ **por álgebra, não por virtude do
método**. Um teste construído sobre essa varredura seria satisfeito por qualquer
implementação, inclusive uma errada.

A não linearidade das equações de Navier–Stokes vive em `mac_transport`, que
não participa deste pipeline.

A linearidade está verificada por teste
(`tests/test_subspace_leakage.py::test_linearidade_do_pipeline`): projetar
$10^6\,\bm w$ devolve exatamente $10^6$ vezes o resultado de projetar $\bm w$.

**Se uma leitura futura quiser reintroduzir a varredura de $m_{pr}$, é este
parágrafo que ela precisa refutar primeiro.**

O que se preserva do teste de Carvalho é o ingrediente que de fato importa: a
entrada de **gradiente puro**, cuja resposta exata é nula. O que muda é o eixo
da varredura — em vez de amplitude, varrem-se $\alpha$, os graus dos espaços de
interface e a razão $H/h$.

---

## 3. Vereditos

Todos os números medidos em `mpiexec -n 4`, `nloc=16`, grau 1, $\alpha=1$,
salvo indicação em contrário.

| # | Hipótese | Veredito | Evidência |
|---|---|---|---|
| **E12a** | O erro se aloja na parcela irrotacional | **✓ confirmada** | `leak(gradient)/leak(solenoidal)` = **60×**, constante ao longo de todo o refino |
| **E12b** | O modo `mixed` reproduz o `gradient` | **✓ confirmada** | desvio relativo **0.4%** |
| **E12c** | O $\alpha$ ótimo diverge entre os modos | **✗ refutada como enunciada** | os argmins divergem (0.1 vs 1.0), mas o do modo `solenoidal` é ruído: aquela curva varia **0.28%** em seis décadas de $\alpha$ |

### 3.1 E12a — o erro é irrotacional

Curva $Hh$, $N_{dom} \in \{2,4,8,16\}$, `leak_ratio`:

| Modo | $H = 0.5$ | $H = 0.25$ | $H = 0.125$ | $H = 0.0625$ | Taxa |
|---|---|---|---|---|---|
| `gradient` | 4.4483e-02 | 1.1255e-02 | 2.8206e-03 | 7.0578e-04 | **1.993** |
| `solenoidal` | 7.4161e-04 | 1.8481e-04 | 4.6163e-05 | 1.1539e-05 | **2.002** |
| `mixed` | 4.4308e-02 | 1.1212e-02 | 2.8097e-03 | 7.0304e-04 | 1.993 |

O resultado é mais forte do que a afirmação do capítulo. As duas parcelas
convergem à **mesma taxa** $O(H^2)$, e a razão entre elas fica **constante em
~60×** ao longo de todo o refino (60.0 no nível grosso, 61.2 no fino).

Refinar $H$ **não muda o balanço**. A predominância irrotacional não é um
transiente que o refino corrige — é estrutural do método. Que é exatamente o que
se espera se a causa for a perda de ortogonalidade no esqueleto, e não erro de
discretização.

**Caveat.** O resíduo do modo `solenoidal` é **piso de amostragem, não defeito**:
o campo de Olshanskii amostrado nas faces MAC tem divergência discreta
$O(h)$ — medida em `1.33e-03` para $N=48$ — que o projetor legitimamente
remove. A razão de 60× é, portanto, um limite **inferior** para a assimetria
real entre as parcelas.

Corroboração independente: enriquecer a interface derruba o vazamento
irrotacional em 234× (grau 1 → grau 3, em $N_{dom}=3$, `nloc=16`:
`4.4333e-02` → `1.8945e-04`) e **não move** o solenoidal (`6.4451e-05` →
`6.4434e-05`, 0.03%). O enriquecimento ataca um dos dois lados, e é o lado que
a tese aponta.

### 3.2 E12b — o erro agregado é o erro irrotacional

`leak_ratio` do modo `mixed` coincide com o do `gradient` em todos os níveis,
com desvio relativo de 0.4%. Isso não é surpresa dada a linearidade — é a sua
consequência quantitativa: como o erro total é a soma das contribuições das duas
parcelas e a solenoidal é 60× menor, o total **é** o irrotacional.

A consequência prática: **qualquer melhoria no erro do método de projeção passa
por atacar a parcela irrotacional.** Trabalhar na solenoidal é otimizar 1.6% do
problema.

### 3.3 E12c — α atua só de um lado

Varredura $\alpha \in \{10^{-3}, 10^{-2}, 0.1, 1, 10, 10^2, 10^3\}$,
$N_{dom}=4$, `nloc=16`:

| Modo | Spread relativo | $\alpha^\ast$ | Leitura |
|---|---|---|---|
| `gradient` | **72.6%** | 0.1 | mínimo interior e genuíno |
| `solenoidal` | **0.28%** | 1.0 | curva plana — mínimo sem significado |
| `mixed` | 72.6% | 0.1 | idêntico a `gradient` |

A hipótese, como enunciada, está refutada: não há "dois ótimos que divergem",
porque um dos dois lados não tem ótimo. O que há é mais nítido — **$\alpha$ atua
exclusivamente sobre a parcela irrotacional**, e ali atua com força.

Isso é coerente com a teoria: $\alpha$ controla o balanço entre continuidade de
pressão ($\alpha\to 0$) e de fluxo ($\alpha\to\infty$), e o defeito de
ortogonalidade é, após a regularização, uma integral do **salto de pressão**.

Curva do modo `gradient`:

| $\alpha$ | 1e-3 | 1e-2 | 0.1 | 1 | 10 | 100 | 1e3 |
|---|---|---|---|---|---|---|---|
| `leak_ratio` | 1.6013e-02 | 1.3735e-02 | **9.2755e-03** | 1.1255e-02 | 1.2653e-02 | 1.2835e-02 | 1.2853e-02 |

O mínimo em $\alpha=0.1$ vale **17.6% de redução** sobre $\alpha=1$ — e, por
E12b, esse é o ganho no erro **total**. Isso corrobora, por outro caminho, o que
`docs/05-estudo-alpha.md` já observara empiricamente; a decomposição por
subespaço mostra **por que**: o ganho vem todo do lado irrotacional.

**Nota metodológica.** O veredito registra `spread` por modo, e não apenas o
`argmin`. Um argmin sobre curva plana não sustenta conclusão nenhuma. E o modo
`mixed` **não** conta como testemunha independente do critério, porque E12b
estabelece que ele reproduz o `gradient` — contá-lo seria dupla contagem da
mesma medição. O teste
`test_convergence_subspace.py::test_curva_plana_nao_vira_veredito` trava isso.

---

## 4. Comandos de execução

| Exp | Objetivo | Comando | Tempo |
|---|---|---|---|
| **E11a** | Campos por subespaço (unitário) | `python -m pytest tests/test_subspace_fields.py -v` | <1 s |
| **E11** | Vazamento entre subespaços | `python -m pytest tests/test_subspace_leakage.py -v` | ~20 s |
| **E12** | Curvas por subespaço | `mpiexec -n 4 python tests/test_convergence_subspace.py --modes gradient,solenoidal,mixed --Ndoms 2,4,8,16` | ~65 s |
| **E12c** | Varredura de $\alpha$ | `mpiexec -n 4 python tests/test_convergence_subspace.py --alphas 1e-3,1e-2,0.1,1,10,100,1e3 --modes gradient,solenoidal,mixed --Ndom 4 --nloc 16` | ~30 s |
| — | Smoke test | `python tests/test_convergence_subspace.py --quick` | ~7 s |

**Saídas:** `results_alpha/convergence_olshanskii/e12_subspace.{json,png}` e
`e12_subspace_alpha.{json,png}`. Todo JSON traz `hypothesis`, `criterion`,
`verdict`, `caveats` e `git_commit` — JSON sem cabeçalho é execução, não
veredito.

---

## 5. O que este estudo não mede

| Excluído | Razão |
|---|---|
| Varredura de amplitude $m_{pr}$ | Vazia por linearidade do pipeline (§2). Passaria por construção |
| Polinômio de pressão de Carvalho | $\nabla p\cdot\bhat n \ne 0$ nas paredes horizontais; incompatível com o caminho impermeável sem tratamento adicional de contorno |
| Marcha temporal, cavidade, von Kármán | Este estudo segue o caminho do Olshanskii, de disparo único |
| Newton/Picard para NS estacionário | Exigiria solver monolítico inexistente |

---

## 6. O defeito de ortogonalidade

As seções anteriores medem o erro por **subespaço**. Esta mede a quantidade que
`sec:erro-decomposicao` identifica como **a** sede do erro, e que
`Resultados.tex` promete medir *diretamente*:

$$\int_\Omega \bm u_h\cdot\tilde{\bm v}_h \;=\; \int_\Gamma \big[\!\big[\,\psi_h\,(\bm u_h\cdot\bhat n)\,\big]\!\big].$$

Nenhuma execução anterior havia calculado esse número.

### 6.1 Por que ele é melhor instrumento que `err_u` e `leak_ratio`

1. **É auto-contido.** Não requer solução exata: é uma inconsistência *interna*
   da solução calculada. `err_u` e `leak_ratio` precisam do campo analítico; este
   não. É o que o torna aplicável à cavidade, ao obstáculo e à decomposição por
   *quadtree*, que não têm solução manufaturada.
2. **É adimensional por normalização.** $\cos\theta = \langle\bm u_h,\tilde{\bm v}_h\rangle/(\|\bm u_h\|\|\tilde{\bm v}_h\|)$
   é o cosseno do ângulo entre as parcelas. Imune à amplitude — não por uma razão
   entre erro e referência, mas por construção.
3. **Tem duas leituras independentes.** Volume à esquerda, esqueleto à direita.
   Confrontá-las verifica, no discreto, a própria derivação do capítulo.

Com a regularização, $\bm u_h\cdot\bhat n$ é unívoco no esqueleto e sai do salto:
resta $\int_\Gamma (\bm u_h\cdot\bhat n)\,[\![\psi_h]\!]$ — o fluxo normal vezes o
**salto de pressão**. A regularização conserta o fluxo, não a pressão. É por isso
que o defeito sobrevive a ela.

#### Ressalva ao item 2: no modo `gradient` a métrica não é independente

Medido: a saída espúria é **ortogonal à entrada**,
$\langle\bm u_h,\bm w\rangle = 7.4\times10^{-15}$. Disso segue, por álgebra,
$\langle\bm u_h,\tilde{\bm v}\rangle = -\|\bm u_h\|^2$ e
$\|\tilde{\bm v}\|=\sqrt{\|\bm w\|^2+\|\bm u_h\|^2}$, logo

$$\cos\theta \;=\; -\,\frac{\texttt{leak\_ratio}}{\sqrt{1+\texttt{leak\_ratio}^2}}.$$

A identidade foi verificada a $10^{-13}$ de erro relativo. **No modo `gradient`,
portanto, o defeito de ortogonalidade não acrescenta informação a `leak_ratio`**
— são a mesma medição reescalada, e é por isso que as duas curvas se sobrepõem na
figura `e13_saturacao.png`.

O valor independente da métrica aparece (i) no modo `mixed`, em que a razão
$|\cos\theta|/\texttt{leak\_ratio}$ vale 0.244 e as curvas se separam; e (ii),
sobretudo, em arranjos **sem solução exata**, em que `leak_ratio` não existe. Este
é o uso que justifica o instrumento, e está fora do escopo deste estudo (§6.5).

#### Um achado ao lado: o operador é uma projeção, só não é a ortogonal

$\|P(P\bm w) - P\bm w\| / \|P\bm w\| = 2.9\times10^{-13}$. O operador multiescala
**é idempotente até a precisão da máquina**.

É a dissociação de `sec:erro-decomposicao` na sua forma mais nítida: o método
entrega uma projeção legítima — idempotente, de divergência nula — que
simplesmente não é a projeção *ortogonal*. Projeta, mas torto. As duas
propriedades que o Teorema HH garante em conjunto aparecem aqui separadas, uma
preservada e outra não, ao nível do operador e não apenas do campo.

### 6.2 A demonstração central: o esqueleto é a causa

$N_{dom}=3$, `nloc=16`, $\alpha=1$:

| Arranjo | `ortho_cos` |
|---|---|
| **Malha fina global** (1 subdomínio, **sem esqueleto**) | `+3.46e-13` |
| Multiescala, `gradient` | `−1.995e-02` |
| Multiescala, `mixed` | `−4.86e-03` |
| Multiescala, `solenoidal` | `−2.53e-03` |

Na malha fina, a decomposição de Helmholtz–Hodge discreta é ortogonal **até a
precisão da máquina**. Introduzir o esqueleto faz o defeito saltar **dez ordens
de grandeza**. Mesma malha, mesmo campo, mesma quadratura — muda apenas a
presença do esqueleto.

Isso é mais específico que "o erro se aloja na parcela irrotacional": é a
atribuição da perda de ortogonalidade à **decomposição de domínio**, e não à
discretização. O sinal é consistentemente negativo nos três modos, o que descarta
ruído numérico.

### 6.3 Vereditos

| # | Hipótese | Veredito | Evidência |
|---|---|---|---|
| **H1** | O defeito é da escala $H$, não da escala $h$ | **✓ confirmada** | taxa vs $h$ = **0.0077** (platô), taxa vs $H$ = **1.993** |
| **H2** | A identidade volume ↔ esqueleto vale no discreto | **✓ confirmada** | resíduo cai $O(h^{0.96})$: `3.73e-01` → `9.83e-02` |
| **H3** | Enriquecer a interface reduz o defeito | **✓ confirmada** | fator **234×** do grau 1 ao 3 |
| **H4** | O $\alpha$ ótimo difere do de `err_u` | ver §3.3 (E12c) | $\alpha$ atua só sobre a parcela irrotacional |

#### H1 — o defeito não vê $h$

$N_{dom}=3$ fixo, `nloc` $\in \{8,16,24,32\}$:

| Modo | taxa de `ortho_cos` vs $h$ | `ortho_cos` no $h$ mais fino |
|---|---|---|
| `gradient` | **0.0077** | 1.988e-02 |
| `mixed` | 0.0156 | 4.831e-03 |

Refinar $h$ em 4× muda o defeito em **1%**. Ele é rigorosamente insensível a $h$.
Sob refino de $H$, por contraste, cai a $O(H^{1.993})$. O defeito é da escala $H$,
como `sec:erro-decomposicao` prevê — e a hipótese era falseável: se ele caísse com
$h$, o erro seria de discretização e o capítulo estaria errado.

**Figura:** `e13_saturacao.png`. Duas retas horizontais contra a referência
$O(h)$, que cai por fator 4 no mesmo intervalo. No modo `gradient` a curva de
`leak_ratio` corre exatamente sobre a de $|\cos\theta|$, pela identidade
algébrica do §6.1; no modo `mixed` elas se separam por fator 4, e é essa
separação que mostra o conteúdo próprio da métrica.

#### H2 — as duas leituras concordam

$N_{dom}=3$, modo `mixed`:

| `nloc` | volume | esqueleto | resíduo relativo |
|---|---|---|---|
| 8 | −2.216676e-03 | −1.389837e-03 | 3.73e-01 |
| 16 | −2.183904e-03 | −1.759823e-03 | 1.94e-01 |
| 24 | −2.172949e-03 | −1.889063e-03 | 1.31e-01 |
| 32 | −2.168622e-03 | −1.955411e-03 | 9.83e-02 |

O resíduo cai por fator 3.79 sob refino 4×, isto é $O(h^{0.96})$: a identidade
vale no limite. Com `nloc` fixo e $N_{dom}$ crescendo ele fica parado em ~0.21 —
é governado por $h$, não por $H$.

**Isso valida a ancoragem da constante *a posteriori*.** O potencial do downscaler
é pinado por patch; a constante de cada subdomínio é ancorada na média de
`projector.last_pT`, que é pós- versus pré-regularização e portanto difere por
mais que uma constante. Se esse fosse um erro estrutural, o resíduo não iria a
zero. Vai. A leitura pelo volume, que não depende de ancoragem nenhuma, é o
árbitro — e foi por isso que ela foi implementada primeiro e de forma
independente.

**Caveat.** A identidade fecha onde o defeito é real (`gradient` 0.196, `mixed`
0.229) e **falha no modo `solenoidal`** (1.87). Ali $\tilde{\bm v}\approx\bm 0$ e
`ortho_abs` vale `−3.2e−08`, nível de ruído: o resíduo relativo entre duas
quantidades desprezíveis não tem significado. O modo de referência para H2 é
`gradient`.

#### H3 — o enriquecimento ataca o defeito

$N_{dom}=3$, `nloc=16`, $\alpha=1$, `ortho_cos`:

| Grau | 1 | 2 | 3 | Fator |
|---|---|---|---|---|
| `gradient` | 1.995e-02 | 1.540e-03 | 8.528e-05 | **234×** |
| `mixed` | 4.864e-03 | 2.129e-05 | 1.138e-07 | 42752× |

Coerente com a teoria: espaço de interface mais rico → pressão mais contínua
através de $\Gamma$ → salto menor → defeito menor. O fator muito maior no modo
`mixed` sugere cancelamento parcial de sinal entre as contribuições; não se
extrai conclusão disso aqui.

**Figura:** `e13_graus.png`. Ambas as curvas são retas em escala semilog, isto é,
a redução é **geométrica no grau** — cada grau a mais divide o defeito por um
fator aproximadamente constante (≈15× no modo `gradient`).

### 6.4 A limitação declarada

**O contraste com e sem regularização não foi medido.**

`sec:erro-decomposicao` afirma que a regularização recupera a divergência nula e
**não** recupera a ortogonalidade. Este estudo mede apenas o lado "com
regularização": confirma que o defeito existe, que vem do esqueleto e que
sobrevive à regularização. Não mede o arranjo sem regularização, e portanto **não
verifica a metade da afirmação que diz que a divergência seria violada sem ela**.

A razão da exclusão é metodológica, não de esforço: construir o campo não
regularizado na malha MAC exigiria reconstruir `projector.last_uT` — BDM₁ em
triângulos — nas faces MAC, e o erro dessa interpolação entraria no próprio
instrumento de medida. Fica como trabalho seguinte.

**Sem este parágrafo, uma leitura futura suporia que a afirmação foi verificada
inteira. Não foi.**

### 6.5 O que o instrumento habilita

Por ser auto-contido (§6.1), o defeito de ortogonalidade se aplica a arranjos
**sem solução manufaturada** — a cavidade de Ghia, o obstáculo em canal, e em
particular a decomposição adaptativa por *quadtree*, cujo capítulo de resultados
ainda não tem instrumento de verificação próprio. Uma decomposição adaptativa tem
mais esqueleto que a uniforme, e a previsão natural é que o defeito cresça com a
área de $\Gamma$. Fora do escopo deste estudo, que segue o caminho do Olshanskii,
de disparo único.

### 6.6 Comandos

| Exp | Objetivo | Comando | Tempo |
|---|---|---|---|
| **E13** | Defeito e identidade (unitário) | `python -m pytest tests/test_ortho_defect.py -v` | ~17 s |
| **H1** | Saturação em $h$ | `mpiexec -n 4 python tests/test_convergence_subspace.py --nlocs 8,16,24,32 --Ndom 3 --modes gradient,mixed` | ~12 s |
| **H3** | Graus de interface | `mpiexec -n 4 python tests/test_convergence_subspace.py --degrees 1,2,3 --Ndom 3 --nloc 16 --modes gradient,mixed` | ~9 s |

**Saídas:** `e13_saturacao.{json,png}` e `e13_graus.{json,png}` no mesmo
diretório, ao lado de `e12_subspace.{json,png}` e `e12_subspace_alpha.{json,png}`.

**Nota de implementação.** O potencial local sai por
`MACDownscaler.keep_potential` / `.last_potential`, um atributo *opt-in*, e não
como terceiro valor de retorno de `solve_patch` — cuja assinatura o plano de
migração RTCF/*quadtree* já altera em outro worktree. Com o default `False`, o
caminho quente (von Kármán, cavidade) não paga nem cópia nem comunicação. A
equivalência entre execução sequencial e `mpiexec -n 4` foi verificada em 12
dígitos significativos.
