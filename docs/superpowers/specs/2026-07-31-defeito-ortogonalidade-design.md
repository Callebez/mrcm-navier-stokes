# Defeito de ortogonalidade da decomposição multiescala — Design Spec

> **Plano:** `docs/superpowers/plans/2026-07-31-decomposicao-por-subespaco.md` (Fase 4, Tasks 6–8)
> **Branch:** `feat/rtcf-quad-mprm`
> **Data:** 2026-07-31

Especificação da medida direta do desvio de ortogonalidade da decomposição de
Helmholtz–Hodge multiescala. Complementa o plano de decomposição por subespaço,
cujas Tasks 1–5 permanecem inalteradas.

---

## 1. Motivação: a quantidade que ninguém mediu

A Seção `sec:erro-decomposicao` do capítulo do MRCM
(`~/Dissertação/Textual/MRCM/MRCM.tex:635-649`) estabelece que, na decomposição
multiescala, divergência nula e ortogonalidade se **dissociam**: a regularização
recupera a primeira e não a segunda, e o erro se aloja inteiramente na parcela
irrotacional. A medida direta desse erro é a equação `eq:ErroOrtogonalidade`,

$$\int_\Omega \bm u_h\cdot\tilde{\bm v}_h \;=\; \int_\Gamma \big[\!\big[\,\psi_h\,(\bm u_h\cdot\hat{\bm n})\,\big]\!\big].$$

O capítulo de resultados (`~/Dissertação/Textual/Resultados/Resultados.tex:272`)
afirma que o benchmark de Olshanskii "permite medir **diretamente**" essa
quantidade. **Nenhuma execução até hoje a calculou.** O que existe é `err_u`, um
erro relativo agregado contra o campo analítico, e o que a Fase 1–2 do plano
acrescenta é o vazamento por subespaço — ambos proxies úteis, nenhum dos dois é
a quantidade que o texto promete.

Esta especificação fecha essa lacuna.

### 1.1 Por que o defeito de ortogonalidade é um instrumento melhor

1. **É auto-contido.** Não requer solução exata: é uma inconsistência interna da
   solução calculada, computável a partir dela mesma. `err_u` e `leak_ratio`
   precisam do campo analítico; o defeito de ortogonalidade não. Isso o torna
   aplicável a arranjos sem solução manufaturada — a cavidade, o obstáculo, a
   decomposição por *quadtree*.
2. **É adimensional depois de normalizado.** O cosseno
   $\langle\bm u_h,\tilde{\bm v}_h\rangle/(\|\bm u_h\|\|\tilde{\bm v}_h\|)$ é o
   cosseno do ângulo entre as duas parcelas. É imune à amplitude da entrada —
   que é exatamente a objeção que o plano levanta contra a varredura $m_{pr}$ da
   tese de Carvalho, e o defeito de ortogonalidade a satisfaz por construção, não
   por álgebra vazia.
3. **Tem duas leituras independentes.** Volume à esquerda, esqueleto à direita.
   Calcular ambas e confrontá-las verifica, no discreto, a própria derivação do
   capítulo.

### 1.2 Por que ele sobrevive à regularização

A regularização torna o fluxo normal contínuo através de $\Gamma$. Então
$\bm u_h\cdot\hat{\bm n}$ é unívoco no esqueleto e sai do salto:

$$\int_\Gamma (\bm u_h\cdot\hat{\bm n})\,[\![\psi_h]\!].$$

O que resta é o fluxo normal vezes o **salto de pressão**. A regularização
conserta o fluxo, não a pressão. É essa a forma que a implementação usa.

---

## 2. As duas leituras

### 2.1 Lado do volume

Direto e sem plumbing novo: `mprm_projection_step` devolve $\bm u_h$, e a parcela
irrotacional é $\tilde{\bm v}_h = \bm w - \bm u_h$ por construção
(`mprm/flow.py:76-77`).

```python
def hh_ortho_defect(uh1, uh2, vt1, vt2, h):
    """Lado do volume de eq:ErroOrtogonalidade.

    Devolve (ortho_abs, ortho_cos). O produto interno usa a mesma norma
    discreta ℓ² ponderada por h² de _l2_mac (Task 2).
    """
```

Quadratura: soma sobre todas as faces com peso uniforme $h^2$, idêntica à de
`_l2_mac` (Task 2). Nas faces externas o arranjo impermeável tem
$\bm u_h\cdot\hat{\bm n}=0$ e portanto $\tilde{\bm v}\cdot\hat{\bm n}=0$, de modo
que o peso de borda não altera o resultado. Registra-se a escolha porque ela
deixa de ser imaterial em arranjos com saída (`outflow`), fora do escopo deste
plano.

### 2.2 Lado do esqueleto

Com $\tilde{\bm v} = \delta t\,\nabla p$ (`mprm/downscaler.py:143-146`), a fórmula
de Green aplicada subdomínio a subdomínio dá

$$\int_\Omega \bm u_h\cdot\tilde{\bm v}_h
= \delta t\sum_i \int_{\partial\Omega_i}(\bm u_h\cdot\hat{\bm n}_i)\,p_i
= \delta t\int_\Gamma (\bm u_h\cdot\hat{\bm n})\,[\![p]\!],$$

pois o termo de volume se anula ($\nabla\cdot\bm u_h = 0$ por subdomínio) e
$\partial\Omega$ não contribui ($\bm u_h\cdot\hat{\bm n}=0$).

Em MAC, para uma interface vertical no índice global de face $I$, com
$\hat{\bm n}_i = +\hat x$ para o subdomínio à esquerda:

$$\delta t \; u_1[I,j]\,\big(p[I-1,j] - p[I,j]\big)\,h_y,$$

somado sobre $j$ e sobre $I \in \{n_{loc}, 2n_{loc},\dots,(N_{dom}-1)n_{loc}\}$.
Análogo para as interfaces horizontais. Custo $O(N\,N_{dom})$.

### 2.3 O problema da constante

O potencial do downscaler é pinado na célula 0 **de cada patch**
(`mprm/downscaler.py:53` e `:92`). Constantes arbitrárias por subdomínio
destruiriam $[\![p]\!]$. A âncora é `projector.last_pT[isub]`, a pressão do MRCM,
globalmente consistente a menos de **uma** constante única:

$$\psi_h^i \;=\; p^i - \overline{p^i} + \overline{\texttt{last\_pT}[i]}.$$

A constante global remanescente é inofensiva: o salto não a vê.

**Esta é a aproximação central e o principal caveat.** O potencial do downscaler
é pós-regularização; `last_pT` é pré-regularização. Eles diferem por mais do que
uma constante — diferem por um resíduo da própria regularização. Daí a ordem das
tasks: o lado do volume (Task 6) vem primeiro, é independente e **não tem
ambiguidade de constante nenhuma**. Se os dois lados concordarem, a ancoragem
está validada *a posteriori*; se discordarem, a Task 7 está medindo o erro da
ancoragem, e não o defeito. O teste de identidade é o árbitro, e o writeup
registra o número medido em qualquer dos dois casos.

### 2.4 Exposição do potencial

O potencial sai como atributo `MACDownscaler.last_potential`, um dict
`isub → p` populado em `mprm_projection_step`. **Não** como terceiro valor de
retorno de `solve_patch`: o plano de migração *quadtree* já altera essa
assinatura em outro worktree, e um atributo não colide com essa mudança.

---

## 3. Métricas

| Chave | Definição | Propriedade |
|---|---|---|
| `ortho_abs` | $\langle\bm u_h,\tilde{\bm v}_h\rangle$ | Com sinal; escala com a amplitude ao quadrado |
| `ortho_cos` | $\langle\bm u_h,\tilde{\bm v}_h\rangle/(\|\bm u_h\|\|\tilde{\bm v}_h\|)$ | **Adimensional, imune à amplitude** — métrica principal |
| `ortho_skel` | $\delta t\int_\Gamma(\bm u_h\cdot\hat{\bm n})[\![\psi_h]\!]$ | Deve igualar `ortho_abs` |
| `ortho_identity_rel` | $\lvert\texttt{abs}-\texttt{skel}\rvert/\max(\lvert\texttt{abs}\rvert,\lvert\texttt{skel}\rvert)$ | Veredito da identidade |

Todas entram em `hh_benchmark_mode` (Task 2) e, portanto, nos três modos
`mixed`/`gradient`/`solenoidal` sem tratamento especial. No modo `gradient`, em
que $\bm u_{ex}=0$ e `err_u` é `None`, `ortho_cos` continua definido — é mais um
argumento a favor dele.

As duas primeiras chaves entram na Task 6; `ortho_skel` e `ortho_identity_rel`
só na Task 7, por força da Constraint 9. Enquanto a Task 7 não existir, ambas
valem `None`, e o consumidor a jusante (`convergence.py`, Task 8) precisa
tolerar `None` — o mesmo tratamento que `err_u` já exige no modo `gradient`.

---

## 4. Hipóteses e critérios

| # | Hipótese | Critério | Consequência se refutada |
|---|---|---|---|
| **H1** | O defeito de ortogonalidade é da escala $H$, não da escala $h$ | Com $H$ fixo e $h\to 0$, `ortho_cos` satura num platô **não nulo** | Se cair com $h$, o erro é de discretização e não da decomposição — contradiz `sec:erro-decomposicao` |
| **H2** | A identidade volume ↔ esqueleto vale no discreto | `ortho_identity_rel` pequeno e decrescente com o refino | Mede o erro da ancoragem da constante (§2.3), não o defeito |
| **H3** | Enriquecer a interface reduz o defeito | `ortho_cos` cai com `degreeU`/`degreeP` | Enriquecimento não ataca a perda de ortogonalidade |
| **H4** | O $\alpha$ ótimo para o defeito não coincide com o de `err_u` | Comparar os mínimos das duas curvas na varredura da Task 4 | A escolha $\alpha=1$ é robusta às duas métricas |

H1 é a hipótese central e é falseável. H4 conversa diretamente com a pergunta da
Task 4 do plano: $\alpha$ controla o balanço entre continuidade de pressão
($\alpha\to0$) e de fluxo ($\alpha\to\infty$), e o defeito de ortogonalidade é,
após a regularização, exatamente uma integral do **salto de pressão**. A previsão
é que $\alpha$ pequeno o reduza — e que isso conflite com o que minimiza `err_u`.

**Nenhum dos vereditos é um requisito.** Hipótese refutada é achado a reportar,
com o número medido, e não critério a relaxar.

---

## 5. Arquivos

| Arquivo | Responsabilidade | Estado |
|---|---|---|
| `mprm/benchmarks/hh.py` | `hh_ortho_defect`, `hh_ortho_skeleton`; quatro chaves em `hh_benchmark_mode` | Modificar |
| `mprm/downscaler.py` | Atributo `last_potential` (só isso) | Modificar |
| `mprm/flow.py` | Popular `downscaler.last_potential` em `mprm_projection_step` | Modificar |
| `mprm/benchmarks/convergence.py` | Carregar `ortho_cos` para cada `case` | Modificar |
| `tests/test_ortho_defect.py` | **Novo** — E13, as duas leituras e a identidade | Criar |
| `tests/test_convergence_subspace.py` | Acrescentar `ortho_cos` às curvas e à varredura de $\alpha$ | Estender (criado na Task 3) |
| `docs/13-decomposicao-por-subespaco.md` | Seção do defeito de ortogonalidade | Estender (criado na Task 5) |

As Tasks 1–5 do plano não são tocadas. `hh_build_w` e `hh_benchmark_mac` mantêm
assinatura e resultado, como já exige a Constraint 7.

---

## 6. Restrições

Valem, sem exceção, as sete Global Constraints do plano de decomposição por
subespaço: disparo único sem laço temporal; `mprm/benchmarks/*` puro sem E/S;
`import mprm` não puxa Firedrake; convenções de `tests/params.py` e
`tests/common.py` com `--quick` obrigatória; envelope autodescritivo em todo
JSON; formas MAC `(N+1,N)`/`(N,N+1)`/`(N,N)`; retrocompatibilidade.

Duas restrições próprias:

8. **`last_potential` é atributo, não valor de retorno.** Ver §2.4. Qualquer
   implementação que altere a assinatura de `solve_patch` está fora de escopo.
9. **O lado do volume não pode depender do lado do esqueleto.** A Task 6 precisa
   passar nos seus testes sem que nada da Task 7 exista, porque é ela que arbitra
   a Task 7.

---

## 7. O que fica de fora

| Excluído | Razão |
|---|---|
| Contraste com/sem regularização | Testaria a metade "sem regularização" de `sec:erro-decomposicao`, mas exigiria reconstruir `projector.last_uT` (BDM em triângulos) nas faces MAC — caminho novo, e o erro de interpolação entraria no próprio instrumento de medida. **Registrar como limitação no writeup.** |
| RHS montado em Firedrake | Confrontaria traços de `last_pT` com $\bm u_h\cdot\hat{\bm n}$ interpolado do MAC; introduz erro entre duas discretizações no instrumento. §2.2 fica só em MAC |
| Varredura de amplitude $m_{pr}$ | Já excluída pelo plano, por linearidade do pipeline. `ortho_cos` é imune à amplitude por normalização, o que torna a exclusão ainda mais clara |
| Defeito de ortogonalidade em cavidade / von Kármán / *quadtree* | O instrumento é auto-contido e se aplicaria a todos eles (§1.1), mas este plano segue o caminho do Olshanskii, de disparo único. Trabalho seguinte |
