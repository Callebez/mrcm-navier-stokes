# Parecer — Especificação MPrM + Quadtree + von Kármán

> **Data:** 2026-07-25
> **Referência:** `docs/superpowers/specs/2026-07-25-mprm-quadtree-von-karman-design.md`
> **Revisor:** análise baseada em leitura completa do código-fonte (`mprm/`,
>   `mprm_core.py`), da documentação (`docs/`), do `mrcm-firedrake` (incluindo
>   `RevisaoQuadtree.md` e `quadtree-design.md`), e da estrutura da dissertação
>   (`~/Dissertação`).

---

## Sumário executivo

A especificação está **muito bem estruturada**, cobre os pontos essenciais com
profundidade e demonstra domínio completo do problema — desde a teoria do MRCM
até as idiossincrasias do `mrcm-firedrake`. Os 6 achados da revisão do código
atual são precisos e bem diagnosticados. As 5 fases de implementação são
logicamente ordenadas e cada uma tem critérios de aceite mensuráveis.

Há, no entanto, **lacunas e riscos** que o documento atual não endereça
explicitamente. Este parecer os organiza em três categorias: (A) o que está
excelente e deve ser preservado, (B) omissões e pontos que merecem atenção antes
da implementação, e (C) riscos de integração com o `mrcm-firedrake` que a
especificação subestima.

---

## A. Pontos fortes (o que está excelente)

### A1. Diagnóstico dos 6 achados (§2)

A revisão do código atual é **cirúrgica e correta**. Destaque para:

- **Achado 1 (🔴):** Sistema global remontado a cada passo. É o gargalo
  dominante e a Fase 1 o ataca corretamente. A separação matriz/RHS é o padrão
  offline/online que o próprio artigo do MPrM prescreve (Algoritmo 1) mas que a
  implementação atual não segue.

- **Achado 5 (🔴):** $\beta$ do Robin inconsistente sob quadtree. Este é o tipo
  de bug que produz resultados **plausíveis mas errados** — exatamente o pior
  cenário. A Fase 3 o resolve corretamente (Robin por folha).

- **Achado 6:** O acoplamento `assert_solid_interior_to_subdomains` + quadtree.
  A decisão D4 (folha do obstáculo travada) é a solução correta e está
  formalizada nas restrições geométricas da §4.3.

### A2. Decisões de design (§3)

As 9 decisões (D1–D9) são bem justificadas e cobrem os eixos críticos:

- D1 ($h$ uniforme, só $H$ varia) é a decisão arquitetural central. Correta:
  isola o transporte MAC e o CFL do quadtree.
- D3 (refinamento geométrico estático) é realista: $K=\delta t$ constante torna
  inaplicável o refinamento por contraste de permeabilidade do mrcm.
- D7 (segmentos pendurados como arestas independentes) está alinhada com a
  teoria mortar e com o que o `Skeleton.from_quadtree` já implementa.
- D8 (hash do skeleton entre ranks) é uma linha de código que previne um
  desastre silencioso.

### A3. Restrições geométricas (§4.3)

As 5 restrições são necessárias e suficientes. A conclusão da restrição 3 —
"o refinamento não pode chegar à superfície do obstáculo" — é **o insight
central** do trabalho e está declarada com honestidade. A configuração de
referência ($320 \times 96$, $H_0=2.67D$, $n_{\text{loc}} \in \{8,16,32\}$)
é bem calibrada.

### A4. Plano de testes (§8)

Os 20 testes cobrem equivalência, corretude, geometria, paralelismo e
regressão. A ênfase em teste de equivalência ("árvore uniforme ≡ cartesiano")
como rede de segurança é a estratégia correta para uma refatoração desta
magnitude.

### A5. Referências (§12)

A bibliografia está completa e corretamente anotada. A separação em 5 eixos
teóricos (formulação, interfaces não conformes, árvores/adaptatividade,
elementos mistos, validação) facilita a redação da dissertação. O alerta sobre
a conversão corrompida do Olshanskii (2013) é um detalhe importante.

---

## B. Omissões e pontos que merecem atenção

### B1. O `mprm_projection_step` atual não está pronto para iteração por folhas

**Problema:** A função `mprm_projection_step` em `mprm/flow.py` hoje extrai
fluxos de interface por indexação cartesiana:

```python
fW = fluxes.get(("v", ix, iy), zeros_m)     # face vertical entre (ix,iy) e (ix-1,iy)
fE = fluxes.get(("v", ix + 1, iy), zeros_m)
fS = fluxes.get(("h", ix, iy), zeros_n)
fN = fluxes.get(("h", ix, iy + 1), zeros_n)
```

Sob quadtree, uma face de folha pode ser composta por **múltiplos segmentos**
com vizinhos diferentes (ex.: uma face de uma folha grossa faz interface com
duas folhas finas). A chave `("v", ix, iy)` não é mais suficiente — é preciso
iterar sobre `Decomposition.gamma` e compor os fluxos dos segmentos que
compõem cada lado da folha.

**Recomendação:** A Fase 3 deve detalhar como o `mprm_projection_step` reconstrói
os 4 fluxos de contorno (W, E, S, N) de uma folha a partir dos múltiplos
segmentos de $\Gamma$ que podem compor cada lado. Isso não é trivial e é o ponto
onde a interface entre `Decomposition` e `MACDownscaler` é mais sensível.

**Sugestão de API:**

```python
def leaf_boundary_fluxes(decomp, flux_avg, isub):
    """Constrói (fW, fE, fS, fN) para o downscaler a partir dos
    fluxos médios por segmento. Lados com múltiplos segmentos
    (nó pendurado) são concatenados."""
    leaf = decomp.leaf(isub)
    fW = np.zeros(leaf.nlocy)  # face vertical esquerda
    fE = np.zeros(leaf.nlocy)  # face vertical direita
    fS = np.zeros(leaf.nlocx)  # face horizontal inferior
    fN = np.zeros(leaf.nlocx)  # face horizontal superior
    for edge in decomp.gamma:
        for role, (isub_edge, side, seg) in [("a", edge.a), ("b", edge.b)]:
            if isub_edge != isub:
                continue
            flux = flux_avg[(isub, side, seg)]
            # posiciona o fluxo no offset correto dentro da face da folha
            offset = int((edge.y0 - leaf.y0) / decomp.h)  # ou x0 para faces horiz.
            if side == 0:   # left
                fW[offset:offset+len(flux)] = flux
            elif side == 1: # right
                fE[offset:offset+len(flux)] = flux
            elif side == 2: # bottom
                fS[offset:offset+len(flux)] = flux
            elif side == 3: # top
                fN[offset:offset+len(flux)] = flux
    return fW, fE, fS, fN
```

### B2. O downscaler com múltiplos tamanhos de patch não está validado

**Problema:** O `MACDownscaler` atual assume `nlocx, nlocy` fixos (do
`MRCMConfig`) e monta a matriz FV uma vez. A Fase 4 propõe cache por `(nlocx,
nlocy, solid_hash)`, o que é correto. Mas há uma sutileza: o **defeito de
compatibilidade** (`last_defect`) é hoje uma variável de instância do
`MACDownscaler`. Com múltiplos tamanhos de patch, o defeito deve ser reportado
**por folha** (ou pelo menos o máximo sobre as folhas), não agregado numa única
variável que é sobrescrita a cada `solve_patch`.

**Recomendação:** `MACDownscaler.solve_patch` deve retornar `(vt1, vt2, defect)`
como tupla, e o chamador (`mprm_projection_step`) agrega os defeitos.

### B3. A Fase 1 (pré-fatoração de $A_\Gamma$) precisa de um plano de fallback

**Problema:** A pré-fatoração LU esparsa (SciPy `splu`) de $A_\Gamma$ escala bem
até algumas dezenas de milhares de DOFs. Para $N_{\text{sub}} \approx 100$ com
`degreeU=degreeP=3` ($k_U=k_P=4$), temos $N_\Gamma \approx 400$ arestas $\times
8$ DOFs = 3200 DOFs — a LU esparsa é instantânea. Mas se o estudo escalar para
malhas maiores ou espaços mais ricos, o fill-in pode explodir.

**Recomendação:** Adicionar um fallback para solver iterativo (GMRES com
precondicionador ILU ou diagonal) quando `cfg.global_solver == "iterative"`.
Isso já está previsto no `mrcm` via `GlobalAssembler`; a Fase 1 pode
simplesmente delegar a esse caminho quando a fatoração direta for inviável.

### B4. O setup canônico (D6) introduz $L_x = 26.67D$ — qual o custo?

**Problema:** A geometria proposta multiplica a malha global por ~2.5× em área
comparado ao setup atual ($16 \times 8$ vs $26.67 \times 8$). Com $h = D/12$,
são $320 \times 96 = 30720$ células MAC — factível, mas o **offline do MRCM**
(precomputação de MsBF para ~100 folhas, cada uma com 8×8 a 32×32 células
triangulares) pode levar minutos.

**Recomendação:** Adicionar uma estimativa de tempo de offline na tabela de
configuração de referência e um plano de contingência (reduzir `nloc_min` para 6
se o offline exceder 30 min).

### B5. A especificação não detalha o formato de saída do estudo comparativo

**Problema:** A Fase 5 lista as métricas, mas não especifica o **formato** do
JSON de saída nem como os plots serão organizados. Isso é importante porque o
JSON é o artefato que entrará na dissertação (tabelas e figuras).

**Recomendação:** Adicionar um schema JSON explícito:

```json
{
  "geometry": {"Lx": 26.67, "Ly": 8.0, "D": 1.0, "h": 0.0833, "Nx": 320, "Ny": 96},
  "flow": {"Re": 100, "U_in": 1.0, "inflow": "parabolic", "walls": "slip"},
  "configs": [
    {
      "label": "cartesian_uniform",
      "N_sub": 30, "n_gamma": 87, "n_gamma_dofs": 696,
      "H_min": 2.67, "H_max": 2.67,
      "t_offline_s": 45.2, "t_online_per_step_s": 0.032,
      "Cd_mean": 1.34, "Cd_std": 0.02,
      "Cl_amp": 0.26, "St": 0.122,
      "div_max": 1.2e-14, "defect": 3.4e-15,
      "err_u_rel": 0.0
    }
  ]
}
```

### B6. Conexão com o `docs/11-plano-validacao-von-karman.md`

**Problema:** A especificação menciona o plano de validação (§12.6), mas não
explicita **como o estudo comparativo da Fase 5 se relaciona** com as fases de
validação daquele documento (Fases 1–4 do plano). Há risco de duplicação de
esforço: a "configuração 4" (referência fina) da Fase 5 é essencialmente a Fase
2 do plano de validação.

**Recomendação:** Adicionar uma seção "Relação com o plano de validação"
explicitando que a Fase 5 **absorve** as Fases 1–2 do plano, e que a varredura
de parâmetros (Fase 3 do plano: $\alpha \in \{10^{-6}, 1, 10^6\}$, graus $k=1,2,3$)
é **ortogonal** e pode ser feita depois, sobre a melhor decomposição identificada.

---

## C. Riscos de integração com o `mrcm-firedrake`

A `RevisaoQuadtree.md` do `mrcm-firedrake` documenta uma série de problemas no
quadtree. É **essencial** entender quais afetam o MPrM e quais são inofensivos.

### C1. Itens que NÃO afetam o MPrM (tranquilos)

| Item da RevisaoQuadtree | Por que não afeta |
|---|---|
| **1.2** `_estimate_contrast` subestima contraste | O MPrM não usa `QuadTree.from_contrast`; a árvore é construída por `build_obstacle_tree` com critério geométrico (D3) |
| **1.4** `CellRefinement` produz subdomínios não retangulares | Não usamos `CellRefinement` |
| **5.1** `from_uniform` ignora razão de aspecto | O MPrM não usa `QuadTree.from_uniform`; `Decomposition.uniform` constrói a grade cartesiana diretamente (R5 da §10) |
| **5.3** Erros por interpolação cross-mesh | O MPrM mede erros na malha MAC (sem interpolação); a referência fina usa `ChannelFinePoisson` (Poisson MAC global) |

### C2. Itens que AFETAM o MPrM (exigem atenção)

| Item da RevisaoQuadtree | Severidade | Ação necessária |
|---|---|---|
| **1.1** `basis_per_side` gera bases inconsistentes | 🟡 Baixa (código morto) | Verificar se `dgk_basis_coeffs` (caminho quente) usa `bbox` do segmento. A RevisaoQuadtree confirma que sim. Nenhuma ação. |
| **1.3** `k_grid` do `CellRefinement` não chega à discretização | 🟡 Baixa | Não usamos `CellRefinement`. Mas `Skeleton.sub_nloc` (que a Fase 2 popula) também nunca foi exercitado com valores heterogêneos. **Ação:** o teste T1 (Fase 2) deve verificar explicitamente que `build_subdomain_workspace` com `sub_nloc[i]` diferente produz a malha local com o número correto de células. |
| **2.1** Deadlock: quadtree + MPI + `use_fine_reference` | 🟢 Nenhuma (não usamos) | O MPrM não usa `use_fine_reference`; a referência fina é `ChannelFinePoisson` (SciPy, não Firedrake). |
| **2.2** `validate=True` + MPI + não uniforme sem guard | 🟡 Média | O MPrM não deve ativar `validate=True` em produção, mas se ativar durante depuração, o caminho MPI pode reportar validação falsamente aprovada. **Ação:** adicionar `validate=False` explícito na construção do `MRCMConfig` quando `use_quadtree=True`. |
| **2.3** Nenhuma verificação de consistência do skeleton entre ranks | 🔴 Alta | A decisão D8 cobre isso com hash. **Ação:** implementar exatamente como descrito na D8. |
| **2.4** `Skeleton.from_quadtree` não valida balanceamento 2:1 | 🟡 Média | O `build_obstacle_tree` chama `balance_2to1()` explicitamente (Fase 2). **Ação:** o teste T11 cobre isso. |
| **4.1** `balance_2to1` é $O(M \cdot N^2)$ | 🟢 Baixa | Com ≤ 150 folhas, custo < 1s. O risco R6 cobre isso. |
| **4.2** `leaves`/@property refaz a árvore a cada acesso | 🟡 Média | O MPrM acessa `leaves` uma vez na construção da `Decomposition`. Mas se `Decomposition.from_quadtree` chamar `qt.leaves` múltiplas vezes, pode haver custo oculto. **Ação:** armazenar `leaves` como atributo na `Decomposition`, não acessar a property repetidamente. |

### C3. O problema do `Skeleton.sub_nloc` merece um teste dedicado

O campo `Skeleton.sub_nloc` (shape `(N_sub,)`) é populado por
`CellRefinement.to_skeleton` mas **nunca foi exercitado com valores
heterogêneos** (item 1.3 da RevisaoQuadtree.md). A Fase 2 vai populá-lo com
$n_{\text{loc}}^i = H_i/h$, potencialmente diferente para cada folha.

`build_subdomain_workspace` (chamado no offline do projetor) lê `cfg.nlocx,
cfg.nlocy` — **não** `sub_nloc`. Isso significa que, **hoje**, mesmo que
`sub_nloc` seja populado corretamente, `build_subdomain_workspace` vai ignorá-lo
e usar `cfg.nlocx, cfg.nlocy` (constantes) para todas as folhas.

**Este é o risco mais sério de integração.** A Fase 2 precisa OU:

(a) Modificar `build_subdomain_workspace` no `mrcm-firedrake` para ler
    `sub_nloc[isub]` quando disponível, OU

(b) Passar `nlocx, nlocy` por folha via `cfg_dict` na `task`, e modificar
    `build_subdomain_workspace` para usar `task["nlocx"]` / `task["nlocy"]`.

A opção (b) é menos invasiva. A `task` já é um dict arbitrário; adicionar
`nlocx_i, nlocy_i` a ela e modificar `build_subdomain_workspace` para usá-los
(com fallback para `cfg.nlocx, cfg.nlocy`) é ~5 linhas de código.

**Recomendação:** Adicionar uma **Fase 0** (pré-requisito no `mrcm-firedrake`)
que implementa a opção (b) e adicionar um teste dedicado no `mrcm-firedrake`
antes de começar a Fase 2 do MPrM.

---

## D. Sugestões de melhoria da especificação

### D1. Adicionar uma Fase 0: pré-requisitos no `mrcm-firedrake`

Antes de começar a implementação no MPrM, são necessárias duas modificações
mínimas no `mrcm-firedrake`:

1. **`build_subdomain_workspace` aceitar `nlocx, nlocy` por folha** (ver §C3
   acima). Impacto: ~5 linhas em `local_solver.py`.
2. **Adicionar `assert` de balanceamento 2:1 no `Skeleton.from_quadtree`**
   (RevisaoQuadtree.md §2.4). Impacto: ~3 linhas em `topology.py`.

Ambas são backward-compatible (comportamento atual preservado quando os novos
campos não são fornecidos).

### D2. Inserir pseudocódigo da `mprm_projection_step` generalizada

A §6 (fluxo de dados) está boa, mas um pseudocódigo mais concreto da função
principal ajudaria a identificar edge cases antes da implementação:

```python
def mprm_projection_step(w1, w2, projector, downscaler, decomp):
    divw = mac_divergence(w1, w2, decomp.h, decomp.h)
    fluxes_by_segment = projector.project(divw)  # dict (isub, side, seg) -> array
    
    u, v = w1.copy(), w2.copy()
    my_patch = {}
    for isub in projector.my_subs:
        leaf = decomp.leaf(isub)
        # Reconstrói os 4 fluxos de contorno da folha a partir dos segmentos
        fW, fE, fS, fN = leaf_boundary_fluxes(decomp, fluxes_by_segment, isub)
        # Override com fluxos de saída (boundary externo)
        fW, fE, fS, fN = _apply_outflow_overrides(projector, isub, fW, fE, fS, fN)
        # Downscaling local
        vt1, vt2, defect = downscaler.solve_patch(
            leaf.nlocx, leaf.nlocy,
            divw[leaf.i0:leaf.i0+leaf.nlocx, leaf.j0:leaf.j0+leaf.nlocy],
            fW, fE, fS, fN,
            solid_loc=solid[leaf.i0:leaf.i0+leaf.nlocx, leaf.j0:leaf.j0+leaf.nlocy]
        )
        my_patch[isub] = (vt1, vt2, defect)
    
    # Allgather e atualização global (como antes, mas por offsets da folha)
    for d in comm.allgather(my_patch):
        for isub, (vt1, vt2, _) in d.items():
            leaf = decomp.leaf(isub)
            u[leaf.i0:leaf.i0+leaf.nlocx+1, leaf.j0:leaf.j0+leaf.nlocy] = ...
            v[leaf.i0:leaf.i0+leaf.nlocx, leaf.j0:leaf.j0+leaf.nlocy+1] = ...
    
    _apply_external_normals_after_step(u, v, projector.problem)
    return u, v
```

### D3. Adicionar uma tabela de tamanhos de sistema de interface

Para calibrar expectativas de performance, uma tabela como esta ajudaria:

| Configuração | $N_{\text{sub}}$ | $N_\Gamma$ | DOFs/aresta ($k_U{=}k_P{=}3$) | DOFs totais | Tamanho LU (estimado) |
|---|---|---|---|---|---|
| Baseline uniforme $10 \times 3$ | 30 | 87 | 8 | 696 | < 1 MB |
| Quadtree $n_\Gamma \approx 87$ | ~50 | ~87 | 8 | ~696 | < 1 MB |
| Quadtree $N_{\text{sub}} \approx 30$ | ~30 | ~50 | 8 | ~400 | < 1 MB |
| Uniforme $20 \times 6$ (escala seguinte) | 120 | 396 | 8 | 3168 | ~10 MB |

### D4. Explicitar a política de branches Git

A especificação não menciona onde o desenvolvimento será feito. Sugestão:

- Branch `feature/quadtree-mprm` no repositório `NavierStokes`
- A Fase 0 (pré-requisitos no `mrcm-firedrake`) em branch separada ou PR
  isolado no `mrcm-firedrake`
- Merges frequentes das fases para evitar divergência prolongada

---

## E. Conclusão e recomendação

A especificação está **pronta para implementação** com as seguintes ressalvas:

1. **Pré-requisito bloqueante:** A Fase 0 no `mrcm-firedrake` (ler `nloc` por
   folha no `build_subdomain_workspace`) deve ser concluída antes de começar a
   Fase 2. Sem isso, a Fase 2 produzirá malhas locais com número errado de
   células e todos os testes de corretude falharão.

2. **A Fase 3 é o ponto de maior complexidade.** A generalização da extração
   de traços e da média de fluxos para arestas multi-segmento exige mais
   detalhamento (ver §B1 e §D2 acima).

3. **A Fase 1 (pré-fatoração de $A_\Gamma$) pode ser implementada
   independentemente,** antes mesmo do quadtree, e já traria ganho de
   performance para os benchmarks atuais. Recomendo começar por ela.

4. **O resultado mais importante é declarado na §9.4:** se o quadtree não
   superar o uniforme a $n_\Gamma$ igual, **isso é um resultado publicável**.
   A limitação $H_{\text{obs}} = H_0 > D$ é uma restrição física real do
   método, não uma falha de implementação.

**Parecer final:** A especificação é sólida, os riscos são conhecidos e
mitigáveis, e o plano de execução é realista. Recomendo prosseguir com a
Fase 0 (pré-requisitos no mrcm) e a Fase 1 (pré-fatoração) em paralelo,
seguidas pelas Fases 2–5 em sequência.
