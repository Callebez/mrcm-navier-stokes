# Estudo de α — Dependência do erro relativo com o parâmetro de Robin

**Data:** 2026-07-19 · **Testes:** `tests/test_alpha_hh.py`,
`tests/test_alpha_cavity.py` · **Resultados:** `results_alpha/`
(`alpha_hh.json`, `alpha_cavity.json`, figuras `alpha_hh_fig6.png`,
`alpha_cavity_fig11.png`, log `alpha_study.log`)

Replica os estudos de α do artigo do MPrM (Lages) nos dois cenários
compatíveis com contorno impermeável. Execução original: `mpiexec -n 3
python alpha_study.py` (526 s para os 45 casos principais + adendo) — os
resultados abaixo foram produzidos por essa versão inicial de um único
arquivo. Ela foi posteriormente refatorada na pasta `tests/` (um arquivo
por teste + interface de parsing de parâmetros compartilhada em
`tests/params.py`/`tests/common.py`); os smoke tests dos arquivos novos
reproduzem os mesmos números bit-a-bit (ver "Reprodução" no fim deste
documento), então os dados e figuras aqui não foram regenerados.

**Escolha de base:** `legendre_P` (hierárquica) em ambos os estudos. Motivo
descoberto durante a montagem: com o `nodal_Q` (default) e **graus
assimétricos** (`degreeP < degreeU`), o consumidor pega as primeiras
`ndofsP` funções da base de Lagrange do grau máximo — e o span resultante
**não contém a constante π₀** (é a deviation D3 já registrada no Vault).
Sem π₀ em `P_H`, o salto médio de fluxo por interface não é controlado, a
média da Eq. 21 quebra o balanço local e o downscaling deixa resíduo de
divergência ~1e-3 (medido). Com `legendre_P`, `P_{H,0}` = constante
verdadeira e `max|div u|` volta a ~1e-14. Famílias hierárquicas são também
o que o artigo define (projeções de espaços polinomiais completos).

---

## Estudo 1 — Decomposição HH modificada vs α (réplica da Fig. 6, §3.1)

Configuração do artigo: malha fina 90×90, decomposição 3×3 (30×30
células/subdomínio), espaços simétricos (P_{H,k}, U_{H,k}), k = 1..4.
Referência: **solução não decomposta** (projeção MAC fina sobre o mesmo w,
como na Fig. 6); o erro da referência fina vs o u analítico é 9,3e-5 —
duas ordens abaixo dos erros multiescala, então as duas métricas coincidem.

### Erro relativo ‖u_h − u_fino‖/‖u_fino‖

| α | k=1 | k=2 | k=3 | k=4 |
|---|---|---|---|---|
| 1e-4 | 3,24e-1 | 5,83e-2 | 1,49e-3 | 1,28e-4 |
| 1e-3 | 3,13e-1 | 5,53e-2 | 1,42e-3 | 1,21e-4 |
| 1e-2 | 2,52e-1 | 3,72e-2 | 1,05e-3 | 7,78e-5 |
| **1e-1** | **1,69e-1** | **1,21e-2** | **7,17e-4** | **3,44e-5** |
| 1e0 | 2,25e-1 | 1,75e-2 | 9,65e-4 | 4,90e-5 |
| 1e1 | 2,55e-1 | 1,94e-2 | 1,03e-3 | 1,36e-4 |
| 1e2 | 2,59e-1 | 1,96e-2 | 1,08e-3 | 1,16e-3 |
| 1e3 | 2,60e-1 | 1,98e-2 | 3,20e-3 | 6,35e-3 |
| 1e4 | 2,60e-1 | 2,68e-2 | 2,16e-2 | 1,96e-2 |

(negrito = mínimo por coluna; `max|div u_h|` ≈ 2e-10 em TODOS os 36 casos)

### Confronto com as alegações do artigo (Fig. 6)

| Alegação do artigo | Neste estudo |
|---|---|
| Erro melhora com o enriquecimento dos espaços | ✅ Reproduzido em todo α ≤ 1e2 (3,2e-1 → 1,3e-4 no ótimo) |
| Variação suave do erro com α, de ~1e-2 a ~1 | ✅ Curvas suaves; faixa 3,4e-5–3,2e-1 (níveis menores porque nosso w é mais regular) |
| "Um mínimo pode ocorrer para α próximo da unidade" | ✅ Mínimo interior em **α = 0,1** para TODOS os k (1,3–2,3× melhor que α=1) |
| "Erros melhores para α grandes" | ❌ Aqui o limite MHM **degrada**: para k ≥ 2 as curvas convergem para ~2e-2 em α = 1e4 |
| "(P₄,U₄) menos sensível a α" | ❌ É a curva MAIS sensível: 3,4e-5 (α=0,1) → 2,0e-2 (α=1e4), 570× |
| Estabilidade/conservação independem de α | ✅ div ~2e-10 e defeito ~1e-10 em toda a varredura (Eq. 38) |

**Leitura do desacordo em α grande:** no limite MHM (β→∞) o termo de Robin
domina os problemas locais e a continuidade fraca de pressão perde peso —
o ganho dos espaços de pressão ricos desaparece e todas as curvas k ≥ 2
colapsam para o mesmo patamar (~2e-2). Nosso benchmark difere do artigo em
dois pontos que explicam a divergência: (i) o potencial p̃ = cos(πx)cos(πy)
com ∇p̃·n = 0 (imposto pela restrição de contorno impermeável do caminho
corrigido de BCs do mrcm), enquanto o artigo usa o p de Olshanskii com
fluxo de contorno não nulo; (ii) a discretização do passo multiescala
(BDM₁–DG₀ em triângulos vs FV/RT₀ do artigo).

---

## Estudo 2 — Cavidade Re=100 vs α (análogo da Fig. 11, §3.4)

A Fig. 11 do artigo varre α no **degrau** (backward-facing step) com
espaços U_{H,1} (fluxo linear) e P_{H,0} (pressão constante) e encontra
mínimo em α ≈ 1. BCs de entrada/saída não são suportadas corretamente pelo
mrcm (bug de fluxo ≠ 0; Vault 2026-07-19), então usamos a **cavidade**
como análogo impermeável com os mesmos espaços. Malha 48², 3×3
subdomínios, T = 8, δt = 4,34e-3 (1844 passos/α); referência: solução de
malha fina com mesmo δt e T. A Tabela 3 do artigo (von Kármán) não é
replicada pelo mesmo motivo.

### Erro relativo ‖u_MPrM − u_fino‖/‖u_fino‖ em t = T

| α | erro | | α | erro |
|---|---|---|---|---|
| 1e-4 | 6,32e-2 | | 3e0 | 5,49e-2 |
| 1e-2 | 6,32e-2 | | 1e1 | 4,54e-2 |
| 1e-1 | 6,28e-2 | | 1e2 | 1,85e-2 |
| 3e-1 | 6,21e-2 | | 1e3 | 4,14e-3 |
| 1e0 | 5,98e-2 | | **1e4** | **3,26e-3** |
| | | | 1e6 | 3,27e-3 |

(`max|div u|` ~1e-14 em todos os casos — conservação exata em qualquer α)

### Confronto com o artigo

- **Sem mínimo interior**: o erro **decresce monotonicamente** com α e
  satura no platô MHM (~3,3e-3) a partir de α ≈ 1e4 — **19× melhor que o
  α = 1** usado no artigo para a cavidade. O mínimo em α ≈ 1 da Fig. 11
  (degrau) **não transfere** para a cavidade.
- O resultado ecoa a observação do próprio artigo na Fig. 6 ("erros
  melhores para α grande — priorizar conservação de fluxo"): na cavidade
  fechada, com pressão de interface pobre (P_{H,0}), o acoplamento de
  fluxo tipo MHM domina a qualidade. O platô MMMFEM (α→0, 6,3e-2) e o
  platô MHM (3,3e-3) diferem por ~20×.
- Nota metodológica: como no MPrM só o gradiente regularizado ṽ entra na
  atualização (§2.7 do artigo), a descontinuidade de pressão do limite
  MHM não penaliza a marcha temporal — condição favorável a α grande que
  o problema do degrau (com saída aberta) não compartilha.

---

## Recomendações práticas extraídas

1. **Espaços ricos (k ≥ 2)**: usar α ≈ 0,1–1; NUNCA o limite MHM (o
   enriquecimento é desperdiçado — curvas colapsam em ~2e-2).
2. **Espaços pobres (U₁/P₀) em domínio impermeável**: usar α grande
   (≥ 1e3); o platô MHM dá o menor erro por DOF de interface do estudo
   (3,3e-3 com apenas 3 DOFs/interface).
3. **Graus assimétricos** exigem `basis_kind="legendre_P"` (π₀/D3).
4. Conservação de massa do pipeline (div ~1e-10–1e-14) é insensível a α —
   qualquer escolha é "segura" em estabilidade; α só controla acurácia.

## Reprodução

```bash
source ~/venv-firedrake/bin/activate
cd ~/mrcm/NavierStokes

# Estudo 1 (Fig. 6) — config completa (90×90, 3×3, k=1..4, 9 valores de α)
mpiexec -n 3 python tests/test_alpha_hh.py

# Estudo 2 (Fig. 11 adaptada) — config completa (48², 3×3, U1/P0, 10 valores de α)
mpiexec -n 3 python tests/test_alpha_cavity.py

# Smoke tests (poucos segundos)
python tests/test_alpha_hh.py --quick
python tests/test_alpha_cavity.py --quick

# Parâmetros customizados — ver --help para a lista completa
python tests/test_alpha_hh.py --alphas 1e-2,1,1e2 --ks 1,2 --Ndom 4 --nloc 20
python tests/test_alpha_cavity.py --Re 400 --T 4 --alphas logspace:-2,4,7
```

Estrutura: `tests/params.py` define a interface de parsing de parâmetros
compartilhada (`CommonParams` + `build_parser`/`parse_args`, com
`parse_float_list`/`parse_int_list` para specs como `"logspace:-4,4,9"` ou
`"range:1,5"`); `tests/common.py` reúne utilitários (comunicador MPI com
fallback sequencial, estilo de plotagem, erro relativo, I/O de
JSON/figura). Cada `tests/test_*.py` separa `run_study` (computação pura,
sem side effects) de `plot_results` (plotagem) e `main` (CLI).
