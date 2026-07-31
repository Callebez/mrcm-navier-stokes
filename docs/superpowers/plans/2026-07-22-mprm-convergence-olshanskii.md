# MPrM — Estudo de Convergência do Problema de Olshanskii — Plano

> **Data:** 2026-07-22
> **Branch alvo:** `mprm-convergence-olshanskii` (a partir de `mprm-opt-parallel` ou `main`)
> **Diagnósticos base:** `docs/06-convergencia-hh.md`, `docs/05-estudo-alpha.md`, `docs/09-dg0-projecao.md`
> **Resultados alvo:** `results_alpha/convergence_olshanskii.json`, `results_alpha/convergence_olshanskii_report.md`

**Goal:** Caracterizar sistematicamente a convergência do MPrM na decomposição
de Helmholtz-Hodge do campo analítico de Olshanskii — o único benchmark com
solução exata conhecida e compatível com a restrição de contorno impermeável do
MRCM. Mapear os **4 modos de convergência** (Hh, H, h, p) em função dos **4
parâmetros livres** (α, degreeU, degreeP, nloc), identificando regimes ótimos,
superconvergência, saturação e limitações assintóticas.

**Tech Stack:** Python 3.12, NumPy, SciPy, Firedrake (via `mrcm`), mpi4py, pytest 9, matplotlib.

**Global Constraints:**
- Ambiente: `source ~/venv-firedrake/bin/activate`; todos os comandos de `~/mrcm/NavierStokes`.
- `mixed_order=0` (BDM₁–DG₀) mantido — ver `docs/09-dg0-projecao.md`.
- `basis_kind="legendre_P"` obrigatório com graus assimétricos (preserva π₀ em P_H).
- Campo de teste: `olshanskii_u_np` + potencial `p̃ = cos(πx)cos(πy)` (∇p̃·n = 0 no contorno).
- Domínio: Ω = [0,1]² (unitário), decomposição Ndom×Ndom.
- Métricas: `err_vs_exact` (vs analítico), `err_vs_fine` (vs projeção MAC não decomposta na mesma N), `divmax`, `defect`.
- Extração de taxa: `mrcm.convergence.compute_convergence_rate` (regressão log-log).

---

## Motivação e estado atual

### O que já se sabe (1 ponto medido)

| Parâmetro | Valor medido | Referência |
|-----------|-------------|------------|
| Modo Hh, (P₁,U₁), α=1, nloc=16 | **O(H^1.99)**, R²=1.0000 | `docs/06-convergencia-hh.md` |
| Modo Hh, (P₂,U₁), α=0.5, nloc=16 | O(H^1.27), R²=0.95 (ruidoso) | mesmo doc, §Ressalvas |
| α-ótimo para erro absoluto (HH) | **α=0.1** (todos os k) | `docs/05-estudo-alpha.md` |
| α-ótimo para erro absoluto (cavidade) | α→∞ (MHM), sem mínimo interior | `docs/05-estudo-alpha.md` |

### O que NÃO se sabe (lacunas que este plano preenche)

1. **A taxa Hh depende de α?** O ponto medido (α=1, taxa≈2) é representativo ou o ótimo α=0.1 produz taxa diferente?
2. **A superconvergência O(H²) é robusta?** Vale para todos os graus simétricos? E para assimétricos?
3. **Existe um regime H (multiscala) com taxa O(H^{ℓ+2})?** O `mrcm` prevê essa taxa para o modo H (h fixo, H→0), mas nunca foi medida no pipeline MPrM (com downscaling).
4. **Qual é a taxa de saturação (modo h)?** Com H fixo e h→0, o erro para de cair — em que nível?
5. **Qual é a taxa de convergência-p (grau)?** Com malha fixa, aumentar degreeU/degreeP reduz o erro exponencialmente (campo C^∞)?
6. **O downscaling SEMPRE recupera O(H²)?** Ou há condições (α, graus, nloc) em que o piso O(H) do MRCM cru reaparece?
7. **Qual a interação nloc × taxa?** nloc=16 foi arbitrário — a taxa Hh é sensível a essa escolha?

---

## Os 4 modos de convergência — definições

Seja H = L/Ndom o tamanho do subdomínio (malha grossa) e h = H/nloc o tamanho
da célula MAC (malha fina). Os 4 modos são:

```
Modo Hh (FEM interior):  nloc fixo, Ndom cresce  → H e h encolhem juntos
                         Mede a taxa de refinamento clássico do pipeline completo
                         (MRCM + downscaling). Esperado: O(H²) com downscaling.

Modo H (multiescala):    h fixo (i.e., nloc ∝ 1/H), Ndom cresce → H→0
                         Mede a taxa multiescala O(H^{ℓ+2}) prevista pelo mrcm.
                         Esperado: O(H^{degreeU+2}) ou O(H^{degreeP+2}).

Modo h (saturação):      H fixo (Ndom fixo), nloc cresce → h→0
                         Mede o piso de erro da decomposição. Quando h ≪ H,
                         o erro satura no limite do espaço de interface.

Modo p (grau):           Ndom, nloc fixos; degreeU, degreeP crescentes.
                         Mede convergência espectral (exponencial para campo C^∞).
```

---

## Plano de experimentos — matriz 5D

### Dimensões do espaço de parâmetros

| Eixo | Símbolo | Valores baseline | Valores a varrer |
|------|---------|-----------------|-------------------|
| α (Robin) | α | 1.0 | {0.01, 0.1, 1.0, 10.0, 100.0, 1000.0} |
| Grau interface (vel) | degreeU | 1 | {1, 2, 3} |
| Grau interface (press) | degreeP | 1 | {1, 2, 3} |
| Razão H/h | nloc | 16 | {8, 16, 32} |
| Nº subdomínios | Ndom | {2,4,8,16} | {2,3,4,6,8,12,16} |

### Casos basais (6 experimentos, ~2–4 h cada)

Cada experimento fixa 3 eixos e varre 1, produzindo uma curva de convergência
com 4–6 pontos. Tempo estimado: ~16–24 h com 4 cores MPI.

```
E1 — Modo Hh × α           : nloc=16, (P₁,U₁), varre α ∈ {0.01,0.1,1,10,100,1000}
                              Ndom ∈ {2,4,8,16} → 6 curvas × 4 pts = 24 jobs
E2 — Modo Hh × graus sim   : nloc=16, α=1, varre (P_k,U_k), k∈{1,2,3}
                              Ndom ∈ {2,4,8,16} → 3 curvas × 4 pts = 12 jobs
E3 — Modo Hh × graus assim : nloc=16, α=1, varre (P_k,U_{k-1}), k∈{2,3}
                              + (P_{k-1},U_k), k∈{2,3}
                              Ndom ∈ {2,4,8,16} → 4 curvas × 4 pts = 16 jobs
E4 — Modo H (multiescala)  : h ≈ 1/256 fixo (malha fina de referência),
                              Ndom ∈ {2,3,4,6,8,12}, (P₁,U₁), α=1
                              nloc = round(1/(h·Ndom)) → ~{128,85,64,43,32,21}
                              → 1 curva × 6 pts = 6 jobs
E5 — Modo h (saturação)    : Ndom=4 fixo (H=0.25), nloc ∈ {2,4,8,16,32,64},
                              (P₁,U₁), α=1
                              → 1 curva × 6 pts = 6 jobs
E6 — Modo p (grau)         : Ndom=4, nloc=16, α=1,
                              varre (P_k,U_k), k∈{1,2,3,4}
                              → 1 curva × 4 pts = 4 jobs
```

### Casos de aprofundamento (condicionais aos resultados basais)

```
E7 — Modo Hh × nloc        : (P₁,U₁), α=1, varre nloc ∈ {8,16,32}
                              Ndom ∈ {2,4,8,16} → 3 curvas × 4 pts = 12 jobs
                              (Só se E1 mostrar sensibilidade a nloc)
E8 — Modo H × graus        : h fixo, varre (P_k,U_k), k∈{1,2}, α=1
                              → 2 curvas × 6 pts = 12 jobs
                              (Só se E4 confirmar taxa > O(H²))
E9 — Cruzamento α-ótimo    : Refinar o ótimo α≈0.1 com malha mais fina
                              Ndom=8, nloc=16, (P₂,U₂), α ∈ {0.05,0.08,0.1,0.12,0.15,0.2}
                              → 1 curva × 6 pts (sem varredura de Ndom)
```

---

## Objetivos detalhados e hipóteses

### Objetivo 1 — Dependência da taxa Hh com α

**Hipótese:** A superconvergência O(H²) observada em α=1 é robusta para uma
faixa de α, mas degrada nos extremos:
- **α → 0 (MMMFEM):** O acoplamento de interface enfraquece — o método tende
  a uma projeção local desacoplada. Erro absoluto sobe (confirmado no estudo α),
  mas a **taxa** pode se manter (o downscaler ainda é O(h²)).
- **α → ∞ (MHM):** A continuidade de pressão domina; o espaço de interface P_H
  perde efeito (curvas colapsam no estudo α). A taxa pode cair para O(H) se o
  downscaler não conseguir recuperar ordem sem a estrutura multiescala.

**Métrica:** Taxa log-log de `err_vs_fine` vs H, para cada α.

**Validação:** Para α=1, deve reproduzir O(H^1.99) medido anteriormente
(teste de regressão). Para α=0.1 (ótimo de erro absoluto), a taxa deve ser
≥ O(H^1.5) — se for menor, a superconvergência depende de α e o ótimo de erro
absoluto não coincide com o ótimo de taxa.

### Objetivo 2 — Taxa Hh para graus simétricos e assimétricos

**Hipótese:** A taxa Hh deve melhorar com o enriquecimento dos espaços de
interface:
- **(P₁,U₁):** O(H²) — já medido, baseline.
- **(P₂,U₂):** O(H³) ou O(H⁴) — o downscaler é O(h²) e pode limitar a taxa
  aparente; mas com nloc fixo, h ∝ H, então espera-se que a taxa do modo Hh
  reflita o grau polinomial dos espaços de interface.
- **(P₂,U₁) assimétrico:** O(H¹) – O(H²) — o espaço de velocidade limita a
  taxa; medição anterior deu 1.27 (ruidoso). Precisa de mais níveis (5–6)
  para confirmar.

**Métrica:** Taxa log-log para cada par (degreeU, degreeP).

**Validação:** Taxa para (P₁,U₁) deve ser ≥ 1.9 (regressão). Taxa para
(P₂,U₂) deve ser ≥ 2.5 (melhora detectável sobre o baseline). Taxa para
assimétricos deve ser dominada pelo menor grau.

### Objetivo 3 — Modo H: taxa multiescala O(H^{ℓ+2})

**Hipótese:** Com h fixo (malha fina de alta resolução), o erro deve cair com
H na taxa O(H^{degreeU+2}) ou O(H^{degreeP+2}), conforme previsto pela teoria
do MRCM para o modo H. O downscaling, operando sempre na mesma malha fina,
adiciona um erro constante ~O(h²) que satura quando H ≪ h — a taxa multiescala
só é observável na faixa intermediária onde H > h.

**Desenho experimental crítico:** h deve ser pequeno o suficiente para que o
piso de saturação fique abaixo do menor erro medido, mas não tão pequeno que
todos os pontos já estejam saturados. h ≈ 1/256 (N=256) com Ndom ∈ {2,...,12}
→ H ∈ {0.5, 0.33, 0.25, 0.17, 0.125, 0.083}. O nloc varia (nloc = N/Ndom).

**Métrica:** Taxa log-log de `err_vs_fine` vs H.

**Validação:** Para (P₁,U₁), espera-se taxa entre O(H²) e O(H³) na faixa
não-saturada. Se todos os pontos já estiverem no piso de saturação (erro
constante ~O(h²)), reduzir h para 1/512 e repetir.

### Objetivo 4 — Modo h: saturação do erro

**Hipótese:** Com H fixo (decomposição fixa) e h→0, o erro de decomposição
converge para um piso determinado pelo espaço de interface. A taxa de
aproximação ao piso deve ser O(h²) (dominada pelo downscaler).

**Métrica:** `err_vs_fine` vs h; identificar o joelho de saturação.

**Validação:** O erro deve cair monotonicamente e estabilizar em um patamar
< 1e-3 para (P₁,U₁) com H=0.25. O patamar deve ser menor para graus maiores
(confirma que o limite é o espaço de interface, não o downscaler).

### Objetivo 5 — Convergência-p (grau)

**Hipótese:** Para o campo C^∞ de Olshanskii, o erro deve cair exponencialmente
com o grau polinomial dos espaços de interface (convergência espectral).
O downscaler (O(h²)) impõe um piso ~(1/64)² ≈ 2.4e-4 para nloc=16, Ndom=4 —
abaixo disso, aumentar o grau não traz benefício.

**Métrica:** `err_vs_fine` vs k (grau).

**Validação:** Erro deve cair por fator ≥ 3× a cada incremento de grau até
atingir o piso do downscaler. O piso deve ser consistente com O(h²).

### Objetivo 6 — Confirmação da hipótese de superconvergência

**Hipótese central:** O downscaling (Poisson FV de 5 pontos, O(h²)) é o
mecanismo que recupera O(H²) do piso O(H) do MRCM cru. Esta hipótese prevê:
- A taxa Hh deve ser ≈ 2 para qualquer configuração onde o downscaling domina.
- A taxa Hh deve cair para ≈ 1 quando o downscaling é "desligado" (i.e., usando
  a reconstrução BDM₁–DG₀ direta do MRCM, sem o passo de downscaling).
- O modo h deve mostrar convergência O(h²) até saturar no piso do espaço de
  interface.

**Experimento crucial:** Comparar `err_vs_fine` COM e SEM downscaling em uma
curva Hh. Sem downscaling, usar o campo `u_h` direto do projetor MRCM
(interpolado para a malha MAC). Espera-se taxa ≈ 1 sem downscaling vs ≈ 2 com.

**Validação:** Este experimento é a prova definitiva de que o downscaling é
o mecanismo da superconvergência. Se a taxa sem downscaling também for ≈ 2,
a hipótese está errada e a superconvergência vem de outra fonte (ex.: o campo
C^∞ ou a projeção FineReference).

---

## Estrutura de arquivos (alvo)

```
mprm/benchmarks/
├── __init__.py
├── hh.py                    # (existente) olshanskii_u_np, hh_benchmark_mac, hh_build_w
└── convergence.py           # NOVO: funções de estudo de convergência

tests/
├── test_convergence_hh.py   # (existente) modo Hh, 1 ponto
├── test_convergence_h.py    # NOVO: modo H (multiescala, h fixo)
├── test_convergence_h_mode.py  # NOVO: modo h (saturação, H fixo)
├── test_convergence_p.py    # NOVO: modo p (grau)
└── test_convergence_sweep.py # NOVO: sweep automatizada (E1–E9)

results_alpha/
├── convergence_hh.json      # (existente)
├── convergence_olshanskii/  # NOVO: diretório para os novos resultados
│   ├── e1_hh_alpha.json
│   ├── e2_hh_degrees_sym.json
│   ├── e3_hh_degrees_asym.json
│   ├── e4_h_multiscale.json
│   ├── e5_h_saturation.json
│   ├── e6_p_degree.json
│   └── ...
└── convergence_olshanskii_report.md  # NOVO: sumário automático

docs/
└── 12-estudo-convergencia-olshanskii.md  # NOVO: writeup completo
```

---

## Tasks

### Task 0: Setup do branch e infraestrutura

**Files:**
- Git: criar branch `mprm-convergence-olshanskii`
- Create: `mprm/benchmarks/convergence.py`
- Create: `results_alpha/convergence_olshanskii/` (diretório)

- [ ] **Step 0.1: Criar branch**

```bash
cd ~/mrcm/NavierStokes && git checkout main && git checkout -b mprm-convergence-olshanskii
```

- [ ] **Step 0.2: Criar diretório de resultados**

```bash
mkdir -p results_alpha/convergence_olshanskii
```

- [ ] **Step 0.3: Criar `mprm/benchmarks/convergence.py`** — funções-core de estudo

Este módulo concentra a lógica comum a todos os experimentos: construção de
curvas de convergência, regressão, e comparação com/sem downscaling.
Reutiliza `hh_build_w`, `MPrMProjector`, `MACDownscaler`, `FineProjector`,
`compute_convergence_rate` dos módulos existentes.

APIs principais:

```python
def convergence_curve_hh(comm, Ndoms, nloc, degreeU, degreeP,
                          alpha, basis_kind="legendre_P",
                          include_raw_mrcm=False) -> dict:
    """Modo Hh: nloc fixo, Ndom varre. Retorna {meta, cases, rate_exact, rate_fine}."""

def convergence_curve_H(comm, Ndoms, h_fine_target, degreeU, degreeP,
                         alpha, basis_kind="legendre_P") -> dict:
    """Modo H: h ~ fixo (aproximado pelo nloc = round(1/(h_fine·Ndom)))."""

def convergence_curve_h(comm, Ndom, nlocs, degreeU, degreeP,
                         alpha, basis_kind="legendre_P") -> dict:
    """Modo h: H fixo (Ndom fixo), nloc varre."""

def convergence_curve_p(comm, Ndom, nloc, ks, alpha,
                         basis_kind="legendre_P") -> dict:
    """Modo p: graus k simétricos (P_k, U_k) crescentes."""
```

### Task 1: Experimento E1 — Modo Hh × α

**Files:**
- Create: `tests/test_convergence_hh_alpha.py`
- Output: `results_alpha/convergence_olshanskii/e1_hh_alpha.json`

**Hipótese:** Taxa ≈ 2 para α ∈ [0.01, 100]; possível degradação nos extremos.

**Config:**
- nloc = 16, (degreeU, degreeP) = (1, 1)
- α ∈ {0.01, 0.1, 1.0, 10.0, 100.0, 1000.0}
- Ndom ∈ {2, 4, 8, 16}
- Total: 6 × 4 = 24 jobs

**Tempo estimado:** ~3 h com 4 cores (≈7 min/job para Ndom=16).

- [ ] **Step 1.1: Criar script de teste**

```bash
# Estrutura similar a test_convergence_hh.py, mas com loop externo em α
python tests/test_convergence_hh_alpha.py --alphas 0.01,0.1,1,10,100,1000 --Ndoms 2,4,8,16
```

- [ ] **Step 1.2: Executar e coletar**

```bash
mpiexec -n 4 python tests/test_convergence_hh_alpha.py
```

- [ ] **Step 1.3: Analisar**

Gerar gráfico: 6 curvas log-log (err vs H), uma por α, com taxa anotada.
Identificar se α=0.1 (ótimo de erro absoluto) também é ótimo de taxa.

### Task 2: Experimento E2 — Modo Hh × graus simétricos

**Files:**
- Create: `tests/test_convergence_hh_degrees.py`
- Output: `results_alpha/convergence_olshanskii/e2_hh_degrees_sym.json`

**Hipótese:** Taxa melhora com k: O(H^{k+1}) ou O(H^{k+2}).

**Config:**
- nloc = 16, α = 1.0 (baseline) e α = 0.1 (ótimo de erro)
- (degreeU, degreeP) ∈ {(1,1), (2,2), (3,3)}
- Ndom ∈ {2, 4, 8, 16}
- Total: 2α × 3 graus × 4 Ndom = 24 jobs

**Tempo estimado:** ~4 h (graus maiores têm mais DOFs → offline mais caro).

- [ ] **Step 2.1: Criar script**

```bash
python tests/test_convergence_hh_degrees.py --ks 1,2,3 --alphas 1.0,0.1 --Ndoms 2,4,8,16
```

- [ ] **Step 2.2: Executar e analisar**

Atenção ao custo: (P₃,U₃) com Ndom=16 → ~48 DOFs de interface por faceta →
sistema de interface ~(48×4×16)×(48×4×16) ≈ 3000×3000. Ainda tratável, mas
pode precisar de `executor_kind="mpi"`.

### Task 3: Experimento E3 — Modo Hh × graus assimétricos

**Files:**
- Modificar: `tests/test_convergence_hh_degrees.py` (adicionar opção `--asymmetric`)
- Output: `results_alpha/convergence_olshanskii/e3_hh_degrees_asym.json`

**Hipótese:** Taxa dominada pelo menor grau. (P₂,U₁) → O(H¹)–O(H²);
(P₁,U₂) → O(H¹)–O(H²).

**Config:**
- nloc = 16, α = 1.0
- (degreeU, degreeP) ∈ {(2,1), (3,2), (1,2), (2,3)}
- Ndom ∈ {2, 4, 8, 16}
- Total: 4 pares × 4 Ndom = 16 jobs

**Validação:** (P₂,U₁) deve ter taxa similar ou inferior a (P₁,U₁).
A medição anterior (1.27, ruidoso) deve ser refinada com 5 níveis
(Ndom ∈ {2,3,4,6,8,12}) se necessário.

- [ ] **Step 3.1: Estender script E2 com flag `--asymmetric`**

- [ ] **Step 3.2: Executar e analisar**

### Task 4: Experimento E4 — Modo H (multiescala)

**Files:**
- Create: `tests/test_convergence_H_mode.py`
- Output: `results_alpha/convergence_olshanskii/e4_H_multiscale.json`

**Hipótese:** Taxa O(H^{degreeU+2}) na faixa não-saturada.

**Config:**
- h_target ≈ 1/256 (malha fina de referência ~256²)
- Ndom ∈ {2, 3, 4, 6, 8, 12} → H ∈ {0.5, 0.33, 0.25, 0.17, 0.125, 0.083}
- nloc = round(1/(h_target · Ndom)) → {128, 85, 64, 43, 32, 21}
- (degreeU, degreeP) = (1,1), α = 1.0
- Total: 6 jobs

**Cuidado:** nloc=128 com Ndom=2 → N=256 (a malha fina inteira). O offline
do MRCM para 2×2 subdomínios com 128×128 células cada é pesado (~16k triângulos
por subdomínio). Começar com Ndom ≥ 3 e subir.

- [ ] **Step 4.1: Criar script**

```bash
python tests/test_convergence_H_mode.py --Ndoms 3,4,6,8,12 --h-target 0.00390625
```

- [ ] **Step 4.2: Executar e analisar**

Identificar a faixa de H onde a taxa multiescala é observável (antes da
saturação em ~O(h²)).

### Task 5: Experimento E5 — Modo h (saturação)

**Files:**
- Create: `tests/test_convergence_h_mode.py`
- Output: `results_alpha/convergence_olshanskii/e5_h_saturation.json`

**Hipótese:** Erro cai O(h²) até saturar no piso do espaço de interface.

**Config:**
- Ndom = 4 (H = 0.25 fixo)
- nloc ∈ {2, 4, 8, 16, 32, 64} → N ∈ {8, 16, 32, 64, 128, 256}
- (degreeU, degreeP) = (1,1), α = 1.0
- Total: 6 jobs

- [ ] **Step 5.1: Criar script**

```bash
python tests/test_convergence_h_mode.py --Ndom 4 --nlocs 2,4,8,16,32,64
```

- [ ] **Step 5.2: Executar e analisar**

Determinar h de saturação: joelho onde `err_vs_fine` para de cair
significativamente.

### Task 6: Experimento E6 — Modo p (grau)

**Files:**
- Create: `tests/test_convergence_p_mode.py`
- Output: `results_alpha/convergence_olshanskii/e6_p_degree.json`

**Hipótese:** Convergência espectral (exponencial) até o piso O(h²) do
downscaler.

**Config:**
- Ndom = 4, nloc = 16 (N = 64, h = 1/64)
- (degreeU, degreeP) ∈ {(1,1), (2,2), (3,3), (4,4)}
- α = 1.0 e α = 0.1
- Total: 2α × 4 graus = 8 jobs

**Piso esperado:** ~(1/64)² ≈ 2.4e-4. Com (P₄,U₄) e α=0.1, o erro absoluto
já medido é 3.4e-5 (estudo α), que está ABAIXO do piso teórico — isso sugere
que o piso real é menor (possivelmente O(h³) ou O(h⁴) para este campo).

- [ ] **Step 6.1: Criar script**

```bash
python tests/test_convergence_p_mode.py --ks 1,2,3,4 --alphas 1.0,0.1
```

- [ ] **Step 6.2: Executar e analisar**

### Task 7: Experimento E7 (aprofundamento) — Modo Hh × nloc

**Condição:** Só executar se E1 mostrar dependência da taxa com α que sugira
interação com nloc.

**Files:**
- Modificar: `tests/test_convergence_hh.py` (adicionar `--nlocs`)
- Output: `results_alpha/convergence_olshanskii/e7_hh_nloc.json`

**Hipótese:** nloc maior → H/h maior → o downscaling tem "mais espaço" para
recuperar ordem → taxa pode ser mais próxima de 2. nloc menor → menos
células por subdomínio → o piso O(H) do MRCM pode reaparecer.

**Config:**
- nloc ∈ {8, 16, 32}, (P₁,U₁), α = 1.0
- Ndom ∈ {2, 4, 8, 16}
- Total: 3 × 4 = 12 jobs

### Task 8: Experimento E8 (aprofundamento) — Modo H × graus

**Condição:** Só executar se E4 confirmar taxa multiescala mensurável.

**Files:**
- Modificar: `tests/test_convergence_H_mode.py`
- Output: `results_alpha/convergence_olshanskii/e8_H_degrees.json`

### Task 9: Experimento E9 (aprofundamento) — Cruzamento do α-ótimo

**Condição:** Executar se E1 mostrar que α=0.1 produz taxa diferente de α=1.

**Files:**
- Criar: `tests/test_convergence_alpha_fine.py`
- Output: `results_alpha/convergence_olshanskii/e9_alpha_cross.json`

Refinar a busca pelo α que minimiza o erro de decomposição na malha mais fina
viável, para confirmar se o ótimo α=0.1 do estudo anterior se mantém.

### Task 10: Experimento crucial — com vs sem downscaling

**Files:**
- Modificar: `mprm/benchmarks/convergence.py` (adicionar `include_raw_mrcm=True`)
- Criar: `tests/test_convergence_downscaling_effect.py`
- Output: `results_alpha/convergence_olshanskii/e10_downscaling_on_off.json`

**Hipótese:** Com downscaling → O(H²); sem downscaling (reconstrução BDM₁–DG₀
crua) → O(H¹).

**Config:**
- nloc = 16, (P₁,U₁), α = 1.0
- Ndom ∈ {2, 4, 8, 16}
- 2 modos (com/sem downscaling) × 4 Ndom = 8 jobs

**Implementação:** Para o modo "sem downscaling", interpolar o campo `u_h` do
MRCM (BDM₁ nos triângulos) para as faces da malha MAC usando os DOFs de fluxo
nas arestas. Comparar o erro dessa interpolação com o erro do pipeline completo.

- [ ] **Step 10.1: Implementar reconstrução sem downscaling**

O `MPrMProjector.project()` já retorna `u_h` (campo BDM₁). É necessário
avaliá-lo nas faces da malha MAC. Usar `firedrake.Function.at()` ou
interpolação manual pelos DOFs de fluxo.

- [ ] **Step 10.2: Executar e analisar**

Este é o experimento que fecha o ciclo de entendimento: confirma ou refuta
a hipótese de que o downscaling é o mecanismo da superconvergência.

### Task 11: Sumário e writeup

**Files:**
- Create: `docs/12-estudo-convergencia-olshanskii.md`
- Create: `results_alpha/convergence_olshanskii_report.md` (automático)
- Create: script `scripts/generate_convergence_report.py`

Gerar automaticamente:
- Tabela-resumo com todos os experimentos: parâmetros, taxa, R², Njobs, wall time
- Gráficos: grid 2×3 com curvas de convergência principais
- Destaque dos achados: regimes de superconvergência, saturação, α-ótimo
- Recomendações práticas: que combinação de (α, graus, nloc) usar para
  diferentes objetivos (erro mínimo vs custo computacional)

---

## Cronograma e estimativas

| Task | Experimento | N jobs | Tempo estimado | Dependência |
|------|------------|--------|---------------|-------------|
| T0 | Infraestrutura | — | 30 min | — |
| T1 | E1 (Hh × α) | 24 | 3 h | T0 |
| T2 | E2 (Hh × graus sim) | 24 | 4 h | T0 |
| T3 | E3 (Hh × graus assim) | 16 | 2.5 h | T2 |
| T4 | E4 (H multiescala) | 6 | 2 h | T0 |
| T5 | E5 (h saturação) | 6 | 1 h | T0 |
| T6 | E6 (p grau) | 8 | 1 h | T0 |
| T10 | E10 (downscaling on/off) | 8 | 1.5 h | T0 |
| T7 | E7 (Hh × nloc) | 12 | 2 h | T1 (condicional) |
| T8 | E8 (H × graus) | 12 | 3 h | T4 (condicional) |
| T9 | E9 (α-ótimo) | 6 | 30 min | T1 (condicional) |
| T11 | Sumário | — | 1 h | T1–T10 |

**Total basal (T0–T6, T10):** ~13.5 h com 4 cores MPI.
**Total com aprofundamentos:** ~21 h.

---

## Estratégia de validação

### Sanity checks por experimento

1. **Regressão:** E1 com α=1 deve reproduzir `convergence_hh.json` existente
   (taxa 1.99, R²=1.0000). Bit-a-bit não é esperado (ordem de operações MPI),
   mas taxa deve diferir < 1%.
2. **Divergência:** `divmax` < 1e-10 em TODOS os casos (conservação de massa).
3. **Defeito:** `defect` < 1e-10 (downscaler convergiu).
4. **Monotonicidade:** `err_vs_fine` deve decrescer com Ndom crescente (Hh, H)
   e com nloc crescente (h).
5. **Consistência:** `err_vs_exact` ≥ `err_vs_fine` sempre (referência fina é
   um subconjunto do erro total).

### Validação cruzada com estudo α

Os experimentos E1 e E6 devem ser consistentes com `docs/05-estudo-alpha.md`:
- Para (P₁,U₁), Ndom=3, nloc=30, o erro vs fino em α=1 é ~0.225 (estudo α).
  Em E1, Ndom=4, nloc=16 → H=0.25 (similar), erro deve ser ~0.1–0.2.
- Para (P₄,U₄), α=0.1, Ndom=3, nloc=30, o erro vs fino é 3.4e-5 (estudo α).
  Em E6, Ndom=4, nloc=16, erro deve ser < 1e-4.

---

## Referências

- Lages, C. "A multiscale projection method for incompressible flows" — §3.1, Eq. 40, Table 1.
- `docs/06-convergencia-hh.md` — Superconvergência O(H²) via downscaling.
- `docs/05-estudo-alpha.md` — Dependência do erro com α.
- `docs/09-dg0-projecao.md` — Por que DG₀ é necessário e suficiente.
- `mrcm.convergence` — Modos Hh, H, h e extração de taxa (regressão log-log).
- `Vault/concepts/Convergence-Rates.md` — Previsões teóricas do MRCM.
