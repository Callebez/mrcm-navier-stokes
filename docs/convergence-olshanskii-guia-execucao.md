# Estudo de Convergência — Problema de Olshanskii — Guia de Execução

> **Plano:** `docs/superpowers/plans/2026-07-22-mprm-convergence-olshanskii.md`
> **Spec:** `docs/superpowers/specs/2026-07-22-mprm-convergence-olshanskii-design.md`
> **Data:** 2026-07-22
> **Ambiente:** `source ~/venv-firedrake/bin/activate && cd ~/mrcm/NavierStokes`

---

## Resumo dos Experimentos

| Exp | Modo | Objetivo | Script | Comando MPI |
|-----|------|----------|--------|-------------|
| **E1** | Hh × α | Taxa de convergência vs parâmetro de Robin. A superconvergência O(H²) sobrevive fora de α=1? | `tests/test_convergence_hh_alpha.py` | `mpiexec -n 4 python tests/test_convergence_hh_alpha.py` |
| **E2** | Hh × graus sim | Taxa para graus simétricos (P_k, U_k), k=1,2,3. Melhora com enriquecimento? | `tests/test_convergence_hh_degrees.py` | `mpiexec -n 4 python tests/test_convergence_hh_degrees.py` |
| **E3** | Hh × graus assim | Taxa para graus assimétricos. Qual grau domina a taxa? | `tests/test_convergence_hh_degrees.py --asymmetric` | `mpiexec -n 4 python tests/test_convergence_hh_degrees.py --asymmetric` |
| **E4** | H (multiescala) | Taxa O(H^{degree+2}) com h~fixo. Mede convergência multiescala. | `tests/test_convergence_H_mode.py` | `mpiexec -n 4 python tests/test_convergence_H_mode.py` |
| **E5** | h (saturação) | H fixo, h→0. Em que h o erro satura? | `tests/test_convergence_h_mode.py` | `mpiexec -n 4 python tests/test_convergence_h_mode.py` |
| **E6** | p (grau) | Malha fixa, graus crescentes. Convergência espectral? | `tests/test_convergence_p_mode.py` | `mpiexec -n 4 python tests/test_convergence_p_mode.py` |
| **E10** | Downscaling on/off | Prova da hipótese: com downscaling → O(H²), sem → O(H¹)? | `tests/test_convergence_downscaling.py` | `mpiexec -n 4 python tests/test_convergence_downscaling.py` |

---

## Detalhamento por Experimento

### E1 — Modo Hh × α

**Objetivo:** Determinar se a taxa de superconvergência O(H²) observada em α=1
se mantém para outros valores de α. O estudo anterior de α (`docs/05-estudo-alpha.md`)
mostrou que α=0.1 minimiza o erro absoluto — mas isso também otimiza a **taxa**?

**Hipóteses:**
- α ≈ 0.1–10: taxa ≈ O(H²) (superconvergência robusta)
- α → 0 (MMMFEM): taxa pode degradar (acoplamento fraco)
- α → ∞ (MHM): taxa pode cair para O(H¹) (colapso dos espaços de pressão)

**Config:** nloc=16, (P₁,U₁), Ndom ∈ {2,4,8,16}, α ∈ {0.01, 0.1, 1, 10, 100, 1000}
**Jobs:** 6α × 4 Ndom = 24
**Tempo estimado:** ~15 min com 4 cores MPI

**Saída:** `results_alpha/convergence_olshanskii/e1_hh_alpha.json`, `.png`

---

### E2 — Modo Hh × graus simétricos

**Objetivo:** Medir como a taxa de convergência Hh melhora com o enriquecimento
dos espaços de interface. Testa (P_k, U_k) para k=1,2,3 com α=1.0 e α=0.1.

**Hipóteses:**
- (P₁,U₁): O(H²) — baseline confirmado
- (P₂,U₂): O(H³) ou O(H⁴) — melhora detectável
- (P₃,U₃): taxa ainda maior, mas limitada pelo downscaler O(h²) = O(H²) em modo Hh

**Config:** nloc=16, Ndom ∈ {2,4,8,16}, k ∈ {1,2,3}, α ∈ {1.0, 0.1}
**Jobs:** 2α × 3k × 4 Ndom = 24
**Tempo estimado:** ~25 min com 4 cores MPI (graus maiores são mais caros)

**Saída:** `results_alpha/convergence_olshanskii/e2_hh_degrees_sym.json`, `.png`

---

### E3 — Modo Hh × graus assimétricos

**Objetivo:** Determinar qual espaço (U_H ou P_H) domina a taxa de convergência
quando os graus são diferentes. Testa (P₂,U₁), (P₃,U₂), (P₁,U₂), (P₂,U₃).

**Hipóteses:**
- A taxa é dominada pelo **menor** grau
- (P₂,U₁) deve ter taxa similar a (P₁,U₁) ≈ O(H²), não a (P₂,U₂)
- (P₁,U₂) deve ter taxa similar a (P₁,U₁) — o grau de pressão sozinho não ajuda

**Config:** nloc=16, Ndom ∈ {2,4,8,16}, α=1.0
**Jobs:** 4 pares × 4 Ndom = 16
**Tempo estimado:** ~15 min com 4 cores MPI

**Saída:** `results_alpha/convergence_olshanskii/e3_hh_degrees_asym.json`, `.png`

---

### E4 — Modo H (multiescala)

**Objetivo:** Medir a taxa multiescala O(H^{degree+2}) prevista pela teoria do
MRCM. Diferente do modo Hh (onde H e h variam juntos), aqui h é mantido
aproximadamente fixo (~1/256) enquanto H diminui (mais subdomínios).

**Hipóteses:**
- Na faixa não-saturada (H ≫ h): taxa O(H³) para (P₁,U₁)
- Quando H → h: saturação no piso O(h²) do downscaler
- Este experimento revela se há benefício em usar MAIS subdomínios (H menor)
  versus refinar a malha fina (h menor)

**Config:** h_target ≈ 1/256, (P₁,U₁), α=1.0, Ndom ∈ {3,4,6,8,12}
**Jobs:** 6
**Tempo estimado:** ~15 min com 4 cores MPI

**Saída:** `results_alpha/convergence_olshanskii/e4_H_multiscale.json`, `.png`

---

### E5 — Modo h (saturação)

**Objetivo:** Com decomposição fixa (Ndom=4, H=0.25), refinar a malha fina
(nloc → ∞) e observar em que ponto o erro para de cair. Determina o piso
de erro imposto pelo espaço de interface.

**Hipóteses:**
- O erro cai O(h²) para h grande (dominado pelo downscaler)
- Satura em ~0.127 para (P₁,U₁) quando h ≲ 1/16
- O patamar de saturação é menor para graus maiores (confirmado por E6)

**Config:** Ndom=4 (H=0.25), (P₁,U₁), α=1.0, nloc ∈ {2,4,8,16,32,64}
**Jobs:** 6
**Tempo estimado:** ~2 min com 4 cores MPI

**Saída:** `results_alpha/convergence_olshanskii/e5_h_saturation.json`, `.png`

---

### E6 — Modo p (grau)

**Objetivo:** Com malha fixa (Ndom=4, nloc=16), aumentar o grau polinomial dos
espaços de interface e medir a queda no erro. Para o campo C^∞ de Olshanskii,
espera-se convergência espectral (exponencial).

**Hipóteses:**
- Erro cai exponencialmente com k (campo suave)
- Piso imposto pelo downscaler: ~O(h²) ≈ (1/64)² ≈ 2.4e-4
- Abaixo desse piso, aumentar o grau não traz benefício

**Config:** Ndom=4, nloc=16, α=1.0, k ∈ {1,2,3,4}
**Jobs:** 4
**Tempo estimado:** ~1 min com 4 cores MPI

**Saída:** `results_alpha/convergence_olshanskii/e6_p_degree.json`, `.png`

---

### E10 — Efeito do Downscaling (crucial)

**Objetivo:** Provar que o downscaling é o mecanismo da superconvergência.
Compara a taxa Hh COM e SEM o passo de downscaling:
- **Com downscaling:** pipeline MPrM completo → esperado O(H²)
- **Sem downscaling:** reconstrução BDM₁-DG₀ crua (avaliada nas faces MAC) → esperado O(H¹)

**Hipóteses:**
- Com downscaling: O(H^1.99) (confirmado em `docs/06-convergencia-hh.md`)
- Sem downscaling: O(H^1.0) — o piso do MRCM cru
- Este experimento fecha o ciclo de entendimento sobre a superconvergência

**Config:** nloc=16, (P₁,U₁), α=1.0, Ndom ∈ {2,4,8,16}
**Jobs:** 4 (cada job produz AMBAS as métricas)
**Tempo estimado:** ~8 min com 4 cores MPI

**Saída:** `results_alpha/convergence_olshanskii/e10_downscaling_on_off.json`, `.png`

---

## Comandos de Execução (MPI 4 cores)

```bash
cd ~/mrcm/NavierStokes && source ~/venv-firedrake/bin/activate

# E1 — Hh × α (~15 min)
mpiexec -n 4 python tests/test_convergence_hh_alpha.py \
  --alphas 0.01,0.1,1.0,10.0,100.0,1000.0 --Ndoms 2,4,8,16

# E2 — Hh × graus simétricos (~25 min)
mpiexec -n 4 python tests/test_convergence_hh_degrees.py \
  --ks 1,2,3 --alphas 1.0,0.1 --Ndoms 2,4,8,16

# E3 — Hh × graus assimétricos (~15 min)
mpiexec -n 4 python tests/test_convergence_hh_degrees.py \
  --asymmetric --Ndoms 2,4,8,16

# E4 — Modo H multiescala (~15 min)
mpiexec -n 4 python tests/test_convergence_H_mode.py \
  --Ndoms 3,4,6,8,12 --h-target 0.00390625

# E5 — Modo h saturação (~2 min)
mpiexec -n 4 python tests/test_convergence_h_mode.py \
  --nlocs 2,4,8,16,32,64 --Ndom 4

# E6 — Modo p grau (~1 min)
mpiexec -n 4 python tests/test_convergence_p_mode.py \
  --ks 1,2,3,4 --Ndom 4 --nloc 16

# E10 — Downscaling on/off (~8 min)
mpiexec -n 4 python tests/test_convergence_downscaling.py \
  --Ndoms 2,4,8,16
```

---

## Smoke Tests (validação rápida, sem MPI)

```bash
# Cada script suporta --quick (2 níveis apenas, ~10s cada)
python tests/test_convergence_hh_alpha.py --quick
python tests/test_convergence_hh_degrees.py --quick
python tests/test_convergence_hh_degrees.py --asymmetric --quick
python tests/test_convergence_H_mode.py --quick
python tests/test_convergence_h_mode.py --quick
python tests/test_convergence_p_mode.py --quick
python tests/test_convergence_downscaling.py --quick
```

---

## Estrutura de Saída

```
results_alpha/convergence_olshanskii/
├── e1_hh_alpha.json          # E1: 6 curvas (uma por α), cada uma com 4 pts
├── e1_hh_alpha.png
├── e2_hh_degrees_sym.json    # E2: 6 curvas (2α × 3k), cada uma com 4 pts
├── e2_hh_degrees_sym.png
├── e3_hh_degrees_asym.json   # E3: 4 curvas (pares assimétricos)
├── e3_hh_degrees_asym.png
├── e4_H_multiscale.json      # E4: 1 curva, 6 pts
├── e4_H_multiscale.png
├── e5_h_saturation.json      # E5: 1 curva, 6 pts
├── e5_h_saturation.png
├── e6_p_degree.json          # E6: 1 curva, 4 pts
├── e6_p_degree.png
├── e10_downscaling_on_off.json  # E10: 2 curvas (com/sem downscaling)
├── e10_downscaling_on_off.png
└── convergence_olshanskii_report.md  # Sumário final (a gerar)
```
