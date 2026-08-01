# Cavidade: sensibilidade ao grau de interface e a α — Design Spec

> **Data:** 2026-08-01 · **Branch:** `feat/rtcf-quad-mprm`
> **Contexto:** Revisão da implementação da cavidade (driven-lid)

## 1. Motivação

O estudo da Fig. 6 (benchmark HH) mostrou que a sensibilidade a α depende
criticamente do grau dos espaços de interface: com BDM₁, k=3,4 são extremamente
sensíveis (spread 30-197×); com BDM₂, são quase planos. A pergunta natural é
se esse comportamento se transfere para a **cavidade**, onde o transporte não
linear (convecção) está presente e pode dominar o erro.

O artigo de Lages (§3.3, Fig. 9) usa apenas (P_H,1, U_H,1) com α=1 para a
cavidade. O estudo de α existente (`docs/05-estudo-alpha.md` §"Estudo 2")
varre α apenas para k=1 e encontra erro **decrescente monotonicamente** com α
— comportamento oposto ao benchmark HH. Não se sabe se esse comportamento
persiste para k≥2.

## 2. Objetivos

### Estudo A: Grau de interface × α para Re=100

Varrer (k, α) ∈ {1,2,3} × {1e-2, 1e-1, 1, 1e1, 1e2, 1e4} e medir:

| Métrica | Definição |
|---|---|
| `err_vs_fine` | ‖u_MPrM − u_fino‖₂ / ‖u_fino‖₂ (mesma malha, mesmo δt, mesmo T) |
| Perfil u(y) | Velocidade horizontal em x=0.5, comparado contra Ghia |
| Perfil v(x) | Velocidade vertical em y=0.5, comparado contra Ghia |
| `divmax` | Divergência máxima (verificação de conservação) |
| `defect` | Defeito de compatibilidade do downscaling |

**Hipóteses:**
- **H1:** O erro decresce monotonicamente com α para qualquer k (como já
  observado para k=1). O limite MHM é sempre o melhor.
- **H2:** Enriquecer k reduz o erro em qualquer α, mas o ganho é menor que
  no benchmark HH (porque o erro de transporte domina).
- **H3:** Os perfis de velocidade convergem para Ghia com o aumento de k.
- **H4:** O α ótimo **não** depende de k (ao contrário do benchmark HH).

### Estudo B: Perfis de velocidade vs Ghia na melhor configuração

Para k=2 (melhor custo-benefício) e α ótimo, rodar Re=100,400,1000 com
N≥128 e comparar perfis u(y), v(x) contra Ghia. Este estudo já foi iniciado
(`tests/test_cavity_ghia.py`, rodando N=128 com k=2, α=1).

## 3. Configuração

| Parâmetro | Estudo A | Estudo B |
|---|---|---|
| Malha | 64² (Ndom=4, nloc=16) | 128² (Ndom=8, nloc=16) |
| Reynolds | 100 | 100, 400, 1000 |
| Tempo final T | 20 | 30 (Re=100), 40 (400), 50 (1000) |
| ks | 1, 2, 3 | 2 |
| αs | 1e-2, 1e-1, 1, 1e1, 1e2, 1e4 | 1 (ou ótimo do Estudo A) |
| Basis | legendre_P | legendre_P |
| Convecção | TVD 2ª ordem, minmod | TVD 2ª ordem, minmod |

**Justificativa para N=64 no Estudo A:** 18 casos (3k × 6α), cada um com
~10-15k passos de transporte. Com N=128 seriam ~50k passos/caso → proibitivo
(~15h). N=64 reduz para ~3k passos/caso → ~15 min total. A qualidade da
comparação é suficiente para identificar tendências.

## 4. Scripts

| Arquivo | Responsabilidade |
|---|---|
| `tests/test_cavity_degree_alpha.py` | Estudo A: varredura (k, α) com N=64 |
| `tests/test_cavity_ghia.py` | Estudo B: validação Ghia com N=128 |
| `results_alpha/cavity_degree_alpha/` | Saída do Estudo A |
| `results_alpha/cavity_ghia/` | Saída do Estudo B |

### Comandos

```bash
# Estudo A: varredura k×α (~15 min com MPI 4 ranks)
mpiexec -n 4 python tests/test_cavity_degree_alpha.py

# Estudo B: Ghia N=128 (~75 min, já rodando)
mpiexec -n 4 python tests/test_cavity_ghia.py --Re 100,400,1000 --Ndom 8 --nloc 16

# Estudo A customizado
python tests/test_cavity_degree_alpha.py --ks 2,3 --alphas 0.1,1,10 --T 30
```

## 5. Figuras geradas (Estudo A)

1. `error_vs_alpha.png` — erro vs α, uma curva por k (log-log, ★ = mínimo)
2. `profiles_u_vs_ghia.png` — u(y) vs Ghia para o melhor α de cada k
3. `profiles_u_k2_all_alpha.png` — u(y) para k=2 em todos os α

## 6. Riscos e limitações

1. **T=20 pode ser insuficiente** para estado estacionário em Re=100 com
   N=64. A validação Ghia (Estudo B) usa T=30 e servirá de referência.
2. **N=64 é marginal** para comparação quantitativa com Ghia. O Estudo A
   foca em tendências (k, α), não em acurácia absoluta.
3. **k=4 omitido** — o custo offline do MRCM com 64 MsBFs por subdomínio
   torna a cavidade (centenas de passos) inviável.
4. **Apenas Re=100** no Estudo A. Reynolds mais altos exigem malha mais
   fina para resolver a camada limite.
