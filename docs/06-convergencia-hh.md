# Convergência Hh do MPrM — superconvergência via downscaling

**Data:** 2026-07-19 · **Teste:** `tests/test_convergence_hh.py` ·
**Resultados:** `results_alpha/convergence_hh.json`, `convergence_hh.png`

## O que o teste mede

Refinamento **Hh** (nomenclatura de `mrcm.convergence`, mode `"Hh"`):
`nloc` fixo (⇒ razão H/h fixa), `Ndom` cresce — H e h encolhem juntos.
Diferente do modo `H` (taxa multiescala O(H^{ℓ+2})) e do modo `h`
(saturação, só a malha fina refina), o modo Hh mede a **taxa de
refinamento clássico de FEM interior**.

Referência: a solução **analítica** de Olshanskii (mesmo campo do
benchmark de Helmholtz-Hodge, `tests/test_alpha_hh.py`), na convenção do
artigo do MPrM (Lages, §3.1, Tabela 1). A extração de taxa reusa
`mrcm.convergence.compute_convergence_rate`/`validate_rate` — mesma
regressão log-log usada pelos estudos T1–T7 do `mrcm`.

## O que a teoria do `mrcm` prevê

`Vault/concepts/Convergence-Rates.md`: "Hh-mode mede a taxa do FEM
interior (≈1.0 com RT₀/BDM₁), não a taxa multiescala." Essa é a taxa da
reconstrução BDM₁-DG₀ **crua** do MRCM — sem nenhum pós-processamento.

## O que foi medido

Config: `(P_{H,1}, U_{H,1})`, α=1, n_loc=16 fixo, Ndom ∈ {2,4,8,16}
(N = 32→256):

| Ndom | H | h | erro vs exato |
|---|---|---|---|
| 2 | 0,5000 | 0,03125 | 5,04e-1 |
| 4 | 0,2500 | 0,01562 | 1,28e-1 |
| 8 | 0,1250 | 0,00781 | 3,20e-2 |
| 16 | 0,0625 | 0,00391 | 8,00e-3 |

**Taxa observada: O(H^1,99)** — R² = 1,0000; taxa por par consecutivo
1,98 / 2,00 / 2,00. O erro cai por um fator ~63 (≈2⁶) quando H cai por
8× — bate quase exatamente com O(H²), **não** O(H) como o modo Hh nativo
do `mrcm` prevê para a reconstrução crua.

## Por que — hipótese

A saída do MPrM que medimos aqui **não é** a reconstrução BDM₁-DG₀ direta
do MRCM: ela passa pelo **downscaling** (`MACDownscaler`, Eqs. 21–23 do
artigo) — um solve de Poisson por volumes finitos de 5 pontos na malha
MAC, usando o fluxo médio de interface como condição de contorno de
Neumann. Esse pós-processamento local é, por si só, O(h²) para campos
suaves (esquema FV centrado clássico). A hipótese é que esse passo extra
**recupera superconvergência** a partir do piso O(H) do MRCM cru — análogo
a técnicas de pós-processamento local que recuperam ordem extra em
métodos mistos (Stenberg e afins).

## Confirmação com a referência fina (erro relativo isolado da decomposição)

O teste também mede `err_vs_fine` — erro relativo vs a projeção MAC **não
decomposta na mesma N** (`FineProjector`), convenção "Fine-reference error"
já usada em `tests/test_alpha_hh.py` e no `mrcm` nativo. Isso isola o erro
de **decomposição** (MSBF/Robin/interface) do erro intrínseco do esquema
MAC naquela resolução.

| Ndom | err vs exato | err vs fino (mesma N) |
|---|---|---|
| 2 | 5,0406e-1 | 5,0406e-1 |
| 4 | 1,2755e-1 | 1,2755e-1 |
| 8 | 3,1963e-2 | 3,1963e-2 |
| 16 | 7,9978e-3 | 7,9977e-3 |

As duas métricas são **idênticas até a 5ª–6ª casa decimal** em todos os
níveis — a mesma taxa O(H^1,99) (R²=1,0000) sai de ambas. Isso descarta a
hipótese de que a superconvergência fosse um artefato da comparação com a
solução analítica (ex.: a referência fina sendo "boa demais" e mascarando
o verdadeiro piso do MRCM): mesmo isolando o erro de decomposição pura, a
taxa medida continua O(H²), não O(H). A referência fina em si é muito
precisa nessas resoluções (erro vs exato ~1e-4, como em
`tests/test_alpha_hh.py`), então ela não contribui de forma mensurável ao
total — o que se mede é, de fato, o erro de decomposição do MPrM.

## Ressalvas

1. **Medido, não uma garantia teórica geral.** O campo de Olshanskii + o
   potencial `cos(πx)cos(πy)` são C^∞. Um problema com fonte de baixa
   regularidade poderia reexpor o piso O(H).
2. **Não é universal em todo (grau, α).** Com `(P_{H,2}, U_{H,1})`
   (assimétrico) e α=0,5, a taxa medida foi só 1,27 (R²=0,95, passos
   0,78→1,76 — ainda pré-assintótico/ruidoso nesse ponto). A
   superconvergência limpa foi confirmada apenas em graus simétricos com
   α=1; não foi testada exaustivamente noutras combinações.
3. Cada nível refaz o estágio offline do zero (sem cache entre valores de
   Ndom, já que a geometria muda) — o teste não reaproveita fatorações
   entre níveis de refinamento.

## Reprodução

```bash
source ~/venv-firedrake/bin/activate
cd ~/mrcm/NavierStokes

python tests/test_convergence_hh.py --Ndoms 2,4,8,16      # config do achado acima
python tests/test_convergence_hh.py --quick                 # smoke test (~4s)
mpiexec -n 3 python tests/test_convergence_hh.py             # paralelo (verificado bit-a-bit igual ao sequencial)

# Explorar outros pontos do espaço de parâmetros
python tests/test_convergence_hh.py --degreeU 2 --degreeP 1 --alpha 0.5
python tests/test_convergence_hh.py --expected-rate 1.0 --tolerance 0.2   # valida contra o piso MRCM cru
```
