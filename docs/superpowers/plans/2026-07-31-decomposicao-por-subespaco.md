# Verificação da decomposição por subespaço: plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Estender o benchmark de Olshanskii para medir **separadamente** as duas componentes do erro da decomposição de Helmholtz–Hodge multiescala — o vazamento solenoidal a partir de entrada irrotacional pura, e o vazamento irrotacional a partir de entrada solenoidal pura —, e acrescentar os testes de verificação no estilo da tese de Carvalho que sobrevivem à análise de linearidade.

**Architecture:** Segue o caminho do Olshanskii, e não o de von Kármán. Toda a construção é de **disparo único**, sem laço temporal: amostra-se um campo analítico nas faces MAC, aplica-se `mprm_projection_step` uma vez e compara-se com a resposta exata. As funções novas entram em `mprm/benchmarks/hh.py` e `mprm/benchmarks/convergence.py` (puras, sem E/S), e os scripts em `tests/` cuidam de CLI, JSON e figuras — a mesma separação já estabelecida.

**Tech Stack:** Python, NumPy, `mpi4py`, Firedrake (só na etapa elíptica, via `mrcm`), `pytest`, matplotlib.

---

## Fundamentação: por que medir por subespaço

O capítulo do MRCM da dissertação estabelece analiticamente que, na decomposição multiescala, **divergência nula e ortogonalidade se dissociam**. A regularização recupera a primeira; a segunda permanece violada, e o resíduo é a integral, sobre o esqueleto, do salto do produto entre pressão e fluxo normal. A conclusão é que o erro se aloja na parcela irrotacional.

Essa é uma afirmação **testável**, e o benchmark atual não a testa. Hoje `hh_benchmark_mac` alimenta o pipeline com $\bm w = \bm u + \nabla\tilde p$, em que ambas as parcelas são não nulas, e mede um único erro agregado `err_u`. Se o erro está todo em uma parcela, isso não aparece.

**A construção que separa.** Como o campo de entrada se decompõe em duas parcelas e o pipeline é linear, basta anular uma de cada vez:

| Caso | Entrada | Resposta exata | O que mede |
|---|---|---|---|
| **G** (gradiente puro) | $\bm w = \nabla\tilde p$ | $\bm u = \bm 0$ | Vazamento **solenoidal espúrio** |
| **S** (solenoidal puro) | $\bm w = \bm u$ | $\bm u_h = \bm w$, $\tilde{\bm v}= \bm 0$ | Vazamento **irrotacional espúrio** |
| **M** (misto, atual) | $\bm w = \bm u + \nabla\tilde p$ | $\bm u$ analítico | Erro agregado, já implementado |

No caso G, **todo** valor não nulo de $\bm u_h$ é erro puro, sem necessidade de solução de referência — a resposta exata é identicamente zero. É a medida mais limpa possível do defeito da decomposição.

### Um alerta que precede a implementação: a varredura de amplitude é vazia

A tese de Carvalho verifica *pressure-robustness* (§3.4.6) multiplicando a pressão exata por um parâmetro $m_{pr}$ e conferindo que o erro de velocidade não cresce. **Essa varredura não deve ser implementada aqui**, porque passaria por construção.

O pipeline `mprm_projection_step` é **linear** em $\bm w$: `mac_divergence` é um operador linear; `MPrMProjector.project` resolve um sistema linear; a média de fluxos e os problemas locais de Neumann do downscaling são lineares; e a correção $\bm u = \bm w - \tilde{\bm v}$ é linear. Logo $\;\mathcal{L}(m\,\bm w) = m\,\mathcal{L}(\bm w)\;$ exatamente, e a razão $\|\bm u_h\|/\|\bm w\|$ independe de $m$ por álgebra, não por virtude do método. Um teste construído sobre essa varredura seria satisfeito por qualquer implementação, inclusive uma errada.

A não linearidade das equações de Navier–Stokes vive em `mac_transport`, que **não** participa deste pipeline.

**O que se preserva do teste de Carvalho** é o ingrediente que de fato importa: a entrada de **gradiente puro**, cuja resposta exata é nula. O que muda é o eixo da varredura: em vez de amplitude, varrem-se os parâmetros que realmente governam o erro — $\alpha$, os graus dos espaços de interface e a razão $H/h$. Isso é exatamente um estudo de convergência, e cabe na infraestrutura existente.

---

## Global Constraints

1. **Disparo único, sem laço temporal.** Nada neste plano usa `run_flow`, `run_cavity` ou `run_von_karman`. O caminho é `hh_build_*` → `mprm_projection_step` → comparação.
2. **Separação de responsabilidades.** `mprm/benchmarks/*.py` são funções puras, sem E/S e sem plotagem. CLI, JSON e figuras ficam em `tests/`.
3. **`import mprm` não puxa Firedrake.** Imports de Firedrake ficam dentro das funções.
4. **Convenções.** `tests/params.py` (`CommonParams`, `build_parser`) e `tests/common.py` (`get_comm`, `banner`, `make_meta`, `save_json`, `save_figure`). Flag `--quick` obrigatória. Saída em `results_alpha/convergence_olshanskii/`.
5. **Envelope autodescritivo.** Todo JSON traz `hypothesis`, `criterion`, `verdict`, `caveats` e `git_commit`. JSON sem cabeçalho é execução, não veredito.
6. **Formas MAC.** `u` é `(N+1, N)`; `v` é `(N, N+1)`; escalares de célula são `(N, N)`.
7. **Retrocompatibilidade.** `hh_build_w` e `hh_benchmark_mac` mantêm assinatura e comportamento. Os testes de convergência existentes devem continuar passando sem alteração.

---

## File Structure

| Arquivo | Responsabilidade | Estado |
|---|---|---|
| `mprm/benchmarks/hh.py` | Campos analíticos e disparo único. Recebe `hh_build_w_mode` e `hh_benchmark_mode` | Modificar |
| `mprm/benchmarks/convergence.py` | Curvas de convergência. Recebe o parâmetro `mode` nas curvas existentes | Modificar |
| `tests/test_subspace_leakage.py` | **Novo** — E11, os dois casos-limite | Criar |
| `tests/test_convergence_subspace.py` | **Novo** — E12, curvas por subespaço | Criar |
| `docs/13-decomposicao-por-subespaco.md` | **Novo** — writeup | Criar |

---

# Fase 1 — Os campos de entrada por subespaço

### Task 1: Generalizar a construção do campo

**Files:**
- Modify: `mprm/benchmarks/hh.py`
- Test: `tests/test_subspace_fields.py` (criar)

**Interfaces:**
- Produz: `hh_build_w_mode(N, mode) -> (w1, w2, u1_ex, u2_ex)` com `mode ∈ {"mixed", "gradient", "solenoidal"}`. `u1_ex, u2_ex` é a **parcela solenoidal exata** da entrada, que é o alvo da comparação em todos os modos. Consumida pelas Tasks 2 e 3.

**Nota de projeto.** O potencial já usado, $\tilde p=\cos(\pi x)\cos(\pi y)$, satisfaz $\nabla\tilde p\cdot\bhat n = 0$ em $\partial(0,1)^2$ — é por isso que ele foi escolhido, e é o que torna a entrada compatível com o caminho impermeável do `mrcm`. **Manter esse potencial**, e não substituí-lo pelo polinômio de Carvalho, cuja derivada normal não se anula nas paredes horizontais e exigiria outro caminho de contorno.

- [ ] **Step 1: Escrever o teste que falha**

```python
# tests/test_subspace_fields.py
import numpy as np
from mprm.benchmarks.hh import hh_build_w, hh_build_w_mode
from mprm.transport import mac_divergence

def test_modo_mixed_reproduz_a_construcao_atual():
    """Retrocompatibilidade exata com hh_build_w."""
    w1a, w2a, u1a, u2a = hh_build_w(24)
    w1b, w2b, u1b, u2b = hh_build_w_mode(24, mode="mixed")
    assert np.allclose(w1a, w1b) and np.allclose(w2a, w2b)
    assert np.allclose(u1a, u1b) and np.allclose(u2a, u2b)

def test_modo_gradient_tem_parcela_solenoidal_nula():
    w1, w2, u1, u2 = hh_build_w_mode(24, mode="gradient")
    assert np.allclose(u1, 0.0) and np.allclose(u2, 0.0)
    # entrada é gradiente puro: sua divergência NÃO é nula
    assert np.abs(mac_divergence(w1, w2, 1/24, 1/24)).max() > 1e-3

def test_modo_solenoidal_tem_entrada_de_divergencia_nula():
    N = 48
    w1, w2, u1, u2 = hh_build_w_mode(N, mode="solenoidal")
    assert np.allclose(w1, u1) and np.allclose(w2, u2)
    # o campo de Olshanskii é solenoidal: div discreta -> 0 com o refino
    d = np.abs(mac_divergence(w1, w2, 1/N, 1/N)).max()
    assert d < 1e-2, f"div discreta do campo analítico: {d:.2e}"

def test_impermeabilidade_em_todos_os_modos():
    for m in ("mixed", "gradient", "solenoidal"):
        w1, w2, _, _ = hh_build_w_mode(16, mode=m)
        assert np.allclose(w1[0, :], 0.0) and np.allclose(w1[-1, :], 0.0), m
        assert np.allclose(w2[:, 0], 0.0) and np.allclose(w2[:, -1], 0.0), m
```

- [ ] **Step 2: Rodar e confirmar que falha**

```bash
cd ~/mrcm/NavierStokes && source ~/venv-firedrake/bin/activate
python -m pytest tests/test_subspace_fields.py -v
```

Esperado: `ImportError: cannot import name 'hh_build_w_mode'`.

- [ ] **Step 3: Implementar**

Em `mprm/benchmarks/hh.py`, acrescentar após `hh_build_w`:

```python
_MODES = ("mixed", "gradient", "solenoidal")


def hh_build_w_mode(N, mode="mixed"):
    """Amostra o campo de entrada da decomposição, por subespaço.

    mode="mixed"      : w = u + grad p̃   (construção original)
    mode="gradient"   : w = grad p̃       (parcela solenoidal exata = 0)
    mode="solenoidal" : w = u            (parcela irrotacional exata = 0)

    Devolve (w1, w2, u1_ex, u2_ex), em que (u1_ex, u2_ex) é a parcela
    SOLENOIDAL EXATA da entrada — o alvo da comparação em qualquer modo.
    """
    if mode not in _MODES:
        raise ValueError(f"mode inválido: {mode!r} (use um de {_MODES})")
    h = 1.0 / N
    xf = np.linspace(0.0, 1.0, N + 1)
    yc = (np.arange(N) + 0.5) * h
    Xu, Yu = np.meshgrid(xf, yc, indexing="ij")
    xc = (np.arange(N) + 0.5) * h
    yf = np.linspace(0.0, 1.0, N + 1)
    Xv, Yv = np.meshgrid(xc, yf, indexing="ij")

    if mode == "gradient":
        u1 = np.zeros_like(Xu); u2 = np.zeros_like(Xv)
    else:
        u1, _ = olshanskii_u_np(Xu, Yu)
        _, u2 = olshanskii_u_np(Xv, Yv)

    if mode == "solenoidal":
        g1 = np.zeros_like(Xu); g2 = np.zeros_like(Xv)
    else:
        g1, _ = hh_gradp_np(Xu, Yu)
        _, g2 = hh_gradp_np(Xv, Yv)

    w1, w2 = u1 + g1, u2 + g2
    w1[0, :] = w1[-1, :] = 0.0
    w2[:, 0] = w2[:, -1] = 0.0
    return w1, w2, u1, u2
```

**Atenção.** A imposição de `w·n = 0` no contorno é exata para $\nabla\tilde p$ e para o campo de Olshanskii separadamente, de modo que zerar as faces externas não introduz inconsistência em nenhum dos três modos. Confirme isso pelo quarto teste do Step 1 antes de seguir.

- [ ] **Step 4: Rodar os testes**

```bash
python -m pytest tests/test_subspace_fields.py -v
python -m pytest tests/test_convergence_hh.py -q
```

- [ ] **Step 5: Commit**

```bash
git add mprm/benchmarks/hh.py tests/test_subspace_fields.py
git commit -m "feat(hh): campo de entrada por subespaço da decomposição

hh_build_w_mode permite anular, na entrada, a parcela solenoidal ou a
irrotacional. Com entrada de gradiente puro a resposta exata é u=0, de modo
que qualquer valor não nulo é erro puro, sem necessidade de referência.

Mantém o potencial cos(pi x)cos(pi y), cuja derivada normal se anula no
contorno — condição do caminho impermeável do mrcm. O polinômio de
Carvalho não tem essa propriedade e exigiria outro tratamento de contorno.

hh_build_w segue intacta."
```

---

### Task 2: Medir as duas componentes do erro

**Files:**
- Modify: `mprm/benchmarks/hh.py` (acrescentar `hh_benchmark_mode`)
- Test: `tests/test_subspace_leakage.py` (criar)

**Interfaces:**
- Consome: `hh_build_w_mode` (Task 1).
- Produz: `hh_benchmark_mode(Ndom, nloc, degreeU, degreeP, alpha, mode, comm) -> dict` com as chaves de `hh_benchmark_mac` mais `err_abs`, `leak_ratio` e `mode`. Consumida pela Task 3.

**As métricas.**

- `err_u` — erro **relativo** à parcela solenoidal exata. Indefinido no modo `gradient`, em que o denominador é zero; nesse modo devolve `None`.
- `err_abs` $=\|\bm u_h - \bm u_{ex}\|_2$, sempre definido, em norma discreta $\ell^2$ ponderada por $h$.
- `leak_ratio` $=\|\bm u_h - \bm u_{ex}\|_2 / \|\bm w\|_2$ — a fração da energia da **entrada** que aparece como erro. É esta a métrica comparável entre modos, e é a que não depende da amplitude.

- [ ] **Step 1: Escrever o teste**

```python
# tests/test_subspace_leakage.py
"""E11 — vazamento entre subespaços da decomposição.

Hipótese (Cap. 3 da dissertação): o erro da decomposição multiescala aloja-se
na parcela irrotacional. Critério: com entrada de gradiente puro, a parcela
solenoidal devolvida é não nula e é a sede do erro; com entrada solenoidal
pura, o erro é comparativamente menor.
"""
import numpy as np
import pytest
from mprm.benchmarks.hh import hh_benchmark_mode


def test_gradiente_puro_devolve_solenoidal_espurio():
    """Entrada = grad p. Resposta exata = 0. Tudo o que sobra é erro."""
    r = hh_benchmark_mode(Ndom=3, nloc=16, alpha=1.0, mode="gradient")
    assert r["err_u"] is None, "erro relativo é indefinido quando u_ex = 0"
    assert r["err_abs"] > 0.0, "erro exatamente nulo indicaria bug no arranjo"
    assert r["leak_ratio"] < 1.0, "vazamento não pode exceder a energia da entrada"


def test_divergencia_nula_em_todos_os_modos():
    """O downscaling deve entregar div u = 0 independentemente da entrada."""
    for m in ("mixed", "gradient", "solenoidal"):
        r = hh_benchmark_mode(Ndom=3, nloc=16, alpha=1.0, mode=m)
        assert r["divmax"] < 1e-9, f"modo {m}: divmax={r['divmax']:.2e}"


def test_vazamento_cai_com_enriquecimento_da_interface():
    """A previsão testável: enriquecer a interface reduz o vazamento."""
    r1 = hh_benchmark_mode(Ndom=3, nloc=16, degreeU=1, degreeP=1,
                           alpha=1.0, mode="gradient")
    r3 = hh_benchmark_mode(Ndom=3, nloc=16, degreeU=3, degreeP=3,
                           alpha=1.0, mode="gradient")
    assert r3["leak_ratio"] < r1["leak_ratio"], (
        f"enriquecer não reduziu o vazamento: "
        f"{r1['leak_ratio']:.3e} -> {r3['leak_ratio']:.3e}")


def test_linearidade_do_pipeline():
    """Registra explicitamente por que a varredura de amplitude é vazia.

    Este teste NÃO verifica o método: verifica a premissa que torna
    desnecessária a varredura de m_pr do teste de pressure-robustness.
    """
    from mprm.benchmarks.hh import hh_build_w_mode
    from mprm.benchmarks.hh import _project_once   # helper da Task 2
    w1, w2, _, _ = hh_build_w_mode(48, mode="gradient")
    a1, a2 = _project_once(w1, w2, Ndom=3, nloc=16, alpha=1.0)
    b1, b2 = _project_once(1e6 * w1, 1e6 * w2, Ndom=3, nloc=16, alpha=1.0)
    assert np.allclose(b1, 1e6 * a1, rtol=1e-8)
    assert np.allclose(b2, 1e6 * a2, rtol=1e-8)
```

- [ ] **Step 2: Rodar e confirmar que falha**

- [ ] **Step 3: Implementar**

Extrair de `hh_benchmark_mac` o miolo que monta o projetor e dispara um passo, em um helper `_project_once(w1, w2, Ndom, nloc, degreeU, degreeP, alpha, comm)`, e escrever `hh_benchmark_mode` sobre ele. **Reescrever `hh_benchmark_mac` para chamar `hh_benchmark_mode(mode="mixed")`**, de modo que exista um único caminho de código e a retrocompatibilidade seja estrutural, e não uma cópia a manter em sincronia.

Norma discreta a usar, coerente com a malha MAC:

```python
def _l2_mac(a1, a2, h):
    return float(np.sqrt(h * h * (np.sum(a1 ** 2) + np.sum(a2 ** 2))))
```

- [ ] **Step 4: Rodar e registrar o resultado**

```bash
python -m pytest tests/test_subspace_leakage.py -v
```

**Instrução a quem executa.** O terceiro teste é uma **hipótese**, não um requisito. Se o vazamento não cair com o enriquecimento, **não relaxe o critério**: isso contradiria a previsão do capítulo e é achado a reportar. Marque `@pytest.mark.xfail(strict=True)` com justificativa e registre o número medido.

- [ ] **Step 5: Commit**

---

# Fase 2 — Curvas de convergência por subespaço

### Task 3: Propagar `mode` às curvas de convergência

**Files:**
- Modify: `mprm/benchmarks/convergence.py`
- Create: `tests/test_convergence_subspace.py`

**Interfaces:**
- Consome: `hh_benchmark_mode` (Task 2).
- Produz: parâmetro `mode="mixed"` em `convergence_curve_hh`, `convergence_curve_H`, `convergence_curve_h` e `convergence_curve_p`; campo `leak_ratio` em cada `case`; e `rate_leak` ao lado de `rate_exact` e `rate_fine`.

- [ ] **Step 1:** Acrescentar `mode="mixed"` às quatro assinaturas, repassar a `_run_one_hh_level`, e este a `hh_benchmark_mode`. O default preserva o comportamento atual (Constraint 7).

- [ ] **Step 2:** Acrescentar `leak_ratio` ao dicionário de cada caso e ajustar `_fit_rate` para produzir também `rate_leak`. No modo `gradient`, `err_vs_exact` é `None`, e as funções de ajuste precisam **ignorar** a curva ausente em vez de quebrar — verifique `_fit_rate` antes de assumir.

- [ ] **Step 3:** Escrever `tests/test_convergence_subspace.py` no padrão dos demais scripts: `build_parser`, `--quick`, JSON e figura. Quatro curvas, uma por modo e por grau, com `Ndom ∈ {2,4,8,16}` e `nloc=16`.

- [ ] **Step 4:** Rodar em modo rápido, depois completo com MPI:

```bash
python tests/test_convergence_subspace.py --quick
mpiexec -n 4 python tests/test_convergence_subspace.py --modes gradient,solenoidal,mixed --Ndoms 2,4,8,16
```

- [ ] **Step 5:** Commit.

---

### Task 4: Varredura de $\alpha$ por subespaço

**Files:**
- Modify: `tests/test_convergence_subspace.py` (acrescentar `--alphas`)

**A pergunta que isto responde.** O capítulo do MRCM fixa $\alpha=1$ a partir de estudos em meios **heterogêneos**, e a dissertação registra que essa escolha não tem justificativa automática em um problema **homogêneo** como a etapa de projeção. Além disso, o parâmetro de Robin controla o balanço entre continuidade de pressão ($\alpha\to 0$) e de fluxo ($\alpha\to\infty$) — precisamente as duas grandezas cujo salto aparece no resíduo de ortogonalidade.

A previsão a testar é, portanto, específica: **o $\alpha$ que minimiza o vazamento no modo `gradient` não precisa coincidir com o que o minimiza no modo `solenoidal`**, porque cada modo penaliza um dos dois saltos. Se os ótimos divergirem, isso é um resultado com consequência direta para a escolha de $\alpha$ no método de projeção.

- [ ] **Step 1:** Varrer `alpha ∈ {1e-3, 1e-2, 0.1, 1, 10, 100, 1e3}` para os modos `gradient` e `solenoidal`, com malha e graus fixos.
- [ ] **Step 2:** Reportar, para cada modo, o $\alpha$ de menor `leak_ratio`, e registrar se coincidem.
- [ ] **Step 3:** Commit.

---

# Fase 3 — Consolidação

### Task 5: Writeup e guia de execução

**Files:**
- Create: `docs/13-decomposicao-por-subespaco.md`

- [ ] **Step 1:** Documentar a construção, incluindo **o argumento de linearidade** e por que a varredura de amplitude foi deliberadamente omitida — de outro modo, uma leitura futura reintroduzirá o teste vazio.
- [ ] **Step 2:** Tabela de vereditos, um por hipótese, com o valor medido.
- [ ] **Step 3:** Comandos de execução e tempos, no formato de `docs/convergence-olshanskii-guia-execucao.md`.
- [ ] **Step 4:** Commit.

---

# Fase 4 — O defeito de ortogonalidade

> **Spec:** `docs/superpowers/specs/2026-07-31-defeito-ortogonalidade-design.md`

As Fases 1–3 medem o erro por **subespaço** — quanto vaza de uma parcela para a
outra. A Fase 4 mede a quantidade que a Seção `sec:erro-decomposicao` da
dissertação identifica como **a** sede do erro, e que o capítulo de resultados
promete medir diretamente:

$$\int_\Omega \bm u_h\cdot\tilde{\bm v}_h \;=\; \int_\Gamma \big[\!\big[\,\psi_h\,(\bm u_h\cdot\hat{\bm n})\,\big]\!\big].$$

Nenhuma execução até hoje calculou esse número. A Task 6 faz a leitura pelo
volume, a Task 7 faz a leitura pelo esqueleto, e a discordância entre as duas é
o veredito sobre a própria derivação do capítulo.

**As Tasks 1–5 não são tocadas.** `hh_build_w` e `hh_benchmark_mac` seguem
intactas, e as Global Constraints 1–7 valem sem exceção. Duas restrições
próprias desta fase:

8. **`last_potential` é atributo, não valor de retorno.** O plano de migração
   RTCF/quadtree já altera a assinatura de `solve_patch` (2 → 3 valores) em outro
   worktree. Nenhum passo desta fase pode alterar essa assinatura.
9. **A Task 6 não pode depender da Task 7.** É ela que arbitra a Task 7; precisa
   passar nos próprios testes sem que nada da Task 7 exista.

| Arquivo | Responsabilidade | Estado |
|---|---|---|
| `mprm/benchmarks/hh.py` | `hh_ortho_defect`, `hh_assemble_psi`, `hh_ortho_skeleton`; chaves novas em `hh_benchmark_mode` | Modificar |
| `mprm/downscaler.py` | `keep_potential`, `last_potential`, `_last_p` | Modificar |
| `mprm/flow.py` | Popular e reunir `last_potential` (opt-in) | Modificar |
| `mprm/benchmarks/convergence.py` | `ortho_cos` em cada `case`; `rate_ortho` | Modificar |
| `tests/test_ortho_defect.py` | **Novo** — E13 | Criar |
| `tests/test_convergence_subspace.py` | `--ortho` nas curvas e na varredura de α | Estender (Task 3) |
| `docs/13-decomposicao-por-subespaco.md` | Seção do defeito de ortogonalidade | Estender (Task 5) |

---

### Task 6: O defeito pelo volume

**Files:**
- Modify: `mprm/benchmarks/hh.py`
- Test: `tests/test_ortho_defect.py` (criar)

**Interfaces:**
- Consome: `_l2_mac(a1, a2, h) -> float` e `hh_benchmark_mode(...) -> dict` (Task 2); `hh_build_w_mode(N, mode)` (Task 1); `FineProjector(Nx, Ny, hx, hy, dt)` de `mprm/flow.py`, cujo `.step(w1, w2) -> (u1, u2)` projeta na malha fina global (um só subdomínio, sem esqueleto).
- Produz: `_dot_mac(a1, a2, b1, b2, h) -> float`; `hh_ortho_defect(uh1, uh2, vt1, vt2, h) -> (ortho_abs, ortho_cos)`; e quatro chaves novas em `hh_benchmark_mode`: `ortho_abs` (float), `ortho_cos` (float ou `None`), `ortho_skel` (`None` até a Task 7) e `ortho_identity_rel` (`None` até a Task 7). Consumidas pelas Tasks 7 e 8.

**Por que o cosseno.** $\langle\bm u_h,\tilde{\bm v}_h\rangle$ escala com o
quadrado da amplitude da entrada; o cosseno não escala com nada. É a única
métrica deste plano inteiro que é adimensional **por normalização**, e não por
uma razão entre erro e referência. No modo `gradient`, em que `err_u` é `None`
por falta de denominador, `ortho_cos` continua definido.

- [ ] **Step 1: Escrever o teste que falha**

```python
# tests/test_ortho_defect.py
"""E13 — defeito de ortogonalidade da decomposição multiescala.

Hipótese (sec:erro-decomposicao da dissertação): as duas parcelas da
decomposição de Helmholtz-Hodge multiescala NÃO são ortogonais, e o resíduo
é a integral, sobre o esqueleto, do salto do produto entre pressão e fluxo
normal. Critério: o defeito medido no pipeline multiescala é ordens de
grandeza maior que o da projeção em malha fina, que não tem esqueleto.
"""
import numpy as np
import pytest

from mprm.benchmarks.hh import (hh_build_w_mode, hh_ortho_defect,
                                hh_benchmark_mode)


def test_cosseno_pequeno_para_o_par_analitico():
    """u de Olshanskii e grad p̃ são ortogonais no contínuo.

    Verifica o INSTRUMENTO, não o método: o cosseno discreto do par exato
    deve ser pequeno e cair com o refino (erro de quadratura, O(h²)).
    """
    cos = {}
    for N in (24, 48, 96):
        _, _, u1, u2 = hh_build_w_mode(N, mode="solenoidal")
        g1, g2, _, _ = hh_build_w_mode(N, mode="gradient")
        _, c = hh_ortho_defect(u1, u2, g1, g2, 1.0 / N)
        cos[N] = abs(c)
    assert cos[96] < 1e-2, f"piso de quadratura alto demais: {cos[96]:.2e}"
    assert cos[96] < cos[24], f"não caiu com o refino: {cos}"


def test_cosseno_imune_a_amplitude():
    """A propriedade que torna o cosseno superior a err_u e a leak_ratio."""
    _, _, u1, u2 = hh_build_w_mode(32, mode="solenoidal")
    g1, g2, _, _ = hh_build_w_mode(32, mode="gradient")
    _, c1 = hh_ortho_defect(u1, u2, g1, g2, 1.0 / 32)
    _, c2 = hh_ortho_defect(1e6 * u1, 1e6 * u2, 1e-3 * g1, 1e-3 * g2, 1.0 / 32)
    assert np.isclose(c1, c2, rtol=1e-10)


def test_defeito_e_maior_no_multiescala_que_na_malha_fina():
    """A hipótese central: o defeito vem do esqueleto.

    A malha fina global é um único subdomínio — não tem esqueleto, logo
    eq:ErroOrtogonalidade não tem sobre o que integrar e só resta o piso de
    quadratura. O multiescala com Ndom=3 tem 4 interfaces internas.
    """
    from mprm.flow import FineProjector

    Ndom, nloc = 3, 16
    N, h = Ndom * nloc, 1.0 / (Ndom * nloc)
    w1, w2, _, _ = hh_build_w_mode(N, mode="mixed")

    uf1, uf2 = FineProjector(N, N, h, h, 1.0).step(w1, w2)
    _, cos_fino = hh_ortho_defect(uf1, uf2, w1 - uf1, w2 - uf2, h)

    r = hh_benchmark_mode(Ndom=Ndom, nloc=nloc, alpha=1.0, mode="mixed")
    cos_ms = r["ortho_cos"]

    assert abs(cos_ms) > 10.0 * abs(cos_fino), (
        f"defeito multiescala não excede o da malha fina: "
        f"{cos_ms:.3e} vs {cos_fino:.3e}")


def test_cosseno_definido_no_modo_gradient():
    """err_u é None por falta de denominador; ortho_cos não é."""
    r = hh_benchmark_mode(Ndom=3, nloc=16, alpha=1.0, mode="gradient")
    assert r["err_u"] is None
    assert r["ortho_cos"] is not None
    assert abs(r["ortho_cos"]) <= 1.0, "cosseno fora de [-1, 1]"


def test_chaves_do_esqueleto_ainda_nao_existem():
    """Constraint 9: a Task 6 é auto-suficiente. Remover na Task 7."""
    r = hh_benchmark_mode(Ndom=2, nloc=8, alpha=1.0, mode="mixed")
    assert r["ortho_skel"] is None
    assert r["ortho_identity_rel"] is None
```

- [ ] **Step 2: Rodar e confirmar que falha**

```bash
cd ~/mrcm/NavierStokes && source ~/venv-firedrake/bin/activate
python -m pytest tests/test_ortho_defect.py -v
```

Esperado: `ImportError: cannot import name 'hh_ortho_defect'`.

- [ ] **Step 3: Implementar**

Em `mprm/benchmarks/hh.py`, junto de `_l2_mac` (criado na Task 2):

```python
def _dot_mac(a1, a2, b1, b2, h):
    """Produto interno L² discreto na malha MAC, peso uniforme h².

    O peso de borda é imaterial no arranjo impermeável: ali u·n = 0 e
    portanto ṽ·n = 0. Deixa de ser imaterial com saída (outflow).
    """
    return float(h * h * (np.sum(a1 * b1) + np.sum(a2 * b2)))


def hh_ortho_defect(uh1, uh2, vt1, vt2, h):
    """Lado do VOLUME de eq:ErroOrtogonalidade.

    Devolve (ortho_abs, ortho_cos):
      ortho_abs = <u_h, ṽ_h>                        — com sinal
      ortho_cos = <u_h, ṽ_h> / (‖u_h‖ ‖ṽ_h‖)        — adimensional

    Zero no contínuo (Teorema HH). Não requer solução exata: é uma
    inconsistência interna da solução calculada.
    """
    dot = _dot_mac(uh1, uh2, vt1, vt2, h)
    nu = _l2_mac(uh1, uh2, h)
    nv = _l2_mac(vt1, vt2, h)
    cos = None if (nu == 0.0 or nv == 0.0) else dot / (nu * nv)
    return dot, cos
```

Em `hh_benchmark_mode`, depois de obter `uh1, uh2` e antes do `return`:

```python
    vt1, vt2 = w1 - uh1, w2 - uh2
    ortho_abs, ortho_cos = hh_ortho_defect(uh1, uh2, vt1, vt2, h)
```

e acrescentar ao dicionário devolvido:

```python
        "ortho_abs": ortho_abs,
        "ortho_cos": ortho_cos,
        "ortho_skel": None,           # Task 7
        "ortho_identity_rel": None,   # Task 7
```

- [ ] **Step 4: Rodar os testes**

```bash
python -m pytest tests/test_ortho_defect.py -v
python -m pytest tests/test_subspace_leakage.py tests/test_subspace_fields.py -q
python -m pytest tests/test_convergence_hh.py -q
```

**Instrução a quem executa.** O terceiro teste é a hipótese central da fase, não
um requisito. Se o defeito multiescala **não** exceder o da malha fina, isso
contradiz `sec:erro-decomposicao` e é achado a reportar: marque
`@pytest.mark.xfail(strict=True)` com justificativa e **registre os dois números
medidos no relatório da task**. Não relaxe o fator 10.

- [ ] **Step 5: Commit**

```bash
git add mprm/benchmarks/hh.py tests/test_ortho_defect.py
git commit -m "feat(hh): defeito de ortogonalidade pelo volume

Mede <u_h, ṽ_h>, que o Teorema HH anula no contínuo e a decomposição
multiescala não anula. É a quantidade que sec:erro-decomposicao identifica
como sede do erro e que o capítulo de resultados promete medir diretamente.

Normalizado pelo produto das normas, é o cosseno do ângulo entre as duas
parcelas: adimensional e imune à amplitude, ao contrário de err_u e de
leak_ratio. Definido também no modo gradient, em que err_u não é.

Não requer solução exata — é inconsistência interna da solução calculada."
```

---

### Task 7: O defeito pelo esqueleto, e a identidade

**Files:**
- Modify: `mprm/downscaler.py` (`__init__` e final de `solve_patch`)
- Modify: `mprm/flow.py` (`mprm_projection_step`)
- Modify: `mprm/benchmarks/hh.py`
- Test: `tests/test_ortho_defect.py` (acrescentar)

**Interfaces:**
- Consome: `hh_ortho_defect` (Task 6); `MACDownscaler.solve_patch(divw_loc, fW, fE, fS, fN, solid_loc=None) -> (vt1, vt2)` — assinatura **inalterada**, ver Constraint 8; `MPrMProjector.last_pT: dict[int, Function]`, preenchido em `projector.py:279` só para os subdomínios do rank.
- Produz: `MACDownscaler.keep_potential: bool` (default `False`), `MACDownscaler.last_potential: dict[int, ndarray(n, m)]`; `hh_assemble_psi(downscaler, projector, Ndom, nloc) -> ndarray(N, N)`; `hh_ortho_skeleton(uh1, uh2, psi, Ndom, nloc, h, dt) -> float`; e as chaves `ortho_skel` e `ortho_identity_rel` deixam de ser `None`. Consumidas pela Task 8.

**A derivação, para quem for implementar.** Com $\tilde{\bm v}=\delta t\,\nabla p$
(`mprm/downscaler.py:143-146`), Green aplicado subdomínio a subdomínio dá

$$\int_\Omega \bm u_h\cdot\tilde{\bm v}_h
= \delta t\sum_i\Big[-\int_{\Omega_i}(\nabla\cdot\bm u_h)\,p_i + \int_{\partial\Omega_i}(\bm u_h\cdot\hat{\bm n}_i)\,p_i\Big].$$

O termo de volume morre porque a regularização impõe $\nabla\cdot\bm u_h=0$ em
cada $\Omega_i$. Em $\partial\Omega$ vale $\bm u_h\cdot\hat{\bm n}=0$. Sobra o
esqueleto, onde cada face é percorrida duas vezes com normais opostas. Numa
interface vertical no índice global de face $I$, com $\hat{\bm n}_i=+\hat x$ para
o subdomínio da esquerda, a contribuição é

$$\delta t\;u_1[I,j]\,\big(p[I-1,j]-p[I,j]\big)\,h.$$

**O problema da constante.** O potencial é pinado na célula 0 **de cada patch**
(`downscaler.py:53` e `:92`), e constantes arbitrárias por subdomínio destruiriam
o salto. A âncora é a média de `projector.last_pT[isub]`, a pressão do MRCM, que
é globalmente consistente a menos de **uma** constante única — e essa constante
única o salto não vê.

**Este é o caveat central da task, e ele precisa aparecer no relatório.** O
potencial do downscaler é pós-regularização; `last_pT` é pré-regularização. Eles
diferem por mais do que uma constante. Se a identidade não fechar, a Task 7 pode
estar medindo o erro da ancoragem em vez do defeito — e é por isso que a Task 6
existe primeiro e é independente: ela não tem ambiguidade de constante nenhuma.

- [ ] **Step 1: Escrever os testes que falham**

Acrescentar a `tests/test_ortho_defect.py`, e **remover**
`test_chaves_do_esqueleto_ainda_nao_existem`, que a Task 6 deixou justamente para
ser removido aqui:

```python
def test_last_potential_e_opt_in():
    """O caminho quente (von Kármán, cavidade) não paga pelo potencial."""
    from mrcm.config import MRCMConfig
    from mprm.downscaler import MACDownscaler

    cfg = MRCMConfig(Ndomx=2, Ndomy=2, nlocx=8, nlocy=8,
                     degreeU=1, degreeP=1, alpha=1.0,
                     mixed_family="BDM", mixed_order=0,
                     executor_kind="sequential")
    d = MACDownscaler(cfg, 1.0)
    assert d.keep_potential is False
    assert d.last_potential == {}


def test_psi_invariante_a_constante_global():
    """O salto não vê a constante global remanescente da ancoragem."""
    from mprm.benchmarks.hh import hh_ortho_skeleton

    Ndom, nloc = 3, 16
    N, h = Ndom * nloc, 1.0 / (Ndom * nloc)
    rng = np.random.default_rng(0)
    uh1 = rng.standard_normal((N + 1, N))
    uh2 = rng.standard_normal((N, N + 1))
    psi = rng.standard_normal((N, N))
    a = hh_ortho_skeleton(uh1, uh2, psi, Ndom, nloc, h, 1.0)
    b = hh_ortho_skeleton(uh1, uh2, psi + 7.25, Ndom, nloc, h, 1.0)
    assert np.isclose(a, b, rtol=1e-12, atol=1e-12)


def test_esqueleto_nulo_sem_interfaces():
    """Ndom=1 não tem esqueleto: eq:ErroOrtogonalidade não tem suporte."""
    from mprm.benchmarks.hh import hh_ortho_skeleton

    nloc = 16
    rng = np.random.default_rng(1)
    uh1 = rng.standard_normal((nloc + 1, nloc))
    uh2 = rng.standard_normal((nloc, nloc + 1))
    psi = rng.standard_normal((nloc, nloc))
    assert hh_ortho_skeleton(uh1, uh2, psi, 1, nloc, 1.0 / nloc, 1.0) == 0.0


def test_identidade_volume_esqueleto():
    """H2 — as duas leituras de eq:ErroOrtogonalidade concordam.

    Tolerância inicial deliberadamente frouxa: o número medido é o
    resultado, não o critério. Ver a instrução do Step 4.
    """
    r = hh_benchmark_mode(Ndom=3, nloc=16, alpha=1.0, mode="mixed")
    assert r["ortho_skel"] is not None
    assert r["ortho_identity_rel"] < 0.2, (
        f"volume={r['ortho_abs']:.6e}  esqueleto={r['ortho_skel']:.6e}  "
        f"rel={r['ortho_identity_rel']:.3e}")


def test_identidade_melhora_com_refino():
    """Se a identidade é discreta, o resíduo cai com nloc."""
    lo = hh_benchmark_mode(Ndom=3, nloc=8, alpha=1.0, mode="mixed")
    hi = hh_benchmark_mode(Ndom=3, nloc=24, alpha=1.0, mode="mixed")
    assert hi["ortho_identity_rel"] < lo["ortho_identity_rel"], (
        f"{lo['ortho_identity_rel']:.3e} -> {hi['ortho_identity_rel']:.3e}")
```

- [ ] **Step 2: Rodar e confirmar que falha**

```bash
python -m pytest tests/test_ortho_defect.py -v
```

Esperado: `AttributeError: 'MACDownscaler' object has no attribute 'keep_potential'`.

- [ ] **Step 3: Expor o potencial**

Em `mprm/downscaler.py`, no fim de `__init__` (junto de `self.last_defect = 0.0`):

```python
        self.keep_potential = False   # opt-in: ver plano Fase 4, Task 7
        self.last_potential = {}      # isub -> p (n, m), preenchido em flow.py
        self._last_p = None           # potencial do ÚLTIMO solve_patch
```

Em `solve_patch`, imediatamente antes de `vt1 = np.empty((n + 1, m))`:

```python
        if self.keep_potential:
            self._last_p = p.copy()
```

**Não** alterar a assinatura nem o valor de retorno de `solve_patch` — o plano
RTCF/quadtree já mexe nisso em outro worktree (Constraint 8).

- [ ] **Step 4: Reunir os potenciais em `mprm_projection_step`**

Em `mprm/flow.py`, antes do laço `for isub in projector.my_subs:` (linha ~48):

```python
    if downscaler.keep_potential:
        downscaler.last_potential = {}
        my_pot = {}
```

dentro do laço, logo após a chamada a `solve_patch`:

```python
        if downscaler.keep_potential:
            my_pot[isub] = downscaler._last_p
```

e depois do `allgather` já existente dos patches (linha ~72-77):

```python
    if downscaler.keep_potential:
        for d in comm.allgather(my_pot):
            downscaler.last_potential.update(d)
```

O `allgather` extra só ocorre com `keep_potential=True`. Com o default `False`,
`mprm_projection_step` executa exatamente o mesmo trabalho de antes.

- [ ] **Step 5: Implementar o lado do esqueleto**

Em `mprm/benchmarks/hh.py`:

```python
def hh_assemble_psi(downscaler, projector, Ndom, nloc):
    """Monta ψ_h global (N, N) a partir dos potenciais locais.

    A constante de cada subdomínio é ancorada na média da pressão do MRCM,
    que é globalmente consistente a menos de UMA constante — invisível ao
    salto. Ver o caveat da Task 7: os dois potenciais diferem por mais do
    que uma constante, e é a identidade com a Task 6 que arbitra isso.
    """
    N = Ndom * nloc
    anchors = {isub: float(np.mean(pT.dat.data_ro))
               for isub, pT in projector.last_pT.items()}
    for d in projector.comm.allgather(anchors):
        anchors.update(d)

    psi = np.zeros((N, N))
    for isub, p in downscaler.last_potential.items():
        ix, iy = isub % Ndom, isub // Ndom
        psi[ix * nloc:(ix + 1) * nloc, iy * nloc:(iy + 1) * nloc] = \
            p - float(p.mean()) + anchors[isub]
    return psi


def hh_ortho_skeleton(uh1, uh2, psi, Ndom, nloc, h, dt):
    """Lado do ESQUELETO de eq:ErroOrtogonalidade.

        δt ∫_Γ (u_h·n̂) [[ψ_h]]

    Com a regularização, u_h·n̂ é unívoco no esqueleto e sai do salto: o que
    resta é o fluxo normal vezes o SALTO DE PRESSÃO. A regularização conserta
    o fluxo, não a pressão — daí o defeito sobreviver a ela.
    """
    N = Ndom * nloc
    total = 0.0
    for I in range(nloc, N, nloc):          # interfaces verticais, n̂ = +x
        total += float(np.sum(uh1[I, :] * (psi[I - 1, :] - psi[I, :])))
    for J in range(nloc, N, nloc):          # interfaces horizontais, n̂ = +y
        total += float(np.sum(uh2[:, J] * (psi[:, J - 1] - psi[:, J])))
    return dt * h * total
```

Em `hh_benchmark_mode`, ligar o opt-in logo após construir o downscaler:

```python
    downscaler.keep_potential = True
```

e, depois do bloco da Task 6, substituir os dois `None`:

```python
    psi = hh_assemble_psi(downscaler, projector, Ndom, nloc)
    ortho_skel = hh_ortho_skeleton(uh1, uh2, psi, Ndom, nloc, h, dt)
    scale = max(abs(ortho_abs), abs(ortho_skel))
    ortho_rel = None if scale == 0.0 else abs(ortho_abs - ortho_skel) / scale
```

com as chaves `"ortho_skel": ortho_skel` e `"ortho_identity_rel": ortho_rel`.

- [ ] **Step 6: Rodar os testes**

```bash
python -m pytest tests/test_ortho_defect.py -v
python -m pytest tests/ -q
mpiexec -n 4 python -m pytest tests/test_ortho_defect.py -q
```

A execução em MPI importa aqui: `last_pT` só existe nos ranks donos, e
`hh_assemble_psi` depende do `allgather` das âncoras. Se o valor de
`ortho_skel` diferir entre 1 rank e 4 ranks, a reunião está errada.

**Instrução a quem executa.** `test_identidade_volume_esqueleto` é a hipótese
H2. Se falhar, **não relaxe a tolerância de 0.2**. Registre no relatório os três
números (`ortho_abs`, `ortho_skel`, `ortho_identity_rel`), marque
`@pytest.mark.xfail(strict=True)` com a justificativa, e diga explicitamente que
o suspeito primário é a ancoragem da constante (§2.3 do spec) e não a derivação
— porque a Task 6, que não depende de ancoragem nenhuma, já terá passado.

- [ ] **Step 7: Commit**

```bash
git add mprm/downscaler.py mprm/flow.py mprm/benchmarks/hh.py \
        tests/test_ortho_defect.py
git commit -m "feat(hh): defeito de ortogonalidade pelo esqueleto

Segunda leitura de eq:ErroOrtogonalidade: δt ∫_Γ (u_h·n̂)[[ψ_h]]. Com a
regularização o fluxo normal é unívoco e sai do salto, restando o fluxo
vezes o salto de PRESSÃO — que a regularização não conserta. Confrontada
com a leitura pelo volume, verifica no discreto a derivação do capítulo.

O potencial sai como atributo opt-in (keep_potential), e não como terceiro
valor de retorno de solve_patch, cuja assinatura o plano RTCF/quadtree já
altera em outro worktree. Com o default o caminho quente é o de antes.

A constante de cada subdomínio é ancorada na média de last_pT. Os dois
potenciais diferem por mais do que uma constante; a identidade com a
leitura pelo volume, que não depende de ancoragem, é o árbitro."
```

---

### Task 8: Propagar às varreduras e ao writeup

**Files:**
- Modify: `mprm/benchmarks/convergence.py`
- Modify: `tests/test_convergence_subspace.py` (criado na Task 3)
- Modify: `docs/13-decomposicao-por-subespaco.md` (criado na Task 5)

**Interfaces:**
- Consome: `hh_benchmark_mode` com as quatro chaves de ortogonalidade (Tasks 6 e 7); `_run_one_hh_level`, `_fit_rate(cases, x_key, err_key)` e as quatro `convergence_curve_*` de `convergence.py`, já com o parâmetro `mode` (Task 3).
- Produz: `ortho_cos` e `ortho_identity_rel` em cada `case`; `rate_ortho` ao lado de `rate_exact`, `rate_fine` e `rate_leak`; flag `--ortho` em `tests/test_convergence_subspace.py`.

**As duas hipóteses que esta task decide.**

**H1 — o defeito é da escala $H$, não da escala $h$.** É a previsão central e é
falseável. Refinar $h$ com $H$ fixo (o modo de saturação, `convergence_curve_h`)
deve fazer `ortho_cos` **saturar num platô não nulo**; refinar $H$
(`convergence_curve_H`) ou enriquecer a interface (`convergence_curve_p`) deve
derrubá-lo. Se `ortho_cos` cair com $h$, o erro é de discretização e não da
decomposição, e isso contradiz `sec:erro-decomposicao`.

**H3 — enriquecer a interface reduz o defeito.** É a mesma previsão que a Task 2
testa para `leak_ratio`, aplicada agora à quantidade que a tese identifica como a
sede do erro. Decidida por `convergence_curve_p`, varrendo
`degreeU = degreeP ∈ {1, 2, 3}` com malha fixa: `rate_ortho` deve ser negativa.

**H4 — o $\alpha$ ótimo não é o mesmo das outras métricas.** Após a
regularização, o defeito é uma integral do **salto de pressão**, e $\alpha$
controla exatamente o balanço entre continuidade de pressão ($\alpha\to0$) e de
fluxo ($\alpha\to\infty$). A previsão é que $\alpha$ pequeno reduza `ortho_cos` e
que isso **conflite** com o $\alpha$ que minimiza `err_u`. A Task 4 já varre
$\alpha \in \{10^{-3},10^{-2},0.1,1,10,10^2,10^3\}$; esta task só acrescenta a
coluna e a comparação dos mínimos:

```python
def _argmin(cases, key):
    valid = [c for c in cases if c.get(key) is not None]
    return min(valid, key=lambda c: abs(c[key]))["alpha"] if valid else None

a_err, a_leak = _argmin(cases, "err_vs_exact"), _argmin(cases, "leak_ratio")
a_ortho = _argmin(cases, "ortho_cos")
print(f"  alpha* : err_u={a_err}  leak={a_leak}  ortho={a_ortho}  "
      f"coincidem={a_err == a_ortho}", flush=True)
```

- [ ] **Step 1: Carregar as métricas nos casos**

Em `mprm/benchmarks/convergence.py`, dentro de `_run_one_hh_level`. **Leia a
função antes de editar**: hoje ela não chama `hh_benchmark_mode` — monta o
próprio `cfg`, o próprio `MPrMProjector`/`MACDownscaler` e chama
`mprm_projection_step` diretamente (`convergence.py:55-80`), porque também
precisa da referência em malha fina (`FineProjector`) para `err_vs_fine`. A Task
3 pode ou não tê-la feito delegar. Os dois caminhos:

**Se a Task 3 fez `_run_one_hh_level` delegar a `hh_benchmark_mode`**, basta ler
as chaves do dicionário devolvido (`res`):

```python
        "ortho_cos": res.get("ortho_cos"),
        "ortho_identity_rel": res.get("ortho_identity_rel"),
```

`.get` e não indexação, para tolerar um `res` sem as chaves.

**Se ela continua montando o pipeline inline**, calcular ali mesmo — é o caminho
que não exige refatoração:

```python
    from mprm.benchmarks.hh import hh_ortho_defect, hh_assemble_psi, hh_ortho_skeleton

    downscaler.keep_potential = True     # antes de mprm_projection_step
    ...
    ortho_abs, ortho_cos = hh_ortho_defect(uh1, uh2, w1 - uh1, w2 - uh2, h)
    psi = hh_assemble_psi(downscaler, projector, Ndom, nloc)
    ortho_skel = hh_ortho_skeleton(uh1, uh2, psi, Ndom, nloc, h, dt)
    scale = max(abs(ortho_abs), abs(ortho_skel))
    ortho_rel = None if scale == 0.0 else abs(ortho_abs - ortho_skel) / scale
```

e acrescentar `"ortho_cos": ortho_cos` e `"ortho_identity_rel": ortho_rel` ao
dicionário do caso. Registre no relatório qual dos dois caminhos encontrou.

- [ ] **Step 2: Ajustar o ajuste de taxa**

`_fit_rate` faz `errs = [c[err_key] for c in ordered]` e passa a
`compute_convergence_rate`, que quebra com `None`. Filtrar antes:

```python
    ordered = [c for c in sorted(cases, key=lambda c: -c[x_key])
               if c.get(err_key) is not None]
    if len(ordered) < 2:
        return None
```

Isso serve a `rate_ortho` e resolve, de uma vez, o `err_vs_exact = None` do modo
`gradient` que a Task 3 Step 2 já apontava. Acrescentar
`rate_ortho = _fit_rate(cases, x_key, "ortho_cos")` ao lado das taxas
existentes, em cada uma das quatro curvas.

- [ ] **Step 3: Verificar que nada quebrou**

```bash
python -m pytest tests/test_convergence_hh.py -q
python tests/test_convergence_subspace.py --quick
```

- [ ] **Step 4: Acrescentar a coluna às curvas e à varredura de α**

Em `tests/test_convergence_subspace.py`: flag `--ortho` (default ligado), coluna
`ortho_cos` nas tabelas impressas e no JSON, e um eixo secundário na figura. Na
varredura de $\alpha$ da Task 4, reportar **dois** mínimos — o $\alpha$ que
minimiza `leak_ratio` e o que minimiza `ortho_cos` — e registrar se coincidem.

O envelope autodescritivo (Constraint 5) ganha as hipóteses novas:

```python
    "hypothesis": {
        "H1": "ortho_cos satura em platô não nulo sob refino de h com H fixo",
        "H3": "enriquecer a interface (degreeU=degreeP) reduz ortho_cos",
        "H4": "o alpha que minimiza ortho_cos difere do que minimiza err_u",
    },
    "criterion": {
        "H1": "|d log ortho_cos / d log h| < 0.2 no modo de saturação",
        "H3": "rate_ortho < 0 na curva p",
        "H4": "argmin_alpha ortho_cos != argmin_alpha err_u",
    },
```

- [ ] **Step 5: Rodar completo**

```bash
mpiexec -n 4 python tests/test_convergence_subspace.py \
    --modes gradient,solenoidal,mixed --Ndoms 2,4,8,16 --ortho
mpiexec -n 4 python tests/test_convergence_subspace.py \
    --alphas 1e-3,1e-2,0.1,1,10,100,1e3 --ortho
mpiexec -n 4 python tests/test_convergence_subspace.py \
    --degrees 1,2,3 --ortho
```

Preencher `verdict` para H1, H3 e H4 com o número medido. **Todos os vereditos
são aceitáveis**: hipótese refutada é resultado, não falha.

- [ ] **Step 6: Estender o writeup**

Acrescentar a `docs/13-decomposicao-por-subespaco.md` uma seção
"O defeito de ortogonalidade", com:

1. A identidade `eq:ErroOrtogonalidade`, as duas leituras, e por que o cosseno é
   preferível a `err_u` e a `leak_ratio` — auto-contido, adimensional por
   normalização, definido também no modo `gradient`.
2. A tabela de vereditos H1–H4 com os valores medidos, integrada à tabela que a
   Task 5 já criou.
3. O caveat da ancoragem da constante (§2.3 do spec) e o que a identidade
   H2 disse sobre ele.
4. **A limitação declarada:** o contraste com/sem regularização não foi medido.
   `sec:erro-decomposicao` afirma que a regularização recupera a divergência nula
   e não a ortogonalidade; esta fase mede apenas o lado "com regularização".
   Medir o outro exigiria reconstruir `projector.last_uT` (BDM em triângulos) nas
   faces MAC, e o erro dessa interpolação entraria no próprio instrumento de
   medida. Registrar como trabalho seguinte, **com a razão** — sem isso, uma
   leitura futura vai supor que a afirmação foi verificada inteira.
5. Que o instrumento é auto-contido e portanto se aplica à cavidade, ao obstáculo
   e à decomposição por *quadtree*, que não têm solução manufaturada. Fora do
   escopo deste plano, que segue o caminho do Olshanskii.

- [ ] **Step 7: Commit**

```bash
git add mprm/benchmarks/convergence.py tests/test_convergence_subspace.py \
        docs/13-decomposicao-por-subespaco.md
git commit -m "feat(convergence): defeito de ortogonalidade nas varreduras

ortho_cos entra nas quatro curvas e na varredura de alpha, com rate_ortho.
_fit_rate passa a ignorar pontos None, o que também resolve err_vs_exact
ausente no modo gradient.

Decide H1 (o defeito é da escala H, não da escala h: satura sob refino de h
com H fixo) e H4 (o alpha que minimiza o defeito difere do que minimiza
err_u, porque o defeito é uma integral do salto de PRESSÃO e alpha controla
o balanço pressão/fluxo).

O writeup registra a limitação: o contraste com/sem regularização não foi
medido, e por quê."
```

---

## Critérios de aceitação

1. `python -m pytest tests/ -q` passa, incluindo os testes preexistentes de convergência.
2. `hh_build_w` e `hh_benchmark_mac` mantêm assinatura e resultado; `hh_benchmark_mac` passa a delegar a `hh_benchmark_mode`, sem código duplicado.
3. Existe medida separada do vazamento solenoidal e do irrotacional, com `leak_ratio` comparável entre modos.
4. A hipótese de que o erro se aloja na parcela irrotacional tem veredito registrado — confirmada ou refutada, ambas aceitáveis.
5. A linearidade do pipeline está verificada por teste, e a omissão da varredura de amplitude está documentada com a razão.
6. Todo JSON traz envelope autodescritivo e `git_commit`.

### Critérios de aceitação — Fase 4

7. O defeito de ortogonalidade `eq:ErroOrtogonalidade` está medido pelas **duas**
   leituras, volume e esqueleto, e a discordância entre elas está registrada com
   o número medido.
8. `solve_patch` mantém assinatura e valor de retorno (Constraint 8); o potencial
   sai por `keep_potential`/`last_potential`, e com o default `False` o caminho
   de `mprm_projection_step` é bit-a-bit o de antes.
9. `ortho_skel` coincide entre execução sequencial e `mpiexec -n 4`.
10. H1 e H4 têm veredito registrado, confirmado ou refutado.
11. O writeup declara explicitamente que o contraste com/sem regularização **não**
    foi medido, e a razão.

## O que fica de fora

| Excluído | Razão |
|---|---|
| Varredura de amplitude $m_{pr}$ | Vazia por linearidade do pipeline; passaria por construção |
| Polinômio de pressão de Carvalho | $\nabla p\cdot\bhat n \ne 0$ nas paredes horizontais; incompatível com o caminho impermeável sem tratamento adicional de contorno |
| Marcha temporal, cavidade, von Kármán | Este plano segue o caminho do Olshanskii, de disparo único |
| Newton/Picard para NS estacionário | Exigiria solver monolítico inexistente |
