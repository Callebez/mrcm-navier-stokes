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

## Critérios de aceitação

1. `python -m pytest tests/ -q` passa, incluindo os testes preexistentes de convergência.
2. `hh_build_w` e `hh_benchmark_mac` mantêm assinatura e resultado; `hh_benchmark_mac` passa a delegar a `hh_benchmark_mode`, sem código duplicado.
3. Existe medida separada do vazamento solenoidal e do irrotacional, com `leak_ratio` comparável entre modos.
4. A hipótese de que o erro se aloja na parcela irrotacional tem veredito registrado — confirmada ou refutada, ambas aceitáveis.
5. A linearidade do pipeline está verificada por teste, e a omissão da varredura de amplitude está documentada com a razão.
6. Todo JSON traz envelope autodescritivo e `git_commit`.

## O que fica de fora

| Excluído | Razão |
|---|---|
| Varredura de amplitude $m_{pr}$ | Vazia por linearidade do pipeline; passaria por construção |
| Polinômio de pressão de Carvalho | $\nabla p\cdot\bhat n \ne 0$ nas paredes horizontais; incompatível com o caminho impermeável sem tratamento adicional de contorno |
| Marcha temporal, cavidade, von Kármán | Este plano segue o caminho do Olshanskii, de disparo único |
| Newton/Picard para NS estacionário | Exigiria solver monolítico inexistente |
