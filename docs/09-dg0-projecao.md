# Por que a projeção MPrM funciona apenas com DG0 (`mixed_order=0`)?

> **Contexto:** `MPrMProjector` em `mprm_core.py` | Assert: `cfg.mixed_order == 0`
> **Formulação:** BDM$_1$–DG$_0$ (mixed finite element, ordem mista 0)

---

## 1. O assert e o que ele protege

```python
assert cfg.mixed_order == 0, "fonte por célula requer DG0"
```

Este assert está no `MPrMProjector.__init__`, logo após o mapeamento das células finas:

```python
ic = np.floor(xcd[:, 0] / hx_f).astype(int)
jc = np.floor(xcd[:, 1] / hy_f).astype(int)
self.cell_map[isub] = (np.clip(ic, 0, cfg.Nglobx - 1),
                       np.clip(jc, 0, cfg.Ngloby - 1))
```

E o uso no `project()`:

```python
ic, jc = self.cell_map[isub]
self.src_fn[isub].dat.data[:] = divw[ic, jc]
```

O mapeamento assume que **cada DOF do espaço DG corresponde exatamente a uma célula da malha MAC**. Isso só é verdade para DG0.

---

## 2. O que é `mixed_order` no contexto do MRCM

O MRCM usa uma formulação mista para o problema de Darcy/projeção:

$$
\begin{aligned}
\mathbf{v} + K \nabla p &= 0 \quad &\text{(Darcy, } K = \delta t \text{)} \\
\nabla \cdot \mathbf{v} &= f \quad &\text{(fonte } f = \nabla \cdot w^* \text{)}
\end{aligned}
$$

A discretização usa o par de elementos finitos **BDM$_k$–DG$_{k-1}$** (Brezzi-Douglas-Marini de ordem $k$ para velocidade, Galerkin descontínuo de ordem $k-1$ para pressão). O parâmetro `mixed_order` controla a **ordem do espaço DG** para a pressão:

| `mixed_order` | Par de elementos | DOFs de velocidade por faceta | DOFs de pressão por célula |
|---------------|-------------------|------------------------------|----------------------------|
| 0 | BDM$_1$–DG$_0$ | 1 (momento $\int \mathbf{u} \cdot \mathbf{n}$) | 1 (valor constante) |
| 1 | BDM$_2$–DG$_1$ | 2 | 3 (linear em triângulo) |
| 2 | BDM$_3$–DG$_2$ | 3 | 6 (quadrático em triângulo) |

---

## 3. Por que DG0 é suficiente (e necessário) para o MPrM

### 3.1 A fonte é constante por célula MAC

No passo de projeção de Chorin, a fonte é $\nabla \cdot w^*$, calculada na malha MAC:
$$
f_{ij} = \frac{w^*_{1, i+1,j} - w^*_{1, i,j}}{h_x} + \frac{w^*_{2, i,j+1} - w^*_{2, i,j}}{h_y}
$$

Este é **um valor por célula** — uma função constante por partes na malha cartesiana. Representá-la num espaço DG0 (constante por elemento) é **exato**: a função constante por célula MAC é isomorfa à função constante por elemento da malha triangulada.

```
Malha MAC (cartesiana)          Malha Firedrake (triangular)
┌───┬───┬───┐                   ┌───╲───╱───┐
│f00│f01│f02│                   │ ╲ ▏╱ ╲ ▏ │
├───┼───┼───┤                   │  ╲▏  ╲▏  │
│f10│f11│f12│                   │  ╱▏  ╱▏  │
├───┼───┼───┤                   │ ╱ ▏╲ ╱ ▏ │
│f20│f21│f22│                   └───╲───╱───┘
└───┴───┴───┘                   
fonte: 1 valor / célula         DG0: 1 DOF / elemento ✓
```

### 3.2 O mapeamento é 1:1

Com DG0, cada triângulo da malha Firedrake tem exatamente 1 DOF de pressão (o valor constante no elemento). O mapa `cell_map` associa cada triângulo ao seu índice de célula MAC correspondente, e a atribuição é direta:

```python
# DG0: self.src_fn[isub].dat.data tem tamanho = número de triângulos
# divw[ic, jc] tem tamanho = número de triângulos
self.src_fn[isub].dat.data[:] = divw[ic, jc]  # ✓ 1:1
```

### 3.3 A base constante ($\pi_0$) é essencial para o balanço de massa

O espaço de interface $P_H$ (pressão no esqueleto) e o downscaler dependem de que a **constante** esteja presente na base local. DG0 garante que $\pi_0$ (função constante) está no espaço de pressão local, o que é necessário para:

1. **Compatibilidade do Neumann**: o problema de Neumann local (Eq. 22) tem um modo constante no núcleo. A projeção da fonte no espaço DG0 garante que o modo constante seja tratado corretamente.

2. **Balanço de fluxos**: a média dos fluxos de interface (Eq. 21) usa a propriedade de que $\int_{\Gamma} \mathbf{v} \cdot \mathbf{n} = \int_{\Omega} f$ para cada subdomínio. Com DG0, essa identidade é exata no nível discreto.

---

## 4. O que quebraria com `mixed_order > 0`

### 4.1 Mapeamento fonte → DOFs

Com DG1 (3 DOFs por triângulo), o mapeamento `divw[ic, jc]` → `src_fn.dat.data[:]` não funciona:

```python
# DG1: 3 DOFs por triângulo (valor nos vértices + modos de borda)
# Mas divw[ic, jc] tem 1 valor por célula MAC
# self.src_fn[isub].dat.data tem 3× mais entradas que divw[ic, jc]
self.src_fn[isub].dat.data[:] = divw[ic, jc]  # ✗ shape mismatch!
```

Seria necessário **projetar** a função constante por célula MAC no espaço DG1, o que envolveria resolver um sistema de massa local:
$$
M_k \, \mathbf{f}_k = \int_{\Omega_k} f_{\text{MAC}} \, q_i \, dx
$$
onde $q_i$ são as funções de base DG1 e $M_k$ é a matriz de massa local.

### 4.2 Modos espúrios no downscaler

O downscaler resolve um problema de Neumann local (FV de 5 pontos) na malha MAC. A solução desse problema é uma pressão $p$ definida nos centros das células. Com DG0, a projeção é trivial (injeção). Com DG1, os DOFs incluem modos lineares que não têm correspondência direta na malha MAC — a reconstrução do fluxo $\tilde{\mathbf{v}} = \delta t \nabla p$ usaria gradientes de funções lineares por elemento, introduzindo inconsistências com o estêncil FV de 5 pontos.

### 4.3 Operadores de traço de interface

Os operadores de traço que extraem o fluxo normal nas faces de interface (`HDiv Trace`, grau 0) assumem 1 DOF por faceta (o momento $\int \mathbf{u} \cdot \mathbf{n}$). Isso é compatível com BDM$_1$ (1 DOF de fluxo por aresta). Com BDM$_k$ ($k > 1$), o espaço de traço teria $k$ DOFs por faceta, e a média de fluxos (Eq. 21) precisaria ser generalizada.

### 4.4 Custo-benefício

Para o problema de projeção ($K = \delta t$ constante e homogêneo), a solução já é essencialmente um problema de Poisson. O par BDM$_1$–DG$_0$ é **estável e ótimo** para este problema. Enriquecer o espaço DG **não melhora a precisão** porque:

1. A fonte já é constante por célula (não há sub-escala para capturar)
2. O erro da decomposição HH é dominado pelos espaços de **interface** ($U_H$, $P_H$), não pelo espaço local
3. O downscaler opera na malha MAC (FV), não na malha de elementos finitos

---

## 5. Diagrama: fluxo da fonte no MPrM

```
∇·w* (MAC, Nx×Ny)
   │
   │  mapeamento 1:1 (DG0) ou projeção (DGk, k>0)
   ▼
src_fn (DG, 1 DOF/tri para DG0)
   │
   │  LocalSolver: resolve (u_h, p_h) com fonte f_h
   ▼
u_h (BDM₁), p_h (DG₀)
   │
   │  Reconstruct: traço de fluxo nas interfaces
   ▼
fluxes (média Eq. 21)
   │
   │  Downscaler: Neumann local FV (5 pontos)
   ▼
ṽ (MAC) → uⁿ⁺¹ = w* − ṽ
```

A seta crítica é o **mapeamento 1:1** entre `∇·w*` (MAC) e `src_fn` (DG). Este mapeamento só é trivial com DG0.

---

## 6. O que seria necessário para suportar `mixed_order > 0`

Caso se desejasse usar DG1 ou DG2, as seguintes mudanças seriam necessárias:

| Componente | Mudança |
|------------|---------|
| `cell_map` | Mapear $N_x \times N_y$ células → $k \times N_{\text{tri}}$ DOFs (projeção $L^2$) |
| `src_fn` | Substituir atribuição direta por projeção: resolver $M_k \mathbf{f} = \int f_{\text{MAC}} \phi_i$ |
| `solve_patch` | Generalizar reconstrução do fluxo para incorporar graus superiores do DG |
| Operadores de traço | Adaptar `HDiv Trace` para grau $k$ em vez de grau 0 |
| Assert | Remover ou condicionar ao `mixed_order` |
| Testes | Validar conservação e balanço de massa com DG1/DG2 |

---

## 7. Conclusão

DG0 (`mixed_order=0`) não é uma limitação acidental — é uma **escolha de design** que explora o fato de que a fonte do MPrM ($\nabla \cdot w^*$) é naturalmente constante por célula na malha MAC. O mapeamento 1:1 entre células MAC e DOFs DG0 torna a implementação simples, exata e eficiente. Enriquecer o espaço DG local **não traria benefícios de precisão** para o problema de projeção, apenas complexidade desnecessária. O ganho de precisão no MPrM vem do enriquecimento dos **espaços de interface** ($U_H$, $P_H$) e do **refinamento da malha** ($n_{\text{loc}}$), não da ordem do espaço DG local.
