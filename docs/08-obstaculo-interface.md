# Por que o obstáculo não pode cruzar interfaces de subdomínio?

> **Contexto:** MPrM (Multiscale Projection Method) com decomposição de domínio MRCM.
> **Branch:** `mprm-von-karman` | `mprm_core.py::assert_solid_interior_to_subdomains`

---

## 1. O problema em uma frase

O MRCM acopla subdomínios através de **fluxos normais nas interfaces**. Se o obstáculo cruza uma interface, as faces sólidas (velocidade zero) coincidem com faces de interface (onde o MRCM impõe continuidade de fluxo), criando um **conflito irresolúvel** entre a condição de contorno do sólido ($u = 0$) e a condição de acoplamento do MRCM ($u^+ \cdot n = u^- \cdot n$ com valor possivelmente não nulo).

---

## 2. Arquitetura do MRCM

O MRCM divide o domínio $\Omega$ em $N_{\text{dom}} \times N_{\text{dom}}$ subdomínios $\Omega_k$:

```
┌────────┬────────┬────────┬────────┐
│  Ω₀    │  Ω₁    │  Ω₂    │  Ω₃    │
│        │        │        │        │
│        │        │  ┌──┐  │        │
│        │        │  │██│  │        │  ← obstáculo em Ω₂
│        │        │  └──┘  │        │
│        │        │        │        │
├────────┼────────┼────────┼────────┤
│  Ω₄    │  Ω₅    │  Ω₆    │  Ω₇    │
│        │        │        │        │
```

Cada subdomínio $\Omega_k$ resolve um problema local com condição de Robin na interface:
$$
\mathbf{u}_k + \beta \, (-\nu \nabla \mathbf{u}_k \cdot \mathbf{n} + p_k \mathbf{n}) = \boldsymbol{\lambda} \quad \text{em } \Gamma_{kl}
$$

O acoplamento entre subdomínios é feito através dos **multiplicadores de interface** $\boldsymbol{\lambda}$ (espaço $U_H$), que impõem **continuidade fraca do fluxo normal** através das interfaces:
$$
\int_{\Gamma_{kl}} (\mathbf{u}_k - \mathbf{u}_l) \cdot \mathbf{n} \; \boldsymbol{\mu} \; ds = 0 \quad \forall \boldsymbol{\mu} \in U_H
$$

---

## 3. O que acontece quando o obstáculo cruza a interface

### 3.1 Caso problemático

```
        Interface Γ (x = 4.0)
              ↓
  ┌──────────┬──────────┐
  │  fluido  │  fluido  │  ← y = 4.0 (fluido)
  │  u ≠ 0   │  u ≠ 0   │
  ├──────────┼──────────┤
  │  fluido  │██ sólido │  ← y = 3.5 (cruzando!)
  │  u ≠ 0   │██ u = 0  │
  ├──────────┼──────────┤
  │  fluido  │██ sólido │  ← y = 2.5 (cruzando!)
  │  u ≠ 0   │██ u = 0  │
  ├──────────┼──────────┤
  │  fluido  │  fluido  │  ← y = 2.0 (fluido)
  │  u ≠ 0   │  u ≠ 0   │
  └──────────┴──────────┘
     Ω_left     Ω_right
```

Nesta situação, a interface vertical em $x = 4.0$ tem:
- **Parte superior e inferior**: faces entre fluido–fluido → condição de Robin normal (acoplamento MRCM)
- **Parte central** (altura do obstáculo): faces entre fluido (esquerda) e sólido (direita)

### 3.2 Os três conflitos

#### Conflito 1: Condições de contorno incompatíveis

| Local | Condição física | Condição MRCM |
|-------|----------------|---------------|
| Face fluido–fluido | Continuidade de fluxo | Robin → OK |
| Face fluido–sólido | $u = 0$ (no-slip) | Robin → **conflito!** |

O MRCM aplica a **mesma** condição de Robin em **todas** as faces de interface, sem distinguir entre fluido–fluido e fluido–sólido. Na face fluido–sólido, a condição correta é $u \cdot n = 0$ (parede), mas o MRCM tentaria impor continuidade de fluxo entre o fluido e o "sólido" (que na verdade é um subdomínio parcialmente sólido).

#### Conflito 2: Downscaling inconsistente

Após resolver o problema de interface global, o **downscaler** resolve problemas de Neumann locais em cada subdomínio. O subdomínio da direita ($\Omega_{\text{right}}$) contém células sólidas. O downscaler com máscara (`solid_loc`) trata essas células corretamente **dentro** do subdomínio. Mas as faces de interface do lado direito do subdomínio esquerdo ($\Omega_{\text{left}}$) recebem fluxos médios (Eq. 21 do artigo) que foram calculados **sem conhecimento da máscara de sólido** no subdomínio vizinho.

O fluxo médio na interface é:
$$
\bar{F}_{\Gamma} = \frac{1}{2} (F_{\text{left}} + F_{\text{right}})
$$

Se o lado direito tem células sólidas na interface, $F_{\text{right}}$ nessas faces deveria ser 0 (no-flow). Mas o traço de fluxo do MRCM no subdomínio direito **não sabe** que há um sólido ali — ele reconstrói o fluxo a partir dos multiplicadores $\boldsymbol{\lambda}$, que foram calculados assumindo fluido em todo o subdomínio.

#### Conflito 3: Projeção da fonte DG0

O projetor MRCM recebe $\nabla \cdot w^*$ como fonte DG0 (1 valor por célula). Células sólidas têm fonte nula. Mas a **matriz de interface** e os **operadores de traço** são construídos no estágio offline **antes** de saber quais células são sólidas. Se o sólido cruza a interface, o traço de fluxo naquela face é calculado a partir de DOFs que incluem contribuições de células sólidas (fonte nula) e fluidas (fonte não nula) no mesmo lado da interface, produzindo um valor inconsistente.

---

## 4. Visualização do problema

```
Caso CORRETO (obstáculo interior ao subdomínio):
┌──────────────┬──────────────┐
│              │              │
│   fluido     │   fluido     │
│              │              │
│   ┌──┐       │              │
│   │██│       │              │  ← obstáculo totalmente em Ω_left
│   └──┘       │              │
│              │              │
├──────────────┼──────────────┤  ← interface limpa (só fluido–fluido)
│              │              │
│   fluido     │   fluido     │

Caso PROIBIDO (obstáculo cruza interface):
┌──────────────┬──────────────┐
│              │              │
│   fluido     │   fluido     │
│              │              │
│        ┌──┼──┐             │  ← obstáculo cruza a interface!
│        │██│██│             │
│        └──┼──┘             │
│              │              │
├──────────────┼──────────────┤  ← interface contaminada (fluido–sólido)
│              │              │
│   fluido     │   fluido     │
```

---

## 5. A validação: `assert_solid_interior_to_subdomains`

```python
def assert_solid_interior_to_subdomains(solid, cfg):
    """Garante que o obstáculo não atravessa interfaces de subdomínio."""
    nx, ny = cfg.nlocx, cfg.nlocy
    xs = np.where(solid.any(axis=1))[0]
    ys = np.where(solid.any(axis=0))[0]
    imin, imax = xs.min(), xs.max() + 1       # [imin, imax) em células
    jmin, jmax = ys.min(), ys.max() + 1

    for i in range(1, cfg.Ndomx):
        if imin < i * nx < imax:              # interface i*nx dentro do bbox
            raise ValueError(...)
    for j in range(1, cfg.Ndomy):
        if jmin < j * ny < jmax:              # interface j*ny dentro do bbox
            raise ValueError(...)
```

O critério é simples: o **bounding box** do obstáculo (em índices de célula) não pode conter nenhuma interface interna ($i \cdot n_{\text{loc}}$ ou $j \cdot n_{\text{loc}}$).

---

## 6. Como evitar o problema

### 6.1 Ajustar a posição do obstáculo

Para um dado $N_{\text{dom}}$, escolha $(x_c, y_c)$ tal que:
$$
x_c + D/2 < \frac{L_x}{N_{\text{dom}}} \quad \text{(cabe no 1º subdomínio)}
$$

Ou, para caber no $k$-ésimo subdomínio:
$$
k \cdot \frac{L_x}{N_{\text{dom}}} < x_c - D/2 \quad \text{e} \quad x_c + D/2 < (k+1) \cdot \frac{L_x}{N_{\text{dom}}}
$$

### 6.2 Ajustar a decomposição

Para um obstáculo fixo em $(x_c, y_c)$, escolha $N_{\text{dom}}$ tal que as interfaces não caiam dentro de $[x_c - D/2, x_c + D/2]$:
$$
N_{\text{dom}} \notin \left( \frac{L_x}{x_c + D/2}, \; \frac{L_x}{x_c - D/2} \right)
$$

### 6.3 Tabela de compatibilidade

Para a geometria default ($L_x=16, L_y=8, D=1$):

| $N_{\text{dom}}$ | Interface em $x$ | Obstáculo $(x_c=3)$ cabe? | Obstáculo $(x_c=4)$ cabe? |
|-------------------|-------------------|----------------------------|----------------------------|
| 2 | $x=8$ | ✅ $[2.5, 3.5] \subset [0,8)$ | ✅ $[3.5, 4.5] \subset [0,8)$ |
| 3 | $x=5.33$ | ✅ $[2.5, 3.5] \subset [0,5.33)$ | ❌ $[3.5, 4.5]$ cruza $5.33$ |
| 4 | $x=4$ | ✅ $[2.5, 3.5] \subset [0,4)$ | ❌ $[3.5, 4.5]$ cruza $4$ |
| 5 | $x=3.2$ | ❌ $[2.5, 3.5]$ cruza $3.2$ | ❌ |
| 6 | $x=2.67$ | ❌ $[2.5, 3.5]$ cruza $2.67$ | ❌ |

> A geometria default usa $x_c=3.0$, compatível apenas com $N_{\text{dom}} \in \{2, 3, 4\}$.

---

## 7. Possíveis soluções futuras

1. **Máscara de interface**: modificar o MRCM para aceitar um `interface_solid_mask` que zera os multiplicadores $\boldsymbol{\lambda}$ nas faces de interface adjacentes ao sólido.

2. **Decomposição adaptativa**: posicionar as interfaces do MRCM de forma a **contornar** o obstáculo (malha não Cartesiana ou recorte dos subdomínios).

3. **Ghost-cell com flag**: tratar as faces de interface fluido–sólido como condição de contorno externa ($u=0$) em vez de condição de interface — essencialmente "furar" a topologia do MRCM onde há sólido.

4. **Pré-processamento automático**: `run_von_karman` poderia calcular automaticamente $N_{\text{dom}}$ e $n_{\text{loc}}$ compatíveis com a geometria, em vez de exigir que o usuário acerte manualmente.
