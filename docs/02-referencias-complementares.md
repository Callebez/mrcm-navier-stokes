# Referências complementares — artigos novos em `~/Dissertação/Refs` (2026-07-19)

Além do artigo principal (MPrM), três artigos novos foram adicionados e
convertidos para markdown. Todos são referências citadas no MPrM e dão o
contexto das abordagens alternativas (homogeneização, estabilização e ROM).

## 1. Abdulle & Budáč (2015) — *An adaptive finite element heterogeneous multiscale method for Stokes flow in porous media*
**SIAM Multiscale Model. Simul. 13(1):256–290.** — ref. [1] do MPrM.

- FE-HMM para Stokes em meios porosos: acopla uma **equação de Darcy efetiva**
  na malha macro com **permeabilidades recuperadas de micro-problemas de
  Stokes** resolvidos em domínios de amostragem centrados nos pontos de
  quadratura de cada elemento macro.
- Custo computacional independente do tamanho ε da estrutura de poros;
  geometria microscópica não periódica (deformação suave de um domínio de
  referência).
- Estimativas a priori limitadas pela regularidade dos micro-problemas de
  Stokes (baixa para poros não convexos) ⇒ esquema **adaptativo** com
  refinamento micro-macro guiado por indicadores residuais; análise a
  posteriori; experimentos 2D e 3D.
- **Relação com o MPrM:** exemplo da via "homogeneização" para multiescala em
  fluidos, em contraste com a via "decomposição de domínio" (MRCM) adotada
  pelo MPrM.

## 2. Araya, Barrenechea & Valentin (2006) — *Stabilized finite element methods based on multiscale enrichment for the Stokes problem*
**SIAM J. Numer. Anal. 44(1):322–348.** — ref. [3] do MPrM.

- Métodos estabilizados para Stokes com pares de interpolação que violam a
  condição inf-sup (P1/P1 contínuo e P1/P0): o espaço de velocidade é
  **enriquecido com funções multiescala** que não se anulam no bordo dos
  elementos; espaço de teste enriquecido com funções tipo bolha
  (Petrov-Galerkin).
- Condensação das bolhas em nível de elemento produz o método estabilizado
  (Galerkin + termos extras); convergência de ordem ótima demonstrada.
- **Relação com o MPrM:** linha dos métodos multiescala *dentro do elemento*
  (enriquecimento) para as equações de fluido, precursora do MHM de Araya et
  al. (2013) — um dos limites do MRCM (α → ∞).

## 3. Baiges, Codina & Idelsohn (2013) — *A domain decomposition strategy for reduced order models. Application to the incompressible Navier–Stokes equations*
**Comput. Methods Appl. Mech. Engrg. 267:23–42.** — ref. [7] do MPrM.

- Decomposição de domínio para **modelos de ordem reduzida (ROM)**
  hiper-reduzidos: as funções da base reduzida (POD) são **restritas aos nós de
  cada subdomínio** — não há nós de interface, e sim elementos de interface, o
  que garante continuidade da solução reconstruída.
- Estratégia híbrida full-order/reduced-order; termos de penalização para
  estabilizar gradientes espúrios nas interfaces; bases locais obtidas
  particionando a base global ou os snapshots.
- **Relação com o MPrM:** representa a via ROM (a "outra família" de redução de
  custo discutida na introdução do MPrM) combinada com DD para o mesmo
  problema-alvo (Navier-Stokes incompressível); o MPrM opta por DD multiescala
  sem base de snapshots, evitando a dependência de conjunto de treinamento.

---

### Onde estão as conversões

| Artigo | Markdown |
|---|---|
| MPrM (Lages) | `Refs/markdown/The_multiscale_projection_method_…​.md` |
| Abdulle & Budáč 2015 | `Refs/markdown/An adaptive finite element heterogeneous multiscale.md` |
| Araya et al. 2006 | `Refs/markdown/STABILIZED FINITE ELEMENT METHODS BASED ON MULTISCALE ENRICHMENT FOR THE STOKES PROBLEM.md` |
| Baiges et al. 2013 | `Refs/markdown/A domain decomposition strategy for reduced order models…​.md` |
