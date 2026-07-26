"""A pré-fatoração de A_Γ não pode mudar nenhum número.

Compara o caminho novo (matriz montada e fatorada no offline) contra o
caminho antigo (remontagem por passo via GlobalAssembler.assemble_from_triplets).

Nota sobre a fixture: a equivalência entre os dois caminhos é comparada num
problema de CANAL (inflow/outflow), não no impermeável default. O
impermeável (Neumann puro em todos os lados) deixa A_Γ deficiente de posto
por exatamente 1 (o modo de pressão constante) — X deixa de ser único e
comparar valores brutos de X entre dois solvers diretos DISTINTOS
(spsolve/UMFPACK no caminho antigo, splu no novo) não é uma comparação
válida de equivalência (ver `test_A_gamma_impermeavel_e_deficiente_de_posto`,
que documenta esse achado à parte, sem afrouxar a tolerância da comparação
principal). Com outflow, A_Γ tem posto cheio e cond ~ 1e2 nesta config —
X_novo bate com X_ref em 1e-12 com folga.
"""
import numpy as np

from mrcm.config import MRCMConfig
from mrcm.global_assembly import compute_subdomain_triplets
from mprm.problem import FlowProblem, SideBC
from mprm.projector import MPrMProjector


def _cfg(Ndom=2, nloc=8):
    return MRCMConfig(Ndomx=Ndom, Ndomy=Ndom, nlocx=nloc, nlocy=nloc,
                      degreeU=1, degreeP=1, alpha=1.0,
                      basis_kind="legendre_P",
                      mixed_family="BDM", mixed_order=0,
                      executor_kind="sequential")


def _channel_problem(cfg, nu=0.01):
    """Canal com inflow/left e outflow/right: quebra o Neumann puro do
    impermeável default, dando posto cheio a A_Γ (ver docstring do módulo)."""
    sides = {
        "left": SideBC("inflow", profile=1.0),
        "right": SideBC("outflow"),
        "top": SideBC("wall", velocity=0.0),
        "bottom": SideBC("wall", velocity=0.0),
    }
    return FlowProblem(Nx=cfg.Nglobx, Ny=cfg.Ngloby, Lx=cfg.Lx, Ly=cfg.Ly,
                       nu=nu, sides=sides)


def _antigo_A_b_X(proj, divw):
    """Reproduz a etapa 2 do project() original: triplas por passo + assemble.

    Retorna (A_old, b_old, X_ref) — a montagem/solve completos pelo caminho
    que existia antes desta tarefa (GlobalAssembler.assemble_from_triplets).
    """
    for isub in proj.my_subs:
        ic, jc = proj.cell_map[isub]
        proj.src_fn[isub].dat.data[:] = divw[ic, jc]
    particular = {isub: proj.solvers[isub].solve(proj.L_forms[isub])
                  for isub in proj.my_subs}
    trip = {isub: compute_subdomain_triplets(
        proj.cfg, proj.skel, proj.dofm, isub,
        particular[isub], proj.ws[isub]["homogeneous"],
        proj.ws[isub]["face_mass"], proj.ws[isub]["cofunc_flux"],
        proj.ws[isub]["cofunc_pres"]) for isub in proj.my_subs}
    return proj.assembler.assemble_from_triplets(trip)


def test_A_gamma_bit_identica_ao_caminho_de_remontagem():
    """A_Γ (montada uma vez no offline) é idêntica à matriz que o caminho
    antigo remontava a cada passo — a prova direta de que separar
    matriz/RHS (Tarefa 1) e mover a montagem para o construtor (esta tarefa)
    não alterou nenhum termo da matriz."""
    cfg = _cfg()
    problem = _channel_problem(cfg)
    proj = MPrMProjector(cfg, dt=0.01, problem=problem)
    rng = np.random.default_rng(3)
    divw = rng.standard_normal((cfg.Nglobx, cfg.Ngloby))

    A_old, b_old, _ = _antigo_A_b_X(proj, divw)

    diff = (proj.A_gamma - A_old)
    max_diff_A = np.abs(diff.data).max() if diff.nnz else 0.0
    assert max_diff_A == 0.0, f"A_Γ diverge do caminho antigo: max|diff|={max_diff_A:.3e}"

    # b muda a cada passo (fonte), mas para ESTA fonte deve bater também —
    # confirma que subdomain_rhs (Tarefa 1) reproduz o RHS completo.
    from mprm.triplets import subdomain_rhs
    particular = {isub: proj.solvers[isub].solve(proj.L_forms[isub])
                  for isub in proj.my_subs}
    b_novo = np.zeros(proj.n_gamma)
    for isub in proj.my_subs:
        b_idx, b_vals = subdomain_rhs(
            cfg, proj.skel, proj.dofm, isub, particular[isub],
            proj.ws[isub]["cofunc_flux"], proj.ws[isub]["cofunc_pres"])
        np.add.at(b_novo, b_idx, b_vals)
    max_diff_b = np.abs(b_novo - b_old).max()
    assert max_diff_b == 0.0, f"b diverge do caminho antigo: max|diff|={max_diff_b:.3e}"


def test_X_identico_ao_caminho_de_remontagem():
    """X do caminho novo (LU do offline) bate com X do caminho antigo
    (assemble_from_triplets) em problema de canal, onde A_Γ tem posto
    cheio (ver docstring do módulo e `test_A_gamma_impermeavel_e_deficiente_de_posto`)."""
    cfg = _cfg()
    problem = _channel_problem(cfg)
    proj = MPrMProjector(cfg, dt=0.01, problem=problem)
    rng = np.random.default_rng(3)
    divw = rng.standard_normal((cfg.Nglobx, cfg.Ngloby))

    _, _, X_ref = _antigo_A_b_X(proj, divw)
    proj.project(divw)
    X_novo = proj.last_X

    err = np.max(np.abs(X_novo - X_ref)) / max(np.max(np.abs(X_ref)), 1.0)
    assert err < 1e-12, f"X divergiu: erro relativo {err:.3e}"


def test_matriz_fatorada_uma_vez_e_reutilizada():
    """A LU é construída no offline e não é reconstruída por passo."""
    cfg = _cfg()
    proj = MPrMProjector(cfg, dt=0.01)
    assert proj.lu_gamma is not None, "A_Γ deveria estar fatorada no construtor"
    lu_id = id(proj.lu_gamma)

    rng = np.random.default_rng(4)
    for _ in range(3):
        divw = rng.standard_normal((cfg.Nglobx, cfg.Ngloby))
        divw -= divw.mean()
        proj.project(divw)
        assert id(proj.lu_gamma) == lu_id, "a LU foi refatorada durante o online"


def test_cavidade_ponta_a_ponta_inalterada():
    """Regressão física: a cavidade continua conservando massa."""
    from mprm.flow import run_cavity
    from mprm.transport import mac_divergence
    r = run_cavity(Re=100.0, Ndom=2, nloc=8, T=0.05, verbose=False)
    dv = np.abs(mac_divergence(r["u"], r["v"], 1.0 / 16, 1.0 / 16)).max()
    assert dv < 1e-11, f"max|div u| = {dv:.3e}"


def test_A_gamma_impermeavel_e_deficiente_de_posto():
    """Achado (não é bug, é estrutural): no problema impermeável default
    (Neumann puro em todos os lados — sem outflow), A_Γ é deficiente de
    posto por exatamente 1 (o modo de pressão constante, indeterminado a
    menos de referência). O número de condição fica no limiar da
    singularidade em ponto flutuante. Por isso a comparação de equivalência
    de X (`test_X_identico_ao_caminho_de_remontagem`) usa um problema de
    canal em vez deste; este teste isola e documenta o achado de forma
    reproduzível, sem afrouxar nenhuma tolerância da suíte principal.
    Ver também `test_impermeavel_nao_quebra_construcao_do_projetor`, que
    verifica que essa deficiência de posto não derruba a fatoração LU."""
    cfg = _cfg()
    proj = MPrMProjector(cfg, dt=0.01)     # problem=None => impermeável default
    A = proj.A_gamma.toarray()

    rank = np.linalg.matrix_rank(A)
    cond = np.linalg.cond(A)
    assert rank == proj.n_gamma - 1, (
        f"esperava deficiência de posto exatamente 1 (modo de pressão "
        f"constante); rank={rank}, n_gamma={proj.n_gamma}")
    assert cond > 1e12, (
        f"esperava A_Γ no limiar da singularidade em ponto flutuante; "
        f"cond={cond:.3e}")


def test_impermeavel_nao_quebra_construcao_do_projetor():
    """Regressão: a deficiência de posto do problema impermeável (achado
    acima) não deve impedir a construção do MPrMProjector nem o online.

    `scipy.sparse.linalg.splu` pode levantar `RuntimeError: Factor is
    exactly singular` num pivô nulo — o caminho antigo (`spsolve`/UMFPACK)
    tolerava matrizes quase singulares emitindo só um aviso. O construtor
    deve capturar essa falha e cair para `lu_gamma = None` (resolvido via
    `_solve_iterative` em `solve_interface`), preservando "ainda produz uma
    solução" em vez de propagar a exceção."""
    cfg = _cfg()
    proj = MPrMProjector(cfg, dt=0.01)     # não deve levantar
    assert proj.A_gamma is not None

    rng = np.random.default_rng(5)
    divw = rng.standard_normal((cfg.Nglobx, cfg.Ngloby))
    divw -= divw.mean()          # fonte compatível (Neumann puro)
    proj.project(divw)           # não deve levantar
    assert np.all(np.isfinite(proj.last_X)), "X contém NaN/Inf"


def test_splu_singular_cai_para_solver_iterativo(monkeypatch):
    """Exercita diretamente o `except RuntimeError` do construtor (Important
    4 da revisão): nas configs de teste deste arquivo, `splu` não chega a
    levantar (o pivô é minúsculo mas não estruturalmente zero — confirmado
    à parte com uma matriz forçadamente singular). Para testar o ramo de
    tratamento em si, sem depender de sorte de pivô, forçamos `splu` a
    levantar via monkeypatch e confirmamos que o construtor não propaga a
    exceção, cai para `lu_gamma = None`, e que `solve_interface`/`project`
    continuam funcionando pelo caminho iterativo (`_solve_iterative`)."""
    import mprm.projector as projmod

    class _FakeSpla:
        """Substitui o módulo `spla` só dentro de `mprm.projector` — não o
        `scipy.sparse.linalg` real, que `mrcm.local_solver` também usa para
        a fatoração local de cada subdomínio (sem relação com A_Γ)."""
        @staticmethod
        def splu(A):
            raise RuntimeError("Factor is exactly singular")

    monkeypatch.setattr(projmod, "spla", _FakeSpla)

    cfg = _cfg()
    proj = MPrMProjector(cfg, dt=0.01)     # não deve propagar o RuntimeError
    assert proj.A_gamma is not None
    assert proj.lu_gamma is None, "deveria cair para o caminho iterativo"

    rng = np.random.default_rng(6)
    divw = rng.standard_normal((cfg.Nglobx, cfg.Ngloby))
    divw -= divw.mean()
    proj.project(divw)           # deve rodar via _solve_iterative
    assert np.all(np.isfinite(proj.last_X)), "X contém NaN/Inf"
