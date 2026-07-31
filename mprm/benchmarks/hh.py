"""Benchmark analítico de Helmholtz-Hodge — MPrM.

hh_benchmark_mac, campos de Olshanskii, etc.
"""
from __future__ import annotations
import numpy as np
from mpi4py import MPI
from mrcm.config import MRCMConfig
from mprm.projector import MPrMProjector
from mprm.downscaler import MACDownscaler
from mprm.flow import mprm_projection_step
from mprm.transport import mac_divergence

# Benchmark analítico da decomposição de Helmholtz-Hodge (artigo §3.1)
# ===========================================================================

A_OLSH = 0.1  # parâmetro do campo de Olshanskii et al. (Eq. 40 do artigo)


def olshanskii_u_np(x, y, a=A_OLSH):
    """Campo solenoidal analítico (Eq. 40), tangente ao contorno (numpy)."""
    E = np.e
    th1 = 2 * np.pi * (np.exp(x) - 1.0) / (E - 1.0)
    th2 = 2 * np.pi * (np.exp(a * y) - 1.0) / (E ** a - 1.0)
    u1 = (a * np.exp(a * y) / (2 * np.pi * (E ** a - 1.0))) \
        * (1.0 - np.cos(th1)) * np.sin(th2)
    u2 = (-np.exp(x) / (2 * np.pi * (E - 1.0))) \
        * (1.0 - np.cos(th2)) * np.sin(th1)
    return u1, u2


def hh_p_np(x, y):
    """Potencial p̃ = cos(πx)cos(πy): ∇p̃·n = 0 no contorno (compatível com
    o caminho impermeável/no-flow forte do mrcm — ver notebook §2)."""
    return np.cos(np.pi * x) * np.cos(np.pi * y)


def hh_gradp_np(x, y):
    gx = -np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)
    gy = -np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
    return gx, gy


_MODES = ("mixed", "gradient", "solenoidal")


def hh_build_w_mode(N, mode="mixed"):
    """Amostra o campo de entrada da decomposição, por subespaço.

    mode="mixed"      : w = u + grad p̃   (construção original)
    mode="gradient"   : w = grad p̃       (parcela solenoidal exata = 0)
    mode="solenoidal" : w = u            (parcela irrotacional exata = 0)

    Devolve (w1, w2, u1_ex, u2_ex), em que (u1_ex, u2_ex) é a parcela
    SOLENOIDAL EXATA da entrada — o alvo da comparação em qualquer modo.

    No modo "gradient" a resposta exata é u = 0, de modo que todo valor não
    nulo devolvido pelo pipeline é erro puro, sem solução de referência.
    """
    if mode not in _MODES:
        raise ValueError(f"mode inválido: {mode!r} (use um de {_MODES})")
    h = 1.0 / N
    xf = np.linspace(0.0, 1.0, N + 1)      # faces verticais
    yc = (np.arange(N) + 0.5) * h
    Xu, Yu = np.meshgrid(xf, yc, indexing="ij")
    xc = (np.arange(N) + 0.5) * h
    yf = np.linspace(0.0, 1.0, N + 1)
    Xv, Yv = np.meshgrid(xc, yf, indexing="ij")

    if mode == "gradient":
        u1, u2 = np.zeros_like(Xu), np.zeros_like(Xv)
    else:
        u1, _ = olshanskii_u_np(Xu, Yu)
        _, u2 = olshanskii_u_np(Xv, Yv)

    if mode == "solenoidal":
        g1, g2 = np.zeros_like(Xu), np.zeros_like(Xv)
    else:
        g1, _ = hh_gradp_np(Xu, Yu)
        _, g2 = hh_gradp_np(Xv, Yv)

    w1, w2 = u1 + g1, u2 + g2
    w1[0, :] = w1[-1, :] = 0.0             # w·n = 0 (exato para p̃; u é tangente)
    w2[:, 0] = w2[:, -1] = 0.0
    return w1, w2, u1, u2


def hh_build_w(N):
    """Amostra o campo do benchmark HH nas faces MAC (malha N×N).

    Retorna (w1, w2, u1, u2): w = u + ∇p̃ e o campo solenoidal analítico u
    nas faces verticais/horizontais, com w·n = 0 imposto no contorno.

    Delega a hh_build_w_mode(N, "mixed"): um único caminho de código, de modo
    que a retrocompatibilidade seja estrutural e não uma cópia a sincronizar.
    """
    return hh_build_w_mode(N, mode="mixed")


def hh_benchmark_mac(Ndom=3, nloc=30, degreeU=1, degreeP=1, alpha=1.0,
                     comm=None):
    """Decomposição HH modificada via pipeline MPrM completo na malha MAC.

    Amostra w = u + ∇p̃ nas faces MAC, projeta com o MRCM (K=1, δt=1),
    aplica o downscaling e compara u_h = w − ṽ com o u analítico.
    Retorna dict(err_u, divmax, defect).
    """
    if comm is None:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    N = Ndom * nloc
    h = 1.0 / N
    dt = 1.0

    w1, w2, u1, u2 = hh_build_w(N)

    cfg = MRCMConfig(Ndomx=Ndom, Ndomy=Ndom, nlocx=nloc, nlocy=nloc,
                     degreeU=degreeU, degreeP=degreeP, alpha=alpha,
                     mixed_family="BDM", mixed_order=0,
                     executor_kind="sequential")
    projector = MPrMProjector(cfg, dt, comm=comm)
    downscaler = MACDownscaler(cfg, dt)
    uh1, uh2 = mprm_projection_step(w1, w2, projector, downscaler)

    num = np.sum((uh1 - u1) ** 2) + np.sum((uh2 - u2) ** 2)
    den = np.sum(u1 ** 2) + np.sum(u2 ** 2)
    return {
        "err_u": float(np.sqrt(num / den)),
        "divmax": float(np.abs(mac_divergence(uh1, uh2, h, h)).max()),
        "defect": downscaler.last_defect,
        "u": (uh1, uh2), "u_exact": (u1, u2), "w": (w1, w2), "N": N,
    }
