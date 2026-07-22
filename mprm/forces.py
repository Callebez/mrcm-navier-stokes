"""Forças, pressão diagnóstica e Strouhal — MPrM.

ChannelFinePoisson, DiagnosticPressureSolver, obstacle_forces, strouhal.
Puro NumPy/SciPy — não importa Firedrake.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from mprm.transport import mac_divergence, solid_faces, _apply_external_normals_after_step

# Referência fina mascarada + diagnósticos (von Kármán)
# ===========================================================================

class ChannelFinePoisson:
    """Poisson MAC global mascarado para o passo de projeção de referência.

    Entrada/paredes/slip: Neumann homogêneo (v·n=0). Saída: Dirichlet p=0.
    Obstáculo: células sólidas excluídas (p=0), faces sólidas no-flow.
    Serve (a) como referência fina de von Kármán e (b) como fonte de pressão
    física para as forças.
    """

    def __init__(self, problem, dt):
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        self.problem = problem
        self.dt = float(dt)
        Nx, Ny = problem.Nx, problem.Ny
        hx, hy = problem.hx, problem.hy
        self.Nx, self.Ny, self.hx, self.hy = Nx, Ny, hx, hy
        solid = problem.solid if problem.solid is not None \
            else np.zeros((Nx, Ny), bool)
        self.solid = solid
        has_out = {s: problem.side(s).kind == "outflow"
                   for s in ("left", "right", "bottom", "top")}
        self.has_out = has_out

        def lid(i, j):
            return i + j * Nx

        rows, cols, vals = [], [], []
        for j in range(Ny):
            for i in range(Nx):
                c = lid(i, j)
                if solid[i, j]:
                    rows.append(c); cols.append(c); vals.append(1.0); continue
                diag = 0.0
                # vizinhos internos (fluido)
                for (di, dj, d, ln) in ((1, 0, hx, hy), (-1, 0, hx, hy),
                                        (0, 1, hy, hx), (0, -1, hy, hx)):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < Nx and 0 <= jj < Ny and not solid[ii, jj]:
                        w = dt * ln / d
                        rows.append(c); cols.append(lid(ii, jj))
                        vals.append(w); diag -= w
                # faces de SAÍDA (Dirichlet p=0): fantasma p=0 a meia célula
                for (side, at_edge, d, ln) in (
                        ("left", i == 0, hx, hy), ("right", i == Nx - 1, hx, hy),
                        ("bottom", j == 0, hy, hx), ("top", j == Ny - 1, hy, hx)):
                    if at_edge and has_out[side]:
                        diag -= dt * ln / (0.5 * d)     # p_ghost=0 (Dirichlet)
                rows.append(c); cols.append(c); vals.append(diag)
        A = sp.csr_matrix((vals, (rows, cols)), shape=(Nx * Ny, Nx * Ny)).tolil()
        self._pure_neumann = not any(has_out.values())
        if self._pure_neumann:
            pin = int(np.argmin(solid.flatten(order="F")))
            A[pin, :] = 0.0; A[pin, pin] = 1.0
            self._pin = pin
        self.lu = spla.splu(A.tocsc())

    def solve(self, w1, w2):
        dt, hx, hy = self.dt, self.hx, self.hy
        Nx, Ny, solid = self.Nx, self.Ny, self.solid
        divw = mac_divergence(w1, w2, hx, hy)
        divw = divw.copy(); divw[solid] = 0.0
        b = (divw * hx * hy).flatten(order="F")
        b[solid.flatten(order="F")] = 0.0
        if self._pure_neumann:
            fluid = ~solid.flatten(order="F")
            b[fluid] -= b[fluid].sum() / fluid.sum()
            b[self._pin] = 0.0
        p = self.lu.solve(b).reshape((Nx, Ny), order="F")
        p[solid] = 0.0
        u = w1.copy(); v = w2.copy()
        u[1:-1, :] -= dt * (p[1:, :] - p[:-1, :]) / hx
        v[:, 1:-1] -= dt * (p[:, 1:] - p[:, :-1]) / hy
        # saída: gradiente de p do lado da célula de borda; entrada/parede v·n
        _apply_external_normals_after_step(u, v, self.problem)
        fx, fy = solid_faces(solid)
        u[fx] = 0.0; v[fy] = 0.0
        return u, v, p

    def step(self, w1, w2):
        u, v, _ = self.solve(w1, w2)
        return u, v


class DiagnosticPressureSolver:
    """Pressão física via equação de Poisson: ∇²p = -∇·((u·∇)u).

    A matriz do sistema (geometria + BCs) é constante no tempo; montada e
    fatorada UMA vez no __init__. A cada passo, apenas a fonte (termo
    convectivo) é recomputada e o sistema resolvido com a fatoração cacheada.

    K=1 (Poisson padrão), Dirichlet p=0 na saída, Neumann homogêneo nas
    demais fronteiras.
    """

    def __init__(self, problem):
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        self.problem = problem
        hx, hy = problem.hx, problem.hy
        Nx, Ny = problem.Nx, problem.Ny
        self.hx, self.hy = hx, hy
        self.Nx, self.Ny = Nx, Ny
        solid = problem.solid if problem.solid is not None else np.zeros((Nx, Ny), bool)
        self.solid = solid

        has_out = {s: problem.side(s).kind == "outflow"
                   for s in ("left", "right", "bottom", "top")}

        def lid(i, j):
            return i + j * Nx

        rows, cols, vals = [], [], []
        for j in range(Ny):
            for i in range(Nx):
                c = lid(i, j)
                if solid[i, j]:
                    rows.append(c); cols.append(c); vals.append(1.0); continue
                diag = 0.0
                for (di, dj, d, ln) in ((1, 0, hx, hy), (-1, 0, hx, hy),
                                        (0, 1, hy, hx), (0, -1, hy, hx)):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < Nx and 0 <= jj < Ny and not solid[ii, jj]:
                        w = 1.0 * ln / d    # K=1
                        rows.append(c); cols.append(lid(ii, jj))
                        vals.append(w); diag -= w
                for (side, at_edge, d, ln) in (
                        ("left", i == 0, hx, hy), ("right", i == Nx-1, hx, hy),
                        ("bottom", j == 0, hy, hx), ("top", j == Ny-1, hy, hx)):
                    if at_edge and has_out[side]:
                        diag -= 1.0 * ln / (0.5 * d)
                rows.append(c); cols.append(c); vals.append(diag)

        A = sp.csr_matrix((vals, (rows, cols)), shape=(Nx * Ny, Nx * Ny)).tolil()
        pure_neumann = not any(has_out.values())
        if pure_neumann:
            pin = int(np.argmin(solid.flatten(order="F"))) if solid.any() else 0
            A[pin, :] = 0.0; A[pin, pin] = 1.0
            self._pin = pin
        self._pure_neumann = pure_neumann
        self.lu = spla.splu(A.tocsc())

    def solve(self, u, v):
        """Resolve ∇²p = -∇·((u·∇)u) com a fatoração cacheada."""
        hx, hy = self.hx, self.hy
        Nx, Ny = self.Nx, self.Ny
        solid = self.solid

        uc = 0.5 * (u[:-1, :] + u[1:, :])
        vc = 0.5 * (v[:, :-1] + v[:, 1:])

        dudx = np.zeros((Nx, Ny)); dudy = np.zeros((Nx, Ny))
        dvdx = np.zeros((Nx, Ny)); dvdy = np.zeros((Nx, Ny))
        dudx[1:-1, :] = (uc[2:, :] - uc[:-2, :]) / (2 * hx)
        dudx[0, :] = (uc[1, :] - uc[0, :]) / hx
        dudx[-1, :] = (uc[-1, :] - uc[-2, :]) / hx
        dudy[:, 1:-1] = (uc[:, 2:] - uc[:, :-2]) / (2 * hy)
        dudy[:, 0] = (uc[:, 1] - uc[:, 0]) / hy
        dudy[:, -1] = (uc[:, -1] - uc[:, -2]) / hy
        dvdx[1:-1, :] = (vc[2:, :] - vc[:-2, :]) / (2 * hx)
        dvdx[0, :] = (vc[1, :] - vc[0, :]) / hx
        dvdx[-1, :] = (vc[-1, :] - vc[-2, :]) / hx
        dvdy[:, 1:-1] = (vc[:, 2:] - vc[:, :-2]) / (2 * hy)
        dvdy[:, 0] = (vc[:, 1] - vc[:, 0]) / hy
        dvdy[:, -1] = (vc[:, -1] - vc[:, -2]) / hy

        conv_x = uc * dudx + vc * dudy
        conv_y = uc * dvdx + vc * dvdy

        source = np.zeros((Nx, Ny))
        source[1:-1, :] = -(conv_x[2:, :] - conv_x[:-2, :]) / (2 * hx)
        source[0, :] = -(conv_x[1, :] - conv_x[0, :]) / hx
        source[-1, :] = -(conv_x[-1, :] - conv_x[-2, :]) / hx
        source[:, 1:-1] += -(conv_y[:, 2:] - conv_y[:, :-2]) / (2 * hy)
        source[:, 0] += -(conv_y[:, 1] - conv_y[:, 0]) / hy
        source[:, -1] += -(conv_y[:, -1] - conv_y[:, -2]) / hy

        if solid is not None:
            source[solid] = 0.0

        b = (source * hx * hy).flatten(order="F")
        if solid is not None:
            b[solid.flatten(order="F")] = 0.0
        if self._pure_neumann:
            fluid = ~solid.flatten(order="F") if solid is not None else np.ones(Nx*Ny, bool)
            b[fluid] -= b[fluid].sum() / fluid.sum()
            b[self._pin] = 0.0

        p = self.lu.solve(b).reshape((Nx, Ny), order="F")
        if solid is not None:
            p[solid] = 0.0
        return p


def _diagnostic_pressure(u, v, problem, dt):
    """Wrapper retrocompatível — usa DiagnosticPressureSolver (refatora a
    cada chamada). Prefira instanciar DiagnosticPressureSolver e chamar
    .solve(u,v) para reutilizar a fatoração LU."""
    solver = DiagnosticPressureSolver(problem)
    return solver.solve(u, v)


def obstacle_forces(u, v, p, problem, U_ref, D, u_prev=None, v_prev=None, dt=None):
    """C_D, C_L por balanço de momento num volume de controle retangular que
    envolve o obstáculo (uma célula de folga). Inclui fluxo convectivo +
    pressão + VISCOSO (σ = -pI + ν(∇u+∇uᵀ)) nas 4 faces do VC e, se
    u_prev/dt fornecidos, o termo transiente d/dt ∫_VC u dV.
    ρ=1; C = 2F/(ρ U² D)."""
    solid = problem.solid
    if solid is None or not solid.any():
        return 0.0, 0.0
    hx, hy, nu = problem.hx, problem.hy, problem.nu
    Nx, Ny = problem.Nx, problem.Ny
    xs = np.where(solid.any(axis=1))[0]; ys = np.where(solid.any(axis=0))[0]
    i0, i1 = max(xs.min() - 1, 0), min(xs.max() + 1, Nx - 1)
    j0, j1 = max(ys.min() - 1, 0), min(ys.max() + 1, Ny - 1)

    uc = 0.5 * (u[:-1, :] + u[1:, :])
    vc = 0.5 * (v[:, :-1] + v[:, 1:])
    # gradientes centrados nos centros das células
    dudx = np.zeros((Nx, Ny)); dudy = np.zeros((Nx, Ny))
    dvdx = np.zeros((Nx, Ny)); dvdy = np.zeros((Nx, Ny))
    dudx[1:-1, :] = (uc[2:, :] - uc[:-2, :]) / (2 * hx)
    dudx[0, :] = (uc[1, :] - uc[0, :]) / hx
    dudx[-1, :] = (uc[-1, :] - uc[-2, :]) / hx
    dudy[:, 1:-1] = (uc[:, 2:] - uc[:, :-2]) / (2 * hy)
    dudy[:, 0] = (uc[:, 1] - uc[:, 0]) / hy
    dudy[:, -1] = (uc[:, -1] - uc[:, -2]) / hy
    dvdx[1:-1, :] = (vc[2:, :] - vc[:-2, :]) / (2 * hx)
    dvdx[0, :] = (vc[1, :] - vc[0, :]) / hx
    dvdx[-1, :] = (vc[-1, :] - vc[-2, :]) / hx
    dvdy[:, 1:-1] = (vc[:, 2:] - vc[:, :-2]) / (2 * hy)
    dvdy[:, 0] = (vc[:, 1] - vc[:, 0]) / hy
    dvdy[:, -1] = (vc[:, -1] - vc[:, -2]) / hy

    Fx = 0.0; Fy = 0.0
    # faces verticais — normal ±x
    for i, s in ((i0, -1.0), (i1 + 1, +1.0)):
        ii = min(max(i, 0), Nx)
        ic = min(max(i - (s > 0), 0), Nx - 1)
        un = u[ii, j0:j1 + 1]
        vc_f = 0.5 * (vc[min(ii, Nx-1), j0:j1+1] + vc[max(ii-1, 0), j0:j1+1])
        pf   = 0.5 * (p[min(ii, Nx-1), j0:j1+1] + p[max(ii-1, 0), j0:j1+1])
        visc_xx = 2.0 * nu * dudx[ic, j0:j1+1]
        visc_yx = nu * (dvdx[ic, j0:j1+1] + dudy[ic, j0:j1+1])
        Fx += -s * np.sum((un * un + pf - visc_xx) * hy)
        Fy += -s * np.sum((vc_f * un        - visc_yx) * hy)
    # faces horizontais — normal ±y
    for j, s in ((j0, -1.0), (j1 + 1, +1.0)):
        jj = min(max(j, 0), Ny)
        jc = min(max(j - (s > 0), 0), Ny - 1)
        vn = v[i0:i1 + 1, jj]
        uc_f = 0.5 * (uc[i0:i1+1, min(jj, Ny-1)] + uc[i0:i1+1, max(jj-1, 0)])
        pf   = 0.5 * (p[i0:i1+1, min(jj, Ny-1)] + p[i0:i1+1, max(jj-1, 0)])
        visc_xy = nu * (dudy[i0:i1+1, jc] + dvdx[i0:i1+1, jc])
        visc_yy = 2.0 * nu * dvdy[i0:i1+1, jc]
        Fx += -s * np.sum((uc_f * vn        - visc_xy) * hx)
        Fy += -s * np.sum((vn * vn + pf     - visc_yy) * hx)

    # termo transiente: d/dt ∫_VC u dV
    if u_prev is not None and dt is not None:
        upc = 0.5 * (u_prev[:-1, :] + u_prev[1:, :])
        vpc = 0.5 * (v_prev[:, :-1] + v_prev[:, 1:])
        dVol = hx * hy
        Fx -= (uc[i0:i1+1, j0:j1+1].sum() - upc[i0:i1+1, j0:j1+1].sum()) * dVol / dt
        Fy -= (vc[i0:i1+1, j0:j1+1].sum() - vpc[i0:i1+1, j0:j1+1].sum()) * dVol / dt

    q = 0.5 * U_ref * U_ref * D
    return Fx / q, Fy / q


def strouhal(signal, dt, U_ref, D, t_skip_frac=0.5, pad=8):
    """St = f·D/U da frequência dominante do sinal.

    Zero-padding (pad=8) refina os bins da FFT para resolução sub-bin.
    Interpolação parabólica de 3 pontos no pico para precisão adicional.
    Descartam-se os primeiros t_skip_frac do sinal (transiente).
    """
    sig = np.asarray(signal, float)
    n0 = int(len(sig) * t_skip_frac)
    s = sig[n0:]
    s = s - s.mean()
    if s.size < 16:
        return float("nan")
    s = s * np.hanning(s.size)
    n = int(2 ** np.ceil(np.log2(s.size * pad)))        # zero-padding
    amp = np.abs(np.fft.rfft(s, n=n))
    amp[0] = 0.0
    freqs = np.fft.rfftfreq(n, d=dt)
    k = int(np.argmax(amp))
    # Interpolação parabólica de 3 pontos para refinar o pico
    if 0 < k < len(amp) - 1:
        a, b, c = amp[k - 1], amp[k], amp[k + 1]
        denom = a - 2 * b + c
        if abs(denom) > 1e-30:
            k = k + 0.5 * (a - c) / denom
    f_dom = k * (freqs[1] - freqs[0]) if k > 0 else freqs[1]
    return float(f_dom * D / U_ref)


def make_frame_saver(output_dir="frames_mprm", dpi=100, Re=100.0, title_prefix="MPrM"):
    """Factory: retorna um callback que salva frames PNG da cavidade.

    Uso:
        saver = make_frame_saver("frames", Re=100)
        result = run_cavity(..., frame_callback=saver, frame_every=50)
        # Depois: make_gif("frames", "cavity.gif")

    Cada frame mostra magnitude |u| + quiver + linha do tempo.
    """
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    frame_idx = [0]  # mutable counter

    def _save(step, t, u, v, hx, hy):
        N = u.shape[0] - 1
        xc = (np.arange(N) + 0.5) * hx
        yc = (np.arange(N) + 0.5) * hy
        uc = 0.5 * (u[:-1, :] + u[1:, :])
        vc = 0.5 * (v[:, :-1] + v[:, 1:])
        mag = np.sqrt(uc**2 + vc**2)

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.pcolormesh(xc, yc, mag.T, shading='auto', cmap='turbo', vmin=0, vmax=1.0)
        skip = max(1, N // 25)
        ax.quiver(xc[::skip], yc[::skip],
                  uc[::skip, ::skip].T, vc[::skip, ::skip].T,
                  scale=4, width=0.002, color='white', alpha=0.7)
        ax.set_title(f"{title_prefix} — Cavidade Re={Re:.0f}  "
                     f"t = {t:.3f}")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='|u|', shrink=0.8)
        fname = os.path.join(output_dir, f"frame_{frame_idx[0]:05d}.png")
        fig.savefig(fname, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        frame_idx[0] += 1

    return _save


def make_gif(frame_dir="frames_mprm", output="cavity_mprm.gif",
             fps=15, duration=None):
    """Cria um GIF animado a partir dos frames PNG salvos.

    Requer: pip install imageio  (ou Pillow)

    Parâmetros:
      frame_dir: diretório com frame_*.png
      output:    arquivo de saída (.gif ou .mp4)
      fps:       quadros por segundo
      duration:  duração por frame em ms (alternativa a fps)
    """
    import os
    import glob

    files = sorted(glob.glob(os.path.join(frame_dir, "frame_*.png")))
    if not files:
        raise FileNotFoundError(f"Nenhum frame encontrado em {frame_dir}")

    if duration is None:
        duration = 1000.0 / fps

    try:
        import imageio.v2 as imageio
        frames = [imageio.imread(f) for f in files]
        imageio.mimsave(output, frames, duration=duration, loop=0)
        print(f"✓ GIF salvo: {output} ({len(frames)} frames)")
        return output
    except ImportError:
        # fallback: Pillow
        try:
            from PIL import Image
            frames = [Image.open(f) for f in files]
            frames[0].save(
                output, save_all=True, append_images=frames[1:],
                duration=duration, loop=0,
            )
            print(f"✓ GIF salvo (Pillow): {output} ({len(frames)} frames)")
            return output
        except ImportError:
            raise ImportError(
                "Instale imageio ou Pillow: pip install imageio"
            )


# ===========================================================================
