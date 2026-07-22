"""Visualização — MPrM.

make_frame_saver, make_gif: salva frames PNG e gera GIF animado.
"""
from __future__ import annotations
import os, glob

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
