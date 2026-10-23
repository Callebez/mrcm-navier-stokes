"""Manifesto das figuras do bloco de Navier-Stokes (Secao 5.7).

Espelha `mrcm-firedrake/studies/_harness/figspec.py`: cada entrada declara,
para uma figura, os arquivos que a alimentam, os campos consumidos, os campos
DELIBERADAMENTE excluidos com o motivo, os eixos e o que a legenda LaTeX
precisa dizer.

O plano de figuras previa este arquivo e ele nao existia; a proveniencia das
tres figuras de NS era um esqueleto escrito a mao, com "ver plot_ghia.py" no
lugar dos campos. Sem manifesto, quem escreve o capitulo tem de abrir o codigo
para saber o que a figura consome, que e exatamente o que o manifesto existe
para evitar.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class Figure:
    slug: str            # nome do arquivo, sem extensao
    section: str         # secao de destino no Cap. 5
    sources: List[str]   # caminhos relativos a raiz do repositorio
    reference: str       # contra o que o erro foi medido
    fields: List[str]    # campos consumidos
    excluded: List[str]  # campos disponiveis e NAO usados, com o motivo
    x: str
    y: str
    panels: str
    caption_hint: str


_INSTRUMENTACAO = [
    "wall_s e timings: custo nao e afirmacao de nenhuma destas tres figuras",
    "rss_peak_mb: memoria residente, usada para dimensionar a campanha",
    "n_iter do solver linear: diagnostico de implementacao, nao de metodo",
]

FIGURES: List[Figure] = [
    Figure(
        slug="fig_ns-olshanskii",
        section="5.7.2",
        sources=["results_alpha/convergence_olshanskii/e1_hh_alpha.json"],
        reference="fina_sem_decomposicao",
        fields=["curves[].alpha", "curves[].cases[].H",
                "curves[].cases[].err_vs_fine"],
        excluded=_INSTRUMENTACAO + [
            "err_vs_exact: a etapa de projecao e medida contra a solucao fina "
            "sem decomposicao, que e o que isola o acoplamento",
        ],
        x="H (log)",
        y="erro relativo contra a solucao fina sem decomposicao (log)",
        panels="",
        caption_hint="Convergencia da etapa de projecao sobre o campo "
                     "analitico de Olshanskii2013, com dt = 1, de modo que a "
                     "etapa eliptica fica independente do avanco temporal. A "
                     "reta pontilhada e a referencia de segunda ordem. A curva "
                     "de alpha = 0,01 NAO converge (taxa 0,64) e cai no H mais "
                     "grosso: e ponto adverso e fica na figura.",
    ),
    Figure(
        slug="fig_ns-par-misto",
        section="5.7.3",
        sources=["results_alpha/alpha_hh.json",
                 "results_alpha/alpha_hh_bdm2.json"],
        reference="fina_sem_decomposicao",
        fields=["cases[].alpha", "cases[].err_vs_fine", "cases[].degree"],
        excluded=_INSTRUMENTACAO + [
            "results_alpha/alpha_hh_rt0.json: o caminho de RT0 roda em "
            "quadrilateros enquanto os demais rodam em triangulacoes, de modo "
            "que par misto e forma de celula variam juntos; e os valores de "
            "alpha a partir de 1e2 sao falha da fatoracao local. A ressalva "
            "pertence ao texto, nao a figura",
            "as ordens 1 a 3 do BDM1: em regime moderado coincidem com as do "
            "BDM2 por construcao, e oito curvas sobrepostas escondem o "
            "argumento. Fica so a ordem mais alta, que e onde as duas se "
            "separam",
        ],
        x="alpha (log)",
        y="erro relativo de velocidade contra a solucao fina sem decomposicao "
          "(log)",
        panels="",
        caption_hint="Cinco curvas: as quatro ordens do espaco de interface "
                     "com BDM2 (solidas) e a ordem mais alta com BDM1 "
                     "(tracejada, mesma cor). A escada e' tracada com BDM2, e "
                     "nao com BDM1, porque BDM1 degrada em alpha grande sob "
                     "espaco de interface rico e contaminaria o efeito do grau "
                     "de interface com um artefato do par misto. Em alpha "
                     "moderado as duas coincidem; a separacao em alpha grande "
                     "e o preco do par de ordem mais baixa.",
    ),
    Figure(
        slug="fig_ns-ghia",
        section="5.7.4",
        sources=["results_alpha/cavity_ghia/ghia_validation.json",
                 "tests/test_cavity_ghia.py"],
        reference="tabelas de Ghia et al. (1982)",
        fields=["cases[].Re", "cases[].y_u", "cases[].u_profile",
                "cases[].x_v", "cases[].v_profile", "GHIA_U", "GHIA_V"],
        excluded=_INSTRUMENTACAO + [
            "campo de vorticidade e linhas de corrente: nao ha valor tabelado "
            "correspondente em Ghia1982 com que compara-los, e a figura existe "
            "para a comparacao",
        ],
        x="u (linha de cima) e x (linha de baixo), ambos lineares",
        y="y (linha de cima) e v (linha de baixo), ambos lineares",
        panels="colunas por Reynolds: (a)/(d) Re=100, (b)/(e) Re=400, "
               "(c)/(f) Re=1000",
        caption_hint="Perfis sobre as linhas de centro da cavidade com tampa "
                     "deslizante, contra os valores tabelados por Ghia1982. "
                     "Malha 128x128 com decomposicao 8x8, resolucao comparavel "
                     "a 129x129 daqueles autores. Os eixos seguem a convencao "
                     "de Ghia, com a velocidade no eixo horizontal do perfil "
                     "vertical. Os valores tabelados vem de "
                     "tests/test_cavity_ghia.py, que e a fonte unica e "
                     "versionada.",
    ),
]


def por_slug(slug: str) -> Figure:
    for f in FIGURES:
        if f.slug == slug:
            return f
    raise KeyError(slug)
