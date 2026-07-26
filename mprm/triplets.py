"""Separação matriz/RHS das triplas do sistema global de interface.

`mrcm.global_assembly.compute_subdomain_triplets` calcula, de uma vez, as
contribuições de um subdomínio para a matriz A_Γ e para o vetor b_Γ. No MPrM
a matriz é CONSTANTE no tempo (K = δt e a geometria não mudam), enquanto b
muda a cada passo com a fonte ∇·w*. Separar as duas permite montar e fatorar
A_Γ uma única vez, no offline.

As duas funções abaixo reproduzem, termo a termo, o que a função do mrcm faz
— ver `mrcm/global_assembly.py:238-320`. Qualquer divergência é um bug, e é
o que `tests/test_triplets.py` verifica.
"""
from __future__ import annotations
import numpy as np


def _arr(x):
    """Array de dados de uma Function do Firedrake, ou o próprio array."""
    return x.dat.data_ro if hasattr(x, "dat") else x


def _spdot(cof, u_arr) -> float:
    """Produto interno esparso: cofunção (idx, vals) contra um array de dofs."""
    idx, v = cof
    return float(np.dot(v, u_arr[idx]))


def subdomain_matrix_triplets(cfg, skeleton, dof_mgr, el,
                              homogeneous_el, face_mass_el,
                              cofunc_flux_el, cofunc_pres_el):
    """Contribuição do subdomínio `el` para A_Γ, em formato COO.

    Depende apenas de dados do offline: as MsBF homogêneas, as matrizes de
    massa de face e as cofunções. Não depende da fonte, logo é constante ao
    longo do laço temporal.

    Returns
    -------
    (rows, cols, vals) : arrays int32, int32, float64
    """
    ndofsU, ndofsP = cfg.ndofsU, cfg.ndofsP
    NunkU = dof_mgr.NunkU
    rows: list = []
    cols: list = []
    vals: list = []

    for inc in skeleton.gamma_incidences(el):
        ell = inc.ell
        key = (inc.side, inc.seg)
        M_face_si = face_mass_el[key]
        cof_flux = cofunc_flux_el[key]
        cof_pres = cofunc_pres_el[key]

        # Linhas de fluxo (Eq. 59): <c_flux_i, u_hat_j> - M_face[i, d_j],
        # o termo de massa apenas para MSBF de fluxo na MESMA aresta de Γ.
        for idf in range(ndofsU):
            iunkG = dof_mgr.flux_index(ell, idf)
            cof_i = cof_flux[idf]
            for junkG, (u_hat, _) in homogeneous_el.items():
                trace_val = _spdot(cof_i, _arr(u_hat))
                mass_val = 0.0
                if junkG < NunkU and junkG // ndofsU == ell:
                    mass_val = float(M_face_si[idf, junkG % ndofsU])
                val = trace_val - mass_val
                if val != 0.0:
                    rows.append(iunkG)
                    cols.append(junkG)
                    vals.append(val)

        # Linhas de pressão (Eq. 58): <c_pres_i, u_hat_j>.
        for idf in range(ndofsP):
            iunkG = dof_mgr.pressure_index(ell, idf)
            cof_i = cof_pres[idf]
            for junkG, (u_hat, _) in homogeneous_el.items():
                val = _spdot(cof_i, _arr(u_hat))
                if val != 0.0:
                    rows.append(iunkG)
                    cols.append(junkG)
                    vals.append(val)

    return (np.asarray(rows, dtype=np.int32),
            np.asarray(cols, dtype=np.int32),
            np.asarray(vals, dtype=np.float64))


def subdomain_rhs(cfg, skeleton, dof_mgr, el, particular_el,
                  cofunc_flux_el, cofunc_pres_el):
    """Contribuição do subdomínio `el` para b_Γ: -<c_i, u_bar>.

    É a única parte que muda a cada passo de tempo, porque `particular_el`
    carrega a solução local da fonte ∇·w*.

    Returns
    -------
    (b_idx, b_vals) : arrays int32, float64
    """
    ndofsU, ndofsP = cfg.ndofsU, cfg.ndofsP
    u_bar_arr = _arr(particular_el[0])
    b_idx: list = []
    b_vals: list = []

    for inc in skeleton.gamma_incidences(el):
        ell = inc.ell
        key = (inc.side, inc.seg)
        cof_flux = cofunc_flux_el[key]
        cof_pres = cofunc_pres_el[key]

        for idf in range(ndofsU):
            b_idx.append(dof_mgr.flux_index(ell, idf))
            b_vals.append(-_spdot(cof_flux[idf], u_bar_arr))
        for idf in range(ndofsP):
            b_idx.append(dof_mgr.pressure_index(ell, idf))
            b_vals.append(-_spdot(cof_pres[idf], u_bar_arr))

    return (np.asarray(b_idx, dtype=np.int32),
            np.asarray(b_vals, dtype=np.float64))
