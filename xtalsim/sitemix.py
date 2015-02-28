"""
sitemix - xtalsim subpackage for modeling cation site mixtures
"""

import numpy as np
import potentials
import geom


def parse_WFEC_coeffs(p, Ncoeff):
    Wcoeff = p[0:Ncoeff[0:1]]
    face_coeff = p[sum(Ncoeff[0:1]):sum(Ncoeff[0:2])]
    edge_coeff = p[sum(Ncoeff[0:2]):sum(Ncoeff[0:3])]
    corner_coeff = p[sum(Ncoeff[0:3]):sum(Ncoeff[0:4])]
    return Wcoeff, face_coeff, edge_coeff, corner_coeff


def eval_cubic_reg_soln_bin(p, Ncoeff, ind_config, X,
                            mixed_occs_FEC, pair_dists_FEC, dist0,
                            get_terms=False):

    pair_dist_face, pair_dist_edge, pair_dist_corner = pair_dists_FEC
    faceshp = pair_dist_face.shape
    avg_dist_face_config = np.mean(pair_dist_face.reshape(
        [faceshp[0], faceshp[1], -1]), axis=2)
    avg_dist_face = np.mean(avg_dist_face_config[ind_config, :], axis=0)

    Wcoeff = p[0:Ncoeff[0:1]]
    face_coeff = p[sum(Ncoeff[0:1]):sum(Ncoeff[0:2])]
    edge_coeff = p[sum(Ncoeff[0:2]):sum(Ncoeff[0:3])]
    corner_coeff = p[sum(Ncoeff[0:3]):sum(Ncoeff[0:4])]

    Wcoeff, face_coeff, edge_coeff, corner_coeff = parse_WFEC_coeffs(p, Ncoeff)

    W = np.polyval(Wcoeff, avg_dist_face-dist0)
    Xtbl = np.tile(X[ind_config], [faceshp[1], 1]).T
    Wtbl = np.tile(W, [len(ind_config), 1])
    mixed_occ_FEC_ind = (mixed_occs_FEC[0][ind_config],
                         mixed_occs_FEC[1][ind_config],
                         mixed_occs_FEC[2][ind_config])
    uniq_id = np.unique(mixed_occ_FEC_ind[0])
    mixed_id = np.setdiff1d(uniq_id, np.array([0]))
    assert mixed_id.size == 1, 'There must be only one mixed_id.'
    face_dist = pair_dist_face[ind_config, :, :, :]
    edge_dist = pair_dist_edge[ind_config, :, :, :]
    corner_dist = pair_dist_corner[ind_config, :, :, :]
    if(get_terms):
        Esite, Eterms = eval_cubic_site_pair_energies(
            (face_coeff, edge_coeff, corner_coeff), mixed_id,
            mixed_occ_FEC_ind, (face_dist, edge_dist, corner_dist), dist0,
            get_terms_FEC=True)
    else:
        Esite = eval_cubic_site_pair_energies(
            (face_coeff, edge_coeff, corner_coeff), mixed_id,
            mixed_occ_FEC_ind, (face_dist, edge_dist, corner_dist), dist0)

    Ereg_soln = Wtbl*Xtbl*(1-Xtbl)
    Etot = Esite + Ereg_soln

    if(get_terms):
        return Etot, Eterms, Ereg_soln
    else:
        return Etot


def eval_mix_pair_coeff(Ecoeff, mixed_id, occ, dist, dist0):
    if(Ecoeff.ndim == 1):
        Ecoeff = Ecoeff.reshape([1, -1])
    #
    dist = dist.reshape(-1)
    occ = occ.reshape(-1)
    Npair = len(dist)
    # ind_contrib = np.where(np.in1d(occ, mixed_id))[0]
    Epair = np.zeros(Npair)
    for ind, id in enumerate(mixed_id):
        Epair[np.where(occ == id)[0]] = np.polyval(
            Ecoeff[ind], dist[np.where(occ == id)[0]]-dist0)
    Etot = 0.5*np.sum(Epair)
    return Etot


def eval_cubic_site_pair_energies(coeff_FEC, mixed_id, mixed_occs_FEC,
                                  pair_dists_FEC, dist0, get_terms_FEC=False):
    face_occs, edge_occs, corner_occs = mixed_occs_FEC
    face_dists, edge_dists, corner_dists = pair_dists_FEC

    Nconfig = face_occs.shape[0]

    Etot = []
    Eterms = []
    for indC in range(Nconfig):
        # if(face_occs.size==0):
        if(coeff_FEC[0].size == 0):
            face_occ = np.empty([0])
            face_dist = np.empty([0])
        else:
            face_occ = face_occs[indC]
            face_dist = face_dists[indC]

        # if(edge_occs.size==0):
        if(coeff_FEC[1].size == 0):
            edge_occ = np.empty([0])
            edge_dist = np.empty([0])
        else:
            edge_occ = edge_occs[indC]
            edge_dist = edge_dists[indC]

        # if(corner_occs.size==0):
        if(coeff_FEC[2].size == 0):
            corner_occ = np.empty([0])
            corner_dist = np.empty([0])
        else:
            corner_occ = corner_occs[indC]
            corner_dist = corner_dists[indC]

        mixed_occ_FEC = (face_occ, edge_occ, corner_occ)
        pair_dist_FEC = (face_dist, edge_dist, corner_dist)

        if(get_terms_FEC):
            iEtot, iEterms = eval_cubic_site_pair_energy(
                coeff_FEC, mixed_id, mixed_occ_FEC, pair_dist_FEC, dist0,
                get_terms_FEC=True)
            Etot.append(iEtot)
            Eterms.append(iEterms)
        else:
            iEtot = eval_cubic_site_pair_energy(
                coeff_FEC, mixed_id, mixed_occ_FEC, pair_dist_FEC, dist0,
                get_terms_FEC=False)
            Etot.append(iEtot)

    if(get_terms_FEC):
        return np.array(Etot), np.array(Eterms)
    else:
        return np.array(Etot)


def eval_cubic_site_pair_energy(coeff_FEC, mixed_id, mixed_occ_FEC,
                                pair_dist_FEC, dist0, get_terms_FEC=False):
    face_occ, edge_occ, corner_occ = mixed_occ_FEC
    face_dist, edge_dist, corner_dist = pair_dist_FEC
    NV = face_dist.shape[0]

    # assert np.unique(np.hstack((face_occ,edge_occ,corner_occ))) == \
    #     np.hstack((0,np.sort(mixed_id))), 'must define all mixed occ present'

    Eface = np.empty(NV)
    Eedge = np.empty(NV)
    Ecorner = np.empty(NV)
    for indV in range(NV):
        if(coeff_FEC[0].size == 0):
            Eface[indV] = 0
        else:
            Eface[indV] = eval_mix_pair_coeff(coeff_FEC[0], mixed_id,
                                              face_occ, face_dist[indV], dist0)
        #
        if(coeff_FEC[1].size == 0):
            Eedge[indV] = 0
        else:
            Eedge[indV] = eval_mix_pair_coeff(coeff_FEC[1], mixed_id,
                                              edge_occ, edge_dist[indV], dist0)
        #
        if(coeff_FEC[2].size == 0):
            Ecorner[indV] = 0
        else:
            Ecorner[indV] = eval_mix_pair_coeff(
                coeff_FEC[2], mixed_id, corner_occ, corner_dist[indV], dist0)
        #
    #
    Etot = Eface + Eedge + Ecorner
    if(get_terms_FEC):
        return Etot, np.vstack((Eface, Eedge, Ecorner))
    else:
        return Etot


def eval_morse_site_pair_energies(ntyp, logEd, logre, logdrs, occs, nn_pairs,
                                  nn_pair_dist_tbl, getpairs=False):
    # nconfig = nn_pair_dist_tbl.shape[0]

    Etot_tbl = np.empty(nn_pair_dist_tbl.shape[0:2])
    Epairs_tbl = np.empty(nn_pair_dist_tbl.shape)
    for ind, (occ, nn_pair_dists) in enumerate(zip(occs, nn_pair_dist_tbl)):
        # import ipdb as pdb;pdb.set_trace()
        iEtot, iEpairs = eval_morse_site_pair_energy(
            ntyp, logEd, logre, logdrs, occ, nn_pairs, nn_pair_dists,
            getpairs=True)
        Etot_tbl[ind, :] = iEtot
        Epairs_tbl[ind, :, :] = iEpairs

    if(getpairs):
        return Etot_tbl, Epairs_tbl
    else:
        return Etot_tbl


def eval_morse_site_pair_energy(ntyp, logEd, logre, logdrs, occ, nn_pairs,
                                nn_pair_dists, getpairs=False):
    """
    Evaluate site pair energy of a single configuration.
    """
    indup = np.array(np.triu_indices(ntyp)).T
    if(np.isscalar(logEd)):
        logEd = np.array([logEd])
    if(np.isscalar(logre)):
        logre = np.array([logre])
    if(np.isscalar(logdrs)):
        logdrs = np.array([logdrs])
    assert logEd.shape[0] == indup.shape[0], \
        'logEd must define 1 param per pair'
    assert logre.shape[0] == indup.shape[0], \
        'logre must define 1 param per pair'
    assert logdrs.shape[0] == indup.shape[0], \
        'logdrs must define 1 param per pair'

    nn_pair_dists = np.squeeze(nn_pair_dists)
    nn_pairs = np.squeeze(nn_pairs)
    occ = np.squeeze(occ)

    # nvol = nn_pair_dists.shape[0]
    # npairs = nn_pairs.shape[0]

    Ed = np.exp(logEd)
    re = np.exp(logre)
    drs = np.exp(logdrs)

    # determine appropriate pair index according to occs and nn_pairs
    pair_occs = occ[nn_pairs]
    pair_occ_inds = np.empty(pair_occs.shape[0], dtype=int)
    for ind, pair_occ in enumerate(pair_occs):
        pair_occ_inds[ind] = np.where(np.all(indup == np.sort(pair_occ),
                                             axis=1))[0]
    #

    Epairs = np.empty(nn_pair_dists.shape)
    for i, (ipair, pair_dist) in enumerate(zip(pair_occ_inds, nn_pair_dists.T)):
        try:
            iEpair = potentials.morse(pair_dist, [Ed[ipair], re[ipair],
                                                  drs[ipair]])
        except:
            iEpair = np.tile(np.Inf, pair_dist.shape)

        Epairs[:, i] = iEpair

    Etot = np.sum(Epairs, axis=1)
    if(getpairs):
        return Etot, Epairs
    else:
        return Etot


#
def eval_ideal_xred(dlogV, X, xred_coeff_endmem):
    assert np.sum(X) == 1, 'component fraction X must sum to one'
    assert np.all(X >= 0) & np.all(X <= 1), \
        'component fractions X must be between 0 and 1'
    assert len(X) == len(xred_coeff_endmem), \
        'Must provide component fraction X for each endmember'

    # Ncomp = len(xred_coeff_endmem)
    Nat = xred_coeff_endmem[0].shape[0]
    Ncoeff = xred_coeff_endmem[0].shape[-1]

    xred = np.zeros([Nat, 3])

    xred_coeff_endmem = np.array(xred_coeff_endmem)

    # Regularize pos0 values so that they are consistent
    #  allowing weighted averages
    for ind in range(Nat):
        ind_pos0_endmem = xred_coeff_endmem[:, ind, :, -1]
        xred_coeff_endmem[:, ind, :, -1] = geom.cont_scl_pos(ind_pos0_endmem)

        # xred_coeff_endmem[:,ind,:,-1] = xtalsim.cont_scl_pos(ind_pos0_endmem)

    xred_coeff = np.zeros([Nat, 3, Ncoeff])
    for ixred_coeff, iX in zip(xred_coeff_endmem, X):
        xred_coeff += iX*ixred_coeff

    for ind in range(Nat):
        xred[ind, 0] = np.polyval(xred_coeff[ind, 0, :], dlogV)
        xred[ind, 1] = np.polyval(xred_coeff[ind, 1, :], dlogV)
        xred[ind, 2] = np.polyval(xred_coeff[ind, 2, :], dlogV)

    return xred
