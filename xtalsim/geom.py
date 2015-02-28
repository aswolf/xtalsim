"""
geom - xtalsim subpackage for manipulating atom positions
"""

import numpy as np
import numpyext as npx


def nearest_neighbor_dist(atoms):
    """
    Return absolute dist and index of nearest neighbor atoms
    """
    Nat = atoms.get_number_of_atoms()

    dist_nn = np.empty(Nat)
    ind_nn = np.empty(Nat)
    for ind in range(Nat):
        dist_pair = atoms.get_distances(ind, range(Nat), mic=True)
        indS = np.argsort(dist_pair)
        dist_nn[ind] = dist_pair[indS[1]]
        ind_nn[ind] = indS[1]

    return dist_nn, ind_nn


def match_nearest_equiv_atom(cell_ref, cell, scl_pos_ref_arr, scl_pos_arr,
                             tol=0.09):
    """
    Return index and dist that matches nearest equiv atom in reference pos array

    Uses absolute distance to determine quality of match
    """

    # Convert tolerance on distance to tol on squared dist
    tolsqr = tol**2

    Nref = scl_pos_ref_arr.shape[0]
    N = scl_pos_arr.shape[0]

    assert Nref >= N, 'Reference xtal must be equal size or supercell'
    assert np.mod(Nref, N) == 0, 'N(ref atoms) must be a multiple of N(atoms)'

    # If supercell, atom pos must be repeated to give equal number of atoms
    if(Nref > N):
        supercellFac = Nref/N
        scl_pos_arr = np.tile(scl_pos_arr, [supercellFac, 1])

    def diff_fun(scl_pos_ref_arr, scl_pos_arr):
        dist_sqr = calc_dist_sqr(cell_ref, cell, scl_pos_ref_arr, scl_pos_arr)
        return dist_sqr

    ind_ref, dist_sqr = match_ref_items(scl_pos_ref_arr, scl_pos_arr,
                                        diff_fun, tolsqr)
    dist = np.sqrt(dist_sqr)

    # If supercell, convert ind_ref back to unitcell references
    if(Nref > N):
        ind_ref = np.mod(ind_ref, N)

    return ind_ref, dist


def match_ref_scaled_pos(ref_pos, pos, tol=0.09):
    """
    Return index and dist that matches nearest atom in reference pos array

    ref_pos and pos should have same number of atoms
    Uses scaled distance (xred) to determine quality of match
    """
    def diff_fun(ref_pos, pos):
        diff = calc_scaled_dist(ref_pos, pos)[0]
        return diff

    return match_ref_items(ref_pos, pos, diff_fun, tol)


def match_ref_site_IDs(xtal_arr, site_ID_ref, pos_tol=0.09):
    ref_pos = xtal_arr[0].get_scaled_positions()
    ref_tags = xtal_arr[0].get_tags()
    # max_pos_diffs = np.empty(N)
    # site_IDs = np.tile(np.array(site_ID_ref), [N, 1])
    site_IDs = []
    max_pos_diffs = []
    for xtal in xtal_arr:
        pos = xtal.get_scaled_positions()
        tags = xtal.get_tags()
        ref_atom_ind, dist = match_ref_scaled_pos(ref_pos, pos, pos_tol)
        assert np.all(tags == ref_tags[ref_atom_ind]), \
            'Atom tags do NOT match. Atoms must have matching tags.'
        site_IDs.append(site_ID_ref[ref_atom_ind])
        max_pos_diffs.append(max(dist))
    assert max(max_pos_diffs) < pos_tol, \
        'max_pos diff not all below tolerance. These pos do NOT match.'
    return site_IDs, max_pos_diffs



# Low level Functions
def cont_scl_pos(xyz):
    xyz = np.mod(xyz-np.tile(xyz[0, :], [xyz.shape[0], 1])+.5, 1)-.5\
        + np.tile(xyz[0, :], [xyz.shape[0], 1])
    return xyz


def img_shft_vecs(nshft):
    shft = np.arange(-nshft, nshft+.1)
    xscl_shft, yscl_shft, zscl_shft = np.meshgrid(shft, shft, shft)
    img_shft = np.hstack((xscl_shft.reshape(-1), yscl_shft.reshape(-1),
                          zscl_shft.reshape(-1)))
    return img_shft


def get_supercell_img_shft(supercell, cell, scl_pos, decimal=6):
    Nat = scl_pos.shape[0]
    indat = np.arange(0, Nat)
    scl_pos_tile, img_shft_tile = tile_scl_pos(scl_pos)
    Nimg = scl_pos_tile.shape[0]/Nat
    #
    abs_pos_tile = np.dot(scl_pos_tile, cell)
    # Invert scaled position from absolute position
    scl_pos_super_tile = np.linalg.solve(supercell.T, abs_pos_tile.T).T
    # NOTE: unclear why modulo must be performed twice, but this makes it
    # work...
    scl_pos_super_tile = np.mod(np.mod(scl_pos_super_tile, 1.0).round(decimal),
                                1.0)
    pos_ind = np.hstack((np.reshape(np.tile(indat, [Nimg]), (-1, 1)),
                         scl_pos_super_tile))
    # When obtaining unique positions inside supercell,
    # be sure to return a list
    # that is sorted the same way every time
    ind_uniq_rows = npx.arg_unique_rows(pos_ind)
    indS = np.argsort(pos_ind[ind_uniq_rows, 0])
    ind_super = ind_uniq_rows[indS].reshape(-1, 2).T.reshape(-1)
    img_shft = img_shft_tile[ind_super, :]
    return img_shft


def calc_scl_pos_super(supercell, cell, scl_pos, img_shft):
    Nat = scl_pos.shape[0]
    Nimg = img_shft.shape[0]/Nat
    scl_pos_shft = np.tile(scl_pos, [Nimg, 1]) + img_shft
    abs_pos_shft = np.dot(scl_pos_shft, cell)
    # Invert scaled position from absolute position
    scl_pos_super = np.mod(np.linalg.solve(supercell.T, abs_pos_shft.T).T, 1.0)
    return scl_pos_super


def tile_scl_pos(scl_pos):
    Nat = scl_pos.shape[0]
    shft = np.arange(-1, 1.1)
    Nimg = len(shft)**3
    xscl_shft, yscl_shft, zscl_shft = np.meshgrid(shft, shft, shft)
    pos_shft = np.hstack((xscl_shft.reshape(-1), yscl_shft.reshape(-1),
                          zscl_shft.reshape(-1)))
    pos_shft = np.vstack(
        (np.tile(xscl_shft.reshape(-1), [Nat, 1]).T.reshape(-1),
         np.tile(yscl_shft.reshape(-1), [Nat, 1]).T.reshape(-1),
         np.tile(zscl_shft.reshape(-1), [Nat, 1]).T.reshape(-1))).T
    scl_pos_tile = np.tile(scl_pos, [Nimg, 1]) + pos_shft
    return scl_pos_tile, pos_shft


def calc_cell_com(cell):
    xcorner, ycorner, zcorner = np.meshgrid([0, 1], [0, 1], [0, 1])
    xcorner = xcorner.reshape(-1)
    ycorner = ycorner.reshape(-1)
    zcorner = zcorner.reshape(-1)

    np.array([xcorner, ycorner, zcorner])
    xyz_corner = np.empty([8, 3])
    xyz_corner[:, 0] = xcorner
    xyz_corner[:, 1] = ycorner
    xyz_corner[:, 2] = zcorner
    pos_corner = np.dot(xyz_corner, cell)
    pos_com = np.mean(pos_corner, 0)
    return pos_com


def calc_cell_dim(cell):
    dim = np.empty([3])
    dim[0] = np.sqrt(sum(cell[0]**2))
    dim[1] = np.sqrt(sum(cell[1]**2))
    dim[2] = np.sqrt(sum(cell[2]**2))
    return dim


def calc_dist_sqr(cell1, cell2, scl_pos1_arr, scl_pos2_arr):
    """
    Return squared absolute min img distance between atoms in two diff xtals

    """

    # ensure that pos_arr are both 2D
    if(scl_pos1_arr.ndim == 1):
        scl_pos1_arr = np.tile(scl_pos1_arr, (1, 1))
    if(scl_pos2_arr.ndim == 1):
        scl_pos2_arr = np.tile(scl_pos2_arr, (1, 1))
    assert scl_pos1_arr.shape == scl_pos2_arr.shape, \
        'scl_pos1_arr and scl_pos2_arr must have same shape'

    # Determine where center-of-mass of cell2 is relative to cell1 lattice
    # vectors
    pos_com2 = calc_cell_com(cell2)
    cell2_disp = np.linalg.solve(cell1.T, pos_com2.T).T
    scl_disp2 = np.round(cell2_disp)

    # Determine absolute position of atoms in xtal1
    pos1_arr = np.dot(scl_pos1_arr, cell1)

    shft = np.arange(-1, 1.1)
    Nimg = len(shft)**3
    xscl_shft, yscl_shft, zscl_shft = np.meshgrid(shft, shft, shft)
    xscl_shft = xscl_shft.reshape(-1)
    yscl_shft = yscl_shft.reshape(-1)
    zscl_shft = zscl_shft.reshape(-1)

    scl_pos2_img = np.empty([Nimg, 3])

    dist_sqr = np.empty(scl_pos1_arr.shape[0])
    for ind, scl_pos2 in enumerate(scl_pos2_arr):
        pos1 = pos1_arr[ind]
        x2_img = scl_pos2[0] + xscl_shft - scl_disp2[0]
        y2_img = scl_pos2[1] + yscl_shft - scl_disp2[1]
        z2_img = scl_pos2[2] + zscl_shft - scl_disp2[2]

        scl_pos2_img[:, 0] = x2_img
        scl_pos2_img[:, 1] = y2_img
        scl_pos2_img[:, 2] = z2_img

        pos2_img = np.dot(scl_pos2_img, cell2)
        pos_img_diff = pos2_img - np.tile(pos1, [Nimg, 1])
        dist_sqr_img = np.sum(pos_img_diff**2, 1)
        dist_sqr[ind] = (np.min(dist_sqr_img))
    return dist_sqr


def calc_neighbor_dist(cell, scl_pos_arr, Nneighbor, Nshft=2):
    """
    Return list of distances and indexes to neighboring atoms for every atom

    """

    # Determine where center-of-mass of cell2 is relative to cell1 lattice
    # vectors
    Natom = scl_pos_arr.shape[0]
    # scl_disp = np.round(cell_disp)

    # Determine absolute position of atoms in xtal
    # pos_arr = np.dot(scl_pos_arr, cell)
    atom_ids = np.arange(0, Natom)

    shft = np.arange(-Nshft, Nshft+.1)
    Nimg = len(shft)**3
    # pos_com = calc_cell_com(len(shft)*cell)
    xscl_shft, yscl_shft, zscl_shft = np.meshgrid(shft, shft, shft)
    xscl_shft = xscl_shft.reshape(-1)
    yscl_shft = yscl_shft.reshape(-1)
    zscl_shft = zscl_shft.reshape(-1)
    scl_pos_shft = np.vstack((xscl_shft, yscl_shft, zscl_shft)).T

    # Construct img positions
    scl_pos_img = np.empty([Nimg*Natom, 3])
    atom_id_img = np.empty([Nimg*Natom])
    for ind, scl_pos in enumerate(scl_pos_arr):
        scl_pos_img[ind*Nimg:(ind+1)*Nimg, :] = \
            np.tile(scl_pos, [Nimg, 1])+scl_pos_shft
        atom_id_img[ind*Nimg:(ind+1)*Nimg] = atom_ids[ind]

    pos_img = np.dot(scl_pos_img, cell)

    neigh_dist = np.empty([Natom, Nneighbor])
    neigh_ind = np.empty([Natom, Nneighbor])
    # Loop over original atoms
    for ind, (scl_pos, atomid) in enumerate(zip(scl_pos_arr, atom_ids)):
        id_match = np.where(atom_id_img == atomid)[0]
        iscl_img_pos = scl_pos_img[id_match]
        # find image closest to origin
        ref_img_ind = np.argmin(
            np.sum((iscl_img_pos-np.tile([.5, .5, .5], [Nimg, 1]))**2, axis=1))
        # calc distsqr to all other atoms
        idist_sqr = np.sum((pos_img-np.tile(pos_img[id_match[ref_img_ind], :],
                                            [Nimg*Natom, 1]))**2, axis=1)
        indS_dist = np.argsort(idist_sqr)[1:Nneighbor+1]
        neigh_dist[ind] = np.sqrt(idist_sqr[indS_dist])
        neigh_ind[ind] = atom_id_img[indS_dist]

    return neigh_dist, neigh_ind


def calc_scaled_dist(pos1_arr, pos2_arr):
    """
    Return scaled distance and relative position vectors

    Calculated for scaled position (reduced atomic positions, xred).
    """
    # ensure that pos_arr are both 2D
    if(pos1_arr.ndim == 1):
        pos1_arr = np.tile(pos1_arr, (1, 1))
    if(pos2_arr.ndim == 1):
        pos2_arr = np.tile(pos2_arr, (1, 1))
    assert pos1_arr.shape == pos2_arr.shape, \
        'pos1_arr and pos2_arr must have same shape'
    # calculate pos diff for each atom
    pos_diff = np.mod(pos1_arr, 1)-np.mod(pos2_arr, 1)
    # account for periodic boundary conditions
    pos_diff[np.where(pos_diff > 0.5)] -= 1.0
    pos_diff[np.where(pos_diff < -0.5)] += 1.0
    dist = np.sqrt(np.sum(pos_diff**2, 1))
    return dist, pos_diff


def match_ref_items(ref_items, items, diff_fun, tol):
    """
    Return index and diff to match items array with a reference items array

    ref_items and items must have same length
    User must provide diff_fun which returns scalar measured difference between
    two items or a list of items
    """
    assert len(items) == len(ref_items), 'items and ref_items must be same len'
    assert type(items) is np.ndarray, 'items must be a numpy array'
    assert type(ref_items) is np.ndarray, 'ref_items must be a numpy array'

    diff = diff_fun(ref_items, items)
    ref_ind = np.arange(len(items))
    if all(diff < tol):
        return ref_ind, diff
    # sort diff in descending order
    indS = np.argsort(diff)[::-1]
    Nmismatch = np.where(diff[indS] > tol)[0][-1] + 1
    ind_mismatch = indS[0:Nmismatch]
    items_mismatch = items[ind_mismatch]
    ref_items_mismatch = ref_items[ind_mismatch]
    swap_hist = np.arange(Nmismatch)
    # determine dimensionality of items to allow proper tiling to compare items
    # and ref_items
    items_tile_reps = np.repeat(1, items.ndim)
    # Dont need to go to end, just end-1
    for swap_ind in range(Nmismatch):
        # Step through each mismatched item and compare with remaining
        # unmatched ref_items
        curr_item = items_mismatch[swap_ind]
        prop_ref_items = ref_items_mismatch[swap_hist[swap_ind:]]
        # ensure tile repeat number matches the number of remaining
        # unmatched ref_items
        items_tile_reps[0] = Nmismatch-swap_ind
        prop_diff = diff_fun(np.tile(curr_item, items_tile_reps),
                             prop_ref_items)
        # swap indices to store item pair with the minimum difference
        swap_target = np.argmin(prop_diff)+swap_ind
        (swap_hist[swap_ind], swap_hist[swap_target]) = \
            (swap_hist[swap_target], swap_hist[swap_ind])
    ref_ind[ind_mismatch] = ind_mismatch[swap_hist]
    diff = diff_fun(ref_items[ref_ind], items)
    assert all(np.abs(diff) < tol), 'diff must all be less than tol'
    return ref_ind, diff
