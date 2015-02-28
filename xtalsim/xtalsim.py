"""
xtalsim - Package for manipulating atom positions for mineral phys applications

Implicit Dependencies:
    - atomic simulation environment (ASE):
        - xtalsim is meant to compliment ase, providing features that are left
        out
        - xtalsim inherits many of the conventions of ase:
            - cell: *rows* defines lattice vectors of xtal
            - scl_pos_arr: *rows* define reduced atomic coordinates
            - Thus absolute positions are given:
                - abs_pos_arr = np.dot(scl_pos_arr, cell)
                        OR
                - abs_pos_arr = np.dot(cell.T, scl_pos_arr.T).T
            - Invert scaled position from absolute position
                - scl_pos_arr = np.linalg.solve(cell.T, abs_pos_arr.T).T

"""

import atsim.potentials.potentialforms as pairpot
import numpy as np
import numpyext as npx
from scipy import optimize as optim

# NOTE: UNITS ARE P=GPa, E = Ha, V = Ang^3
#  This requires conversion of dE/dV to get pressure units
#  dE/dV = Ha/Ang^3 * eV/Ha * GPa/(eV/Ang^3)
EV_PER_ANG3_IN_GPA = 160.2176487
EV_PER_HA = 27.211396
PUNIT_CONV = EV_PER_HA*EV_PER_ANG3_IN_GPA

# High level potential modeling





#def eval_ideal_xred(dlogV, X, xred_coeff_endmem):
#    assert np.all(X>=0) & np.all(X<=1), \
#        'component fractions X must be between 0 and 1'
#    assert len(X)==len(xred_coeff_endmem), \
#        'Must provide component fraction X for each endmember'
#
#    xred = np.zeros(xred_coeff_endmem[0].shape)
#    Nat = xred.shape[0]
#    Ncomp = len(xred_coeff_endmem)
#
#    xred_endmems = []
#    for ixred_coeff in xred_coeff_endmem:
#        ixred_endmem = np.zeros([Nat, 3])
#        for ind in range(Nat):
#            ixred_endmem[ind, 0] = np.polyval(ixred_coeff[0,:], dlogV)
#            ixred_endmem[ind, 1] = np.polyval(ixred_coeff[1,:], dlogV)
#            ixred_endmem[ind, 2] = np.polyval(ixred_coeff[2,:], dlogV)
#
#        xred_endmems.append(ixred_endmem)
#
#    xred_endmems = np.array(xred_endmems)
#    X_wtfac = np.tile(X,[Ncomp,1]).T
#    for ind in range(Nat):
#        ind_xred_endmems = cont_scl_pos(xred_endmems[:,ind,:])
#        xred[ind,:] = X_wtfac*ind_xred_endmems
#
#    return xred

# High level atomic operations

# def neighbor_dists(atoms, Nneighbor):
#     """
#     Return set of absolute distances and index of N neighbor atoms
#     """
#     Nat = atoms.get_number_of_atoms()
#     Ncell = 3
#
#     atoms.repeat([Ncell,Ncell,Ncell])
#
#     dist_nn = np.empty(Nat)
#     ind_nn = np.empty(Nat)
#     for ind in range(Nat):
#         dist_pair = atoms.get_distances(ind, range(Nat), mic=True)
#         indS = np.argsort(dist_pair)
#         dist_nn[ind] = dist_pair[indS[1]]
#         ind_nn[ind] = indS[1]
#
#     return dist_nn, ind_nn
#



# pair potential Functions
