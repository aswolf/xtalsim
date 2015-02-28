"""
potentials -
"""

import numpy as np
import atsim.potentials.potentialforms as pairpot


def morse(r, params):
    Ed = params[0]
    re = params[1]
    drs = params[2]

    gam = 1.0/(drs*re)
    E = map(pairpot.morse(gam, re, Ed), r)
    return np.array(E)
