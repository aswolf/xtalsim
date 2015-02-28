"""
eos - xtalsim subpackage for evaluating equations of state

"""

import numpy as np
from scipy import optimize as optim
import const


def log_eos_E(V, eos_coeff, logVref):
    return np.polyval(eos_coeff, np.log(V)-logVref)


def log_eos_P(V, eos_coeff, logVref):
    Ederiv = np.polyval(np.polyder(eos_coeff), np.log(V)-logVref)
    return -const.PUNIT_CONV*Ederiv/V


def inv_log_eos_P(P, eos_coeff, logVref):
    Vbnds = np.exp(logVref+np.array([-.3, .3]))

    def calcPdiff(V): return log_eos_P(V, eos_coeff, logVref)-P

    return optim.brentq(calcPdiff, Vbnds[0], Vbnds[1])
