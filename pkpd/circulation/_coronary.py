import numpy as np

from ._common import _psi


#########################
# Circulation Pressures #
#########################

def _p_eq(P):
    """
    Equations A.77 and A.83

    :param P:
    :return:
    """
    return P


def _pcor(V, V0, C, Kxp, Kxv):
    """
    Equations from A.78 to A.80

    :param V:
    :param V0:
    :param C:
    :param Kxp:
    :param Kxv:
    :return:
    """
    psi = _psi(V, Kxp, Kxv)
    return (V - V0) / C - psi


def _pim(Plv):
    """
    Equation A.84

    :param Plv:
    :return:
    """
    return np.abs(Plv / 2)


def _pcorc(P, Plv):
    """
    Equations A.81 and A.82

    :param P:
    :param Pim:
    :return:
    """
    return P + _pim(Plv)


############################
# Circulation Forward Flow #
############################

def _fcor(Ppos, Pneg, R):
    """
    Equations from A.85 to A.88

    :param Ppos:
    :param Pneg:
    :param R:
    :return:
    """
    return (Ppos - Pneg) / R


###########################
# Circulation Radial Flow #
###########################

def _vcor(Fin, Fout):
    """
    Equations from A.89 to A.92

    :param Fin:
    :param Fout:
    :return:
    """
    return Fin - Fout
