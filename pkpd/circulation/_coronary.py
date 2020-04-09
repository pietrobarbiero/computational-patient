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


def _equations(t,
               Pcorintrac, Pcorcapc, Pcorvnc,
               Pcorepi, Pcorintra, Pcorcap, Pcorvn,
               Vcorepi, Vcorintra, Vcorcap, Vcorvn, Vcorcirc,
               Fcorepi, Fcorintra, Fcorcap, Fcorvn,
               # heart
               Plv, Pra,
               # systemic
               Paop,
               # parameters
               Rcorepi, Rcorintra, Rcorcap, Rcorvn,
               Ccorepi, Ccorintra, Ccorcap, Ccorvn,
               Vcorepi0, Vcorintra0, Vcorcap0, Vcorvn0,
               Kxp, Kxp1, Kxv, Kxv1):

    # pressures
    Pcorepi = _p_eq(Paop)
    Pcorintra = _pcor(Vcorintra, Vcorintra0, Ccorintra, Kxp1, Kxv1)
    Pcorcap = _pcor(Vcorcap, Vcorcap0, Ccorcap, Kxp1, Kxv1)
    Pcorvn = _pcor(Vcorvn, Vcorvn0, Ccorvn, Kxp, Kxv)
    Pcorintrac = _pcorc(Pcorintra, Plv)
    Pcorcapc = _pcorc(Pcorcap, Plv)
    Pcorvnc = _p_eq(Pcorvn)

    # flows
    Fcorepi = _fcor(Pcorepi, Pcorintrac, Rcorepi)
    Fcorintra = _fcor(Pcorintrac, Pcorcapc, Rcorintra)
    Fcorcap = _fcor(Pcorcapc, Pcorvnc, Rcorcap)
    Fcorvn = _fcor(Pcorvnc, Pra, Rcorvn)

    # differential equations
    # d_Vcorepi_dt = _vcor(Flv, d_Vaop_dt + Faop + Fcorepi)
    d_Vcorintra_dt = _vcor(Fcorepi, Fcorintra)
    d_Vcorcap_dt = _vcor(Fcorintra, Fcorcap)
    d_Vcorvn_dt = _vcor(Fcorcap, Fcorvn)

    return d_Vcorintra_dt, d_Vcorcap_dt, d_Vcorvn_dt
