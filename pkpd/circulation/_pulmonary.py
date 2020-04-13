from ._common import _psi


#########################
# Circulation Pressures #
#########################

def _ppap1(Prv, Rtpap, Rrv, Fpap, Vpap, Vpap0, Cpap, Kxp, Kxv):
    psi = _psi(Vpap, Kxp, Kxv)
    return ((Rtpap * Prv) - (Rrv * Fpap * Rtpap) +
            (Rrv * (((Vpap - Vpap0) / Cpap) - psi))) / (Rtpap + Rrv)


def _ppap2(Rtpap, Rrv, Fpap, Vpap, Vpap0, Cpap, Kxp, Kxv):
    psi = _psi(Vpap, Kxp, Kxv)
    # TODO: double check equation A.61.b
    return (-(Rrv * Fpap * Rtpap) + (Rrv * (((Vpap - Vpap0) / Cpap) - psi))) / Rrv


def _ppap(Prv, Rtpap, Rrv, Fpap, Vpap, Vpap0, Cpap, Kxp, Kxv):
    """
    Equation A.61

    :param Prv:
    :param Rtpap:
    :param Rrv:
    :param Fpap:
    :param Vpap:
    :param Vpap0:
    :param Cpap:
    :param Kxp:
    :param Kxv:
    :return:
    """
    ppap1 = _ppap1(Prv, Rtpap, Rrv, Fpap, Vpap, Vpap0, Cpap, Kxp, Kxv)
    if Prv > ppap1:
        return ppap1

    ppap2 = _ppap2(Rtpap, Rrv, Fpap, Vpap, Vpap0, Cpap, Kxp, Kxv)
    return ppap2


def _ppad(Vpad, Vpad0, Kxp, Kxv, Fpap, Rtpad, Fpad, Cpad):
    """
    Equation A.62

    :param Vpad:
    :param Vpad0:
    :param Kxp:
    :param Kxv:
    :param Fpap:
    :param Rtpad:
    :param Fpad:
    :param Cpad:
    :return:
    """
    psi = _psi(Vpad, Kxp, Kxv)
    return Fpap * Rtpad - Fpad * Rtpad + (Vpad - Vpad0) / Cpad - psi


def _pp(Vp, Vp0, Cp, Kxp, Kxv):
    """
    Equations from A.63 to A.65

    :param Vp:
    :param Vp0:
    :param Cp:
    :param Kxp:
    :param Kxv:
    :return:
    """
    psi = _psi(Vp, Kxp, Kxv)
    return (Vp - Vp0) / Cp - psi


############################
# Circulation Forward Flow #
############################

def _fp(Pp1, Pp2, Rp):
    """
    Equations from A.66 to A.69

    :param Pp1:
    :param Pp2:
    :param Rp:
    :return:
    """
    return (Pp1 - Pp2) / Rp


###########################
# Circulation Radial Flow #
###########################

def _vp(Fin, Fout):
    """
    Equations from A.70 to A.74

    :param Fin:
    :param Fout:
    :return:
    """
    return Fin - Fout


def _fpa(Ppos, Pneg, Lp):
    """
    Equations A.75 and A.76

    :param Ppos:
    :param Pneg:
    :param Lp:
    :return:
    """
    return (Ppos - Pneg) / Lp
