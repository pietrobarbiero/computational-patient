import numpy as np

from ._common import _psi


#########################
# Circulation Pressures #
#########################


def _ppap(Prv, Ppap, Rtpap, Rrv, Fpap, Vpap, Vpap0,
          Cpap, Kxp, Kxv):
    """
    Equation A.61

    :param Prv:
    :param Ppap:
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
    psi = _psi(Vpap, Kxp, Kxv)
    if Prv > Ppap:
        return ((Rtpap * Prv) - (Rrv * Fpap * Rtpap) + (
                Rrv * (((Vpap - Vpap0) / Cpap) - psi))) / (Rtpap + Rrv)

    # TODO: double check equation A.61.b
    return (-(Rrv * Fpap * Rtpap) + (Rrv * (((Vpap - Vpap0) / Cpap) - psi))) / Rrv


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


def _equations(t,
               Ppap, Ppad, Ppa, Ppc, Ppv,
               Vpap, Vpad, Vpa, Vpc, Vpv,
               Fpap, Fpad, Fps, Fpa, Fpc, Fpv,
               # heart
               Prv, Pla, Frv,
               #parameters
               Rtpap, Rtpad, Rpap, Rpad, Rps, Rpa, Rpc, Rpv,
               Cpap, Cpad, Cpa, Cpc, Cpv,
               Vpap0, Vpad0, Vpa0, Vpc0, Vpv0,
               Lpap, Lpad,
               # heart
               Rrv,
               # systemic
               Kxp, Kxv):

    # pressures
    Ppap = _ppap(Prv, Ppap, Rtpap, Rrv, Fpap, Vpap, Vpap0, Cpap, Kxp, Kxv)
    Ppad = _ppad(Vpad, Vpad0, Kxp, Kxv, Fpap, Rtpad, Fpad, Cpad)
    Ppa = _pp(Vpa, Vpa0, Cpa, Kxp, Kxv)
    Ppc = _pp(Vpc, Vpc0, Cpc, Kxp, Kxv)
    Ppv = _pp(Vpv, Vpv0, Cpv, Kxp, Kxv)

    # flows
    Fps = _fp(Ppa, Ppv, Rps)
    Fpa = _fp(Ppa, Ppc, Rpa)
    Fpc = _fp(Ppc, Ppv, Rpc)
    Fpv = _fp(Ppv, Pla, Rpv)

    # differential equations
    d_Vpad_dt = _vp(Fpap, Fpad)
    d_Vpap_dt = _vp(Frv, Fpap)
    d_Vpa_dt = _vp(Fpad, Fps + Fpa)
    d_Vpc_dt = _vp(Fpa, Fpc)
    d_Vpv_dt = _vp(Fpc + Fps, Fpv)
    d_Fpap_dt = _fpa(Ppap, Ppad + Fpap*Rpap, Lpap)
    d_Fpad_dt = _fpa(Ppad, Ppa + Fpad*Rpad, Lpad)

    return d_Vpad_dt, d_Vpap_dt, d_Vpa_dt, d_Vpc_dt, \
           d_Vpv_dt, d_Fpap_dt, d_Fpad_dt
