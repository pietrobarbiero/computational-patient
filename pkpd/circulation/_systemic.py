import numpy as np

from ._common import _psi


def _aortic_flow_change(MAPmeas, MAPmod, KCOMAP):
    """
    Equation A.27

    :param MAPmeas:
    :param MAPmod:
    :param KCOMAP:
    :return:
    """
    return KCOMAP * (MAPmeas - MAPmod)


def _pulmonary_valve_flow_change(Frv, Frv_sm, tauC0):
    """
    Equation A.28

    :param Frv:
    :param Frv_sm:
    :param tauC0:
    :return:
    """
    return (Frv - Frv_sm) / tauC0


def _cardiac_output(Frv_sm):
    """
    Equation A.29

    :param Frv_sm:
    :return:
    """
    return Frv_sm


def _stroke_volume(COmod, HR):
    """
    Equation A.30

    :param COmod:
    :param HR:
    :return:
    """
    # TODO: double check: "COmod / HR" or "COmod * HR" ??
    # [L] * [min^-1] / [min^-1] = [L]
    # it looks like Eq. A.30 is wrong in the original paper!
    return COmod / HR


def _arterial_blood_pressure(ABPmeas, t, offv):
    """
    Equation A.31

    :param ABPmeas:
    :param t:
    :param offv:
    :return:
    """
    # TODO: what if ABPmeas was a function or a stream?
    if t >= offv:
        return ABPmeas[t + offv]

    return 0


def _abp_change(ABPshift, ABPfol, tauABP):
    """
    Equation A.32

    :param ABPshift:
    :param ABPfol:
    :param tauABP:
    :return:
    """
    return (ABPshift - ABPfol) / tauABP


def _kv(Kv1, Ksv):
    """
    Equation A.33

    :param Kv1:
    :param Ksv:
    :return:
    """
    return Kv1 * Ksv


def _aortic_afterload(ABPshift):
    """
    Equation A.34

    :param ABPshift:
    :return:
    """
    return ABPshift


##################################
# Systemic Circulation Pressures #
##################################

def _pressure_aop_change(Flv, d_Vaop_dt, Faop, Fcorepi, Ccorepi):
    """
    Equation A.35

    :param Flv:
    :param d_Vaop_dt:
    :param Faop:
    :param Fcorepi:
    :param Ccorepi:
    :return:
    """
    return (Flv - d_Vaop_dt - Faop - Fcorepi) / Ccorepi


def _map(Rtaod, Rcrb, AOFmod, Faod, Vaod, Vaod0, Caod, Kxp, Kxv, Pvc):
    """
    Equation A.36

    :param Rtaod:
    :param Rcrb:
    :param AOFmod:
    :param Faod:
    :param Vaod:
    :param Vaod0:
    :param Caod:
    :param Kxp:
    :param Kxv:
    :param Pvc:
    :return:
    """
    psi = _psi(Vaod, Kxp, Kxv)
    return (Rtaod * Rcrb * AOFmod - Rtaod * Rcrb * Faod + ((Vaod - Vaod0) * Rcrb / Caod)
            - Rcrb * psi + Pvc * Rtaod) / (Rcrb + Rtaod)


def _psap(Vsap, Vsap0, Csap, Kxp, Kxv):
    """
    Equation A.37

    :param Vsap:
    :param Vsap0:
    :param Csap:
    :param Kxp:
    :param Kxv:
    :return:
    """
    psi = _psi(Vsap, Kxp, Kxv)
    return (Vsap - Vsap0) / Csap - psi


def _psa_a(Kc, Vsa, Vsa0, Do):
    """
    Equation A.38

    :param Kc:
    :param Vsa:
    :param Vsa0:
    :param Do:
    :return:
    """
    return Kc * np.log(((Vsa - Vsa0) / Do) + 1)


def _psa_p(Kp1, Kp2, tau_p, Vsa, Vsa0):
    """
    Equation A.39

    :param Kp1:
    :param Kp2:
    :param tau_p:
    :param Vsa:
    :param Vsa0:
    :return:
    """
    return Kp1 * np.exp(tau_p * (Vsa - Vsa0)) + Kp2 * (Vsa - Vsa0) ** 2


def _psa(f_vaso, Psa_a, Psa_p):
    """
    Equation A.40

    :param f_vaso:
    :param Psa_a:
    :param Psa_p:
    :return:
    """
    return f_vaso * Psa_a + (1 - f_vaso) * Psa_p


def _psc(Vsc, Vsc0, Csc, Kxp, Kxv):
    """
    Equation A.41

    :param Vsc:
    :param Vsc0:
    :param Csc:
    :param Kxp:
    :param Kxv:
    :return:
    """
    psi = _psi(Vsc, Kxp, Kxv)
    return (Vsc - Vsc0) / Csc - psi


def _psv(Kv, Vmax_sv, Vsv):
    """
    Equation A.42

    :param Kv:
    :param Vmax_sv:
    :param Vsv:
    :return:
    """
    return -Kv * np.log((Vmax_sv / Vsv) - 0.99)


def _pvc(Vvc, Vvc0, Vmin_vc, K1, K2, D2, Kxp, Kxv):
    psi = _psi(Vvc, Kxp, Kxv)
    if Vvc > Vvc0:
        return K1 * (Vvc - Vvc0) - psi

    else:
        return D2 + K2 * np.exp(Vvc / Vmin_vc) - psi


#####################################
# Systemic Circulation Forward Flow #
#####################################

def _forward_flow(Fin, Fout, norm):
    """
    Equations from A.44 to A.51

    :param Fin:
    :param Fout:
    :param norm:
    :return:
    """
    # TODO: double check Equation A.44
    # (Paop - Faop * Raop - ABPmeas) / Laop ???
    # ABPmeas [mmHg]
    return (Fin - Fout) / norm


####################################
# Systemic Circulation Radial Flow #
####################################

def _vaop_change(Paop, Vaop, Vaop0, Caop, Rtaop):
    """
    Equation A.52

    :param Paop:
    :param Vaop:
    :param Vaop0:
    :param Caop:
    :param Rtaop:
    :return:
    """
    return (Paop - ((Vaop - Vaop0) / Caop)) / Rtaop


def _v_change(Vpos, Vneg):
    """
    Equations from A.53 to A.58

    :param Vpos:
    :param Vneg:
    :return:
    """
    # TODO: double check equation A.58
    # Fsv + Fcrb - Fvc OR Fsv - Fcrb - Fvc
    return Vpos - Vneg


def _mapmeas_change(ABPshift, MAPmeas, tauMAP):
    """

    :param ABPshift:
    :param MAPmeas:
    :param tauMAP:
    :return:
    """
    return (ABPshift - MAPmeas) / tauMAP


def _comea_change(PAFmeas, COmea, tauCO):
    """

    :param PAFmeas:
    :param COmea:
    :param tauCO:
    :return:
    """
    return (PAFmeas - COmea) / tauCO


##################################
# Nonlinear systemic resistances #
##################################

def _rsa(Kr, f_vaso, Vsa, Vsa_max, Rsa0):
    """
    Equation A.59

    :param Kr:
    :param f_vaso:
    :param Vsa_max:
    :param Rsa0:
    :return:
    """
    return (Kr * np.exp(4 * f_vaso)) + (Kr * (Vsa_max / Vsa) ** 2) + Rsa0


def _rvc(KR, Vmax_vc, Vvc, R0):
    """
    Equation A.60

    :param KR:
    :param Vmax_vc:
    :param Vvc:
    :param R0:
    :return:
    """
    return (KR * (Vmax_vc / Vvc) ** 2) + R0


def _equations(t, HR, PAFmeas, ABPmeas,
               Rsa, Rvc,
               Paop, Paod, Psa_a, Psa_p, Psa, Psap, Psc, Psv, Pvc,
               Faop, Faod, Fsa, Fsap, Fsc, Fsv, Fvc, Fcrb,
               Vaop, Vaod, Vsa, Vsap, Vsc, Vsv, Vvc,
               Kv, AOFmod,
               ABPshift, ABPfol, MAPmeas, MAPmod, COmea,
               # heart
               Frv, Frv_sm, Flv, Pra,
               # coronary
               Fcorepi,
               # parameters
               KCOMAP, tauC0, offv,
               Raop, Rtaop, Rtaod, Rcrb, Rsap, Raod, Rsc, Rsv,
               Caop, Caod, Csap, Csc,
               Vaop0, Vaod0, Vsap0, Vsc0,
               Laop, Laod,
               Kc, Do, Vsa0, Vsa_max,
               Kp1, Kp2, Kr, Rsa0, tau_p, Ksv, Kv1,
               Vmax_sv, D2, K1, K2, KR, R0,
               Vvc0, Vmax_vc, Vmin_vc, tauCO,
               Kxp, Kxv, Kxv1, Kxp1,
               tauMAP, tauABP,
               # coronary
               Ccorepi,
               # baroreceptor
               f_vaso,
               ):
    # misc
    COmod = _cardiac_output(Frv_sm)
    SV = _stroke_volume(COmod, HR)
    ABPshift = _arterial_blood_pressure(ABPmeas, t, offv)
    Kv = _kv(Kv1, Ksv)
    MAPmod = _map(Rtaod, Rcrb, AOFmod, Faod,
                  Vaod, Vaod0, Caod, Kxp, Kxv, Pvc)

    # pressures
    Paod = _aortic_afterload(ABPshift)
    Psap = _psap(Vsap, Vsap0, Csap, Kxp, Kxv)
    Psa_a = _psa_a(Kc, Vsa, Vsa0, Do)
    Psa_p = _psa_p(Kp1, Kp2, tau_p, Vsa, Vsa0)
    Psa = _psa(f_vaso, Psa_a, Psa_p)
    Psc = _psc(Vsc, Vsc0, Csc, Kxp, Kxv)
    Psv = _psv(Kv, Vmax_sv, Vsv)
    Pvc = _pvc(Vvc, Vvc0, Vmin_vc, K1, K2, D2, Kxp, Kxv)

    # flows
    Fcrb = _forward_flow(MAPmod, Pvc, Rcrb)
    Fsap = _forward_flow(Psap, Psa, Rsap)
    Fsa = _forward_flow(Psa, Psc, Rsa)
    Fsc = _forward_flow(Psc, Psv, Rsc)
    Fsv = _forward_flow(Psv, Pvc, Rsv)
    Fvc = _forward_flow(Pvc, Pra, Rvc)

    # differential equations
    d_ABPfol_dt = _abp_change(ABPshift, ABPfol, tauABP)
    d_AOFmod_dt = _aortic_flow_change(MAPmeas, MAPmod, KCOMAP)
    d_Vaop_dt = _vaop_change(Paop, Vaop, Vaop0, Caop, Rtaop)
    d_Vaod_dt = _v_change(AOFmod, Faod + Fcrb)
    d_Vsa_dt = _v_change(Fsap, Fsa)
    d_Vsap_dt = _v_change(Faod, Fsap)
    d_Vsc_dt = _v_change(Fsa, Fsc)
    d_Vsv_dt = _v_change(Fsc, Fsv)
    d_Vvc_dt = _v_change(Fsv + Fcrb, Fvc)
    d_Frv_sm_dt = _pulmonary_valve_flow_change(Frv, Frv_sm, tauC0)
    d_Faop_dt = _forward_flow(Paop, Faop * Raop + Paod, Laop)
    d_Faod_dt = _forward_flow(MAPmod, Faod * Raod - Psap, Laod)
    d_Paop_dt = _pressure_aop_change(Flv, d_Vaop_dt, Faop, Fcorepi, Ccorepi)

    return d_ABPfol_dt, d_AOFmod_dt, d_Vaop_dt, d_Vaod_dt, \
           d_Vsa_dt, d_Vsap_dt, d_Vsc_dt, \
           d_Vsv_dt, d_Vvc_dt, d_Frv_sm_dt, \
           d_Faop_dt, d_Faod_dt, d_Paop_dt
