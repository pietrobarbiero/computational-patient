import numpy as np


def _yi(t_rel, Ts):
    """
    Equation A.1

    :param t_rel:
    :param Ts:
    :return:
    """
    if 0.0 <= t_rel < Ts:
        return (1.0 - np.cos(np.pi * t_rel / Ts)) / 2.0
    elif Ts <= t_rel < 1.5 * Ts:
        return (1.0 + np.cos(2.0 * np.pi * (t_rel - Ts) / Ts)) / 2.0
    else:
        return 0.0


def t_rel(t, twave):
    """
    Equations A.2 and A.3

    :param t:
    :param twave:
    :return:
    """
    return t - twave


# Equations 4, 5, 6 are identities


def _trigger_A(t, tHB, HP, PRint, offv, Ts1a, Ts2):
    """
    Equations from A.7 to A.10

    :param t:
    :param tHB:
    :param HP:
    :param PRint:
    :param offv:
    :param Ts1a:
    :param Ts2:
    :return:
    """
    if t >= tHB - PRint - offv:
        HRa = 1 / HP
        Tsa = Ts1a * np.sqrt(Ts2 / HRa)
        tPwave = tHB - PRint - offv
        return HRa, Tsa, tPwave


def _trigger_B(t, tHB, HP, offv, Ts1v, Ts2, n,
               Vlv, Vlvd0, EDVLV, Vlvs0,
               Vrv, Vrvd0, EDVRV, Vrvs0,
               af_con):
    """
    Equations from A.11 to A.17

    :param t:
    :param tHB:
    :param HP:
    :param offv:
    :param Ts1v:
    :param Ts2:
    :param n:
    :param Vlv:
    :param Vlvd0:
    :param EDVLV:
    :param Vlvs0:
    :param Vrv:
    :param Vrvd0:
    :param EDVRV:
    :param Vrvs0:
    :param af_con:
    :return:
    """
    if t >= tHB - offv:
        HRv = 1 / HP
        Tsv = Ts1v * np.sqrt(Ts2 / HRv)
        tRwave = tHB - offv
        m = n

        if Vlv < Vlvd0:
            Vvarlvs0 = Vlvd0
        elif Vlv > EDVLV:
            Vvarlvs0 = Vlvs0
        else:
            Vvarlvs0 = (Vlvs0 - Vlvd0) * ((Vlv - Vlvd0) / (EDVLV - Vlvd0)) + Vlvd0

        if Vrv < Vrvd0:
            Vvarrvs0 = Vrvd0
        elif Vlv > EDVLV:
            Vvarrvs0 = Vrvs0
        else:
            Vvarrvs0 = (Vrvs0 - Vrvd0) * ((Vrv - Vrvd0) / (EDVRV - Vrvd0)) + Vrvd0

        af_con2 = af_con

        return HRv, Tsv, tRwave, Vvarlvs0, Vvarrvs0, af_con2


# Equation 18 is an identity


def _psi(V, Kxp, Kxv):
    """
    Psi

    :param V:
    :param Kxp:
    :param Kxv:
    :return:
    """
    return Kxp * (1 / (np.exp((V) / Kxv) - 1))


def _pv(Ev, Vv, Vv0, af_con2, Kxp, Kxv):
    """
    Equation A.19

    :param Ev:
    :param Vv:
    :param Vv0:
    :param af_con2:
    :param Kxp:
    :param Kxv:
    :return:
    """
    psi = _psi(Vv, Kxp, Kxv)
    return Ev * (Vv - Vv0) * af_con2 - psi


def _pa(Ea, Va, Va0, Kxp, Kxv):
    """
    Equation A.20

    :param Ea:
    :param Va:
    :param Va0:
    :param Kxp:
    :param Kxv:
    :return:
    """
    psi = _psi(Va, Kxp, Kxv)
    return Ea * (Va - Va0) - psi


def _ei(Emax, Emin, y):
    """
    Equation A.21

    :param Emax:
    :param Emin:
    :param y:
    :return:
    """
    return (Emax - Emin) * y + Emin


def _emaxv(Kev, Emaxv1):
    """
    Equation A.22

    :param Kev:
    :param Emaxv1:
    :return:
    """
    return Kev * Emaxv1


def _v0(y, Vd0, Vs0):
    """
    Equations A.23 and A.24

    :param y:
    :param Vd0:
    :param Vs0:
    :return:
    """
    return (1 - y) * (Vd0 - Vs0) + Vs0


def _fi(Pi, Pj, Ri):
    """
    Equation A.25

    :param Pi:
    :param Pj:
    :param Ri:
    :return:
    """
    if Pi > Pj:
        return (Pi - Pj) / Ri
    return 0


def _volume_change(Fin, Fout):
    """
    Equation A.26
    Vra:t = (Fvc - Fra + Fcorvn)
    Vrv:t = (Fra - Frv)

    :param Fin:
    :param Fout:
    :return:
    """
    return Fin - Fout


def _equations(t, n, tHB, HP,
               tPwave, tRwave,
               Tsa, Tsv,
               Vvarlvs0, Vvarrvs0,
               Vrv, Vra, Vla, Vlv,  # volumes
               Ppap, Paop,  # pressures
               Fvc, Fpv, Fcorvn,  # flows
               af_con2, af_con,
               # fixed parameters
               Emaxra, Eminra, Emaxla, Eminla, Emaxlv, Eminlv,
               Vrad0, Vras0, Vlad0, Vlas0,
               Kxp, Kxv,
               Rra, Rrv, Rla, Rlv,
               offv, Ts1a, Ts1v, Ts2, PRint,
               Vlvd0, EDVLV, Vlvs0, Vrvd0, EDVRV, Vrvs0):
    resultA = _trigger_A(t, tHB, HP, PRint, offv, Ts1a, Ts2)
    if resultA is not None:
        HRa, Tsa, tPwave = resultA

    resultB = _trigger_B(t, tHB, HP, offv, Ts1v, Ts2, n,
                         Vlv, Vlvd0, EDVLV, Vlvs0,
                         Vrv, Vrvd0, EDVRV, Vrvs0,
                         af_con)
    if resultB is not None:
        HRv, Tsv, tRwave, Vvarlvs0, Vvarrvs0, af_con2 = resultB

    #
    ta_rel = t_rel(t, tPwave)
    tv_rel = t_rel(t, tRwave)

    # activation functions
    ya = _yi(ta_rel, Tsa)
    yv = _yi(tv_rel, Tsv)

    # elastances
    Era = _ei(Emaxra, Eminra, ya)
    Erv = _ei(Emaxra, Eminra, yv)
    Ela = _ei(Emaxla, Eminla, ya)
    Elv = _ei(Emaxlv, Eminlv, yv)
    # unstressed volumes
    Vra0 = _v0(ya, Vrad0, Vras0)
    Vrv0 = _v0(yv, Vrvd0, Vvarrvs0)
    Vla0 = _v0(ya, Vlad0, Vlas0)
    Vlv0 = _v0(yv, Vlvd0, Vvarlvs0)
    # pressures
    Pra = _pa(Era, Vra, Vra0, Kxp, Kxv)
    Prv = _pv(Erv, Vrv, Vrv0, af_con2, Kxp, Kxv)
    Pla = _pa(Ela, Vla, Vla0, Kxp, Kxv)
    Plv = _pv(Elv, Vlv, Vlv0, af_con2, Kxp, Kxv)
    # flows
    Fra = _fi(Pra, Prv, Rra)
    Frv = _fi(Prv, Ppap, Rrv)
    Fla = _fi(Pla, Plv, Rla)
    Flv = _fi(Plv, Paop, Rlv)
    # volumes
    d_Vra_dt = _volume_change(Fvc + Fcorvn, Fra)
    d_Vrv_dt = _volume_change(Fra, Frv)
    d_Vla_dt = _volume_change(Fpv, Fla)
    d_Vlv_dt = _volume_change(Fla, Flv)

    return d_Vra_dt, d_Vrv_dt, d_Vla_dt, d_Vlv_dt
