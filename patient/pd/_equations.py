import numpy as np


def GLU(t, glu):
    if t > 24:
        day = np.ceil(t / 24)
        t = t - 24 * (day - 1)

    # (mmol / L)
    if glu == 1:
        # Normal subject
        glucose_conc = (3049270060749109 * t ** 12) / 154742504910672534362390528 - \
                       (8573627330812761 * t ** 11) / 2417851639229258349412352 + \
                       (5184092078348791 * t ** 10) / 18889465931478580854784 - \
                       (222554360563333 * t ** 9) / 18446744073709551616 + \
                       (3080243600503657 * t ** 8) / 9223372036854775808 - \
                       (7013297429851209 * t ** 7) / 1152921504606846976 + \
                       (5321658630653787 * t ** 6) / 72057594037927936 - \
                       (5325990862474837 * t ** 5) / 9007199254740992 + \
                       (6786421942359735 * t ** 4) / 2251799813685248 - \
                       (2564107225361311 * t ** 3) / 281474976710656 + \
                       (3998409413514137 * t ** 2) / 281474976710656 - \
                       (4690707683073767 * t) / 562949953421312 + \
                       1033786866424801 / 140737488355328

    elif glu == 2:
        # Diabetic subject
        glucose_conc = - (3194535292912431 * t ** 12) / 77371252455336267181195264 + \
                       (6601616942431553 * t ** 11) / 1208925819614629174706176 - \
                       (5782046365470279 * t ** 10) / 18889465931478580854784 + \
                       (5516279079400991 * t ** 9) / 590295810358705651712 - \
                       (6023660115892281 * t ** 8) / 36893488147419103232 + \
                       (6699337853108719 * t ** 7) / 4611686018427387904 - \
                       (6438576095458023 * t ** 6) / 9223372036854775808 - \
                       (282392831455481 * t ** 5) / 2251799813685248 + \
                       (5946501511689545 * t ** 4) / 4503599627370496 - \
                       (6951870006681155 * t ** 3) / 1125899906842624 + \
                       (3642220872320963 * t ** 2) / 281474976710656 - \
                       (2001597022128155 * t) / 281474976710656 + \
                       4564289255684401 / 562949953421312

    else:
        glucose_conc = glu

    return glucose_conc


def mass_balance_AGT(k_AGT, c_Renin, AGT_conc, k_degr_AGT):
    """
    Equation 2

    :param k_AGT:
    :param c_Renin:
    :param AGT_conc:
    :param k_degr_AGT:
    :return:
    """
    return k_AGT - c_Renin * AGT_conc - k_degr_AGT * AGT_conc


def mass_balance_Renin(baseline_prod_Renin, k_feedback, AngII_conc_t0,
                       AngII_conc, feedback_capacity, k_degr_Renin, Renin_conc):
    """
    Equation 3

    :param baseline_prod_Renin:
    :param k_feedback:
    :param AngII_conc_t0:
    :param AngII_conc:
    :param feedback_capacity:
    :param k_degr_Renin:
    :param Renin_conc:
    :return:
    """
    # Baseline production of Renin + negative feedback from AngII to Renin
    # production using logistic function dependence on change of AngII_conc
    # from steady state set point
    renin_feedback = k_feedback * (AngII_conc_t0 - AngII_conc) * (
                1 - (AngII_conc_t0 - AngII_conc) / (feedback_capacity))

    # Degradation of Renin
    renin_degradation = k_degr_Renin * Renin_conc

    return baseline_prod_Renin + renin_feedback - renin_degradation


def _catalized_AngI(c_ACE, AngI_conc, Inhibition):
    return c_ACE * AngI_conc * (1 - (Inhibition / 100))


def mass_balance_AngI(c_Renin, AGT_conc, k_cat_Renin,
                      Renin_conc, Renin_conc_t0,
                      k_degr_AngI, k_NEP, k_ACE2, AngI_conc,
                      c_ACE, Inhibition):
    """
    Equation 5

    :param c_Renin:
    :param AGT_conc:
    :param k_cat_Renin:
    :param Renin_conc:
    :param Renin_conc_t0:
    :param k_degr_AngI:
    :param k_NEP:
    :param k_ACE2:
    :param AngI_conc:
    :param c_ACE:
    :param Inhibition:
    :return:
    """

    # Production rate of Ang I from angiotensinogen --> Ang I in presence
    # of Renin with baseline and variable contributions. Only Renin changes
    # due to drug presence.
    baseline_prod_AngI = c_Renin * AGT_conc
    variable_prod_AngI = k_cat_Renin * (Renin_conc - Renin_conc_t0)
    angI_production = variable_prod_AngI + baseline_prod_AngI

    # Degradation of Ang I
    # Considering ANGI --> ANG(1-9), ANG(1-7) and half life degradation.
    # Refer to Fig 2. from Pilvankar et al. 2018
    angI_degradation = (k_degr_AngI + k_NEP + k_ACE2) * AngI_conc

    # Rate of Ang I --> Ang II catalyzed by ACE with AngI_conc and I/KI
    # changing due to drug presence
    # peptide = ODE_glucose_RAS
    angI_catalized = _catalized_AngI(c_ACE, AngI_conc, Inhibition)

    return angI_production - angI_degradation - angI_catalized


def mass_balance_AngII(h_ANGII, c_AT1, k_APA, k_ACE2, k_AT2, AngII_conc,
                       c_ACE, AngI_conc, Inhibition):
    """
    Equation 6

    :param h_ANGII:
    :param c_AT1:
    :param k_APA:
    :param k_ACE2:
    :param k_AT2:
    :param AngII_conc:
    :param c_ACE:
    :param AngI_conc:
    :param Inhibition:
    :return:
    """
    # Consumption rate of Ang II --> with AngII_conc being the only term
    # that changes due to drug presence
    k_degr_AngII = (np.log(2) / h_ANGII)
    baseline_cons_AngII = (c_AT1 + k_APA + k_ACE2 + k_AT2 + k_degr_AngII) * AngII_conc

    angI_catalized = _catalized_AngI(c_ACE, AngI_conc, Inhibition)

    return angI_catalized - baseline_cons_AngII
