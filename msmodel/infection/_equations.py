import numpy as np

from msmodel.pd._equations import _catalized_AngI


def mass_balance_AngI_infection(c_Renin, AGT_conc, k_cat_Renin,
                                Renin_conc, Renin_conc_t0,
                                k_degr_AngI, k_NEP, k_ACE2, AngI_conc,
                                c_ACE, Inhibition, k_infection):
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
    angI_degradation = (k_degr_AngI + k_NEP + (k_ACE2 * k_infection)) * AngI_conc

    # Rate of Ang I --> Ang II catalyzed by ACE with AngI_conc and I/KI
    # changing due to drug presence
    # peptide = ODE_glucose_RAS
    angI_catalized = _catalized_AngI(c_ACE, AngI_conc, Inhibition)

    return angI_production - angI_degradation - angI_catalized


def mass_balance_AngII_infection(h_ANGII, c_AT1, k_APA, k_ACE2, k_AT2, AngII_conc,
                                 c_ACE, AngI_conc, Inhibition, k_infection):
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
    baseline_cons_AngII = (c_AT1 + k_APA + (k_ACE2 * k_infection) + k_AT2 + k_degr_AngII) * AngII_conc

    angI_catalized = _catalized_AngI(c_ACE, AngI_conc, Inhibition)

    return angI_catalized - baseline_cons_AngII


def mass_balance_ANG17(k_NEP, AngI_conc, k_ACE2, k_SARS, AngII_conc, h_ANGII, ANG17_conc):
    """

    :param k_NEP:
    :param AngI_conc:
    :param k_ACE2:
    :param k_SARS:
    :param AngII_conc:
    :param h_ANGII:
    :param ANG17_conc:
    :return:
    """
    return k_NEP * AngI_conc + (k_ACE2 * k_SARS) * AngII_conc - np.log(2) / (17.5*h_ANGII) * ANG17_conc


def blood_pressure_change(AngII_conc, ANG17_conc, t, sbp_spikes):
    SSS_ANG2 = 0.0316
    III_ANG17 = 0.0316
    sum_list = [
        sbp_spikes * np.sin(2 * np.pi * t / 24),
        0.2 * (1 + SSS_ANG2 * AngII_conc),
        -0.2 * (1 + III_ANG17 * ANG17_conc),
        # - K_out_SBP * SBP,
    ]
    return sum(sum_list)
