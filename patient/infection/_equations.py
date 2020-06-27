import numpy as np

from patient.pd._equations import _catalized_AngI


def mass_balance_AngI_infection(c_Renin, AGT_conc, k_cat_Renin,
                                Renin_conc, Renin_conc_t0,
                                k_degr_AngI, k_NEP, k_ACE2, AngI_conc,
                                c_ACE, Inhibition, drug_type):
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
    if drug_type == "ACEi":
        angI_catalized = _catalized_AngI(c_ACE, AngI_conc, Inhibition)
    else:
        angI_catalized = c_ACE * AngI_conc

    return angI_production - angI_degradation - angI_catalized


def mass_balance_AngII_infection(h_ANGII, c_AT1, k_APA, k_ACE2, k_AT2, AngII_conc,
                                 c_ACE, AngI_conc, Inhibition, drug_type):
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

    if drug_type == "ACEi":
        angI_catalized = _catalized_AngI(c_ACE, AngI_conc, Inhibition)
    else:
        angI_catalized = c_ACE * AngI_conc

    return angI_catalized - baseline_cons_AngII


def mass_balance_ANG17(k_NEP, AngI_conc, k_ACE2, AngII_conc, h_ANG17, ANG17_conc):
    """

    :param k_NEP:
    :param AngI_conc:
    :param k_ACE2:
    :param AngII_conc:
    :param h_ANG17:
    :param ANG17_conc:
    :return:
    """
    return k_NEP * AngI_conc + k_ACE2 * AngII_conc - np.log(2) / h_ANG17 * ANG17_conc


def mass_balance_AT1R(c_ATR, AngII_conc, h_ATR, ATR, Inhibition, drug_type):
    """

    :param c_ATR:
    :param AngII_conc:
    :param h_ATR:
    :param ATR:
    :return:
    """
    if drug_type == "ACEi":
        return c_ATR * AngII_conc - np.log(2) / h_ATR * ATR
    else:
        return c_ATR * AngII_conc * (1 - (Inhibition / 100)) - np.log(2) / h_ATR * ATR


def mass_balance_AT2R(c_ATR, AngII_conc, h_ATR, ATR):
    """

    :param c_ATR:
    :param AngII_conc:
    :param h_ATR:
    :param ATR:
    :return:
    """
    return c_ATR * AngII_conc - np.log(2) / h_ATR * ATR
