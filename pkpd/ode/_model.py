import numpy as np

from ..pd import GLU, mass_balance_AGT, mass_balance_Renin, mass_balance_AngI, mass_balance_AngII
from ..pk import analytical_PK


def ODE(t, conc, drugdose, ke_diacid,
        VF_diacid, ka_diacid, feedback_capacity,
        k_cat_Renin, k_feedback, C50,
        n_Hill, tau, tfinal_dosing, AngI_conc_t0,
        AngII_conc_t0, Renin_conc_t0, AGT_conc_t0,
        baseline_prod_Renin,
        k_degr_Renin, k_degr_AngI, k_degr_AGT,
        k_cons_AngII, tstart_dosing, glu):
    # Input concentration vector conc contains species AngI, AngII & Renin
    AngI_conc, AngII_conc, Renin_conc, AGT_conc = conc

    # PK model explicit functions
    diacid_conc = analytical_PK(drugdose,
                                ka_diacid,
                                VF_diacid,
                                ke_diacid,
                                t,
                                tau,
                                tfinal_dosing,
                                tstart_dosing)

    Inhibition = (100 * (diacid_conc ** n_Hill)) / (diacid_conc ** n_Hill + C50 ** n_Hill)

    # Glucose-dependent rate params for enzymatic/binding reactions
    # from Pilvankar et. al 2018 from Approach 1
    Rate_params = np.array([
        1.527482117056147e-07,
        1.705688364046031e-05,
        2.472978807773762e-04,
        4.533794480918563e-03,
        7.072930413876994e-04,
        1.296703909210782e-02
    ])

    # Units conversion:
    # c_X_a: (L/mmol/s) to (L/mmol/hr)
    # c_X_b: (1/s) to (1/hr)
    c_Renin_a, c_Renin_b, c_ACE_a, c_ACE_b, c_AT1_a, c_AT1_b = Rate_params * 3600

    # # TODO: double check
    # c_Renin_b = 6.16e10-11
    # c_ACE_b = 163
    # c_AT1_b = 464

    # Glucose-dependent rate params
    c_Renin = c_Renin_a * GLU(t, glu) + c_Renin_b
    c_ACE = c_ACE_a * GLU(t, glu) + c_ACE_b
    c_AT1 = c_AT1_a * GLU(t, glu) + c_AT1_b

    # Non-Glucose-dependent rate constants for enzymatic/binding reactions
    # from Pilvankar et. al 2018 from Approach 1
    Rate_cons = np.array([
        1.210256981930063e-02,
        1.069671574938187e-04,
        6.968146259597334e-03,
        1.628277841850352e-04,
        6.313823632053240E+02
    ])
    # Units converted from
    # k_APA, k_ACE2, k_AT2, k_NEP: from (1/s) to (1/hr)
    # k_AGT: from (nmol/L/s) to (umol/L/hr)
    k_APA, k_ACE2, k_AT2, k_NEP, k_AGT = Rate_cons * 3600
    h_ANGII = 18 / 3600  # from (s) to (hr)

    # mass balance AGT
    d_AGT_conc_dt = mass_balance_AGT(k_AGT, c_Renin, AGT_conc, k_degr_AGT)

    # ODEs for the three changing hormone/enzyme concentrations
    d_Renin_conc_dt = mass_balance_Renin(baseline_prod_Renin,
                                         k_feedback,
                                         AngII_conc_t0,
                                         AngII_conc,
                                         feedback_capacity,
                                         k_degr_Renin,
                                         Renin_conc)
    d_AngI_conc_dt = mass_balance_AngI(c_Renin,
                                       AGT_conc,
                                       k_cat_Renin,
                                       Renin_conc,
                                       Renin_conc_t0,
                                       k_degr_AngI,
                                       k_NEP,
                                       k_ACE2,
                                       AngI_conc,
                                       c_ACE,
                                       Inhibition)
    d_AngII_conc_dt = mass_balance_AngII(h_ANGII,
                                         c_AT1,
                                         k_APA,
                                         k_ACE2,
                                         k_AT2,
                                         AngII_conc,
                                         c_ACE,
                                         AngI_conc,
                                         Inhibition)

    # concentration derivative vector has entries for Ang I, Ang II, and Renin
    d_conc_dt = np.array([
        d_AngI_conc_dt,
        d_AngII_conc_dt,
        d_Renin_conc_dt,
        d_AGT_conc_dt,
    ])

    return d_conc_dt
