import argparse
from typing import Tuple, Dict

import scipy.io
import numpy as np
from scipy.integrate import solve_ivp


def _load_configuration() -> argparse.Namespace:
    """
    Parse command line arguments.

    Parameters
    ----------
    :return: configuration object
    """
    h = 24

    parser = argparse.ArgumentParser()
    parser.add_argument("--dose", help="Drug dose (mg). 5 is nominal dose for Benazepril and 1.25 for Cilazapril.",
                        default=5, required=False, type=int)
    parser.add_argument("--n-dose", help="Number of doses per day (cannot be zero).", default=1, required=False,
                        type=int)
    parser.add_argument("--drug-name", help="Drug name.", default="benazepril",
                        required=False, choices=["benazepril", "cilazapril"], nargs=1)
    parser.add_argument("--renal-function", help="Type of renal function.", default="normal",
                        required=False, choices=["normal", "impaired"], nargs=1)
    parser.add_argument("--tstart-dosing", help="Initial dosing time.", default=h * 1, required=False, type=int)
    parser.add_argument("--tfinal-dosing", help="Final dosing time.", default=h * 2, required=False, type=int)
    parser.add_argument("--sim-time-end", help="Simulation end time.", default=h * 5, required=False, type=int)
    parser.add_argument("--glu", help="Glucose concentration (mmol/L). "
                                      "To have normal subject glucose dyanmics as input,"
                                      "use glu = 1; for diabetic subjects, glu = 2. Rest all values will be"
                                      "used directly in mmol/L as steady state glucose input.",
                        default=1, required=False, type=float)
    parser.add_argument("--show-plots", help="Show plots.", default=True, required=False, type=bool)
    parser.add_argument("--linestyle", help="Plot line style.", default="-", required=False, type=str)
    parser.add_argument("--linewidth", help="Plot line width.", default=2, required=False, type=int)
    parser.add_argument("--legendloc", help="Plot legend location.", default="NorthEast", required=False, type=str)
    args = parser.parse_args()

    return args


def analytical_PK(drugdose, ka, VF, ke, t, tau, tfinal_dosing, tstart_dosing, glu=None):
    """
    Equation 1

    :param drugdose:
    :param ka:
    :param VF:
    :param ke:
    :param t:
    :param tau:
    :param tfinal_dosing:
    :param tstart_dosing:
    :param glu:
    :return:
    """
    if t > tstart_dosing:
        if t < tfinal_dosing:
            n = np.floor(t / tau) + 1

        else:
            n = np.floor(tfinal_dosing / tau)

        tprime = t - tau * (n - 1)
        drug_conc_theo = drugdose * ka / (VF * (ka - ke)) * (
                (1 - np.exp(-n * ke * tau)) * (np.exp(-ke * tprime)) / (1 - np.exp(-ke * tau))
                - (1 - np.exp(-n * ka * tau)) * (np.exp(-ka * tprime)) / (1 - np.exp(-ka * tau)))

    else:
        drug_conc_theo = 0

    return drug_conc_theo


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


def catalized_AngI(c_ACE, AngI_conc, Inhibition):
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
    angI_catalized = catalized_AngI(c_ACE, AngI_conc, Inhibition)

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

    angI_catalized = catalized_AngI(c_ACE, AngI_conc, Inhibition)

    return angI_catalized - baseline_cons_AngII


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


def combinedRAS_ACE_PKPD(coefficients, drug_dose, tau,
                         tfinal_dosing, ka_drug, VF_drug, ke_drug, ke_diacid,
                         VF_diacid, ka_diacid, C50, n_Hill,
                         AngI_conc_t0, AngII_conc_t0, Renin_conc_t0, diacid_conc_t0,
                         drug_conc_t0, AGT_conc_t0, k_degr_Renin, k_degr_AngI,
                         k_degr_AGT, Mw_AngI, Mw_AngII, Mw_Renin, Mw_AGT,
                         sim_time_end, tstart_dosing, glu):
    c_Renin, k_cat_Renin, k_feedback, feedback_capacity, k_cons_AngII = coefficients

    # impose constraining assumption that the initial values are steady-state
    baseline_prod_Renin = k_degr_Renin * Renin_conc_t0

    time = np.arange(0, sim_time_end, tau / 500)  # hours

    # initial condition for the ODE solver
    conc_t0 = np.array([AngI_conc_t0, AngII_conc_t0, Renin_conc_t0, AGT_conc_t0])

    ODE_args = (
        drug_dose, ke_diacid,
        VF_diacid, ka_diacid, feedback_capacity,
        k_cat_Renin, k_feedback, C50,
        n_Hill, tau, tfinal_dosing, AngI_conc_t0,
        AngII_conc_t0, Renin_conc_t0, AGT_conc_t0,
        baseline_prod_Renin,
        k_degr_Renin, k_degr_AngI, k_degr_AGT,
        k_cons_AngII, tstart_dosing, glu
    )
    sol = solve_ivp(fun=ODE, t_span=[0, sim_time_end], y0=conc_t0,
                    args=ODE_args,
                    method="RK23")
    # method="RK45", rtol=1e-12, atol=1e-16)

    # Concentrations of each species at each time
    drug_conc_list = []
    diacid_conc_list = []
    for i in range(0, len(sol["t"])):
        drug_conc_list.append(analytical_PK(drug_dose,
                                            ka_drug,
                                            VF_drug,
                                            ke_drug,
                                            sol["t"][i],
                                            tau,
                                            tfinal_dosing,
                                            tstart_dosing))
        diacid_conc_list.append(analytical_PK(drug_dose,
                                              ka_diacid,
                                              VF_diacid,
                                              ke_diacid,
                                              sol["t"][i],
                                              tau,
                                              tfinal_dosing,
                                              tstart_dosing))

    drug_conc = np.array(drug_conc_list)
    diacid_conc = np.array(diacid_conc_list)

    # Units converted from (umol/l) to (pg/ml)
    conv_rate = 10 ** 6 / 1000
    AngI_conc = sol["y"][0, :] * Mw_AngI * conv_rate
    AngII_conc = sol["y"][1, :] * Mw_AngII * conv_rate
    Renin_conc = sol["y"][2, :] * Mw_Renin * conv_rate
    AGT_conc = sol["y"][3, :] * Mw_AGT * conv_rate

    Inhibition = (100. * (diacid_conc ** n_Hill)) / (diacid_conc ** n_Hill + C50 ** n_Hill)

    return sol["t"], diacid_conc, AngII_conc, AngI_conc, \
           Inhibition, Renin_conc, drug_conc, AGT_conc


def call_combinedRAS_ACE_PKPD(args, params):
    # compute dependent parameters
    drug_dose = args.dose * 1e6
    tau = 24 / args.n_dose

    # ODE coefficients
    coefficients = [
        params["c_Renin"][0][0],
        params["k_cat_Renin"][0][0],
        params["k_feedback"][0][0],
        params["feedback_capacity"][0][0],
        params["k_cons_AngII"][0][0],
    ]
    coefficients = np.array(coefficients)

    # load PK parameters
    pk_params_file = "".join(["PK_params_", args.drug_name, args.renal_function, ".mat"])
    pk_params = scipy.io.loadmat(pk_params_file)

    solution = combinedRAS_ACE_PKPD(coefficients,
                                    drug_dose,
                                    tau,
                                    args.tfinal_dosing,

                                    pk_params["ka_drug"][0][0],
                                    pk_params["VF_drug"][0][0],
                                    pk_params["ke_drug"][0][0],

                                    params["ke_diacid"][0][0],
                                    params["VF_diacid"][0][0],
                                    params["ka_diacid"][0][0],

                                    pk_params["C50"][0][0],
                                    pk_params["n_Hill"][0][0],

                                    pk_params["AngI_conc_t0"][0][0],
                                    pk_params["AngII_conc_t0"][0][0],
                                    pk_params["Renin_conc_t0"][0][0],
                                    pk_params["diacid_conc_t0"][0][0],
                                    pk_params["drug_conc_t0"][0][0],
                                    pk_params["AGT_conc_t0"][0][0],

                                    pk_params["k_degr_Renin"][0][0],
                                    pk_params["k_degr_AngI"][0][0],
                                    pk_params["k_degr_AGT"][0][0],

                                    pk_params["Mw_AngI"][0][0],
                                    pk_params["Mw_AngII"][0][0],
                                    pk_params["Mw_Renin"][0][0],
                                    pk_params["Mw_AGT"][0][0],

                                    args.sim_time_end,
                                    args.tstart_dosing,
                                    args.glu
                                    )

    t, diacid_conc, AngII_conc, AngI_conc, \
        Inhibition, Renin_conc, drug_conc, AGT_conc = solution

    return


def covid19_dkd_model():
    args = _load_configuration()

    # load parameters
    params_file = "".join(["params_", args.drug_name, args.renal_function, ".mat"])
    params = scipy.io.loadmat(params_file)

    call_combinedRAS_ACE_PKPD(args, params)

    return
