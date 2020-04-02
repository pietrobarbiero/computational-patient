from scipy.integrate import solve_ivp

import numpy as np

from ._model import ODE
from ..pk import analytical_PK


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

    t_eval = np.arange(0, sim_time_end, tau / 500)  # hours

    # # TODO: double check
    # AGT_conc_t0 = 1.7e7
    # AngI_conc_t0 = 271
    # AngII_conc_t0 = 21
    # feedback_capacity = 0.397 * 21 / 1.65e-2

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
                    args=ODE_args, t_eval=t_eval,
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
