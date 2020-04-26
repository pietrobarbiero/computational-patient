import os
import pickle

from scipy.integrate import solve_ivp
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from ..infection._equations import mass_balance_AngI_infection, mass_balance_AngII_infection, mass_balance_ANG17, \
    blood_pressure_change
from ..pd import GLU, mass_balance_AGT, mass_balance_Renin, mass_balance_AngI, mass_balance_AngII
from ..pk import analytical_PK


def ODE(t, conc, drugdose, ke_diacid,
        VF_diacid, ka_diacid, feedback_capacity,
        k_cat_Renin, k_feedback, C50,
        n_Hill, tau, tfinal_dosing,
        AngII_conc_t0, Renin_conc_t0,
        baseline_prod_Renin,
        k_degr_Renin, k_degr_AngI, k_degr_AGT,

        tstart_dosing, glu):
    # Input concentration vector conc contains species AngI, AngII & Renin
    AngI_conc, AngII_conc, Renin_conc, AGT_conc, ANG17_conc, SBP = conc

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

    k_SARS = 0.5
    d_AngI_conc_dt = mass_balance_AngI_infection(c_Renin,
                                                 AGT_conc,
                                                 k_cat_Renin,
                                                 Renin_conc,
                                                 Renin_conc_t0,
                                                 k_degr_AngI,
                                                 k_NEP,
                                                 k_ACE2,
                                                 AngI_conc,
                                                 c_ACE,
                                                 Inhibition,
                                                 k_SARS)
    d_AngII_conc_dt = mass_balance_AngII_infection(h_ANGII,
                                                   c_AT1,
                                                   k_APA,
                                                   k_ACE2,
                                                   k_AT2,
                                                   AngII_conc,
                                                   c_ACE,
                                                   AngI_conc,
                                                   Inhibition,
                                                   k_SARS)
    d_ANG17_conc_dt = mass_balance_ANG17(k_NEP, AngI_conc, k_ACE2, k_SARS, AngII_conc, h_ANGII, ANG17_conc)
    k_1 = 0.8
    k_2 = 0.75
    d_SBP_dt = blood_pressure_change(k_1, AngII_conc, k_2, ANG17_conc)

    # concentration derivative vector has entries for Ang I, Ang II, and Renin
    d_conc_dt = np.array([
        d_AngI_conc_dt,
        d_AngII_conc_dt,
        d_Renin_conc_dt,
        d_AGT_conc_dt,
        d_ANG17_conc_dt,
        d_SBP_dt,
    ])

    return d_conc_dt


def local_RAS_model(coefficients, drug_dose, tau,
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

    ANG17_conc_t0 = AngII_conc_t0
    SBP_t0 = 150

    # initial condition for the ODE solver
    conc_t0 = np.array([AngI_conc_t0, AngII_conc_t0, Renin_conc_t0, AGT_conc_t0, ANG17_conc_t0, SBP_t0])

    ODE_args = (
        drug_dose, ke_diacid,
        VF_diacid, ka_diacid, feedback_capacity,
        k_cat_Renin, k_feedback, C50,
        n_Hill, tau, tfinal_dosing,
        AngII_conc_t0, Renin_conc_t0,
        baseline_prod_Renin,
        k_degr_Renin, k_degr_AngI, k_degr_AGT,
        tstart_dosing, glu
    )
    sol = solve_ivp(fun=ODE, t_span=[0, sim_time_end], y0=conc_t0,
                    args=ODE_args,
                    t_eval=t_eval,
                    method="LSODA")
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
    Ang17_conc = sol["y"][4, :] * Mw_AngII * conv_rate
    SBP = sol["y"][5, :]

    Inhibition = (100. * (diacid_conc ** n_Hill)) / (diacid_conc ** n_Hill + C50 ** n_Hill)

    plt.figure()
    plt.plot(sol["t"], AngII_conc, label="ANG-II")
    plt.plot(sol["t"], Ang17_conc, label="ANG-(1-7)")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(sol["t"], SBP, label="SBP")
    plt.legend()
    plt.show()

    return sol["t"], diacid_conc, AngII_conc, AngI_conc, \
           Inhibition, Renin_conc, drug_conc, AGT_conc, Ang17_conc, SBP


def call_infection(args, params):
    # compute dependent parameters
    drug_dose = args.dose * 1e6
    pill_mg = args.dose * 1e-6
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

    solution = local_RAS_model(coefficients,
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
                               args.glu)

    t, diacid_conc, AngII_conc, AngI_conc, \
    Inhibition, Renin_conc, drug_conc, AGT_conc, Ang17_conc, SBP = solution

    ANGII_Plot = 0.021001998652419
    # y_angII = ((AngII_conc / (pk_params["Mw_AngII"][0][0] * 10**6/1000)) / ANGII_Plot) * 100
    y_angII_norm = ((AngII_conc / (pk_params["Mw_AngII"][0][0] * 10 ** 6)) / ANGII_Plot) * 100
    tplot = t / 24
    y_angII = AngII_conc / (pk_params["Mw_AngII"][0][0] * 10 ** 6 / 1000)
    y_ang17 = Ang17_conc / (pk_params["Mw_AngII"][0][0] * 10 ** 6 / 1000)

    save_var = {
        "params": args,
        "t": tplot,
        "angII": y_angII,
        "angII_norm": y_angII_norm,
        "diacid": diacid_conc,
        "ang17": y_ang17,
        "sbp": SBP,
    }
    dose = str(args.dose).replace(".", "-")
    out_dir = "_".join(["./data/" + args.drug_name,
                        dose,
                        str(args.n_dose),
                        str(args.glu)])
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    file_name = f"{args.renal_function}.dat"
    pickle.dump(save_var, open(os.path.join(out_dir, file_name), 'wb'))

    return solution
