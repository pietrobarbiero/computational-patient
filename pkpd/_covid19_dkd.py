import os
import pickle

import scipy.io
import numpy as np

from ._config import load_configuration
from .ode.local_RAS import combinedRAS_ACE_PKPD


def call_combinedRAS_ACE_PKPD(args, params):
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
                                    args.glu)

    t, diacid_conc, AngII_conc, AngI_conc, \
        Inhibition, Renin_conc, drug_conc, AGT_conc = solution

    ANGII_Plot = 0.021001998652419
    # y_angII = ((AngII_conc / (pk_params["Mw_AngII"][0][0] * 10**6/1000)) / ANGII_Plot) * 100
    y_angII_norm = ((AngII_conc / (pk_params["Mw_AngII"][0][0] * 10**6)) / ANGII_Plot) * 100
    tplot = t / 24
    y_angII = AngII_conc/(pk_params["Mw_AngII"][0][0] * 10**6/1000)

    save_var = {
        "params": args,
        "t": tplot,
        "angII": y_angII,
        "angII_norm": y_angII_norm,
        "diacid": diacid_conc
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

    return


def covid19_dkd_model():
    args = load_configuration()
    args.renal_function = args.renal_function[0]

    # load parameters
    params_file = "".join(["params_", args.drug_name, args.renal_function, ".mat"])
    params = scipy.io.loadmat(params_file)

    call_combinedRAS_ACE_PKPD(args, params)

    return
