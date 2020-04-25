import os
import pickle

from scipy.integrate import solve_ivp
import numpy as np
import scipy
import pandas as pd

from ..hypertension._equations import transit_compartment_model, change_S_ANG17, change_S_ANG2, change_S_SBP


def ODE(t, conc,
        K_in_ANG17, K_out_ANG17, K_tr, SS_ANG2, n,
        K_in_ANG2, K_out_ANG2, SS_ANG17, m,
        K_in_SBP, K_out_SBP, SSS_ANG2, III_ANG17,
        ES_p, EI_p, ES_ANG2, ES_ANG17):
    # Input concentration vector conc contains species AngI, AngII & Renin
    S_ANG17, S_ANG2, S_SBP = conc

    # transit compartment models
    ES_ANG2_n = transit_compartment_model(ES_ANG2, K_tr, n)
    ES_ANG17_m = transit_compartment_model(ES_ANG17, K_tr, m)

    d_S_ANG17_dt = change_S_ANG17(K_in_ANG17, ES_ANG2_n, ES_p, K_out_ANG17, S_ANG17)
    d_S_ANG2_dt = change_S_ANG2(K_in_ANG2, EI_p, K_out_ANG2, ES_ANG17_m, S_ANG2)
    d_S_SBP_dt = change_S_SBP(K_in_SBP, SSS_ANG2, S_ANG2, III_ANG17, S_ANG17, K_out_SBP, S_SBP)

    # concentration derivative vector has entries for Ang I, Ang II, and Renin
    d_conc_dt = np.array([
        d_S_ANG17_dt,
        d_S_ANG2_dt,
        d_S_SBP_dt
    ])

    return d_conc_dt


def hypertension_model(K_in_ANG17, K_out_ANG17, K_tr, SS_ANG2, n,
                       K_in_ANG2, K_out_ANG2, SS_ANG17, m,
                       K_in_SBP, K_out_SBP, SSS_ANG2, III_ANG17,
                       ES_p, EI_p):
    # initial conditions
    # TODO: double check
    #  see fig. 4, first week
    C_ANG17_mean = 9
    C_ANG2_mean = 0.8
    # equation 7 and 10
    ES_ANG2 = SS_ANG2 * C_ANG2_mean
    ES_ANG17 = SS_ANG17 * C_ANG17_mean
    # equation from 4 to 6
    S_ANG17 = K_in_ANG17 * (1 + ES_ANG2) / K_out_ANG17
    S_ANG2 = K_in_ANG2 / (K_out_ANG2 * (1 + ES_ANG17))
    S_SBP = K_in_SBP * (1 + SSS_ANG2 * S_ANG2 - III_ANG17 * S_ANG17) / K_out_SBP

    # initial condition for the ODE solver
    conc_t0 = np.array([S_ANG17, S_ANG2, S_SBP])

    ODE_args = (
        K_in_ANG17, K_out_ANG17, K_tr, SS_ANG2, n,
        K_in_ANG2, K_out_ANG2, SS_ANG17, m,
        K_in_SBP, K_out_SBP, SSS_ANG2, III_ANG17,
        ES_p, EI_p, ES_ANG2, ES_ANG17
    )
    sol = solve_ivp(fun=ODE, t_span=[0, 1000], y0=conc_t0,
                    args=ODE_args, method="LSODA")

    return


def call_hypertension(args):
    # parameters (table I)
    # S -> stimulation
    # I -> inhibition
    # pg -> picograms

    # ANG-(1-7) parameters
    K_in_ANG17 = 117 # TODO: double check
    # K_in_ANG17 = 17
    K_out_ANG17 = 1.59
    K_tr = 1.63
    SS_ANG2 = 0.0726
    n = 29

    # ANG II parameters
    K_in_ANG2 = 27.5
    K_out_ANG2 = 0.215
    SS_ANG17 = 0.0711
    m = 4

    # systolic blood pressure parameters
    K_in_SBP = 2670
    K_out_SBP = 56
    SSS_ANG2 = 0.0316
    III_ANG17 = 0.00956

    # drug parameters
    ES_p = 0.2097
    EI_p = 0.4199
    ES_p = 0
    EI_p = 0

    hypertension_model(
        K_in_ANG17, K_out_ANG17, K_tr, SS_ANG2, n,
        K_in_ANG2, K_out_ANG2, SS_ANG17, m,
        K_in_SBP, K_out_SBP, SSS_ANG2, III_ANG17,
        ES_p, EI_p)

    return
