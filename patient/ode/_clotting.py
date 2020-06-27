import os
import pickle

import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
import scipy
import pandas as pd
import seaborn as sns


def ODE(t, y, stiffness, vessel_volume):
    clotting, vD, heparin = y

    d_vD_dt = 0
    d_heparin_dt = 0
    d_clotting_dt = 0

    d_dt = np.array([d_vD_dt, d_heparin_dt, d_clotting_dt])

    return d_dt


def clotting_model(days,
                   clotting_t0,
                   vD_t0,
                   heparin_t0,
                   stiffness,
                   vessel_volume):
    # initial condition for the ODE solver
    conc_t0 = np.array([clotting_t0, vD_t0, heparin_t0])
    tau = 0.001

    ODE_args = (stiffness, vessel_volume)
    sol = solve_ivp(fun=ODE, t_span=[0, days],
                    y0=conc_t0,
                    max_step=tau,
                    args=ODE_args, method="LSODA")

    return sol


def call_clotting(args):
    out_dir = f"./data/{args.age}"
    file_name = f"CLOTTING_drug-{args.dose}_glu-{int(args.glu)}_infection-{int(args.infection)}_renal-{args.renal_function}.csv"
    file = os.path.join(out_dir, file_name)
    if os.path.isfile(file):
        return

    days = int(args.sim_time_end / 24)

    # initial conditions
    clotting_t0 = 1
    vD_t0 = 1
    heparin_t0 = 1
    stiffness = 1
    vessel_volume = 1

    sol = clotting_model(days,
                         clotting_t0,
                         vD_t0,
                         heparin_t0,
                         stiffness,
                         vessel_volume)

    x = np.concatenate([sol["t"].reshape((1, -1)), sol["y"]])
    sol_df = pd.DataFrame(x.T, columns=["t", "I", "G", "beta", "IR", "MTOR", "Tt", "C"])

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    file = os.path.join(out_dir, file_name)
    sol_df.to_csv(file)

    return
