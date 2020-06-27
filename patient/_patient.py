import os
import pickle

import scipy.io
import numpy as np
import pandas as pd

from ._config import load_configuration
from .ode._circulation import call_cardio
from .ode._diabetes import call_diabetes
from .ode._infection import call_infection
from .ode._local_RAS import call_combinedRAS_ACE_PKPD
from .ode._hypertension import call_hypertension


def computational_patient(model="infection"):
    args = load_configuration()

    if model == "diabetes":
        call_diabetes(args)

    elif model == "infection":
        args.renal_function = args.renal_function[0]
        try:
            params_file = "".join(["params_", args.drug_name, args.renal_function, ".mat"])
            params = scipy.io.loadmat(params_file)
        except:
            params_file = "".join(["params_", "benazepril", args.renal_function, ".mat"])
            params = scipy.io.loadmat(params_file)

        call_diabetes(args)
        call_infection(args, params)

        cardio_params = pd.read_csv('circulation.csv', index_col=0, squeeze=True)
        call_cardio(args, cardio_params)

    elif model == "hypertension":
        call_hypertension(args)

    elif model == "cardio":
        params = pd.read_csv('circulation.csv', index_col=0, squeeze=True)
        call_cardio(args, params)

    elif model == "dkd":
        # args.renal_function = args.renal_function[0]
        params_file = "".join(["params_", args.drug_name, args.renal_function, ".mat"])
        params = scipy.io.loadmat(params_file)
        call_combinedRAS_ACE_PKPD(args, params)

    return
