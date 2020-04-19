import os
import pickle

import scipy.io
import numpy as np
import pandas as pd

from ._config import load_configuration
from .ode._circulation import call_cardio
from .ode._local_RAS import call_combinedRAS_ACE_PKPD


def covid19_dkd_model(model="cardio"):
    args = load_configuration()

    if model == "cardio":
        params = pd.read_csv('circulation.csv', index_col=0, squeeze=True)
        call_cardio(args, params)

    elif model == "dkd":
        args.renal_function = args.renal_function[0]
        params_file = "".join(["params_", args.drug_name, args.renal_function, ".mat"])
        params = scipy.io.loadmat(params_file)
        call_combinedRAS_ACE_PKPD(args, params)

    return
