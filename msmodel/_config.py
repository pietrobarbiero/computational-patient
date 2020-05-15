import argparse


def load_configuration() -> argparse.Namespace:
    """
    Parse command line arguments.

    Parameters
    ----------
    :return: configuration object
    """
    h = 24
    days = 10

    parser = argparse.ArgumentParser()
    parser.add_argument("--age", help="Patient age (years).",
                        default=20, required=False, type=int)
    parser.add_argument("--infection", help="Whether the patient has a SARS infection or not.",
                        default=True, required=False, type=bool)
    parser.add_argument("--dose", help="Drug dose (mg). 5 is nominal dose for Benazepril and 1.25 for Cilazapril.",
                        default=0, required=False, type=int)
    parser.add_argument("--renal-function", help="Type of renal function.", default="normal",
                        required=False, choices=["normal", "impaired"], nargs=1, type=str)
    parser.add_argument("--glu", help="Glucose concentration (mmol/L). "
                                      "To have normal subject glucose dyanmics as input,"
                                      "use glu = 1; for diabetic subjects, glu = 2. Rest all values will be"
                                      "used directly in mmol/L as steady state glucose input.",
                        default=1, required=False, type=float)
    parser.add_argument("--drug-name", help="Drug name.", default="benazepril",
                        required=False)

    parser.add_argument("--n-dose", help="Number of doses per day (cannot be zero).", default=1, required=False,
                        type=int)
    parser.add_argument("--tstart-dosing", help="Initial dosing time.", default=h * 0, required=False, type=int)
    parser.add_argument("--tfinal-dosing", help="Final dosing time.", default=h * days, required=False, type=int)
    parser.add_argument("--sim-time-end", help="Simulation end time.", default=h * days, required=False, type=int)

    parser.add_argument("--show-plots", help="Show plots.", default=True, required=False, type=bool)
    parser.add_argument("--linestyle", help="Plot line style.", default="-", required=False, type=str)
    parser.add_argument("--linewidth", help="Plot line width.", default=2, required=False, type=int)
    parser.add_argument("--legendloc", help="Plot legend location.", default="NorthEast", required=False, type=str)
    args = parser.parse_args()

    return args
