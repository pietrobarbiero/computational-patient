import sys

import patient


def main():

    sys.argv.extend(["--renal-function", "impaired"])
    patient.computational_patient(model="dkd")

    sys.argv.extend(["--renal-function", "normal"])
    patient.computational_patient(model="dkd")

    return


if __name__ == "__main__":
    main()
