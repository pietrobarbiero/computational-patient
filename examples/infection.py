import sys

import msmodel


def main():

    # sys.argv.extend(["--renal-function", "impaired"])
    # msmodel.covid19_dkd_model(model="infection")

    sys.argv.extend(["--renal-function", "normal"])
    msmodel.covid19_dkd_model(model="infection")

    return


if __name__ == "__main__":
    main()
