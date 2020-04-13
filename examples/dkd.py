import sys

import pkpd


def main():

    sys.argv.extend(["--renal-function", "impaired"])
    pkpd.covid19_dkd_model()

    sys.argv.extend(["--renal-function", "normal"])
    pkpd.covid19_dkd_model()

    return


if __name__ == "__main__":
    main()
