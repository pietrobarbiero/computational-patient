import sys

import msmodel


def main():

    sys.argv.extend(["--glu", "1"])
    msmodel.covid19_dkd_model(model="diabetes")
    sys.argv = [sys.argv[0]]

    sys.argv.extend(["--glu", "2"])
    msmodel.covid19_dkd_model(model="diabetes")
    sys.argv = [sys.argv[0]]

    return


if __name__ == "__main__":
    main()
