import sys

import numpy as np

import msmodel


def main():

    for age in [70]:#np.arange(20, 70, 10):
        sys.argv.extend(["--age", str(age)])
        sys.argv.extend(["--infection", False])
        sys.argv.extend(["--dose", "0"])
        sys.argv.extend(["--renal-function", "normal"])
        sys.argv.extend(["--glu", "5"])
        msmodel.covid19_dkd_model(model="infection")
        sys.argv = [sys.argv[0]]

        sys.argv.extend(["--age", str(age)])
        sys.argv.extend(["--infection", "True"])
        sys.argv.extend(["--dose", "0"])
        sys.argv.extend(["--renal-function", "normal"])
        sys.argv.extend(["--glu", "5"])
        msmodel.covid19_dkd_model(model="infection")
        sys.argv = [sys.argv[0]]

        sys.argv.extend(["--age", str(age)])
        sys.argv.extend(["--infection", False])
        sys.argv.extend(["--dose", "5"])
        sys.argv.extend(["--renal-function", "impaired"])
        sys.argv.extend(["--glu", "25"])
        msmodel.covid19_dkd_model(model="infection")
        sys.argv = [sys.argv[0]]

        sys.argv.extend(["--age", str(age)])
        sys.argv.extend(["--infection", "True"])
        sys.argv.extend(["--dose", "0"])
        sys.argv.extend(["--renal-function", "impaired"])
        sys.argv.extend(["--glu", "25"])
        msmodel.covid19_dkd_model(model="infection")
        sys.argv = [sys.argv[0]]

        sys.argv.extend(["--age", str(age)])
        sys.argv.extend(["--infection", "True"])
        sys.argv.extend(["--dose", "5"])
        sys.argv.extend(["--renal-function", "impaired"])
        sys.argv.extend(["--glu", "25"])
        msmodel.covid19_dkd_model(model="infection")
        sys.argv = [sys.argv[0]]

        break

    return


if __name__ == "__main__":
    main()
