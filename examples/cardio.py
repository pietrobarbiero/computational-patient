import sys

import patient


def main():

    sys.argv.extend(["--age", "20"])
    sys.argv.extend(["--infection", "False"])
    sys.argv.extend(["--dose", "0"])
    sys.argv.extend(["--renal_function", "normal"])
    sys.argv.extend(["--glu", "1"])
    patient.computational_patient(model="cardio")

    sys.argv.extend(["--age", "60"])
    sys.argv.extend(["--infection", "True"])
    sys.argv.extend(["--dose", "0"])
    sys.argv.extend(["--renal_function", "impaired"])
    sys.argv.extend(["--glu", "2"])
    patient.computational_patient(model="cardio")

    return


if __name__ == "__main__":
    main()
