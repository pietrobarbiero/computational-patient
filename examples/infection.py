import sys

import msmodel


def main():

    # young
    sys.argv.extend(["--age", "20"])
    sys.argv.extend(["--infection", False])
    sys.argv.extend(["--dose", "0"])
    sys.argv.extend(["--renal-function", "normal"])
    sys.argv.extend(["--glu", "5"])
    msmodel.covid19_dkd_model(model="infection")
    sys.argv = [sys.argv[0]]

    # old
    sys.argv.extend(["--age", "60"])
    sys.argv.extend(["--infection", False])
    sys.argv.extend(["--dose", "0"])
    sys.argv.extend(["--renal-function", "normal"])
    sys.argv.extend(["--glu", "5"])
    msmodel.covid19_dkd_model(model="infection")
    sys.argv = [sys.argv[0]]

    # old + drug
    sys.argv.extend(["--age", "60"])
    sys.argv.extend(["--infection", False])
    sys.argv.extend(["--dose", "5"])
    sys.argv.extend(["--renal-function", "normal"])
    sys.argv.extend(["--glu", "5"])
    msmodel.covid19_dkd_model(model="infection")
    sys.argv = [sys.argv[0]]

    # old + glucose + drug
    sys.argv.extend(["--age", "60"])
    sys.argv.extend(["--infection", False])
    sys.argv.extend(["--dose", "5"])
    sys.argv.extend(["--renal-function", "normal"])
    sys.argv.extend(["--glu", "25"])
    msmodel.covid19_dkd_model(model="infection")
    sys.argv = [sys.argv[0]]

    # old + infection + drug
    sys.argv.extend(["--age", "60"])
    sys.argv.extend(["--infection", "True"])
    sys.argv.extend(["--dose", "5"])
    sys.argv.extend(["--renal-function", "normal"])
    sys.argv.extend(["--glu", "5"])
    msmodel.covid19_dkd_model(model="infection")
    sys.argv = [sys.argv[0]]

    # old + glucose + renal + infection + drug
    sys.argv.extend(["--age", "60"])
    sys.argv.extend(["--infection", "True"])
    sys.argv.extend(["--dose", "5"])
    sys.argv.extend(["--renal-function", "impaired"])
    sys.argv.extend(["--glu", "25"])
    msmodel.covid19_dkd_model(model="infection")
    sys.argv = [sys.argv[0]]

    # # old + glucose + renal + infection + drug
    # sys.argv.extend(["--age", "60"])
    # sys.argv.extend(["--infection", "True"])
    # sys.argv.extend(["--dose", "5"])
    # sys.argv.extend(["--renal-function", "impaired"])
    # sys.argv.extend(["--glu", "25"])
    # msmodel.covid19_dkd_model(model="infection")
    # sys.argv = [sys.argv[0]]





    # # young + drug
    # sys.argv.extend(["--age", "20"])
    # sys.argv.extend(["--infection", False])
    # sys.argv.extend(["--dose", "5"])
    # sys.argv.extend(["--renal-function", "normal"])
    # sys.argv.extend(["--glu", "5"])
    # msmodel.covid19_dkd_model(model="infection")
    # sys.argv = [sys.argv[0]]

    # # old + renal
    # sys.argv.extend(["--age", "60"])
    # sys.argv.extend(["--infection", False])
    # sys.argv.extend(["--dose", "0"])
    # sys.argv.extend(["--renal-function", "impaired"])
    # sys.argv.extend(["--glu", "5"])
    # msmodel.covid19_dkd_model(model="infection")
    # sys.argv = [sys.argv[0]]

    # # old + glucose + renal + infection + drug++
    # sys.argv.extend(["--age", "60"])
    # sys.argv.extend(["--infection", "True"])
    # sys.argv.extend(["--dose", "10"])
    # sys.argv.extend(["--renal-function", "impaired"])
    # sys.argv.extend(["--glu", "25"])
    # msmodel.covid19_dkd_model(model="infection")
    # sys.argv = [sys.argv[0]]

    return


if __name__ == "__main__":
    main()
