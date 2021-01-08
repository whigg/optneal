import dimod

import optneal as opn


def main():
    sampleset = dimod.SampleSet.from_samples([[1, 0, 1], [0, 1, 0], [0, 1, 0]], 'BINARY', 0)
    print(sampleset)

    path_json = 'sampleset.json'
    opn.save_sampleset(path_json, sampleset)

    sampleset_loaded = opn.load_sampleset(path_json)

    print(sampleset == sampleset_loaded)


if __name__ == '__main__':
    main()