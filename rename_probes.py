from segmentation import segment_electrodes, electrode, point_of_electrode, representative, probe, load_probes
import argparse
import numpy as np
import pickle


def rename_probes(probe_locs_file, probe_names, probe_file):
    probe_locs = np.loadtxt(f'{probe_locs_file}')

    if np.shape(probe_locs)[0] != np.shape(probe_names)[0]:
        raise ValueError(f"{np.shape(probe_locs)[0]} locations does not match {np.shape(probe_names)[0]} names")

    list_of_probes = load_probes(probe_file)
    num_probes = len(list_of_probes)
    if len(probe_names) != num_probes:
        raise ValueError("Number of names and number of probes do not match")

    for i in range(0, num_probes):
        loc = list_of_probes[i]._electrodes[0].get_midpoint()
        if loc == tuple(probe_locs[i]):
            list_of_probes[i].set_name(probe_names[i])

    for i in range(0, num_probes):
        print(list_of_probes[i].get_name())
        print(list_of_probes[i]._electrodes[0].get_label())

    with open(probe_file, 'wb') as fp:
        pickle.dump(list_of_probes, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename probes based on user input",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("probe_locs", help="List of probe locations")
    parser.add_argument("probe_names",
                        nargs='*',
                        type=str,
                        default=[],
                        help="Names of the probes")
    parser.add_argument("probe_file", help="File containing the probes")
    args = parser.parse_args()
    rename_probes(args.probe_locs, args.probe_names, args.probe_file)
