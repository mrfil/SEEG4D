import argparse
import numpy as np
import mne


def get_edf_annotations(edf_file, outfile):
    # Load edf file
    raw = mne.io.read_raw_edf(edf_file, preload=False)
    # read annotations
    anno = raw.annotations.description
    print(anno)
    print(type(anno))
    np.save(outfile, anno)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve and save edf annotations",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("edf_file", help="EDF file containing a seizure")
    parser.add_argument("outfile", help="name of text file to write the events to")
    args = parser.parse_args()
    get_edf_annotations(args.edf_file, args.outfile)
