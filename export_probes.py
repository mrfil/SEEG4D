import argparse
# Requires these imports even if not being actively used due to linking issue
from segmentation import export_probes_as_obj, segment_electrodes, electrode, point_of_electrode, representative, probe


def export_probes(segmented, probefile, path):
    """
    segmented is the segmented by probe file
    probefile is the pickled probe file
    path is the path for the outfiles
    """
    export_probes_as_obj(segmented, probefile, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Turn probe data into object files",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("segmented", help="")
    parser.add_argument("probe_file", help="")
    parser.add_argument("output_folder", help="")
    args = parser.parse_args()
    export_probes(args.segmented, args.probe_file, args.output_folder)
