import argparse
import os
import numpy as np
from glob import glob
import math
import random
import pandas as pd
import os.path as op
import sys
import mne
import pickle

import nibabel as nib

import os
import re

from segmentation import segment_electrodes, electrode, point_of_electrode, representative, probe, load_probes

# num_cpu = '1' # Set as a string (for linear algebra functions to use only one CPU per process)
# os.environ['OMP_NUM_THREADS'] = num_cpu

import sys
from matplotlib import pyplot as plt


def find_event_time(events, event_ids, event_name, info):
    if event_name in event_ids:
        return mne.pick_events(events, event_ids[event_name])[0,0] / info['sfreq']
    else:
        return None


def pick_seeg_channels(ch_names):
    return np.array(ch_names)[mne.pick_channels_regexp(ch_names, '^(?!.*(ref|Sref|DC|Ref|EKG|\$)).*$')]


def process(edf_file, probe_file, annotation, lp_freq, hp_freq):
    # Load edf file
    raw = mne.io.read_raw_edf(edf_file, preload=False)
    # Load the probes
    list_of_probes = load_probes(probe_file)
    # read annotations
    events, event_id = mne.events_from_annotations(raw)
    # crop events around the seizure
    print(annotation)
    sz_start = find_event_time(events, event_id, annotation, raw.info)
    print(sz_start)
    print(f'Seizure start time: {sz_start}')
    min = sz_start - 2
    max = sz_start + 2
    if min < 0:
        min = 0
    sz_raw = raw.copy().crop(tmin=min,
                             tmax=max)

    new_names = dict((ch_name, re.sub(r"-[a-zA-Z]+", "", ch_name).replace('EEG ', '').replace('POL ', ''))
                     for ch_name in sz_raw.ch_names)
    sz_raw = sz_raw.rename_channels(new_names)
    # drop bad channel names
    sz_raw = sz_raw.pick(pick_seeg_channels(sz_raw.info.ch_names)).load_data()

    fs = sz_raw.info['sfreq']
    # Notch filter
    filter_freqs = (60, 120, 180, 240)
    filtered = sz_raw.copy().notch_filter(freqs=filter_freqs)
    # default should be 80, 250
    filtered = filtered.copy().filter(l_freq=lp_freq, h_freq=hp_freq)
    filtered_pd = filtered.to_data_frame()
    # GOAL: instead of saving to CSV, save the data into the probe objects themselves (see segmentation.py)
    filtered_pd.to_csv('/seeg_vol/SEEG_Viewer/Outputs/Filtered.csv')
    # loop through the probes
    for i in range(0, len(list_of_probes)):
        electrodes = list_of_probes[i].get_electrodes()
        # loop through the electrodes
        for j in range(0, len(electrodes)):
            name = electrodes[j].get_label()
            # if the name is a column in the dataframe, add it to the electrode
            if name in filtered_pd.columns:
                print(f'ts data set: {name}')
                electrodes[j].set_timeseries(filtered_pd[name])
        # list_of_probes[i]._electrodes = electrodes
    with open(probe_file, 'wb') as fp:
        pickle.dump(list_of_probes, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and EDF file and generate a filtered signal file",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("edf", help="EDF file containing a seizure")
    parser.add_argument("probe_file", help="probefile pickle file containing the probe data")
    parser.add_argument("annotation", help="annotation of the seizure selected")
    parser.add_argument("lpf", help="High pass frequency")
    parser.add_argument("hpf", help="High pass frequency")
    args = parser.parse_args()
    process(args.edf, args.probe_file, args.annotation, args.lpf, args.hpf)
