# File to generate a blender FBX from the SEEG data
# Load libraries
import argparse
import os
import bpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from sklearn.preprocessing import normalize
from segmentation import export_probes_as_obj, segment_electrodes, electrode, point_of_electrode, representative, probe

def compute_power(STEP_SIZE, obj_directory, EEG_data):
    power = []
    half_step = int(np.floor(STEP_SIZE / 2))
    for filename in os.listdir(obj_directory):
        f = os.path.join(obj_directory, filename)
        print(f'Loading: {f}')
        # checking if it is a file
        if os.path.isfile(f):
            fname = os.path.splitext(f)[0]
            fname = fname.rsplit('/', 1)[1]
            # For each file in the eeg list
            if str(fname) in EEG_data.columns:
                print(f'Processing {fname}')
                windowed_pwr = []
                windowed_signal = None
                eeg_signal = EEG_data[fname]
                signal_len = len(eeg_signal)
                # For each time point in the series
                for j in range(0, signal_len):
                    # check beginning
                    if j < half_step:
                        windowed_signal = eeg_signal[0:STEP_SIZE]
                    # check n-5 - n
                    elif j > signal_len - half_step:
                        windowed_signal = eeg_signal[signal_len - half_step:-1]
                    # allow rest
                    else:
                        windowed_signal = eeg_signal[j - half_step:j + half_step]

                    tmp_pwr = (sum(windowed_signal ** 2)) / signal_len
                    windowed_pwr.append(tmp_pwr)
                power.append(windowed_pwr)
    return power


def plot_eeg(x, electrode_data, labels, outfile):
    num_lines = len(labels)
    fig, axs = plt.subplots(num_lines, 1, figsize=(num_lines, 2*num_lines), sharey=True)
    for i, ax in enumerate(axs):
        ax.plot(x, electrode_data[i])
        ax.set_ylabel(f"{labels[i]}")
        ax.set_aspect('equal')
        ax.grid('on', linestyle='--')
        #if i != len(axs) - 1:
        #    ax.set_xticklabels([])
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.title("Time centered on seizure (ms)", ha="center", va='top')

    plt.savefig(outfile, bbox_inches='tight')


def generate_fbx(obj_directory, filtered_csv, out_name, mode):
    print(mode)
    if mode != 'power' and mode != 'voltage':
        raise ValueError('Incorrect mode type selected')
    EEG_data = pd.read_csv(filtered_csv)
    EEG_data.drop(EEG_data.columns[[0, 1]], axis=1, inplace=True)

    EEG_data_np = EEG_data.to_numpy()
    EEG_data_abs = np.abs(EEG_data_np)
    EEG_data_size = np.size(EEG_data_np)

    # DO NOT SET > 3. CAUSES ENTIRE PROGRAM TO CRASH UNTIL SYSTEM RESTART
    scale_factor = None

    num_frames = 4 * 1000 # 4 seconds at 1000 samples/second
    STEP_SIZE = 24
    START_FRAME = 0
    eeg_data_list = list(range(START_FRAME, num_frames, STEP_SIZE))  # can add in a half_step size here
    frame_list = eeg_data_list
    for j in range(0, len(frame_list)):
        frame_list[j] = frame_list[j] - START_FRAME
    print(eeg_data_list)

    # EEG_data.loc[idx, 'name'] yields values in time series
    def delete_object(obj_name):
        object_to_delete = bpy.data.objects[obj_name]
        bpy.data.objects.remove(object_to_delete, do_unlink=True)

    delete_object('Cube')
    delete_object('Camera')
    delete_object('Light')
    # Compute Power
    power = None
    if mode == 'power':
        scale_factor = 20
        power = compute_power(STEP_SIZE, obj_directory, EEG_data)
        power = normalize(power, norm='l2', axis=1)
        power = (power - np.min(power)) / (np.max(power) - np.min(power))
    else:
        scale_factor = 4
    k = 0
    brain_matter_names = ['T1_fast_pve_0_CT.obj', 'T1_fast_pve_1_CT.obj', 'T1_fast_pve_2_CT.obj']
    contact_names = []
    contact_eeg_data = []
    eeg_x = None
    print(eeg_x)
    for filename in os.listdir(obj_directory):
        f = os.path.join(obj_directory, filename)
        # checking if it is a file
        obj = None
        mat = None
        fname = None
        x_set = None
        if os.path.isfile(f):
            # isolate the filename as it is the object name
            fname = os.path.splitext(f)[0]
            fname = fname.rsplit('/', 1)[1]

            # import obj files
            bpy.ops.wm.obj_import(filepath=f)
            # generate new material
            mat = bpy.data.materials.new(name=fname + '_mat')
            # # activate new material
            obj = bpy.data.objects[fname]
            obj.active_material = mat

        # colors = []
        scales = []

        if str(fname) in EEG_data.columns:
            # grab first two contacts on electrode
            if (str(fname)[-1] == '1' or str(fname)[-1] == '2') and str(fname)[-2].isalpha():
                contact_names.append(str(fname))
                contact_eeg_data.append(EEG_data[str(fname)])
            if x_set is None:
                eeg_x = [i for i in range(0,len(EEG_data[str(fname)]))]
                x_set = True

            # set color to gold for active electrodes
            mat.diffuse_color = (0.687, 0.366, 0.012, 1)
            for i in range(0, len(eeg_data_list)):
                # compute color scaling, double sum for 2D data
                if mode == 'power':
                    scales.append((power[k][eeg_data_list[i]] * scale_factor, power[k][eeg_data_list[i]] * scale_factor,
                                   power[k][eeg_data_list[i]] * scale_factor))
                else:
                    EEG_data_sum = \
                       sum( sum( EEG_data_abs < np.abs(EEG_data.loc[eeg_data_list[i], str(fname)]) ) ) / EEG_data_size
                    scales.append((scale_factor * EEG_data_sum, scale_factor * EEG_data_sum, scale_factor * EEG_data_sum))

            k = k+1

        # setting color of gm/wm/csf
        elif filename == brain_matter_names[0]:  # csf
            mat.diffuse_color = (1, 1, 1, 0.1)
        elif filename == brain_matter_names[1]:  # gm
            mat.diffuse_color = (0, 0, 1, 0.1)
        elif filename == brain_matter_names[2]:  # wm
            mat.diffuse_color = (0.136, 0.001, 0.001, 0.1)
        # catch all else primarily for electrodes without data
        else:
            # set to light grey and transparent otherwise with no animations
            mat.diffuse_color = (0, 0, 0, 0.1)


        # animate keyframes
        # for fr, c in zip(frame_list, colors):
        #     mat.diffuse_color = c
        #     mat.keyframe_insert(data_path='diffuse_color', frame=fr, index=-1)
        # Apply the transform
        obj.data.update()
        # Resize the object
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')
        for fr, s in zip(frame_list, scales):
            # insert keyframe for object                 .scale[:] = s
            obj.scale[:] = s
            obj.keyframe_insert(data_path='scale', frame=fr, index=-1)
    # call plotting function
    #plot_eeg(eeg_x, contact_eeg_data, contact_names, f'{out_name}.png')
    bpy.ops.wm.save_mainfile(filepath=f'{out_name}.blend')
    # bpy.ops.export_scene.fbx(filepath=f"{out_name}.fbx")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Turn data into object files",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_dir", help="Directory of input object files")
    parser.add_argument("in_csv", help="CSV of filtered data")  # likely will be deleted as the project moves forward
    parser.add_argument("out_name", help="Outfile animation file name")
    parser.add_argument("mode", help="Valid options: power, voltage")
    args = parser.parse_args()
    generate_fbx(args.in_dir, args.in_csv, args.out_name, args.mode)
