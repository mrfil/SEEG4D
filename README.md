main.py ->   Run in a VENV (not container), Handles GUI, call docker and pipe everything through the Vol

process_subjects.py -> Run in mne docker, process and register mri/ct

segmentation.py ->   Run in mne docker, segments the electrodes/probes. saves data as txt/p files

MANUAL STEP -> LABEL THE ELECTRODES (refactor fbx_generator.py when auto naming is done) HIGHEST PRIORITY

process_seizure_data.py ->  Run in mne docker, filters eeg data, currently saves as csv - needs fixing

niigz_to_obj.py  -> convert GM/WM segmentation to obj file

export_probes.py -> converts pickled probes to obj file

fbx_generator.py -> converts all object files in a directory to an animation file
