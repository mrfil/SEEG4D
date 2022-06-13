import subprocess
import argparse
from convert_to_iso import convert_file


def call_fsl(cmd, pipe):
    proc = subprocess.Popen(cmd.split(), stdout=pipe, stderr=pipe, shell=False)
    return proc.communicate()


def process(ct_file, mri_file):
    output_folder = '/seeg_vol/SEEG_Viewer/Outputs/'
    SEEG_CT = ct_file
    file_mri_struct = mri_file
    pipe = subprocess.PIPE
    print("CT Filename loaded: " + str(SEEG_CT))
    print("MRI Filename loaded: " + str(file_mri_struct))

    #  Threshold CT
    call_fsl(f"fslmaths {SEEG_CT} -thr 0 {output_folder}SEEG_CT.nii.gz", pipe)
    # Run BET on MRI
    print("Extracting brain. This may take some time.")
    call_fsl(f"bet {file_mri_struct} {output_folder}mri_struct -s -B -m -f 0.3",pipe)
    print("Brain extraction finished")
    # Generate partial volume estimates
    print("Extracting GM/WM/CSF. This may take some time.")
    call_fsl(f"fsl_anat -i {file_mri_struct} -o {output_folder}brain_seg --nosubcortseg",pipe) #removes --nosubcortseg --nononlinreg --betfparam 0.3
    print("GM/WM/CSF segmentation finished")
    # Register CT to MRI Space, temp is for debugging
    call_fsl(f"flirt -in {SEEG_CT} -ref {file_mri_struct} -dof 6 -cost mutualinfo -omat {output_folder}SEEG2MRI.mat -out {output_folder}temp.nii.gz", pipe)
    # Erode MRI mask - Added 3rd erosion back in
    call_fsl(f"fslmaths {output_folder}mri_struct_mask -kernel 3D -ero -ero -ero {output_folder}ero_mri_struct_mask.nii.gz", pipe)
    # Create transform matrices
    call_fsl(f"convert_xfm -omat {output_folder}MRI2SEEG.mat -inverse {output_folder}SEEG2MRI.mat",pipe)
    # Apply mask to CT
    print("Applying mask to the CT")
    call_fsl(f"flirt -interp nearestneighbour -in {output_folder}ero_mri_struct_mask.nii.gz -ref {output_folder}SEEG_CT.nii.gz -applyxfm -init {output_folder}MRI2SEEG.mat -out {output_folder}mriMaskinSEEG.nii.gz", pipe)

    # New idea: Threshold -> filter -> iso -> threshold
    # 03/13/2024 lowering threshold to 90% from 99.95
    THR = call_fsl(f"fslstats {output_folder}SEEG_CT.nii.gz -P 99.95", pipe)
    thresh_out = THR[0]
    thr_out_arr = thresh_out.split()
    thr_val = thr_out_arr[0]
    thr_val = float(thr_val.decode('ascii'))
    print(f"Threshold chosen: {thr_val}")
    call_fsl(f"fslmaths {output_folder}SEEG_CT.nii.gz -thr {thr_val} -mas {output_folder}mriMaskinSEEG.nii.gz {output_folder}SEEG_CT_thr.nii.gz", pipe)

    # apply median filtering
    print("Applying filtering to the CT")
    call_fsl(f"fslmaths {output_folder}SEEG_CT_thr.nii.gz -kernel sphere 0.5 -fmedian {output_folder}filtered_CT.nii.gz", pipe)

    # convert to iso before threshold
    convert_file(f'{output_folder}filtered_CT.nii.gz', f'{output_folder}filtered_CT_iso.nii.gz')
    convert_file(f'{output_folder}mriMaskinSEEG.nii.gz', f'{output_folder}mriMaskinSEEG_iso.nii.gz')

    print("Thresholding CT")
    # Double Threshold - old 99.50
    THR = call_fsl(f"fslstats {output_folder}filtered_CT_iso.nii.gz -P 40", pipe)
    thresh_out = THR[0]
    thr_out_arr = thresh_out.split()
    thr_val = thr_out_arr[0]
    thr_val = float(thr_val.decode('ascii'))
    print(f"Threshold chosen: {thr_val}")
    call_fsl(f"fslmaths {output_folder}filtered_CT_iso.nii.gz -thr {thr_val} -mas {output_folder}mriMaskinSEEG_iso.nii.gz {output_folder}mri_mas_thr_SEEG_in_SEEG_rethr.nii.gz", pipe)
    print("Thresholding finished")
    # Convert PVE
    print("Converting PVEs")
    brain_seg = f'{output_folder}brain_seg.anat/'
    for i in range(0, 3):
        # Align PVE TO MRI
        call_fsl(f"flirt -in {brain_seg}T1_fast_pve_{i}.nii.gz -ref {file_mri_struct} -applyxfm -usesqform -out {output_folder}T1_fast_pve_{i}_MRI.nii.gz", pipe)
        # Align PVE_MRI to CT
        call_fsl(f"flirt -in {output_folder}T1_fast_pve_{i}_MRI.nii.gz -ref {output_folder}SEEG_CT.nii.gz -applyxfm -init {output_folder}MRI2SEEG.mat -out {output_folder}T1_fast_pve_{i}_CT.nii.gz", pipe)

    call_fsl(f"flirt -interp nearestneighbour -in {file_mri_struct} -ref {output_folder}SEEG_CT.nii.gz -applyxfm -init {output_folder}MRI2SEEG.mat -out {output_folder}mri_struct_CT.nii.gz", pipe)

    print("Conversion finished")


if __name__ == "__main__":
    print('entered file')
    parser = argparse.ArgumentParser(description="Pre-Process the CT and MRI files",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("ct", help="CT file as nii.gz", type=str)
    parser.add_argument("mri", help="T1 Weighted MRI as nii.gz", type=str)
    args = parser.parse_args()
    print('parsed')
    process(args.ct, args.mri)
    print('File finished')
