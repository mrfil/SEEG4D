import numpy as np
import nibabel as nib
import argparse
import scipy.ndimage as ndimage


def convert_file(inname, outname):
    img = nib.load(inname)
    header = img.header
    img_data = img.get_fdata()
    iso_voxels = np.min(header.get_zooms())
    zoom_ratio = tuple(np.array(header.get_zooms()) / iso_voxels)
    print(zoom_ratio)
    print(np.shape(img_data))
    data = ndimage.zoom(img_data, zoom=zoom_ratio, order=1)
    print(np.shape(data))
    iso_img = nib.Nifti1Image(data, None)
    nib.save(iso_img, outname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="convert a nifti file to isometric",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("infile", help="nii.gz file")
    parser.add_argument("outfile", help="output.nii.gz")
    args = parser.parse_args()
    print(args.infile, args.outfile)
    convert_file(args.infile, args.outfile)
