import argparse
import matplotlib.image
import nibabel as nib
import numpy as np

def find_center_of_image(image):
    vox_center = (np.array(image.shape) - 1) / 2
    aff = image.affine
    M = image.affine[:3, :3]
    abc = image.affine[:3, 3]
    f = M.dot([vox_center[0], vox_center[1], vox_center[2]]) + abc
    r = [ (f[0]/aff[0,0] + aff[3,3]), (f[1]/aff[1,1] + aff[3,3]), (f[2]/aff[2,2] + aff[3,3]) ]
    return np.round(vox_center - r)


def get_slice(image, axis):
    img = nib.load(image)
    data = img.get_fdata()
    ctr = find_center_of_image(img)
    if axis == 0:
        sl = data[int(ctr[0])][:][:]
    elif axis == 1:
        sl = data[:][int(ctr[1])][:]
    elif axis == 2:
        sl = data[:][:][int(ctr[2])]

    else:
        raise ValueError("Axis may only contain values 0, 1, 2")

    return sl


def axial_slice(image):
    return get_slice(image, axis=0)


def coronal_slice(image):
    return get_slice(image, axis=1)


def saggital_slice(image):
    return get_slice(image, axis=2)


