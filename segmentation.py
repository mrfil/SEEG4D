import argparse
import sys
import copy
import nibabel as nib
from os.path import exists
import math
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import resource
import scipy.ndimage as ndimage
from nibabel.affines import apply_affine
import numpy.linalg as npl
from skimage.segmentation import watershed
from skimage.morphology import opening
from skimage.morphology import remove_small_objects
import skimage.morphology
from scipy import ndimage as ndi
from scipy.ndimage import zoom
from skimage.feature import peak_local_max
from skimage.measure import marching_cubes


class electrode:
    """
    Class to represent a single electrode in the brain
    """
    def __init__(self):
        self._electrode_points = []
        self._midpoint = None
        self._name = ""
        self._number = None
        self._time_series_data = []

    def __eq__(self, obj):
        """
        Set equals to compare the midpoint of two electrodes
        :param obj: another instance of an electrode object
        """
        if isinstance(obj, electrode):
            return self._midpoint == obj.get_midpoint()
        return False

    def __len__(self):
        return len(self._elecrode_points)

    def add_electrode_point(self, pt):
        """
        Add a point to the electrode
        :param pt: [x,y,z] point being added
        """
        self._electrode_points.append(pt)

    def set_number(self, n):
        self._number = n

    def get_number(self):
        return self._number

    def set_name(self, name):
        """
        Name of the electrode, required for source localization
        :param name: name of the electrode
        """
        self._name = name

    def get_name(self):
        return self._name

    def get_label(self):
        return f'{self._name}{self._number}'

    def set_timeseries(self, ts):
        """
        set time series data to the electrode
        function under construction
        :param ts: time series data as an array
        """
        self._time_series_data = ts

    def get_timeseries(self):
        """
        retrieve time series data list
        :return: time series data point
        """
        return self._time_series_data

    def _calculate_midpoint(self):
        """
        calculate midpoint of the electrode and store it in the object as a *tuple*
        """
        mid = [0, 0, 0]
        dims = len(mid)
        num_electrodes = len(self._electrode_points)
        if num_electrodes == 0:
            print('Attempted to calculate midpoint of an empty set')
            self._midpoint = mid
        else:
            # loop through all the electrode points, add their x,y,z coordinates and divide by the number of points
            for e in self._electrode_points:
                loc = e.show_loc()
                for j in range(0,dims):
                    mid[j] += loc[j]
            for i in range(0, dims):
                mid[i] = round(mid[i] / num_electrodes)
            self._midpoint = tuple(mid)

    def get_electrode_points(self):
        """
        :return: array of points within the electrode
        """
        return self._electrode_points

    def get_midpoint(self):
        """
        If the midpoint does not yet exist, calculate it. Then return the midpoint
        :return: midpoint of the electrode as a tuple
        """
        if self._midpoint is None:
            self._calculate_midpoint()
        return self._midpoint

    def set_midpoint(self,val):
        """
        Set the midpoint to a specified value
        :param val: val of the midpoint as a tuple
        """
        # if self._midpoint == None:
        #     self._calculate_midpoint()
        self._midpoint = val

    def get_length(self):
        """
        :return: the number of points in the electrode
        """
        return len(self._electrode_points)
    
    def get_electrode_points_values(self):
        """
        Get all the points associated in the electrode as a list of (x,y,z) coordinates
        :return: list of tuples represented [x,y,z] coordinates
        """
        tmp = []
        if self._electrode_points:
            for e in self._electrode_points:
                tmp.append(e.show_loc())
        return tmp

    # Please don't call this function unless you know what you're doing
    # Pretty please
    # No seriously, you'll delete data >:(
    def purge_electrodes(self, d):
        """
        Removes all points from the electrode and deletes the points from the main data array
        This is only to be used when replacing a misshapen, or small electrode with a representative
        Please be very careful when using
        :param d: the data array of the image
        """
        for pt in self._electrode_points:
            l = pt.show_loc()
            d[l[0],l[1],l[2]] = 0
        self._midpoint = None
        self._electrode_points = []

    def erode(self, d, representative):
        """
        Erodes the electrode in two directions.
        :param d: the data array of the image
        """
        rep_x = 3
        rep_y = 3
        rep_z = 3
        # create two lists of electrodes, the master list and the mutable
        master = []
        for pt in self._electrode_points:
            master.append(pt.show_loc())
        mutable = np.copy(master).tolist()
        for i in range(0, len(mutable)):
            mutable[i] = tuple(mutable[i])

        # loop through master, splicing into 3 new lists. all x all y all z coords
        x_list = []
        y_list = []
        z_list = []
        for pts in master:
            x_list.append(pts[0])
            y_list.append(pts[1])
            z_list.append(pts[2])

        x_list.sort()
        y_list.sort()
        z_list.sort()

        x_dist = np.abs(x_list[0] - x_list[-1])
        y_dist = np.abs(y_list[0] - y_list[-1])
        z_dist = np.abs(z_list[0] - z_list[-1])

        # if a point in the master list does not have 8 adjacent points, remove it from the mutable list
        master_shape = np.shape(master)

        for pts in range(0, master_shape[0]):
            base_loc = master[pts]

            # only search in x,y direction to try and prevent z from becoming too thin
            if z_dist > rep_z:  # erode z
                """for z in [(base_loc[2] + k) for k in (-1, 0, 1) if k != 0]:
                     # point = tuple([x,y,base_loc[2]])
                     point = tuple([base_loc[0], base_loc[1], z])
                     if point not in master:
                         mutable.remove(base_loc)
                         break"""
                for x in [(base_loc[0] + i) for i in (-1, 0, 1) if i != 0]:
                    # point = tuple([x,y,base_loc[2]])
                    point = tuple([x, base_loc[1], base_loc[2]])
                    if point not in master:
                        mutable.remove(base_loc)
                        break
            elif x_dist > rep_x: #erode y
                for y in [(base_loc[1] + j) for j in (-1, 0, 1) if j != 0]:
                    # point = tuple([x,y,base_loc[2]])
                    point = tuple([base_loc[0], y, base_loc[2]])
                    if point not in master:
                        mutable.remove(base_loc)
                        break
            elif y_dist > rep_y: #erode x
                for x in [(base_loc[0] + i) for i in (-1, 0, 1) if i != 0]:
                    # point = tuple([x,y,base_loc[2]])
                    point = tuple([x, base_loc[1], base_loc[2]])
                    if point not in master:
                        mutable.remove(base_loc)
                        break

        # then, delete all the electrode points
        self.purge_electrodes(d)
        # rebuild the electrode with new values
        for new_points in mutable:
            new_x, new_y, new_z = new_points
            d[new_x, new_y, new_z] = 3000
        # rebuild the probe
        for coords in mutable:
            self.add_electrode_point(point_of_electrode(coords[0], coords[1], coords[2]))
        self._midpoint = None

    def remove_standalone(self, d):
        """
        Removes electrodes with a single voxel in them. Intended to be used only once per file during the first pass
        erosions to remove noise and 'bridges'
        """
        if len(self._electrode_points) == 1:
            # then, delete all the electrode points
            print(f'removed single: {self._electrode_points[0].show_loc()}')
            self.purge_electrodes(d)
        self._midpoint = None

    def export_as_obj(self, path, shape, pxdims):
        """
        param shape: [x,y,z] size of data array
        NOTE: THIS FUNCTION IS CURRENTLY WORKING WITH THE DATA ARRAY IN CT SPACE, IT EVENTUALLY NEEDS TO PORT TO MRI SPACE
        THE TEMPORARY FIX IS TO PLOT THE MRI IN CT SPACE FOR THE VR DEMO, HOWEVER THIS MUST BE FIXED FOR THE FINAL RELEASE
        """
        # resample to isotropic first, then generate the square
        # try nearest neighbor interpolation upscaling of midpoints
        d = np.zeros(shape)
        for x,y,z in [(self._midpoint[0]+i,self._midpoint[1]+j,self._midpoint[2]+k) for i in (-1,0,1) for j in (-1,0,1) for k in (-1, 0, 1)]:
            d[x,y,z] = 5000
        # 1mm isotropic
        # Not needed anymore since data is converted to isotropic elsewhere
        # d = zoom(d, (pxdims[0], pxdims[1], pxdims[2]))

        # flip in x dir
        d = np.flip(d, 0)

        # generate obj
        verts, faces, normals, values = marching_cubes(d, method='lewiner')
        faces = faces+1
        # file = open(str(fname)+'_'+str(self._midpoint[0])+'_'+str(self._midpoint[1])+'_'+str(self._midpoint[2])+'.obj', 'w')
        file = open(f'{path}{self.get_label()}.obj', 'w')
        for item in verts:
            file.write("v {0} {1} {2}\n".format(item[0],item[1],item[2]))

        for item in normals:
            file.write("vn {0} {1} {2}\n".format(item[0],item[1],item[2]))

        for item in faces:
            file.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(item[0],item[1],item[2]))  

        file.close()


class point_of_electrode:
    """
    class to represent a single point in space belonging to an electrode
    """
    def __init__(self, x: int, y: int, z: int):
        """
        initialize the class as a set of x,y,z coordiantes. inputs must be integers
        """
        self._x = x
        self._y = y
        self._z = z

    def show_loc(self):
        """
        Return the x,y,z coordinates of the point
        :return: tuple(x,y,z) of the points
        """
        return tuple( [self._x, self._y, self._z] )


class representative(electrode):
    """
    Class which extends the electrode class to create a 'representative' electrode
    The representative is designed to be automatically calculated and is the 'perfect' electrode
    It's used to wipe out bad electrodes and replace them with the representative so that better calculations can be made
    """
    def __init__(self, e):
        """
        Intialize the representative
        :param e: electrode to be used as the representative
        """
        super(representative, self).__init__()
        self._electrode_points = e._electrode_points
        self._midpoint = e._midpoint
        self._distances = None

    @classmethod
    def from_electrode(cls, e):
        """
        Class method to return functions from the inherited class
        :return: return value of inherited function
        """
        return cls(e)

    def create_distance_matrix(self):
        """
        Create a distance matrix of the distance from each point in the matrix to the midpoint
        """
        if self._distances == None:
            self._distances = []
            for point in self._electrode_points:
                self._distances.append(tuple(np.subtract(self._midpoint,point.show_loc() )))

    def get_distances(self):
        """
        Returns the distance matrix
        :return: list of distances from each point in the electrode to the midpoint of the electrode
        """
        if self._distances == None:
            self.create_distance_matrix()
        return self._distances


class probe:
    """
    class to represent a full probe which contains multiple electrode contacts which each have multiple points
    """
    def __init__(self):
        self._electrodes = []
        self._name = None
        self._electrode_data = None

    def get_name(self):
        return self._name

    def set_name(self, n):
        self._name = n
        self.add_name_to_electrodes(self._name)

    def get_electrodes(self):
        """
        Return the electrodes
        :return: list of electrodes
        """
        return self._electrodes

    def add_electrode(self,e):
        """
        Add an electrode to the probe
        :param e: electrode to be added
        """
        self._electrodes.append(e)
    
    def segment(self, n, d):
        """
        Set the intensity values of the probe so that it can be segmented distinctly from the other probes
        :param n: scaling factor for intensity segmentation
        :param d: the data mtrix of the image
        """
        for e in self._electrodes:
            points = e.get_electrode_points_values()
            for pt in points:
                d[pt[0],pt[1],pt[2]] = (n * 100) + 2000

    def sort_electrodes(self, c, pixdims):
        """
        Sorts electrodes based on the distance from the outside of the image
        :param c: center of the image
        """
        # get list of midpoints
        sorted_pairs = []
        unsorted_pairs = copy.deepcopy(self._electrodes)
        num_electrodes = len(self._electrodes)
        # for i in range(0, num_electrodes):
        #    print(unsorted_pairs[i].get_midpoint())
        # calculate distance between midpoint and center of image
        distances = list(map(lambda e:calc_euclidean_dist_3D(tuple(e.get_midpoint()), tuple(c)), self._electrodes))
        # Find max distance, and name probe at the furthest distance from the center to be electrode contact N
        furthest = distances.index(np.max(distances))
        sorted_pairs.append(self._electrodes[furthest])
        unsorted_pairs.remove(self._electrodes[furthest])
        # Build the rest of the sorted list by finding the closest point to it
        for i in range(0, num_electrodes - 1):
            # get distances of the unsorted compared to furthest
            distances = list(
                map(lambda e: calc_euclidean_dist_3D(
                    tuple(e.get_midpoint()), tuple(self._electrodes[furthest].get_midpoint())),
                    unsorted_pairs))
            # calculate new furthest point based on proximity to previous furthest point
            furthest = distances.index(np.min(distances))
            # add furthest point to list
            sorted_pairs.append(unsorted_pairs[furthest])
            # remove furthest point from unsorted list
            unsorted_pairs.remove(unsorted_pairs[furthest])
        # Number the electrodes outermost to innermost and reassign back to the class property
        self._electrodes = self.number_electrodes(sorted_pairs)
        """for i in range(0, num_electrodes):
            print(self._electrodes[i].get_midpoint(), self._electrodes[i].get_name())"""
        return

    def number_electrodes(self, list_of_e):
        """
        Making the assumption that list_of_e[0] is the outermost electrode on the probe and that the list is sorted
        outwards to inwards, i.e. that list_of_e[n] is the electrode at the tip of the probe inside the brain.
        """
        length = len(list_of_e)
        for i in range(0, length):
            list_of_e[i].set_number(str(length - i))
        return list_of_e

    def add_name_to_electrodes(self, name):
        """
        name is the alphabet combination associated with the probe. e.g. Probe A, Probe B. and is applied across the
        probe to each contact via the functions downstream.
        """
        num_electrodes = len(self._electrodes)
        for i in range(0, num_electrodes):
            self._electrodes[i].set_name(f'{name}')

    def export_as_obj(self, path, shape, pxdims):
        for e in self._electrodes:
            e.export_as_obj(path, shape, pxdims)


def add_electrode_points(p, e, u):
    """
    Recursive function to build an electrode
    :param p: Electrode being built
    :param e: point_of_electrode being added
    :param u: global search space of x,y,z points
    """
    base_loc = e.show_loc()
    # check the nearby voxels to find neighbors
    for x,y,z in [(base_loc[0]+i, base_loc[1]+j, base_loc[2]+k)
                  for i in (-1, 0, 1)
                  for j in (-1, 0, 1)
                  for k in (-1, 0, 1)
                  if abs(i) + abs(j) + abs(k) == 1]:

        # log neighboring postitions
        point = tuple([x, y, z])

        # if this point is in the space of x,y,z points, add it to the electrode
        if point in u:
            # print(f"added electrode {point}")
            new_e = point_of_electrode(point[0], point[1], point[2])
            p.add_electrode_point(new_e)
            u.remove(point)
            add_electrode_points(p, new_e,u)


def show_slices(slices):
    """
    view and plot a brain in 3 different slices
    :param slices:list of 3 slices of views
    """
    fig, axes = plt.subplots(1, len(slices))
    axes[0].set_xlabel('Y values')
    axes[0].set_ylabel('Z values')
    axes[1].set_xlabel('X values')
    axes[1].set_ylabel('Z values')
    axes[2].set_xlabel('X values')
    axes[2].set_ylabel('Y values')
    for i, slice in enumerate(slices):  
        axes[i].imshow(slice.T, cmap="gray", origin="lower")


def get_electrodes(data):
    """
    Recursively retrive the electrodes from the image data
    :param data: 3D image data
    """
    nz = np.nonzero(data)
    nz_x = nz[0]
    nz_y = nz[1]
    nz_z = nz[2]
    unchecked = set()
    electrodes = []

    # add every voxel with a nonzero value to a list of unchecked points
    for i in range(0, len(nz_x)):
        unchecked.add( tuple( [nz_x[i], nz_y[i], nz_z[i]] ) )

    while unchecked:
        # initialize the first point
        pt = unchecked.pop()
        e = point_of_electrode(pt[0],pt[1],pt[2])
        p = electrode()
        p.add_electrode_point(e)
        # kick off recursion
        add_electrode_points(p, e, unchecked)
        electrodes.append(p)
    return electrodes


def get_midpoints(e, data):
    """
    Retrieve midpoints of every electrode and set their intensities on the image plot
    :param e: list of electrodes
    :param data: 3D image data
    """
    mpts = []
    for elect in e:
        mid = elect.get_midpoint() # tuple
        data[ mid[0] ][ mid[1] ][ mid[2] ] = 8000
        mpts.append(mid)
    print(f"Number of electrodes: {len(mpts)}")
    print(f"Length of e: {len(e)}")
    return mpts, data, e
    

def remove_outlier_electrodes(num_p, e, e_min, e_max):
    """
    Remove outlier null electrodes
    :param num_p:
    :param e:
    :param e_min:
    :param e_max:
    """
    length = len(num_p)
    new_num_p = []
    new_e = []
    
    for i in range(0, length):
        if num_p[i] < e_max and num_p[i] > e_min:
            new_num_p.append(num_p[i])
            new_e.append(e[i])
    return new_num_p, new_e# data[s<m]


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def replace_electrode(e, r, d):
    mid = e.get_midpoint()
    e.purge_electrodes(d)
    for dist in r.get_distances():
        x, y, z = np.add(mid, dist)
        point = point_of_electrode(x, y, z)
        d[x,y,z] = 3000
        e.add_electrode_point(point)
    # reset the midpoint highlight
    d[mid[0], mid[1], mid[2]] = 8000


def calc_euclidean_dist_3D(pt_1, pt_2):
    # removing usage of pixdims as image should be isometric anyway
    # assume inputs are tuples
    return np.sqrt( ((pt_1[0] - pt_2[0])**2) + ((pt_1[1] - pt_2[1])**2) + ((pt_1[2] - pt_2[2])**2) )


def sort_electrodes(electrodes, c):
    """
    Sorts electrodes based on the distance from the outside of the image
    :param electrodes: list of electrodes
    :param c: center of the image
    """
    # get list of midpoints
    sorted_pairs = []
    unsorted_pairs = copy.deepcopy(electrodes)
    num_electrodes = len(electrodes)
    # for i in range(0, num_electrodes):
    #    print(unsorted_pairs[i].get_midpoint())
    # calculate distance between midpoint and center of image
    distances = list(map(lambda e:calc_euclidean_dist_3D(tuple(e.get_midpoint()), tuple(c)), electrodes))
    # Find max distance, and name probe at the furthest distance from the center to be electrode contact N
    furthest = distances.index(np.max(distances))
    sorted_pairs.append(electrodes[furthest])
    unsorted_pairs.remove(electrodes[furthest])
    # Build the rest of the sorted list by finding the closest point to it
    for i in range(0, num_electrodes - 1):
        # get distances of the unsorted compared to furthest
        distances = list(
            map(lambda e: calc_euclidean_dist_3D(
                tuple(e.get_midpoint()), tuple(electrodes[furthest].get_midpoint())),
                unsorted_pairs))
        # calculate new furthest point based on proximity to previous furthest point
        furthest = distances.index(np.min(distances))
        # add furthest point to list
        sorted_pairs.append(unsorted_pairs[furthest])
        # remove furthest point from unsorted list
        unsorted_pairs.remove(unsorted_pairs[furthest])
    # Number the electrodes outermost to innermost and reassign back to the class property
    # sorted_pairs[0] is the outermost electrode
    return sorted_pairs


def calc_theta(pr, p2, p3):
    p1 = pr._electrodes[-2].get_midpoint()
    v1 = np.subtract(p2, p1)
    v2 = np.subtract(p3, p1)

    arccos = np.arccos( (v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2])
                       /np.sqrt( (v1[0]**2 + v1[1]**2 + v1[2]**2)*(v2[0]**2 + v2[1]**2 + v2[2]**2) ) )
    angle = arccos * 180/np.pi
    return angle


def build_probe(pr, e_1, l, n, pixdims, theta):
    if len(l) == 0:
        return
    distances = []
    thetas = []
    for e_2 in l:
        pt_1 = e_1.get_midpoint()
        pt_2 = e_2.get_midpoint()
        distances.append(calc_euclidean_dist_3D(pt_1, pt_2))
        if len(pr.get_electrodes()) >= 2:
            thetas.append(calc_theta(pr, pt_1, pt_2))
    min_dist = min(distances)
    minpos = distances.index(min_dist)

    if min_dist <= n and (len(thetas) == 0 or thetas[minpos] <= theta):
        if len(thetas) > 0:
            print(thetas[minpos])
        e_2 = l.pop(minpos)
        pr.add_electrode(e_2)
        build_probe(pr, e_2, l, n, pixdims, theta)


def save_probes(fname, e, c, pixdims):
    list_of_probes = []
    search_dist = 15
    theta = 20 # theta is in degrees
    print("Starting")
    while e:
        pr = probe()
        e = sort_electrodes(e, c)
        e_1 = e.pop(0)
        pr.add_electrode(e_1)
        print("Start recursion")
        build_probe(pr, e_1, e, search_dist, pixdims, theta) # 5 was old value, with new iso space conversions, upping to 8, 8 too much. lower to 6?
        for elect in pr._electrodes:
            print(elect.get_midpoint())
        print("end recursion")
        list_of_probes.append(pr)

    print(f"Number of probes is: {len(list_of_probes)}")
    for p in list_of_probes:
        p.sort_electrodes(c, pixdims)
    with open(fname, 'wb') as fp:
        pickle.dump(list_of_probes, fp)
    print('Saved Probes')
    return list_of_probes


def load_probes(fname):
    probes = []
    with open(fname, 'rb') as fp:
        probes = pickle.load(fp)
    print(f"Number of probes is: {len(probes)}")
    return probes


def filter_isolated_cells(array, struct):
    """ Return array with completely isolated single cells removed
    :param array: Array with completely isolated single cells
    :param struct: Structure array for generating unique regions
    :return: Array with minimum region size > 1
    """

    filtered_array = np.copy(array)
    id_regions, num_ids = ndimage.label(filtered_array, structure=struct)
    id_sizes = np.array(ndimage.sum(array, id_regions, range(num_ids + 1)))
    area_mask = (id_sizes == 1)
    filtered_array[area_mask[id_regions]] = 0
    return filtered_array


def find_center_of_image(image):
    vox_center = (np.array(image.shape) - 1) / 2
    aff = image.affine
    M = image.affine[:3, :3]
    abc = image.affine[:3, 3]
    f = M.dot([vox_center[0], vox_center[1], vox_center[2]]) + abc
    r = [ (f[0]/aff[0,0] + aff[3,3]), (f[1]/aff[1,1] + aff[3,3]), (f[2]/aff[2,2] + aff[3,3]) ]
    return np.round(vox_center - r)


def create_representative(fname, structure):
    elect = electrode()
    rep = representative(elect)
    shape = np.shape(structure)
    if (len(shape) != 3):
        raise ValueError('Structure shape invalid - must be 3 dimensional')

    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            for k in range(0, shape[2]):
                if structure[i][j][k] == 1:
                    rep.add_electrode_point(point_of_electrode(i, j, k))
    rep._calculate_midpoint()
    rep.create_distance_matrix()
    file_rep = open(fname, 'wb')
    pickle.dump(rep, file_rep)


def load_representative(fname):
    filehandler = open(fname, 'rb')
    return pickle.load(filehandler)


def delete_close_electrodes(e, d):
    # e is a list of electrodes
    dist = 4
    checked = []
    while len(e) != 0:
        print(len(e))
        p = e.pop()
        close = False
        for elect in e:
            euclid_dist = calc_euclidean_dist_3D(elect.get_midpoint(), p.get_midpoint())
            print(p.get_midpoint(), elect.get_midpoint(), euclid_dist)

            if euclid_dist < dist:
                print('purge me')
                p.purge_electrodes(d)
        if not close:
            checked.append(p)

    return checked, d

def segment_electrodes(f, rep_file, probefile, probelocationsfile, segmentedfile, electrodelabelsfile, datafile):
    # Assumes file input is isotropic
    sys.setrecursionlimit(100000)
    resource.setrlimit(resource.RLIMIT_STACK, [int(1.6e+7), resource.RLIM_INFINITY])
    img = nib.load(f)
    pixdim = img.header['pixdim']
    header = img.header

    data = img.get_fdata()
    print(pixdim)
    print(np.shape(data))
    print(not(exists(probefile)))
    print(not(exists(datafile)))
    """
    struct = [[[1,1,1],[1,1,1],[1,1,1]],
            [[1,1,1],[1,1,1],[1,1,1]],
            [[1,1,1],[1,1,1],[1,1,1]]]
    create_representative('/seeg_vol/SEEG_Viewer/cube_representative_3D.p',struct)
    """
    rep = load_representative(rep_file)
    print(rep.get_length())
    ctr = find_center_of_image(img)
    e = get_electrodes(data)
    mpts, d, e = get_midpoints(e, data)

    # list_of_probes is used, ignore the linter saying it's not.
    list_of_probes = []

    if not(exists(probefile)):
        erosion = True
        erosion_counter = 0
        # Loop breaks when no more erosions are performed
        while erosion:
            print('Eroded')
            erosion = False
            # replace the smaller electrodes with the representative
            # erode larger electrodes
            print(f'Electrode count: {len(e)}')
            for elect in e:
                # A cube representative has length 27
                if elect.get_length() <= 64 and erosion_counter >= 0:
                    replace_electrode(elect, rep, d)
                else:
                    elect.erode(d, rep)
                    erosion = True

            if erosion:
                erosion_counter += 1
                print(f'erosion counter: {erosion_counter}')
            e = get_electrodes(d)
            mpts, d, e = get_midpoints(e, d)
            mid_img = nib.Nifti1Image(d, None)
            nib.save(mid_img, f'/seeg_vol/SEEG_Viewer/Outputs/{erosion_counter}_pass_iso.nii.gz')

            print('end of loop')
        # build probes
        e = get_electrodes(d)
        mpts, d, e = get_midpoints(e, d)

        #if electrode contacts touch, delete one
        for elect in e:
            replace_electrode(elect, rep, d)
        e = get_electrodes(d)
        mpts, d, e = get_midpoints(e, d)

        list_of_probes = save_probes(probefile, e, ctr, pixdim)
    else:
        list_of_probes = load_probes(probefile)

    num_probes = len(list_of_probes)
    probe_locs = []
    for i in range(0, num_probes):
        list_of_probes[i].segment(i+1,d)
        probe_locs.append(list_of_probes[i]._electrodes[0].get_midpoint())

    if not(exists(probelocationsfile)):
        # save midpoints into text file
        np.savetxt(probelocationsfile, probe_locs, fmt='%d')
    if not(exists(segmentedfile)):
        # Save image as nifti
        new_header = img.header.copy()
        mid_img = nib.Nifti1Image(d, None)
        nib.save(mid_img, segmentedfile)
    if not(exists(electrodelabelsfile)):
        myfile = open(electrodelabelsfile, 'w')
        for i in range(0, len(list_of_probes)):
            myfile.write(f"Probe {i+1}: \n")
            for e in list_of_probes[i].get_electrodes():
                myfile.write(str(e.get_midpoint()))
                myfile.write('\n')

    if not(exists(datafile)):
        np.save(datafile, d)
    return d


# This function has to stay in this file due to some weird linking issue. Advise not refactoring
def export_probes_as_obj(segmented, probefile, path):
    """
    segmented is the segmented by probe file
    probefile is the pickled probe file
    path is the path for the outfiles
    """
    # path assumes format of "folder/"
    img = nib.load(segmented)
    data = img.get_fdata()
    shape = np.shape(data)
    pxdims = img.header['pixdim'][1:4]
    list_of_probes = load_probes(probefile)
    for i in range(0, len(list_of_probes)):
        list_of_probes[i].export_as_obj(path, shape, pxdims)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment the electrode contacts and their midpoints",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("ct", help="CT file of just the electrodes. default: fmri_mas_thr_SEEG_in_SEEG_rethr.nii.gz")
    parser.add_argument("rep", help="Representative file")
    parser.add_argument("probefile", help="File to save the pickled electrode location data")
    parser.add_argument("probelocationsfile", help="Text file of the probe locations")
    parser.add_argument("segmentedfile", help="Text file of the segmented data")
    parser.add_argument("electrodelabelsfile", help="Text file of the electrode labels")
    parser.add_argument("datafile", help="Data file for plotting")

    args = parser.parse_args()
    segment_electrodes(args.ct, args.rep, args.probefile, args.probelocationsfile, args.segmentedfile, args.electrodelabelsfile, args.datafile)
    # segment_electrodes(f, probefile, probelocationsfile, segmentedfile, electrodelabelsfile)