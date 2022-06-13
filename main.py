import sys
from sys import stdin, stdout, stderr
import re
import os
import pathlib
import subprocess
import nibabel as nib
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import json
from threading import *
from os.path import exists
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import get_display_slices
import time


def call_cmd(cmd, pipe):
    proc = subprocess.Popen(cmd.split(), stdout=pipe, stderr=pipe, shell=False)
    return proc.communicate()


def call_docker(path_to_volume, commands):
    try:
        t = time.time()
        print('calling container')
        subprocess.run(["docker", "run", "--gpus", "all",
                        "--cpus=4", "--name=mne_bpy", "-it", "--rm", "-d", "-v",
                       path_to_volume+":/seeg_vol", "mne_bpy"], check=True)
        for cmd in commands:
            subprocess.call(cmd, stdin=stdin, stdout=stdout, stderr=stderr, shell=True)
        subprocess.run(["docker", "stop", "mne_bpy"], check=True)
        print(f'Time elapsed: {time.time() - t}')
    except Exception as e:
        print(e)
        subprocess.run(["docker", "stop", "mne_bpy"], check=True)


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111, projection='3d')
        fig.subplots_adjust(wspace=0)
        super(MplCanvas, self).__init__(fig)


class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'SEEG_GUI'
        self.left = 0
        self.top = 0
        self.width = 900
        self.height = 900

        self.pipe = subprocess.PIPE
        self.ct_file = None
        self.ct_path = None
        self.mri_file = None
        self.mri_path = None
        self.edf_file = None
        self.edf_path = None
        self._data = None
        self.vol_path = '/seeg_vol/SEEG_Viewer/'
        self.output = f'{self.vol_path}Outputs/'
        self.obj = f'{self.output}obj/'
        self.real_path = f'{pathlib.Path(__file__).parents[1]}'
        self.initUI()
        self.name_boxes = []

    def thread(self, funct):
        t1 = Thread(target=funct)
        t1.start()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # Button box contains all the buttons
        button_box = QVBoxLayout(self)
        button_box.setAlignment(Qt.AlignTop)

        # Button to load CT file
        ct_choice = QPushButton("Select the CT file", self)
        ct_choice.clicked.connect(self.getCTText)
        button_box.addWidget(ct_choice, alignment=Qt.AlignLeft)

        # Button to load MRI file
        mri_choice = QPushButton("Select the MRI file", self)
        mri_choice.clicked.connect(self.getMRIText)
        button_box.addWidget(mri_choice, alignment=Qt.AlignLeft)

        # Button to load EDF File
        eeg_choice = QPushButton("Select the EDF file", self)
        eeg_choice.clicked.connect(self.getEEGText)
        button_box.addWidget(eeg_choice, alignment=Qt.AlignLeft)

        ct_idx = self.layout().indexOf(ct_choice)

        self.ct_label = QLabel("CT file: ", self)
        button_box.insertWidget(ct_idx + 1, self.ct_label)

        self.mri_label = QLabel("MRI file: ", self)
        button_box.insertWidget(ct_idx + 3, self.mri_label)

        self.eeg_label = QLabel("EEG file: ", self)
        button_box.insertWidget(ct_idx + 8, self.eeg_label)

        # Begin the preprocessing, segmentation steps prior to electrodes needing labels
        process_button = QPushButton("Process Subjects")
        process_button.clicked.connect(lambda: self.thread(self.process_sub))
        button_box.addWidget(process_button, alignment=Qt.AlignLeft)
        
        self.process_label = QLabel("", self)
        button_box.insertWidget(self.layout().indexOf(process_button)+1, self.process_label)

        # Add graphing/labelling segment of the UI
        plot_button = QPushButton("Plot Segmentation")
        button_box.addWidget(plot_button, alignment=Qt.AlignLeft)
        self.plot_label = QLabel("", self)
        button_box.insertWidget(self.layout().indexOf(plot_button) + 1, self.plot_label)

        self.plot = MplCanvas()
        plotlayout = QHBoxLayout(self)
        plotlayout.addWidget(self.plot)

        self.naming_layout = QFormLayout(self)

        plot_button.clicked.connect(lambda: self.thread(self.plot_segmentation(self.naming_layout)))
        plotlayout.addLayout(self.naming_layout)
        button_box.addLayout(plotlayout)

        # add layout for options
        self.options_layout = QHBoxLayout(self)


        """ blender_mode_label = QLabel("Select visualization mode: ")
        self.options_layout.addWidget(blender_mode_label, alignment=Qt.AlignLeft)
        self.blender_mode_cb = QComboBox()
        self.blender_mode_cb.addItems(['power', 'voltage'])
        self.options_layout.addWidget(self.blender_mode_cb, alignment=Qt.AlignLeft)"""

        annotation_mode_label = QLabel("Select seizure annotation")
        self.options_layout.addWidget(annotation_mode_label, alignment=Qt.AlignLeft)
        self.annotation_cb = QComboBox()
        self.options_layout.addWidget(self.annotation_cb, alignment=Qt.AlignLeft)

        self.lowlabel = QLabel("Bandpass low frequency (Hz)")
        self.options_layout.addWidget(self.lowlabel, alignment=Qt.AlignLeft)
        self.lpf = QLineEdit()
        self.lpf.setValidator(QIntValidator())
        self.lpf.setText('80')
        self.options_layout.addWidget(self.lpf, alignment=Qt.AlignLeft)

        self.highlabel = QLabel("Bandpass high frequency (Hz)")
        self.options_layout.addWidget(self.highlabel, alignment=Qt.AlignLeft)
        self.hpf = QLineEdit()
        self.hpf.setValidator(QIntValidator())
        self.hpf.setText('250')
        self.options_layout.addWidget(self.hpf, alignment=Qt.AlignLeft)


        button_box.addLayout(self.options_layout)

        rename_button = QPushButton("Rename Electrodes")
        rename_button.clicked.connect(lambda: self.thread(self.rename_probes()))
        button_box.addWidget(rename_button, alignment=Qt.AlignLeft)

        process_edf_button = QPushButton("Process EDF and Generate FBX")
        # add connect function and threading
        process_edf_button.clicked.connect(lambda: self.thread(self.finalize_patient()))
        button_box.addWidget(process_edf_button, alignment=Qt.AlignLeft)

        self.show()

    def finalize_patient(self):
        self.process_edf()
        self.create_obj_files()
        self.generate_fbx()

    def process_edf(self):
        print(self.get_eeg_annotations())
        anno = self.get_eeg_annotations().replace(' ', '\ ')
        print(anno)
        commands = [
            f'docker exec mne_bpy bash -c \" python3.9 -u {self.vol_path}process_seizure_data.py '
            f'{self.vol_path}Inputs/{self.edf_file} '
            f'{self.output}probes.p '
            f'{anno} '
            f'{self.lpf.text()} '
            f'{self.hpf.text()}\"'
        ]
        print(commands)
        call_docker(self.real_path, commands)

    def create_obj_files(self):
        # First, create the CSF/GM/WM obj files
        #TODO: If the /obj folder doesn't exist, create it
        command = []
        for i in range(0, 3):
            cmd = f'docker exec mne_bpy bash -c \" python3.9 -u {self.vol_path}convert_to_iso.py ' \
                  f'{self.output}T1_fast_pve_{i}_CT.nii.gz ' \
                  f'{self.output}T1_fast_pve_{i}_CT_iso.nii.gz\"'
            command.append(cmd)
            cmd = f'docker exec mne_bpy bash -c \" python3.9 -u {self.vol_path}niigz_to_obj.py ' \
                  f'{self.output}T1_fast_pve_{i}_CT_iso.nii.gz ' \
                  f'{self.obj}T1_fast_pve_{i}_CT.obj\"'
            command.append(cmd)
            # Now, convert each probe to an obj file
            # export_probes.py export_probes(f, probefile, path)
        cmd = f'docker exec mne_bpy bash -c \" python3.9 -u {self.vol_path}export_probes.py ' \
              f'{self.output}segmented_by_probe.nii.gz ' \
              f'{self.output}probes.p ' \
              f'{self.obj}\"'
        command.append(cmd)
        call_docker(self.real_path, command)

    def generate_fbx(self):
        # print(self.blender_mode_cb.currentText())
        # mode = self.blender_mode_cb.currentText()
        mode = 'power'
        cmd = [f'docker exec mne_bpy bash -c \" python3.10 -u {self.vol_path}fbx_generator.py '
              f'{self.obj} '
              f'{self.output}Filtered.csv ' 
              f'{self.output}Animation '
              f'{mode} \"']
        call_docker(self.real_path, cmd)

    def rename_probes(self):
        names = []
        # Load names of probes
        for i in range(0, len(self.name_boxes)):
            names.append(self.name_boxes[i].text())

        names = ' '.join(names)
        # handle escape characters
        names = names.replace("'", "\\\'")
        print(names)
        cmd = [f'docker exec mne_bpy bash -c \"python3.9 -u {self.vol_path}rename_probes.py '
              f'{self.output}probelocations.txt '
              f'{names} '
              f'{self.output}probes.p\"']
        print(cmd)
        call_docker(self.real_path, cmd)

    def clear_layout(self, layout):
        for i in reversed(range(layout.count())):
            widgetToRemove = layout.itemAt(i).widget()
            # remove it from the layout list
            layout.removeWidget(widgetToRemove)
            # remove it from the gui
            widgetToRemove.setParent(None)

    def process_sub(self):
        if (self.ct_file is None) & (self.mri_file is None):
            self.process_label.setText("Error: CT and MRI files not found")
        elif self.ct_file is None:
            self.process_label.setText("Error: CT file not found")
        elif self.mri_file is None:
            self.process_label.setText("Error: MRI file not found")
        else:
            self.process_label.setText("Processing. This will take some time. Check the terminal for progress reports.")
            commands = [

                f'docker exec mne_bpy bash -c \"python3.9 -u '
                f'{self.vol_path}process_subjects.py '
                f'{self.vol_path}Inputs/{self.ct_file} '
                f'{self.vol_path}Inputs/{self.mri_file}\"',

                f'docker exec mne_bpy bash -c \" python3.9 -u {self.vol_path}segmentation.py '
                f'{self.output}mri_mas_thr_SEEG_in_SEEG_rethr.nii.gz '
                f'{self.vol_path}cube_representative_3D.p '
                f'{self.output}probes.p '
                f'{self.output}probelocations.txt '
                f'{self.output}segmented_by_probe.nii.gz '
                f'{self.output}electrode_labels.txt '
                f'{self.output}datafile.npy\"'
            ]
            cmd = f'docker exec mne_bpy bash -c \" python3.9 -u {self.vol_path}convert_to_iso.py ' \
                  f'{self.output}mri_struct_CT.nii.gz ' \
                  f'{self.output}mri_struct_CT_iso.nii.gz\"'
            commands.append(cmd)
            cmd = f'docker exec mne_bpy bash -c \" python3.9 -u {self.vol_path}convert_to_iso.py ' \
                  f'{self.output}SEEG_CT.nii.gz ' \
                  f'{self.output}SEEG_CT_iso.nii.gz\"'
            commands.append(cmd)
            cmd = f'docker exec mne_bpy bash -c \" python3.9 -u {self.vol_path}convert_to_iso.py ' \
                  f'{self.output}filtered_CT.nii.gz ' \
                  f'{self.output}filtered_CT_iso.nii.gz\"'
            commands.append(cmd)
            call_docker(self.real_path, commands)

            self.process_label.setText("Done Processing!")


    # Get the filenames
    def getCTText(self):
        userInput, _ = QFileDialog.getOpenFileName(self, "Select your CT file", "",
                                                   "All Files (*.*);;NIFTI Files (*.nii.gz)",
                                                   options=QFileDialog.Options())
        self.ct_path = userInput
        if _:  # and userInput != '':
            if str(userInput).strip():
                self.CTniftiCheck(str(userInput))
            else:
                self.ct_label.setText("Error, no file selected")

    def getMRIText(self):
        userInput, _ = QFileDialog.getOpenFileName(self, "Select your MRI file", "",
                                                   "All Files (*.*);;NIFTI Files (*.nii.gz)",
                                                   options=QFileDialog.Options())
        self.mri_path = userInput
        if _:  # and userInput != '':
            if str(userInput).strip():
                self.MRIniftiCheck(str(userInput))
            else:
                self.mri_label.setText("Error, no file selected")

    def getEEGText(self):
        # Hard requirement from mne.io.read_raw_edf that the file extension is lowercase
        userInput, _ = QFileDialog.getOpenFileName(self, "Select your EEG file", "",
                                                   "All Files (*.*);;EDF Files (*.edf)", options=QFileDialog.Options())
        self.edf_path = userInput
        if _:  # and userInput != '':
            if str(userInput).strip():
                self.EEGniftiCheck(str(userInput))
            else:
                self.eeg_label.setText("Error, no file selected")

    # Checking file extensions
    def CTniftiCheck(self, userInput):
        pieces = userInput.split("/")
        file = pieces[len(pieces) - 1]
        print(file)
        if re.search(".nii.gz", file):
            self.ct_file = file
            print("CT file: ", self.ct_file)
            '''ct = {"CT_file": self.ct_file}
            with open('config1.json', 'w') as f:
                json.dump(ct, f)
                print("Save success!")'''
        else:
            file = "Not a NIFTI file, choose again"

        self.ct_label.setText("CT file: " + file)

    def MRIniftiCheck(self, userInput):
        pieces = userInput.split("/")
        file = pieces[len(pieces) - 1]
        print(file)
        if re.search(".nii.gz", file):
            self.mri_file = file
            print("MRI file: ", self.mri_file)
            '''mri = {"MRI_file": self.mri_file}
            with open('config1.json', 'w') as f:
                json.dump(mri, f)
                print("Save success!")

            print("MRI file: ", self.mri_file)'''
        else:
            file = "Not a NIFTI file, choose again"

        self.mri_label.setText("MRI file: " + file)

    def load_EEG_annotations(self):
        commands = [
            f'docker exec mne_bpy bash -c \" python3.9 -u {self.vol_path}get_edf_annotations.py '
            f'{self.vol_path}Inputs/{self.edf_file} '
            f'{self.output}edf_annotations \"'
        ]
        call_docker(self.real_path, commands)
        annotations = np.load(f'{self.real_path}/SEEG_Viewer/Outputs/edf_annotations.npy')
        self.annotation_cb.addItems(annotations)

    def get_eeg_annotations(self):
        return self.annotation_cb.currentText()

    def EEGniftiCheck(self, userInput):
        pieces = userInput.split("/")
        file = pieces[len(pieces) - 1]
        print(file)
        if re.search(".edf", file):
            self.edf_file = file
            print("EEG file: ", self.edf_file)
            '''
            eeg = {"EEG_file": self.eeg_file}
            with open('config1.json', 'w') as f:
                json.dump(eeg, f)
                print("Save success!")'''

            # print("EEG file: ", self.eeg_file)
        else:
            file = "Not an edf file! Choose again"

        self.thread(self.eeg_label.setText("Loading " + file + "..."))
        self.thread(self.load_EEG_annotations())
        self.eeg_label.setText("EDF file: " + file)

    def update_data(self):
        output = f'./Outputs/'
        self._data = np.load(f'{output}datafile.npy')

    def plot_segmentation(self, layout):
        local_output = f'./Outputs/'
        stride = 13
        # Generate slices of the MRI to plot
        img = nib.load(f'{local_output}mri_struct_CT_iso.nii.gz')
        data = img.get_fdata()
        ctr = get_display_slices.find_center_of_image(img)
        pxdims = img.header['pixdim'][1:4]
        colormap = plt.cm.Greys

        n_x, n_y, n_z = np.shape(data)
        dim_size = np.max([n_x, n_y, n_z])
        min_val = data.min()
        max_val = data.max()
        # plot Y data (middle slice) at the back - cor
        """y_cut = data[:, n_y // 2, :]
        X, Z = np.mgrid[0:n_x, 0:n_z]
        Y = 100 * np.ones((n_x, n_z))
        self.plot.axes.plot_surface(X, Y, Z, rstride=stride, cstride=stride, facecolors=colormap((y_cut-min_val)/(max_val-min_val)), shade=False, alpha=0.5, zorder=0)"""
        # plot Z data (middle slice) at the bottom - ax
        z_cut = data[:, :, int(ctr[0])]
        X, Y = np.mgrid[0:n_x, 0:n_y]
        Z = 50 * np.ones((n_x, n_y))
        self.plot.axes.plot_surface(X, Y, Z, rstride=stride, cstride=stride, facecolors=colormap((z_cut-min_val)/(max_val-min_val)), shade=False, alpha=0.5, zorder=0)
        # plot X data middle slice in the middle - sag
        x_cut = data[int(ctr[0]), :, :]
        Y, Z = np.mgrid[0:n_y, 0:n_z]
        X = n_x // 2 * np.ones((n_y, n_z))
        self.plot.axes.plot_surface(X, Y, Z, rstride=stride, cstride=stride,
                                    facecolors=colormap((x_cut - min_val) / (max_val - min_val)), shade=False,
                                    alpha=0.5, zorder=10)

        if self._data is None:
            self.update_data()
        self.clear_layout(layout)
        probe_locs = np.loadtxt(f'{local_output}probelocations.txt')
        x, y, z = self._data.nonzero()
        # basic example
        self.plot.axes.scatter(x, y, z, c='red', zorder=100)

        # MAYBE WORKS?
        # self.axes.plot(coronal)
        # self.axes.plot(saggital)

        for p in range(0, len(probe_locs)):
            self.name_boxes.append(QLineEdit(self))
                # self.axes.text(probe_locs[p,0],probe_locs[p,1],probe_locs[p,2], 'Probe %s' % str(p + 1), size=10 )
            self.plot.axes.text(probe_locs[p, 0], probe_locs[p, 1], probe_locs[p, 2], '%s' % str(p + 1), size=10, zorder=500)
            layout.addRow(f'Probe {p + 1}', self.name_boxes[p])

        self.plot.axes.set_xlim(100, dim_size-100)
        self.plot.axes.set_ylim(100, dim_size-100)
        self.plot.axes.set_zlim(100, dim_size-100)
        self.plot.axes.set_xlabel('x')
        self.plot.axes.set_ylabel('y')
        self.plot.axes.set_zlabel('z')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
