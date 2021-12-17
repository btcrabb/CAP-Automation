"""

Viewer tkinter application to display MRI images and assign view classification labels.

Author: Brendan Crabb
Email: brcrabb@eng.ucsd.edu

"""

# import statements
import tkinter as tk
from tkinter import filedialog as fd
from PIL import ImageTk, Image, ImageOps
import os
import pydicom
import pandas as pd
import cv2
import numpy as np


class MainApplication(tk.Frame):

    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        # start application by selecting files
        self.select_files()

    def select_files(self):

        """
        Opens directory of images in viewer.
        """

        # clear old grid objects

        self.path = fd.askdirectory(title='Select Patient Directory',
                               initialdir='./')

        # path to directory containing images
        self.image_list = []
        self.file_list = []
        self.autosave_counter = 0

        # find available series
        try:
            self.path
        except NameError:
            self.select_files()

        self.series_list = []
        for series in os.listdir(self.path):
            self.series_list.append(os.path.join(self.path, series))

        # start on the first series
        self.series_number = 0
        for file in os.listdir(self.series_list[0]):
            if '.dcm' in file:
                self.dcm = pydicom.dcmread(os.path.join(self.series_list[0], file), force=True)

                # windowing
                self.window_center = float(self.dcm[0x0028, 0x1050].value)
                self.window_width = float(self.dcm[0x0028, 0x1051].value)

                self.img = self.dcm.pixel_array
                self.window_image()

                img = ImageTk.PhotoImage(self.pad_and_resize_image())
                self.image_list.append(img)
                self.file_list.append(file)

        # generate current labels list
        self.current_view_label = {}
        self.patient_name = self.path.split('/')[-1]
        try:
            df = pd.read_csv('./output/{}_annotations.csv'.format(self.patient_name), header=[0])
            for index, row in df.iterrows():
                self.current_view_label[row[0]] = row[1]
        except FileNotFoundError:
            print('No previously saved labels...')

        try:
            self.preds_path
        except:
            self.select_predictions()

        # DICOM info
        self.info_var = tk.StringVar()
        self.info_var.set('DICOM Info:')
        self.info_header = tk.Label(self.parent, textvariable=self.info_var, )

        self.frames_var = tk.StringVar()
        self.frames_var.set('Number of frames: {}    '.format(len(self.file_list)))
        self.my_frames = tk.Label(self.parent, textvariable=self.frames_var, anchor='w', justify=tk.LEFT)

        self.desc_var = tk.StringVar()
        self.desc_var.set('Series Description: {}  '.format(self.dcm.SeriesDescription))
        self.my_desc = tk.Label(self.parent, textvariable=self.desc_var, anchor='w', justify=tk.LEFT)

        self.pulse_var = tk.StringVar()
        self.pulse_var.set('Pulse Sequence: {}  '.format(self.dcm.ScanningSequence))
        self.my_pulse = tk.Label(self.parent, textvariable=self.pulse_var, anchor='w', justify=tk.LEFT)

        # generate predicted labels list
        self.pred_view_labels = {}
        self.confidence = {}

        try:
            df = pd.read_csv(self.preds_path, header=[0])
            for index, row in df.iterrows():
                self.pred_view_labels[row['Series ID']] = row['Predicted View']
                self.confidence[row['Series ID']] = row['Confidence']
        except FileNotFoundError:
            print('Could not find predictions file')

        self.series_id = self.dcm.SeriesInstanceUID
        self.my_img = self.image_list[0]
        self.my_label = tk.Label(image=self.my_img)

        self.file_var = tk.StringVar()
        self.file_var.set(self.series_id)
        self.my_file = tk.Label(self.parent, textvariable=self.file_var)

        self.ms_delay = int(1000 / len(self.image_list))
        self.cancel_id = None

        self.frames_var = tk.StringVar()
        self.frames_var.set('Number of frames: {}    '.format(len(self.file_list)))
        self.my_frames = tk.Label(self.parent, textvariable=self.frames_var, anchor='w', justify=tk.LEFT)

        self.desc_var = tk.StringVar()
        self.desc_var.set('Series Description: {}  '.format(self.dcm.SeriesDescription))
        self.my_desc = tk.Label(self.parent, textvariable=self.desc_var, anchor='w', justify=tk.LEFT)

        self.pulse_var = tk.StringVar()
        self.pulse_var.set('Pulse Sequence: {}     '.format(self.dcm.ScanningSequence))
        self.my_pulse = tk.Label(self.parent, textvariable=self.pulse_var, anchor='w', justify=tk.LEFT)

        self.pred_view = tk.StringVar()
        if self.series_id in self.pred_view_labels.keys():
            self.pred_view.set(
                'Predicted View Label: {} ({})        '.format(self.pred_view_labels[self.series_id], self.confidence[
                    self.series_id]))  # predicted_views[file_list[0]].upper()))
        else:
            self.pred_view.set('Predicted View Label: {}     '.format('None'))
        self.my_pred = tk.Label(self.parent, textvariable=self.pred_view, fg='red')

        self.cur_view = tk.StringVar()
        if self.file_list[0] in self.current_view_label.keys():
            self.cur_view.set(
                'Accepted View Label: {}       '.format(self.current_view_label[self.file_list[0]].upper()))
        else:
            self.cur_view.set('Accepted View Label: {}       '.format('None'))
        self.my_view = tk.Label(self.parent, textvariable=self.cur_view, anchor='w', justify=tk.LEFT, fg='red')

        self.main()

    def window_image(self):

        """
        Opens an image with correct window level and width
        :return: none
        """

        img_min = self.window_center - self.window_width // 2
        img_max = self.window_center + self.window_width // 2
        self.img[self.img < img_min] = img_min
        self.img[self.img > img_max] = img_max

        # normalize
        self.img = self.img / np.max(self.img)
        self.img = self.img * 255
        self.img.astype(np.uint8)

    def pad_and_resize_image(self):

        """
        Pads an image to square and resizes to the desired size.
        :return: new_im (array) - resized/padded image
        """

        # resize so max size is 256
        desired_size = 356
        old_size = np.array(self.img).shape  # old_size[0] is in (width, height) format
        ratio = desired_size / np.max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        self.img = Image.fromarray(cv2.resize(np.array(self.img), new_size, cv2.INTER_CUBIC))

        # pad to square
        delta_w = desired_size - new_size[0]
        delta_h = desired_size - new_size[1]
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        new_im = ImageOps.expand(self.img, padding)

        return new_im

    def select_predictions(self):

        """
        Loads csv file containing model view predictions.
        :return:
        """

        self.preds_path = fd.askopenfilename(title='Select Predictions File',
                                        initialdir='./')

        # generate predicted labels list
        self.pred_view_labels = {}
        self.confidence = {}

        try:
            df = pd.read_csv(self.preds_path, header=[0])
            for index, row in df.iterrows():
                self.pred_view_labels[row['Series ID']] = row['Predicted View']
                self.confidence[row['Series ID']] = row['Confidence']
        except FileNotFoundError:
            print('Could not find predictions file')

        self.series_id = self.dcm.SeriesInstanceUID
        self.my_img = self.image_list[0]
        self.my_label = tk.Label(image=self.my_img)

        self.file_var = tk.StringVar()
        self.file_var.set(self.series_id)
        self.my_file = tk.Label(self.parent, textvariable=self.file_var)

        self.ms_delay = int(1000 / len(self.image_list))
        self.cancel_id = None

        #self.series_number += 1

        # DICOM info
        self.info_var = tk.StringVar()
        self.info_var.set('DICOM Info:')
        self.info_header = tk.Label(self.parent, textvariable=self.info_var, )

        self.frames_var = tk.StringVar()
        self.frames_var.set('Number of frames: {}    '.format(len(self.file_list)))
        self.my_frames = tk.Label(self.parent, textvariable=self.frames_var, anchor='w', justify=tk.LEFT)

        self.desc_var = tk.StringVar()
        self.desc_var.set('Series Description: {}  '.format(self.dcm.SeriesDescription))
        self.my_desc = tk.Label(self.parent, textvariable=self.desc_var, anchor='w', justify=tk.LEFT)

        self.pulse_var = tk.StringVar()
        self.pulse_var.set('Pulse Sequence: {}  '.format(self.dcm.ScanningSequence))
        self.my_pulse = tk.Label(self.parent, textvariable=self.pulse_var, anchor='w', justify=tk.LEFT)

        self.frames_var = tk.StringVar()
        self.frames_var.set('Number of frames: {}    '.format(len(self.file_list)))
        self.my_frames = tk.Label(self.parent, textvariable=self.frames_var, anchor='w', justify=tk.LEFT)

        self.desc_var = tk.StringVar()
        self.desc_var.set('Series Description: {}  '.format(self.dcm.SeriesDescription))
        self.my_desc = tk.Label(self.parent, textvariable=self.desc_var, anchor='w', justify=tk.LEFT)

        self.pulse_var = tk.StringVar()
        self.pulse_var.set('Pulse Sequence: {}     '.format(self.dcm.ScanningSequence))
        self.my_pulse = tk.Label(self.parent, textvariable=self.pulse_var, anchor='w', justify=tk.LEFT)

        self.pred_view = tk.StringVar()
        if self.series_id in self.pred_view_labels.keys():
            self.pred_view.set(
                'Predicted View Label: {} ({})        '.format(self.pred_view_labels[self.series_id], self.confidence[
                    self.series_id]))  # predicted_views[file_list[0]].upper()))
        else:
            self.pred_view.set('Predicted View Label: {}     '.format('None'))
        self.my_pred = tk.Label(self.parent, textvariable=self.pred_view, fg='red')

        self.cur_view = tk.StringVar()
        if self.file_list[0] in self.current_view_label.keys():
            self.cur_view.set(
                'Accepted View Label: {}       '.format(self.current_view_label[self.file_list[0]].upper()))
        else:
            self.cur_view.set('Accepted View Label: {}       '.format('None'))
        self.my_view = tk.Label(self.parent, textvariable=self.cur_view, anchor='w', justify=tk.LEFT, fg='red')

        self.main()

    def back(self):

        """
        Moves main viewer back one series.
        :return: none
        """

        self.series_number -= 1

        self.image_list = []
        self.file_list = []

        # start on the first series
        for file in os.listdir(self.series_list[self.series_number]):
            if '.dcm' in file:
                self.dcm = pydicom.dcmread(os.path.join(self.series_list[self.series_number], file), force=True)

                # windowing
                self.window_center = float(self.dcm[0x0028, 0x1050].value)
                self.window_width = float(self.dcm[0x0028, 0x1051].value)

                self.img = self.dcm.pixel_array
                self.window_image()

                img = ImageTk.PhotoImage(self.pad_and_resize_image())
                self.image_list.append(img)
                self.file_list.append(file)

        self.series_id = self.dcm.SeriesInstanceUID
        self.my_img = self.image_list[0]
        self.my_label = tk.Label(image=self.my_img)

        # DICOM info
        self.info_var = tk.StringVar()
        self.info_var.set('DICOM Info:')
        self.info_header = tk.Label(self.parent, textvariable=self.info_var, )

        self.frames_var = tk.StringVar()
        self.frames_var.set('Number of frames: {}    '.format(len(self.file_list)))
        self.my_frames = tk.Label(self.parent, textvariable=self.frames_var, anchor='w', justify=tk.LEFT)

        self.desc_var = tk.StringVar()
        self.desc_var.set('Series Description: {}  '.format(self.dcm.SeriesDescription))
        self.my_desc = tk.Label(self.parent, textvariable=self.desc_var, anchor='w', justify=tk.LEFT)

        self.pulse_var = tk.StringVar()
        self.pulse_var.set('Pulse Sequence: {}  '.format(self.dcm.ScanningSequence))
        self.my_pulse = tk.Label(self.parent, textvariable=self.pulse_var, anchor='w', justify=tk.LEFT)

        self.file_var.set(self.series_id)
        self.my_file = tk.Label(self.parent, textvariable=self.file_var)

        self.ms_delay = int(1000 / len(self.image_list))
        self.cancel_id = None

        self.pred_view = tk.StringVar()
        if self.series_id in self.pred_view_labels.keys():
            self.pred_view.set(
                'Predicted View Label: {} ({})        '.format(self.pred_view_labels[self.series_id], self.confidence[
                    self.series_id]))  # predicted_views[file_list[0]].upper()))
        else:
            self.pred_view.set('Predicted View Label: {}     '.format('None'))

        self.my_pred = tk.Label(self.parent, textvariable=self.pred_view, fg='red')
        self.cur_view = tk.StringVar()
        if self.file_list[0] in self.current_view_label.keys():
            self.cur_view.set(
                'Accepted View Label: {}       '.format(self.current_view_label[self.file_list[0]].upper()))
        else:
            self.cur_view.set('Accepted View Label: {}       '.format('None'))
        self.my_view = tk.Label(self.parent, textvariable=self.cur_view, anchor='w', justify=tk.LEFT, fg='red')

        self.main()

    def forward(self):

        """
        Moves main viewer forward one series.
        :return: none
        """

        self.image_list = []
        self.file_list = []

        for file in os.listdir(self.series_list[self.series_number + 1]):
            if '.dcm' in file:
                self.dcm = pydicom.dcmread(os.path.join(self.series_list[self.series_number + 1], file), force=True)

                # windowing
                self.window_center = float(self.dcm[0x0028, 0x1050].value)
                self.window_width = float(self.dcm[0x0028, 0x1051].value)

                self.img = self.dcm.pixel_array
                self.window_image()

                img = ImageTk.PhotoImage(self.pad_and_resize_image())
                self.image_list.append(img)
                self.file_list.append(file)

        self.series_id = self.dcm.SeriesInstanceUID
        self.my_img = self.image_list[0]
        self.my_label = tk.Label(image=self.my_img)

        self.file_var.set(self.series_id)
        self.my_file = tk.Label(self.parent, textvariable=self.file_var)

        self.ms_delay = int(1000 / len(self.image_list))
        self.cancel_id = None

        self.series_number += 1

        self.frames_var = tk.StringVar()
        self.frames_var.set('Number of frames: {}    '.format(len(self.file_list)))
        self.my_frames = tk.Label(self.parent, textvariable=self.frames_var, anchor='w', justify=tk.LEFT)

        self.desc_var = tk.StringVar()
        self.desc_var.set('Series Description: {}  '.format(self.dcm.SeriesDescription))
        self.my_desc = tk.Label(self.parent, textvariable=self.desc_var, anchor='w', justify=tk.LEFT)

        self.pulse_var = tk.StringVar()
        self.pulse_var.set('Pulse Sequence: {}     '.format(self.dcm.ScanningSequence))
        self.my_pulse = tk.Label(self.parent, textvariable=self.pulse_var, anchor='w', justify=tk.LEFT)

        self.pred_view = tk.StringVar()
        if self.series_id in self.pred_view_labels.keys():
            self.pred_view.set('Predicted View Label: {} ({})        '.format(self.pred_view_labels[self.series_id], self.confidence[
                self.series_id]))  # predicted_views[file_list[0]].upper()))
        else:
            self.pred_view.set('Predicted View Label: {}     '.format('None'))
        self.my_pred = tk.Label(self.parent, textvariable=self.pred_view, fg='red')

        self.cur_view = tk.StringVar()
        if self.file_list[0] in self.current_view_label.keys():
            self.cur_view.set('Accepted View Label: {}       '.format(self.current_view_label[self.file_list[0]].upper()))
        else:
            self.cur_view.set('Accepted View Label: {}       '.format('None'))
        self.my_view = tk.Label(self.parent, textvariable=self.cur_view, anchor='w', justify=tk.LEFT, fg='red')

        self.main()

    # function to handle gif images
    def update_image(self):

        """
        Updates the image when a cine is playing.
        :return: none
        """

        self.my_label.configure(image=self.image_list[self.frame_num])
        self.frame_num = (self.frame_num+1) % len(self.image_list)
        self.cancel_id = self.parent.after(self.ms_delay, self.update_image)

    def cancel_animation(self):

        """
        Stops a cine that is playing.
        :return: none
        """

        if self.cancel_id is not None:  # Animation started?
            self.parent.after_cancel(self.cancel_id)
            self.cancel_id = None
        self.main()

    def enable_animation(self):

        """
        Starts a cine playing.
        :return: none.
        """

        self.frame_num = 0
        if self.cancel_id is None:  # Animation not started?
            self.ms_delay = 1000 // len(self.image_list)  # Show all frames in 1000 ms.
            self.cancel_id = self.parent.after(
                self.ms_delay, self.update_image)

        self.enable = tk.Button(self.parent, text="play", command=lambda: self.enable_animation(), state=tk.DISABLED)
        self.disable = tk.Button(self.parent, text="stop", command=lambda: self.cancel_animation())

        self.button_forward = tk.Button(root, text=">>", command=lambda: self.forward(), state=tk.DISABLED)
        self.button_back = tk.Button(root, text="<<", command=lambda: self.back(), state=tk.DISABLED)
        self.button_exit = tk.Button(root, text="EXIT PROGRAM", command=self.parent.quit)
        self.button_save = tk.Button(root, text="Save Labels", command=lambda: self.save_output())

        self.button_4ch = tk.Button(root, text="   4CH   ", command=lambda: self.pick_label('4ch'), state=tk.DISABLED)
        self.button_3ch = tk.Button(root, text="   3CH   ", command=lambda: self.pick_label('3ch'), state=tk.DISABLED)
        self.button_lvot = tk.Button(root, text="  LVOT  ", command=lambda: self.pick_label('lvot'), state=tk.DISABLED)
        self.button_rvot = tk.Button(root, text="  RVOT  ", command=lambda: self.pick_label('rvot'), state=tk.DISABLED)
        self.button_2chl = tk.Button(root, text=" 2CH LT ", command=lambda: self.pick_label('2ch lt'), state=tk.DISABLED)
        self.button_2chr = tk.Button(root, text=" 2CH RT ", command=lambda: self.pick_label('2ch rt'), state=tk.DISABLED)
        self.button_sa = tk.Button(root, text="    SA    ", command=lambda: self.pick_label('sa'), state=tk.DISABLED)
        self.button_other = tk.Button(root, text=" OTHER ", command=lambda: self.pick_label('other'), state=tk.DISABLED)
        self.button_accept = tk.Button(root, text="Accept Prediction",
                                       command=lambda: self.pick_label(self.pred_view_labels[self.series_id]), state=tk.DISABLED)
        self.preds_button = tk.Button(root, text='Select Predictions File', command=lambda: self.select_predictions())

        # button positions
        self.button_back.grid(row=17, column=4)
        self.button_exit.grid(row=17, column=5, columnspan=3)
        self.button_forward.grid(row=17, column=8)
        self.button_4ch.grid(row=1, column=10)
        self.button_3ch.grid(row=2, column=10)
        self.button_lvot.grid(row=3, column=10)
        self.button_rvot.grid(row=4, column=10)
        self.button_2chl.grid(row=5, column=10)
        self.button_2chr.grid(row=6, column=10)
        self.button_sa.grid(row=7, column=10)
        self.button_other.grid(row=8, column=10)
        self.button_accept.grid(row=17, column=0, sticky='w')
        self.button_save.grid(row=1, column=2, sticky='w', padx=12)
        self.enable.grid(row=4, column=3)
        self.disable.grid(row=5, column=3)

    def save_output(self):

        """
        Saves label annotations to a patient-specific csv file.
        :return: none.
        """

        output = []
        for key in self.current_view_label.keys():
            output.append([key, self.current_view_label[key]])
        df = pd.DataFrame(output, columns=['Series ID', 'Label'])
        df.to_csv('./output/{}_annotations.csv'.format(self.patient_name), header=['File', 'Label'], index=False)
        print('Done!')

    def pick_label(self, key):

        """
        Selects a label to be saved and moves the viewer forward one series.
        :param key: (str) the assigned label.
        :return: none.
        """

        file = self.file_list[0]
        if file in self.current_view_label.keys():
            if key.upper() == self.current_view_label[file].upper():
                pass
                print("Labels already correct")
            else:
                for fname in self.file_list:
                    self.current_view_label[fname] = key.upper()
                print('Updating labels!')
        else:
            for fname in self.file_list:
                self.current_view_label[fname] = key.upper()
            print('Adding new labels..')

        self.autosave_counter += 1
        if self.autosave_counter % 10 == 0:
            self.save_output()
            print('Autosaving')

        self.image_list = []
        self.file_list = []

        for file in os.listdir(self.series_list[self.series_number + 1]):
            if '.dcm' in file:
                self.dcm = pydicom.dcmread(os.path.join(self.series_list[self.series_number + 1], file), force=True)

                # windowing
                self.window_center = float(self.dcm[0x0028, 0x1050].value)
                self.window_width = float(self.dcm[0x0028, 0x1051].value)

                self.img = self.dcm.pixel_array
                self.window_image()

                img = ImageTk.PhotoImage(self.pad_and_resize_image())
                self.image_list.append(img)
                self.file_list.append(file)

        self.series_id = self.dcm.SeriesInstanceUID
        self.my_img = self.image_list[0]
        self.my_label = tk.Label(image=self.my_img)

        self.file_var = tk.StringVar()
        self.file_var.set(self.series_id)
        self.my_file = tk.Label(self.parent, textvariable=self.file_var)

        self.ms_delay = int(1000 / len(self.image_list))
        self.cancel_id = None
        self.series_number += 1

        self.frames_var = tk.StringVar()
        self.frames_var.set('Number of frames: {}    '.format(len(self.file_list)))
        self.my_frames = tk.Label(self.parent, textvariable=self.frames_var, anchor='w', justify=tk.LEFT)

        self.desc_var =tk.StringVar()
        self.desc_var.set('Series Description: {}  '.format(self.dcm.SeriesDescription))
        self.my_desc = tk.Label(self.parent, textvariable=self.desc_var, anchor='w', justify=tk.LEFT)

        self.pulse_var = tk.StringVar()
        self.pulse_var.set('Pulse Sequence: {}  '.format(self.dcm.ScanningSequence))
        self.my_pulse = tk.Label(self.parent, textvariable=self.pulse_var, anchor='w', justify=tk.LEFT)

        self.pred_view = tk.StringVar()
        if self.series_id in self.pred_view_labels.keys():
            self.pred_view.set('Predicted View Label: {} ({})        '.format(self.pred_view_labels[self.series_id], self.confidence[
                self.series_id]))  # predicted_views[file_list[0]].upper()))
        else:
            self.pred_view.set('Predicted View Label: {}     '.format('None'))

        self.my_pred = tk.Label(self.parent, textvariable=self.pred_view, fg='red')
        self.cur_view = tk.StringVar()
        if self.file_list[0] in self.current_view_label.keys():
            self.cur_view.set('Accepted View Label: {}       '.format(self.current_view_label[self.file_list[0]].upper()))
        else:
            self.cur_view.set('Accepted View Label: {}       '.format('None'))
        self.my_view = tk.Label(self.parent, textvariable=self.cur_view, anchor='w', justify=tk.LEFT, fg='red')

        self.main()

    def main(self):

        """
        Renders the main GUI
        :return: none.
        """

        # main GUI
        # set title
        self.parent.title('CAP Automation Image Viewer')

        # buttons
        self.button_save = tk.Button(self.parent, text="Save Labels", command=lambda: self.save_output())

        if self.series_number == 0:
            self.button_back = tk.Button(self.parent, text="<<", state=tk.DISABLED)
        else:
            self.button_back = tk.Button(self.parent, text="<<", command=self.back)

        self.button_exit = tk.Button(self.parent, text="EXIT PROGRAM", command=self.parent.quit)

        if self.series_number == len(self.series_list) - 1:
            self.button_forward = tk.Button(self.parent, text=">>", command=self.forward, state=tk.DISABLED)
        else:
            self.button_forward = tk.Button(self.parent, text=">>", command=self.forward)

        self.button_4ch = tk.Button(self.parent, text="   4CH   ", command=lambda: self.pick_label('4ch'))
        self.button_3ch = tk.Button(self.parent, text="   3CH   ", command=lambda: self.pick_label('3ch'))
        self.button_lvot = tk.Button(self.parent, text="  LVOT  ", command=lambda: self.pick_label('lvot'))
        self.button_rvot = tk.Button(self.parent, text="  RVOT  ", command=lambda: self.pick_label('rvot'))
        self.button_2chl = tk.Button(self.parent, text=" 2CH LT ", command=lambda: self.pick_label('2ch lt'))
        self.button_2chr = tk.Button(self.parent, text=" 2CH RT ", command=lambda: self.pick_label('2ch rt'))
        self.button_sa = tk.Button(self.parent, text="    SA    ", command=lambda: self.pick_label('sa'))
        self.button_other = tk.Button(self.parent, text=" OTHER ", command=lambda: self.pick_label('other'))
        self.button_accept = tk.Button(self.parent, text="Accept Prediction", command=lambda: self.pick_label(
            self.pred_view_labels[self.series_id]))
        self.enable = tk.Button(self.parent, text="play", command=lambda: self.enable_animation())
        self.disable = tk.Button(self.parent, text="stop", command=lambda: self.cancel_animation)
        self.open_button = tk.Button(self.parent, text='Select Directory', command=self.select_files)
        self.preds_button = tk.Button(self.parent, text='Select Predictions File', command=self.select_predictions)

        # button positions
        self.button_back.grid(row=17, column=4)
        self.button_exit.grid(row=17, column=5, columnspan=3)
        self.button_forward.grid(row=17, column=8)
        self.button_4ch.grid(row=1, column=10)
        self.button_3ch.grid(row=2, column=10)
        self.button_lvot.grid(row=3, column=10)
        self.button_rvot.grid(row=4, column=10)
        self.button_2chl.grid(row=5, column=10)
        self.button_2chr.grid(row=6, column=10)
        self.button_sa.grid(row=7, column=10)
        self.button_other.grid(row=8, column=10)
        self.button_accept.grid(row=17, column=0, sticky='w')
        self.button_save.grid(row=1, column=2, sticky='w', padx=12)
        self.enable.grid(row=4, column=3)
        self.disable.grid(row=5, column=3)

        # other positions
        self.my_label.grid(row=1, column=4, rowspan=14, columnspan=5)
        self.my_file.grid(row=0, column=0, columnspan=10, sticky='w')
        self.info_header.grid(row=2, column=0, columnspan=3, sticky='w')
        self.my_frames.grid(row=3, column=0, columnspan=3, sticky='w')
        self.my_desc.grid(row=4, column=0, columnspan=3, sticky='w')
        self.my_pulse.grid(row=5, column=0, columnspan=3, sticky='w')
        self.my_view.grid(row=9, column=0, columnspan=3, sticky='w')
        self.my_pred.grid(row=8, column=0, columnspan=3, sticky='w')
        self.open_button.grid(row=1, column=0, sticky='w')
        self.preds_button.grid(row=1, column=1, sticky='w')

        # grid sizes / formatting
        self.parent.grid_columnconfigure(3, minsize=50)

if __name__ == "__main__":
    root = tk.Tk()
    MainApplication(root)
    root.mainloop()
