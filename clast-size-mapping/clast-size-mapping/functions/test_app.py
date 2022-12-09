import PySimpleGUI as sg
import os.path

#LAYOUT DEFINITION

###################################
#            Main page            #
###################################

layout1 = [[sg.Text('Welcome to the Clast Measuring and Mapping Tool')],
           [sg.Text('This Software was developped by Antoine Soloy',)]]

###################################
#      Terrestrial detection      #
###################################

column1 = [
                        [sg.Text("Input Folder"),
                         sg.In(size=(25, 1), enable_events=True, key="input_folder_lay3"),
                         sg.FolderBrowse(),],
                        [sg.Listbox(select_mode=sg.SELECT_MODE_EXTENDED,
                         values=[], enable_events=True, size=(40, 21), key="input_filelist_lay3")],
                        [sg.Button("Select", key="filelist_selected_lay3")],
                   ]

column2 = [
                        [sg.Text("List of images to be processed")],
                        [sg.Listbox(select_mode='single',
                         values=[], enable_events=True, size=(40, 10), key="output_filelist_lay3")],
                        [sg.Text("Output Folder"),
                         sg.In(size=(25, 1), enable_events=True, key="output_folder_lay3"),
                         sg.FolderBrowse(),],
                        [sg.Text("Ground Sampling Distance: "), sg.In("0.0001", size=(13, 1), enable_events=True, key="GSD"), sg.Text(" m/px")],
                        [sg.Text("Device Mode: "),sg.Radio(text="GPU", group_id='devicemode', key="GPU_enabled", default=True),sg.Radio(text="CPU", group_id='devicemode', key="CPU_enabled", default=False)],
                        [sg.Text("Device Number: "), sg.In("0", size=(23, 1), enable_events=True, key="device_number")],
                        [sg.Text("Display Figures: "),sg.Radio(text="Yes", group_id='display_plot', key="display_plot_enabled", default=False),sg.Radio(text="No", group_id='display_plot', key="display_plot_disabled", default=True)],
                        [sg.Text("Save Figures: "),sg.Radio(text="Yes", group_id='save_plot', key="save_plot_enabled", default=False),sg.Radio(text="No", group_id='save_plot', key="save_plot_disabled", default=True)],
                        [sg.Text("Save Results: "),sg.Radio(text="Yes", group_id='save_results', key="save_results_enabled", default=True),sg.Radio(text="No", group_id='save_results', key="save_results_disabled", default=False)],
                        [sg.Button("Process Data", key="process_data")],
                   ]


layout3 = [[sg.Column(column1),
            sg.VSeperator(),
            sg.Column(column2),]]


#TABS DEFINITION
tabgrp = [[
               sg.TabGroup([[
                   sg.Tab('Main page', layout1),
                   sg.Tab('Terrestrial clast detection', layout3)]], tab_location='centertop', border_width=5)]]


     


#WINDOW DEFINITION
window = sg.Window("Clast Measuring and Mapping Tool",tabgrp)

while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break

    # Folder name was filled in, make a list of files in the folder
    if event == "input_folder_lay3":
        folder = values["input_folder_lay3"]
        try:
            # Get list of files in folder
            file_list1 = os.listdir(folder)
        except:
            file_list1 = []

        fnames = [
            f
            for f in file_list1
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith((".png", ".jpg", "jpeg", "bmp", "tif"))
        ]
        window["input_filelist_lay3"].update(fnames)
    if event == "filelist_selected_lay3":  # A file was chosen from the listbox
        files = values["input_filelist_lay3"]
        try:
            # Get list of files in folder
            window["output_filelist_lay3"].update(files)
        except:
            pass
    if event == "Process_Data":
            filelist = values["output_filelist_lay3"]
            import os
            import numpy as np
            from os import listdir
            from os.path import isfile, join
            import clasts_detection
            import pandas as pd

            # Root directory of the project
            RT_DIR = os.getcwd()

            #Detection & measurement operation
            clasts = clasts_detection.clasts_detect(imlist = filelist, imdirectory = folder, mode = "terrestrial", resolution = values["GSD"], plot = True, saveplot = False, saveresults = False)
            
            
            
            
            
            


#Read  values entered by user
event,values = window.read()
#access all the values and if selected add them to a string
window.close()    