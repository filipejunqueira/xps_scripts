import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from importnexus import get_nexus_data
from nexusformat.nexus import nxload


def plot_xas_data(data_list, metadata_list):
    for i, _ in enumerate(data_list):
        print(data_list[i].keys())
        try:
            x = data_list[i]["jenergy"]
        except:
            x = data_list[i]["energies"]
        try:
            y = data_list[i]["xas_spectra"]
        except:
            y = data_list[i]["spectrum_data"]
        try:
            i0 = data_list[i]["i0_xax"]
        except:
            i0 = data_list[i]["i0"]
        try:
            y_corrected = y/i0
        except:
            #Average of all non zero values
            i0_single_value = np.mean(i0[np.nonzero(i0)])
            y_corrected = y/i0_single_value
            print('Could not correct the data')

        #Normalize y and y_corrected from 0 to 100
        y_min, y_max = np.min(y), np.max(y)
        y_corrected_min, y_corrected_max = np.min(y_corrected), np.max(y_corrected)
        y = ((y - y_min) / (y_max - y_min)) * 100
        y_corrected = ((y_corrected - y_corrected_min) / (y_corrected_max - y_corrected_min)) * 100
        
        x_corrected = np.arange(0, len(x), 1)


        region = metadata_list[i]["region_name"]
        acquisition_mode = metadata_list[i]["acquisition_mode"]
        iterations = metadata_list[i]["number_of_iterations"]
        excitation_energy = metadata_list[i]["excitation_energy"]
        pass_energy = metadata_list[i]["pass_energy"]
        start_time = metadata_list[i]["start_time"].split("T")
        date = start_time[0]
        try:
            time = start_time[1].split(".")[0]
        except:
            time = ""

        figure_title = f"XAS Spectrum for {file_name}"

        fig = plt.subplots(figsize=figure_size)
        plt.title(figure_title, fontsize=font_size)
        plt.xlabel("Photon emission energy (eV)", fontsize=font_size_label)
        plt.ylabel("Counts [Arb. Units]", fontsize=font_size_label)
        plt.xticks(fontsize=font_size_label)
        plt.yticks(fontsize=font_size_label)

        #DEBUGING

        print('x_corrected shape:', np.array(x_corrected).shape)
        print('y shape:', np.array(y).shape)
        print('y_corrected shape:', np.array(y_corrected).shape)

        sns.lineplot(x=x_corrected, y=y, color="#FFB347", label="Raw Data")
        sns.scatterplot(x=x_corrected, y=y, s=marker_size, edgecolor="#FFB347", facecolor="None", linewidth=edge_width, alpha=marker_transparency)
        sns.lineplot(x=x_corrected, y=y_corrected, color="#ff0800",label="Normalized Data")
        sns.scatterplot(x=x_corrected, y=y_corrected, s=marker_size, edgecolor="#ff0800", facecolor="None", linewidth=edge_width, alpha=marker_transparency)

        text_string = ("")
        plt.annotate(text_string, xy=(0.65, 0.65), xycoords="axes fraction", fontsize=font_size_text, color="#808080")
        figure_name = f"{file_name.split('.nxs')[0]}_XAS.png"
        plt.savefig(f"{save_path}{figure_name}", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Figure saved as {figure_name}")


def write_list(list, folder_path):
    n_list = ["{}\n".format(i) for i in list]
    with open(f'{folder_path}/files_with_errors.txt', 'w') as fp:
        fp.writelines(n_list)

user = os.environ['USER']

# INPUT: path to the nexus file############################################
prefix = f"i09-"
experiment_id = "si31574-2"
folder_path = f"/home/{user}/i09/{experiment_id}/"

# INPUT: Detector entry ############################################
# This is necessary because the nexus file can have multiple entries

plot_all_flag = True  # if True, all files in the folder will be plotted
file_list = [237140]  # if plot_all_flag is True, this will be ignored, if True only these will be plotted.

# INPUT: Graph  ####################################################
figure_size = (16, 10)  # size
font_size = 36
font_size_label = 20
font_size_text = 16
label_coefficient = 0.8
marker_size = 30
edge_width = 0.8
marker_transparency = 0.15
save_path = f"{folder_path}graphs/"
### Check if save_path exists, if not create it
if not os.path.exists(save_path):
    os.makedirs(save_path)
# Data and Metadata #########################################################
if plot_all_flag is True:
    file_list = []
    print(folder_path)
    list_files = os.listdir(folder_path)
    list_files = sorted(list_files)
    for _fstring in list_files:
        if _fstring.endswith(".nxs"):
            #grabs everything from the first - to the last .nxs (could be -6 or -5)
            print(_fstring)
            try:
                _temp_id = _fstring.strip(".nxs")
                _temp_id = _temp_id.split("-")[1]
                file_list.append(_temp_id)
            except:
                file_list.append(_fstring.strip(".nxs")[-6:])

print(file_list)
print(len(file_list))
print(f"Files to be processed: {file_list}")

error_list = []

for id in file_list:
    file_name = f"{prefix}{id}.nxs"
    full_path = f"{folder_path}{file_name}"

    file = nxload(full_path)
    data_list, metadata_list = get_nexus_data(file)
    plot_xas_data(data_list, metadata_list)
    print(f"I was able to load {file_name}")

    try:
        plot_image_data(data_list, metadata_list)  # print(metadata_list)  # plt.show()
    except:
        error_list.append(file_name)
        print(f"File {file_name} came out with an error")

error_path = f"{folder_path}/graphs"
write_list(error_list, error_path)