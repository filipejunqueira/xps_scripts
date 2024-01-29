import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from importnexus import get_nexus_data
from nexusformat.nexus import nxload

def plot_xps_data(data_list, metadata_list):
    for i, _ in enumerate(data_list):
        x = data_list[i]["energies"][0]
        y = data_list[i]["spectrum_data"][0]
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

        figure_title = f"XPS Spectrum for {file_name}"

        fig = plt.subplots(figsize=figure_size)
        plt.title(figure_title, fontsize=font_size)
        plt.xlabel("Energy (eV)", fontsize=font_size_label)
        plt.ylabel("Counts [Arb. Units]", fontsize=font_size_label)
        plt.xticks(fontsize=font_size_label)
        plt.yticks(fontsize=font_size_label)
        sns.lineplot(x=x, y=y, color="#845EC2")
        sns.scatterplot(x=x, y=y, s=marker_size, edgecolor="#845EC2", facecolor="None", linewidth=edge_width, alpha=marker_transparency)
        text_string = (f"Region: {region} \n"
                       f"Excitation Energy: {round(excitation_energy, 2)} eV\n"
                       f"Pass Energy: {round(pass_energy, 2)} eV\n"
                       f"Acquisition Mode: {acquisition_mode}\n"
                       f"Iterations: {iterations}\n"
                       f"Date: {date}\n"
                       f"Time: {time}")
        plt.annotate(text_string, xy=(0.65, 0.65), xycoords="axes fraction", fontsize=font_size_text, color="#808080")
        figure_name = f"{file_name.split('.nxs')[0]}_{region}.png"
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
session_code = "si31574-3"
folder_path = f"/home/{user}/i09/{session_code}/"

# INPUT: Detector entry ############################################
entry_string = "entry1"
detector_string = "ew4000"
# This is necessary because the nexus file can have multiple entries

plot_all_flag = True  # if True, all files in the folder will be plotted
file_list = [254656]  # if plot_all_flag is True, this will be ignored, if True only these will be plotted.

# INPUT: Graph  ####################################################
figure_size = (16, 10)  # size
font_size = 36
font_size_label = 20
font_size_text = 16
label_coefficient = 0.8
marker_size = 30
edge_width = 0.8
marker_transparency = 0.15
save_path = f"/home/{user}/i09/{session_code}/graphs/"
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
    try:
        file = nxload(full_path)
        data_list, metadata_list = get_nexus_data(file, entry_string=entry_string, detector=detector_string)
        plot_xps_data(data_list, metadata_list)
        print(f"File {file_name} saved")
        #plot_image_data(data_list, metadata_list)
        print(metadata_list)
        #plt.show()
    except:
        error_list.append(file_name)
        print(f"File {file_name} came out with an error")

error_path = f"{folder_path}/graphs"
write_list(error_list, error_path)
