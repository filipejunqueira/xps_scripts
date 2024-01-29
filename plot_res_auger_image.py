import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
import seaborn as sns
import os
from importnexus import get_nexus_data
from nexusformat.nexus import nxload
from tempfile import TemporaryFile
import h5py
import scipy
import json
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import dash


def fit_peaks():
    pass

def index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
def crop_z(x,y,z, x_min, x_max, y_min, y_max):
    x_croped = x[index(x,x_min):index(x,x_max)]
    y_croped = y[index(y,y_min):index(y,y_max)]
    z_croped = z[index(y,y_min):index(y,y_max),index(x,x_min):index(x,x_max)]
    return x_croped, y_croped, z_croped


def export_xyz_blender(x,y,z):
    vertices = []
    faces = []

    #Create the vertices by looping through the x and y and z arrays.
    print(len(x))
    print(len(y))
    print(len(z))
    for j  in range(len(y)):
        for i in range(len(y)):
            vertices.append((x[i],y[j],z[i,j]))

    #Given that the vertices are in a grid, we can create the faces by connecting the vertices!
    #Faces are defined by the indices of the vertices that form them.
    #Faces will be rectangles, so we need to connect 4 vertices to form each face.
    for i in range(len(x)-1):
        for j in range(len(y)-1):
            #First triangle
            faces.append((i*len(x)+j,i*len(x)+j+1,(i+1)*len(x)+j))
            #Second triangle
            faces.append(((i+1)*len(x)+j,(i+1)*len(x)+j+1,i*len(x)+j+1))

    return vertices, faces



class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def plot_auger_image(x,y,z, metadata_list, cmap="viridis", midpoint_norm = 0.5, vmin=10, vmax = 90, i0_corrected = True, less_text = False, heat_map_flag = True):
    for i, _ in enumerate(data_list):
        region = metadata_list[i]["region_name"]
        acquisition_mode = metadata_list[i]["acquisition_mode"]
        iterations = metadata_list[i]["number_of_iterations"]
        excitation_energy = metadata_list[i]["excitation_energy"]
        pass_energy = metadata_list[i]["pass_energy"]
        start_time = metadata_list[i]["start_time"].split("T")
        date = start_time[0]
        time = start_time[1].split(".")[0]

        figure_title = f"Res Auger Spectrum for {file_name}"

        fig,ax = plt.subplots(figsize=figure_size)

        c = ax.pcolormesh(x,y,z, cmap=cmap, norm=MidpointNormalize(midpoint=midpoint_norm,vmin=vmin, vmax=vmax), shading='nearest')

        plt.xlabel("Energy [eV]", fontsize=font_size_label)
        plt.ylabel("Excitation Energy [eV]", fontsize=font_size_label)
        plt.xticks(fontsize=font_size_label)
        plt.yticks(fontsize=font_size_label)
        #sns.lineplot(x=x, y=y, color="#845EC2")
        #sns.scatterplot(x=x, y=y, s=marker_size, edgecolor="#845EC2", facecolor="None", linewidth=edge_width, alpha=marker_transparency)
        if i0_corrected is True:
            i0_corrected_string = "I0_Corrected"
        else:
            i0_corrected_string = "I0_NOT_Corrected"

        text_string = (f"Region: {region} \n"
                       f"Pass Energy: {round(pass_energy, 2)} eV\n"
                       f"Acquisition Mode: {acquisition_mode}\n"
                       f"Iterations: {iterations}\n"
                       f"Date: {date}\n"
                       f"Time: {time}\n"
                       f"{i0_corrected_string}")
        if less_text is True:
            less_text_string = f"NO_metadata"
        else:
            plt.annotate(text_string, xy=(0.65, 0.7), xycoords="axes fraction", fontsize=font_size_text, color="#808080", alpha=0.5)
            ax.set_title(figure_title, fontsize=font_size)
            less_text_string = f"WITH_metadata"
        ax.axis([x.min(), x.max(), y.min(), y.max()])
        #add color bar and set the limits of the colorbar manually from vmin to vmax
        if heat_map_flag is True:
            cb = fig.colorbar(c, ax=ax)
            cb.set_label("Normalized Log [Arb. Units]", fontsize = font_size_label)
            heat_map_string = f"WITH_colorbar"
        else:
            heat_map_string = f"NO_colorbar"

        figure_name = f"{file_name.split('.nxs')[0]}_{region}_{i0_corrected_string}_{less_text_string}_{heat_map_string}_{energy_cutoff_string}_{excitation_energy_string}.png"
        plt.savefig(f"{save_path}{figure_name}", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        print(f"Figure saved as {figure_name}")


def write_list(list, folder_path):
    n_list = ["{}\n".format(i) for i in list]
    with open(f'{folder_path}/files_with_errors.txt', 'w') as fp:
        fp.writelines(n_list)


user = os.environ['USER']

# INPUT: path to the nexus file############################################
prefix = f"i09-"
folder_id = "si31574-3"
folder_path = f"/home/{user}/i09/{folder_id}/"

# INPUT: Detector entry ############################################
# This is necessary because the nexus file can have multiple entries

#file_list = [254696]  # if plot_all_flag is True, this will be ignored, if True only these will be plotted.
file_list = [254696]  # if plot_all_flag is True, this will be ignored, if True only these will be plotted.

# INPUT: Graph  ####################################################
figure_size = (16, 10)  # size
font_size = 30
font_size_label = 20
font_size_text = 16
label_coefficient = 0.8
marker_size = 30
edge_width = 0.8
marker_transparency = 0.15
save_path = f"/home/{user}/i09/{folder_id}/graphs/"
less_text = False
correct_for_i0 = True
heat_map_flag = True

### Check if save_path exists, if not create it
if not os.path.exists(save_path):
    os.makedirs(save_path)
# Data and Metadata #########################################################

print(f"Files to be processed: {file_list}")
error_list = []

for id in file_list:
    file_name = f"{prefix}{id}.nxs"
    full_path = f"{folder_path}{file_name}"

    file = nxload(full_path)
    data_list, metadata_list = get_nexus_data(file)
    excitation_energy = metadata_list[0]["excitation_energy"]
    energies = data_list[0]["energies"][0]
    spectrum_data = data_list[0]["spectrum_data"]
    z_raw = spectrum_data
    spectrum_data = np.log10(spectrum_data)
    z = spectrum_data
    # rescale z to 0-100
    z_min, z_max = np.min(z), np.max(z)
    z = ((z - z_min) / (z_max - z_min)) * 100

    # reduce the number of points of energies and excitation energy and z by 70% (do a convolution)
    reduce_factor = 2

    energies_blender = energies[::reduce_factor]
    excitation_energy_blender = excitation_energy[::reduce_factor]
    z_blender = z[::reduce_factor, ::reduce_factor]
    z_cutoff_max = 102
    z_cutoff_min = -2

    color_map = 'RdYlBu_r'
    mid_point_norm = 50
    limit_min = 0
    limit_max = 100

    energy_cutoff_min = 0
    energy_cutoff_max = 22216.0
    excitation_energy_min = 0
    excitation_energy_max = 22245.70

    #energy_cutoff_min = 204.0
    #energy_cutoff_max = 216.0
    #excitation_energy_min = 244.80
    #excitation_energy_max = 245.70

    if energy_cutoff_max < max(energies) or energy_cutoff_min > min(energies):
        energy_cutoff_string = f"Energy_cutoff_{energy_cutoff_min}and{energy_cutoff_max}_eV"
    else:
        energy_cutoff_string = f"Energy_cutoff_None"\

    if excitation_energy_max < max(excitation_energy) or excitation_energy_min > min(excitation_energy):
        excitation_energy_string = f"Excitation_energy_cutoff_{excitation_energy_min}and{excitation_energy_max}_eV"
    else:
        excitation_energy_string = f"Excitation_energy_cutoff_None"


    if correct_for_i0 is True:
        i0 = data_list[0]["i0"]
        for i in range(len(excitation_energy)):
            z[i,:] = z[i,:] / i0[i]

    x, y, z = crop_z(energies,excitation_energy,z,energy_cutoff_min,energy_cutoff_max,excitation_energy_min,excitation_energy_max)
    # for all z bigger than limit_max set it to limit_max    # for all z bigger than limit_max set it to limit_max
    z[z > z_cutoff_max] = z_cutoff_max
    # for all z smaller than limit_min set it to limit_min
    z[z < z_cutoff_min] = z_cutoff_min

    # rescale z to 0-100
    z_min, z_max = np.min(z), np.max(z)
    z = ((z - z_min) / (z_max - z_min)) * 100



    # save x,y,z_raw to a .h5 file metadata_list and the name of the file
    h5f = h5py.File(f'{folder_path}/data_{id}_i0{correct_for_i0}_.h5', 'w')
    h5f.create_dataset('x', data=x)
    h5f.create_dataset('y', data=y)
    h5f.create_dataset('z', data=z)
    h5f.create_dataset('z_raw', data=z_raw)
    h5f.create_dataset('file_name', data=file_name)
    h5f.close()


    plot_auger_image(x, y, z, metadata_list, cmap=color_map, midpoint_norm=mid_point_norm, vmin=limit_min, vmax=limit_max, i0_corrected = correct_for_i0, less_text=less_text, heat_map_flag = heat_map_flag)
    #peaks = fit_peak(x,y)
    #subtract_background(x,y,peaks)
    #plot interactive 3D plot using x,y,z using plotly









    #plot_auger_image(energies_blender, excitation_energy_blender, z_blender, metadata_list, cmap=color_map, midpoint_norm=mid_point_norm)
    #normalize energies_blender and excitation_energy_blender in such a way that maintains the same ratio between them
    size_xy = 10
    energies_blender = energies_blender / np.max(energies_blender) * size_xy
    excitation_energy_blender = excitation_energy_blender / np.max(excitation_energy_blender) * size_xy
    z_blender = z_blender / np.max(z_blender) * size_xy

    print(energies)
    vertices, faces = export_xyz_blender(energies_blender, excitation_energy_blender,z_blender)

    np.save("vertices", vertices)
    np.save("vertices", vertices)
    np.save("faces", faces)


error_path = f"{folder_path}/graphs"
write_list(error_list, error_path)

