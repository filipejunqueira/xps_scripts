import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as colors
import matplotlib as mpl
import os
from importnexus import get_nexus_data
from nexusformat.nexus import nxload
from tempfile import TemporaryFile
import h5py
import scipy
import json
from lmfit.models import LorentzianModel, QuadraticModel, ConstantModel, LinearModel, ExponentialModel, GaussianModel, VoigtModel



def index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
from lmfit import Model, Parameters, fit_report
def crop_z(x,y,z, x_min, x_max, y_min, y_max):
    x_croped = x[index(x,x_min):index(x,x_max)]
    y_croped = y[index(y,y_min):index(y,y_max)]
    z_croped = z[index(y,y_min):index(y,y_max),index(x,x_min):index(x,x_max)]
    return x_croped, y_croped, z_croped

def create_background(background="quadratic"):
    # Create the background model
    if background == "quadratic":
        background_model = QuadraticModel(prefix="bkg_")
    elif background == "linear":
        background_model = LinearModel(prefix="bkg_")
    elif background == "constant":
        background_model = ConstantModel(prefix="bkg_")
    elif background == "exponential":
        background_model = ExponentialModel(prefix="bkg_")

    # Initialize the composite model with the background model
    model = background_model
    params = background_model.make_params()

    return model, params


def plot_xps_data(x_data, z_data, index, metadata_list):
    x = x_data
    # We need to look at the z_data and extract the y_data (energy) for a given value of excitation excitation energy
    # Z

    region = metadata_list[0]["region_name"]
    acquisition_mode = metadata_list[0]["acquisition_mode"]
    iterations = metadata_list[0]["number_of_iterations"]
    excitation_energy = excitation_energy
    pass_energy = metadata_list[0]["pass_energy"]
    start_time = metadata_list[0]["start_time"].split("T")
    date = start_time[0]
    time = start_time[1].split(".")[0] if len(start_time) > 1 else ""
    figure_title = f"RES AUGER spec for {file_name} at photon energy: {excitation_energy}eV "

    fig, ax1 = plt.subplots(1, 1, figsize=figure_size)
    sns.lineplot(x=x, y=y, color="#845EC2", linewidth=edge_width, alpha=1, ax=ax1)
    sns.scatterplot(x=x, y=y, s=marker_size, edgecolor="#845EC2", facecolor="None", linewidth=edge_width, alpha=marker_transparency, ax=ax1)
    ax1.set_title(figure_title, fontsize=font_size)
    ax1.set_xlabel("Energy (eV)", fontsize=font_size_label)
    ax1.set_ylabel("Counts [Arb. Units]", fontsize=font_size_label)
    ax1.tick_params(labelsize=font_size_label)

    # Add text box for metadata
    text_string = (f"Region: {region} \n"
                   f"Excitation Energy: {np.round(excitation_energy, 2)} eV\n"
                   f"Pass Energy: {np.round(pass_energy, 2)} eV\n"
                   f"Acquisition Mode: {acquisition_mode}\n"
                   f"Iterations: {iterations}\n"
                   f"Date: {date}\n"
                   f"Time: {time}")
    ax1.text(0.65, 0.65, text_string, transform=ax1.transAxes, fontsize=font_size_text, color="#808080")

    # Save the plot
    figure_name = f"{file_name.split('.nxs')[0]}_{region}.png"
    plt.savefig(f"{save_path}{figure_name}", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Figure saved as {figure_name}")



user = os.environ['USER']
prefix = f"i09-"
folder_id = "si31574-3"
folder_path = f"/home/{user}/i09/{folder_id}/"
file_id = 254696

figure_size = (18, 12)  # size
font_size = 32
font_size_label = 20
font_size_text = 16
label_coefficient = 0.8
marker_size = 32
edge_width = 0.9
marker_transparency = 0.90
save_path = f"/home/{user}/i09/{folder_id}/graphs/fitted_res_auger/"
less_text = False
correct_for_i0 = True
heat_map_flag = False

### Check if save_path exists, if not create it
if not os.path.exists(save_path):
    os.makedirs(save_path)
# Data and Metadata #########################################################

print(f"File to be processed: {file_id}")
error_list = []

file_name = f"{prefix}{file_id}.nxs"
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
reduce_factor = 3

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
energy_cutoff_max = 4000.0
excitation_energy_min = 0
excitation_energy_max = 40000

# energy_cutoff_min = 204.0
# energy_cutoff_max = 216.0
# excitation_energy_min = 244.80
# excitation_energy_max = 245.70

if energy_cutoff_max < max(energies) or energy_cutoff_min > min(energies):
    energy_cutoff_string = f"Energy_cutoff_{energy_cutoff_min}and{energy_cutoff_max}_eV"
else:
    energy_cutoff_string = f"Energy_cutoff_None"
if excitation_energy_max < max(excitation_energy) or excitation_energy_min > min(excitation_energy):
    excitation_energy_string = f"Excitation_energy_cutoff_{excitation_energy_min}and{excitation_energy_max}_eV"
else:
    excitation_energy_string = f"Excitation_energy_cutoff_None"

if correct_for_i0 is True:
    i0 = data_list[0]["i0"]
for i in range(len(excitation_energy)):
    z[i, :] = z[i, :] / i0[i]

x, y, z = crop_z(energies, excitation_energy, z, energy_cutoff_min, energy_cutoff_max, excitation_energy_min, excitation_energy_max)
# for all z bigger than limit_max set it to limit_max    # for all z bigger than limit_max set it to limit_max
z[z > z_cutoff_max] = z_cutoff_max
# for all z smaller than limit_min set it to limit_min
z[z < z_cutoff_min] = z_cutoff_min

# rescale z to 0-100
z_min, z_max = np.min(z), np.max(z)
z = ((z - z_min) / (z_max - z_min)) * 100

print(f"dimension of x is {np.ndim(x)} and dimession of y is {np.ndim(y)} and dimension of z is {np.ndim(z)}")



#NOW WE START THE FITTING.
# Create the background model

background_model = create_background(background="quadratic")

#Now we fit the background to the first and last background_percentage% of points of the spectrum
background_percentage = 0.1
number_of_points = len(x)
number_of_points_background = int(number_of_points * background_percentage)




