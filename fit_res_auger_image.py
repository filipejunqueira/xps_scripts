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
from lmfit.models import LorentzianModel, QuadraticModel, ConstantModel, LinearModel, ExponentialModel, GaussianModel, VoigtModel, DoniachModel,


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

def create_composite_model(num_peaks, peaks_type="lorentzian", previous_model=None, previous_params=None):
    # We iniate model with the previous model
    model = previous_model
    params = previous_params
    #determnines the number of peaks already fitted
    initial_peak_count = len(model.components) - 1
    print(f"Initial peak count: {initial_peak_count}")

    # Function to create a specific peak model with a unique prefix
    def create_peak_model(model_type, prefix):
        if model_type == "lorentzian":
            return LorentzianModel(prefix=prefix)
        elif model_type == "gaussian":
            return GaussianModel(prefix=prefix)
        elif model_type == "voigt":
            return VoigtModel(prefix=prefix)
        # Add other models as needed

    # Add peak models to the composite model
    for i in range(num_peaks):
        prefix = f"{peaks_type}{i+initial_peak_count}_"  # Unique prefix for each peak model
        peak_model = create_peak_model(peaks_type, prefix)
        model += peak_model
        print(f"Model updated with peak model with prefix: {prefix}" )
        # Update the parameters object
        params.update(peak_model.make_params())
        print(f"Parameters updated with peak model with prefix: {prefix}" )

    return model, params

def filter_range_to_fit(x, z, range_to_fit, fit_over_range=True):
    #This function takes a range to fit list of tuples and returns the z and x data cropped to that range.
    #Each tuple corresponds of a minimum and a maximum value of the range to fit.
    #Both x and z are 1D arrays of the same length.

    #Initiate the np.arrays
    x_filtered = np.array([])
    z_filtered = np.array([])

    for tuple in range_to_fit:
        x_min = tuple[0]
        x_max = tuple[1]
        #We now crop x in the range x_min and x_max (note that the min and max might not be exactly the same as the values in x so we need to look at the values that are in between or equal)
        #Note that we must append the values to the array and not just assign them to the array. Otherwise we will overwrite the previous values.
        #x[(x>=x_min) & (x<=x_max)]
        x_filtered = np.append(x_filtered,x[(x>=x_min) & (x<=x_max)])
        z_filtered = np.append(z_filtered,z[(x>=x_min) & (x<=x_max)])
    if fit_over_range is True:
        return x_filtered,z_filtered
    else:
        return x, z


def plot_xps_data(x_data,z_data,excitation_energy, metadata_list):
    x = x_data
    y = z_data
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
    #Need to add the scatterplot here but the legend for this should be DATA
    sns.scatterplot(x=x, y=y, s=marker_size, edgecolor="#845EC2", facecolor="None", linewidth=edge_width, alpha=marker_transparency, ax=ax1)    # Plot the fitted model
    # Set titles, labels, etc.

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

def find_peak_center(x_data,z_data,range_guess):
    #For a range guess tuple (initialx, finalx) we find the index of the 3 highest points in z_data within that range
    #We then take the average of the 3 highest points and return the closest value in x_data to that average. GO!
    x_min = range_guess[0]
    x_max = range_guess[1]
    #Croping the data to the range
    x_data_cropped = x_data[(x_data>=x_min) & (x_data<=x_max)]
    z_data_cropped = z_data[(x_data>=x_min) & (x_data<=x_max)]
    #Finding the 3 highest points
    z_data_cropped_sorted = np.sort(z_data_cropped)
    z_data_cropped_sorted = z_data_cropped_sorted[::-1]
    z_data_cropped_sorted = z_data_cropped_sorted[:3]
    #Finding the average of the 3 highest points
    average = np.mean(z_data_cropped_sorted)
    #Finding the closest value in x_data that corresponds to that average in this cropped range
    center_guess = x_data_cropped[index(z_data_cropped,average)]

    return center_guess


def plot_xps_data_with_fitting(x_data,z_data,excitation_energy, metadata_list, model, result,subtract_background=False):
    x = x_data
    y = z_data
    region = metadata_list[0]["region_name"]
    acquisition_mode = metadata_list[0]["acquisition_mode"]
    iterations = metadata_list[0]["number_of_iterations"]
    excitation_energy = excitation_energy
    pass_energy = metadata_list[0]["pass_energy"]
    start_time = metadata_list[0]["start_time"].split("T")
    date = start_time[0]
    time = start_time[1].split(".")[0] if len(start_time) > 1 else ""
    figure_title = f"RES AUGER spec for {file_name} at photon energy: {excitation_energy}eV "

    # Create main plot and residuals subplot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figure_size, gridspec_kw={'height_ratios': [3, 1]})

    # Turn background grey
    #ax1.set_facecolor("#2c2525")
    #ax2.set_facecolor("#2c2525")

    # Plot each peak component with complementary colors
    #peak_colors = ["#F7B9D4", "#FFEF6E", "#B6EF67","#ED89B5","#FAE64D","#A2E546"]  # Complementary colors
    peak_colors = ["#F7B9D4", "#FFEF6E", "#B6EF67","#ED89B5","#FAE64D","#A2E546"]  # Complementary colors


    #Since the background is always set first we loop over all components
    if len(model.components) >= 1:
        for i, component in enumerate(model.components):
            component_y = component.eval(params=result.params, x=x)
            ax1.plot(x, component_y, color=peak_colors[i % len(peak_colors)])

    #plot the background:
    background_y = model.components[0].eval(params=result.params, x=x)

    # Subtract background if flag is set
    if subtract_background is True:
        y_copy_subtracted = y - background_y
    else:
        y_copy_subtracted = y

    # Plot the original data (if it's subtracted then y will be the subtracted data). If not then y_subtracted = y.
    sns.scatterplot(x=x, y=y_copy_subtracted, s=marker_size, edgecolor="#845EC2", facecolor="None", linewidth=edge_width, alpha=marker_transparency, ax=ax1)    # Plot the fitted model

    total_fit = result.eval(x=x)
    sns.lineplot(x=x, y = (total_fit), color="black", linewidth=edge_width * 2, alpha=1, ax=ax1)
    # Plot the original data (if it's subtracted then y will be the subtracted data). If not then y_subtracted = y.
    # Plot residuals
    residuals = y_copy_subtracted - total_fit
    ax2.plot(x, residuals, color="#C11D65")
    ax2.axhline(0, color='black', lw=1)  # Zero line for reference


    # Set titles, labels, etc.
    ax1.set_title(figure_title, fontsize=font_size)
    ax1.set_xlabel("Energy (eV)", fontsize=font_size_label)
    ax1.set_ylabel("Counts [Arb. Units]", fontsize=font_size_label)
    ax2.set_xlabel("Energy (eV)", fontsize=font_size_label)
    ax2.set_ylabel("Residuals", fontsize=font_size_label)
    ax1.tick_params(labelsize=font_size_label)
    ax2.tick_params(labelsize=font_size_label)

    #Add legend for all the peaks including the background
    #Add after the name of the peak, also the value of the peak center and the fwhm
    # Put the legend in the upper left corner.
    legend_labels = []
    for i, component in enumerate(model.components):
        #Check if component is a peak or the background by checking if it contains the word "bkg"
        if "bkg" in component.prefix:
            legend_labels.append(f"Background, a = {np.round(result.params[f'{component.prefix}a'].value,2)}, b = {np.round(result.params[f'{component.prefix}b'].value,2)}, c = {np.round(result.params[f'{component.prefix}c'].value,2)}")
        #Check if component is a peak or the background by checking if it contains the word "lorentzian"
        if "lorentzian" in component.prefix:
            legend_labels.append(f"{component.prefix[:-1]} Center: {np.round(result.params[f'{component.prefix}center'].value,2)}eV, FWHM: {np.round(result.params[f'{component.prefix}fwhm'].value,2)}eV")
    legend_labels.append("Raw Data")
    ax1.legend(legend_labels, loc='upper left', fontsize=font_size_label*0.95)


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
    figure_name = f"{file_name.split('.nxs')[0]}_{region}_pe{excitation_energy}.png"
    plt.savefig(f"{save_path}{figure_name}", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Figure saved as {figure_name}")

user = os.environ['USER']
prefix = f"i09-"
folder_id = "si31574-3"
folder_path = f"/home/{user}/i09/{folder_id}/"
file_list = [254696]

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

print(f"Files to be processed: {file_list}")
error_list = []

#This technically is wrong. It only loops once over the file_list. But it's ok for now.

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



#NOW WE START THE FITTING.
# Example usage

num_peaks = 3  # Adjust the number of peaks as needed
peaks_list = ["lorentzian", "gaussian", "voigt", "doniach"]  # Add other peaks as needed
peaks_type = "lorentzian"
background_type = "quadratic"
#We fisrt create the background model and we fit this background model to the data.
model_background, params_background = create_background(background=background_type)


# Generate some data for testing - grab the z data in which y is equal to 245.3eV#
excitation_energy = 245.45
y_index = np.where(y >= excitation_energy)

# Then pass that to the 2D array z and get a 1D array. First we find the index of the value we want!
# Then we get the 1D array,we get the x array (energies) is all the same for all y values
z_data = z[y_index, :][0][0]
x_data = x

# Manually set initial values and bounds for some parameters
# Make so that the intercept of the background is always the average of the first 10% of points. GO!

#initiate a parameters object

params = model_background.make_params()
#params["bkg_c"].set(value=np.mean(z_data[:int(len(z_data) * 0.1)]), vary=True)
model = model_background


#FITING STARTS HERE #######################################################################
###########################################################################################


# Fit the background model to your data
range_to_fit_background = [(204,207),(214,216)]
x_data_background,z_data_background = filter_range_to_fit(x_data, z_data, range_to_fit_background)


#Background inital values
params["bkg_a"].set(value=-0.17431169, min=-100000, max=4000, vary=False)
params["bkg_b"].set(value=75.6076453, min=-100000, max=4000, vary=False)
params["bkg_c"].set(value=-8163.06762, min=-100000, max=4000, vary=False)
result = model.fit(z_data_background, params, x=x_data_background)
#Now very carefully we fit one peak at a time (manually) by creating a composite model with the previous model with 1 peak.
#We use the previous fit as a starting point for the next one.

model, params = create_composite_model(1,peaks_type=peaks_type, previous_model=model, previous_params=params)
range_guess = (204,210)
peak_range_guess = [range_guess]
peak_center_guess = find_peak_center(x_data,z_data,range_guess)
print(f"Peak center guess: {peak_center_guess}")
params["lorentzian0_center"].set(value=peak_center_guess, vary=False, min=208, max=216)
params["lorentzian0_sigma"].set(value=0.44352257, vary=False, min=0.004, max=40)
params["lorentzian0_amplitude"].set(value=54.7683478, vary=False, min=5, max=80)
x_data_peak,z_data_peak = filter_range_to_fit(x_data, z_data, peak_range_guess, fit_over_range=True)
result = model.fit(z_data_peak, params, x=x_data_peak)

model, params = create_composite_model(1,peaks_type=peaks_type, previous_model=model, previous_params=params)
range_guess = (210,212)
peak_range_guess = [range_guess]
peak_center_guess = find_peak_center(x_data,z_data,range_guess)
print(f"Peak center guess: {peak_center_guess}")
params["lorentzian0_center"].set(value=peak_center_guess, vary=False, min=208, max=216)
params["lorentzian0_sigma"].set(value=0.44352257, vary=False, min=0.004, max=40)
params["lorentzian0_amplitude"].set(value=54.7683478, vary=False, min=5, max=80)
x_data_peak,z_data_peak = filter_range_to_fit(x_data, z_data, peak_range_guess, fit_over_range=False)
result = model.fit(z_data_peak, params, x=x_data_peak)

model, params = create_composite_model(1,peaks_type=peaks_type, previous_model=model, previous_params=params)
range_guess = (212,216)
peak_range_guess = [range_guess]
peak_center_guess = find_peak_center(x_data,z_data,range_guess)
print(f"Peak center guess: {peak_center_guess}")
params["lorentzian0_center"].set(value=peak_center_guess, vary=False, min=208, max=216)
params["lorentzian0_sigma"].set(value=0.44352257, vary=False, min=0.004, max=40)
params["lorentzian0_amplitude"].set(value=54.7683478, vary=False, min=5, max=80)
x_data_peak,z_data_peak = filter_range_to_fit(x_data, z_data, peak_range_guess, fit_over_range=False)
result = model.fit(z_data_peak, params, x=x_data_peak)





# Print the fitting results
print(result.fit_report())
plot_xps_data(x_data,z_data,excitation_energy, metadata_list)

peaks_eval_dic = result.eval_components(x=x_data)
plot_xps_data_with_fitting(x_data,z_data,excitation_energy,metadata_list, model, result, subtract_background=False)

#Now one peak at a time we fit the data to a lorentzian model.
#We use the previous fit as a starting point for the next one.

'''

params["lorentzian1_center"].set(value=211, min=210.4, max=212)
params["lorentzian2_center"].set(value=213, min=212.3, max=213.6)
params["lorentzian3_center"].set(value=208, min=206.3, max=213.6)

params["lorentzian1_fwhm"].set(value=0.1, min=0.05, max=4)
params["lorentzian2_fwhm"].set(value=0.1, min=0.05, max=4)
params["lorentzian3_fwhm"].set(value=0.1, min=0.1, max=2)

params["lorentzian1_amplitude"].set(value=80, min=70, max=100)
params["lorentzian2_amplitude"].set(value=65, min=55, max=75)
params["lorentzian3_amplitude"].set(value=30, min=10, max=40)
'''
