from nexusformat.nexus import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def get_nexus_data(file, entry_string="entry1", detector="ew4000"):
    try:
        start_time = file[entry_string]["start_time"].nxvalue
    except:
        print("Could not get the start time")
        start_time = ""
    try:
        end_time = file[entry_string]['end_time'].nxvalue
    except:
        print("Could not get the end time")
        end_time = ""
    try:
        region_name_list = file[entry_string]["instrument"][detector]["region_list"].nxvalue
        region_name_list = region_name_list.split(",")
    except:
        print("Could not get the region list")
        region_name_list = []
        # Get the region which is the first parameter after entry1 in the Nexus file structure
        temp_region = list(file[entry_string].keys())[0]
        region_name_list.append(temp_region)

    metadata_region_list = []
    data_region_list = []

    for region in region_name_list:
        try:
            attributes = file[entry_string]["instrument"][region]
        except:
            print("Could not get the attributes")
            attributes = ""
        try:
            spectrum_data = file[entry_string]["instrument"][region].spectrum_data.nxvalue  # Y data
        except:
            print("Could not get the spectrum data")
            spectrum_data = ""
        try:
            energies = file[entry_string]["instrument"][region].energies.nxvalue  # X data
        except:
            print("Could not get the energies")
            energies = ""
        try:
            acquisition_mode = file[entry_string]["instrument"][region].acquisition_mode.nxvalue
        except:
            print("Could not get the acquisition mode")
            acquisition_mode = ""
        try:
            angles = file[entry_string]["instrument"][region].angles.nxvalue
        except:
            angles = ""
        try:
            energy_mode = file[entry_string]["instrument"][region].energy_mode.nxvalue
        except:
            energy_mode = ""
        try:
            energy_step = file[entry_string]["instrument"][region].energy_step.nxvalue
        except:
            energy_step = ""
        try:
            excitation_energy = file[entry_string]["instrument"][region].excitation_energy.nxvalue
        except:
            excitation_energy = ""
        try:
            fixed_energy = file[entry_string]["instrument"][region].fixed_energy.nxvalue
        except:
            fixed_energy = ""
        try:
            high_energy = file[entry_string]["instrument"][region].high_energy.nxvalue
        except:
            high_energy = ""
        try:
            image_data = file[entry_string]["instrument"][region].image_data.nxvalue
        except:
            image_data = ""
        try:
            lens_mode = file[entry_string]["instrument"][region].lens_mode.nxvalue
        except:
            lens_mode = ""
        try:
            local_name = file[entry_string]["instrument"][region].local_name.nxvalue
        except:
            local_name = ""
        try:
            low_energy = file[entry_string]["instrument"][region].low_energy.nxvalue
        except:
            low_energy = ""
        try:
            number_of_iterations = file[entry_string]["instrument"][region].number_of_iterations.nxvalue
        except:
            number_of_iterations = ""
        try:
            number_of_slices = file[entry_string]["instrument"][region].number_of_slices.nxvalue
        except:
            number_of_slices = ""
        try:
            pass_energy = file[entry_string]["instrument"][region].pass_energy.nxvalue
        except:
            pass_energy = ""
        try:
            step_time = file[entry_string]["instrument"][region].step_time.nxvalue
        except:
            step_time = ""
        try:
            total_steps = file[entry_string]["instrument"][region].total_steps.nxvalue
        except:
            total_steps = ""
        try:
            total_time = file[entry_string]["instrument"][region].total_time.nxvalue
        except:
            total_time = ""
        try:
            external_io_data = file[entry_string]["scaler2"]["sm5amp8"].nxvalue
        except:
            print("Could not get the external_io_data")
            external_io_data = ""

        #print(f"Metadata found are for file {file_name}: acquisition mode: {acquisition_mode}, angle: {angles}, energy mode: {energy_mode}, energy step: {energy_step}, excitation energy: {excitation_energy}, fixed energy: {fixed_energy}, high energy: {high_energy}, lens mode: {lens_mode}, local name: {local_name}, low energy: {low_energy}, number of iterations: {number_of_iterations}, number of slices: {number_of_slices}, pass energy: {pass_energy}, step time: {step_time}, total steps: {total_steps}, total time: {total_time}")

        metadata_region_list.append({"acquisition_mode": acquisition_mode, "angles": angles, "energy_mode": energy_mode,
                    "energy_step": energy_step, "excitation_energy": excitation_energy, "fixed_energy": fixed_energy,
                    "high_energy": high_energy, "lens_mode": lens_mode, "local_name": local_name,
                    "low_energy": low_energy, "number_of_iterations": number_of_iterations,
                    "number_of_slices": number_of_slices, "pass_energy": pass_energy, "step_time": step_time,
                    "total_steps": total_steps, "total_time": total_time, "start_time": start_time, "end_time": end_time, "energies": energies,
                    "spectrum_data": spectrum_data, "region_name": region, "attributes": attributes})
        data_region_list.append({"energies": energies, "spectrum_data": spectrum_data, "image_data": image_data, "i0": external_io_data})
    return data_region_list, metadata_region_list

if __name__ == "__main__":

    print("DONT RUN ME, I AM A MODULE")



