#XPS SCRIPTS 

## Disclaimer
This is a very quick, partily AI generated documentation. 
Eventually me "Filipe" will try to make this more "comprehensive"
There are a few important files/scripts in this repository. 

They are: 

- `importnexus.py`
- `plot_xps.py`
- `plot_xas.py` 
- `plot_res_auger_image.py`
- `fit_res_auger_image.py` 

# importnexus.py (Nexus Data Reader Module) 

## Introduction
This module is designed to facilitate the extraction and handling of data from NeXus file formats. It leverages the `nexusformat` library to access and read NeXus files, primarily focusing on extracting various metadata and data regions related to spectroscopy and imaging instruments.

## Installation
To use this module, you need to have Python installed on your system along with several dependencies:
- `nexusformat`
- `numpy`
- `matplotlib`
- `seaborn`

You can install these dependencies using pip (but you should really be using conda as much as possible):
```bash
pip install nexusformat numpy matplotlib seaborn
```

## Usage

This module is intended to be used as a library and not as a standalone script. Here's a basic guide on how to use it:

### Import the Module: 

First, import the module into your Python script where you need to handle NeXus files.
```python
from importnexus import get_nexus_data
from nexusformat.nexus import nxload
```

You are not going to run this file normally. The files that you are going to run (plot_xps.py, plot_xas etc) are going to call this
module though. 

### Run this (You are not really running this anywhere, I'm just showing you how to do it)

```python
print(file_list)
print(len(file_list))
print(f"Files to be processed: {file_list}")

error_list = []

for id in file_list:
    file_name = f"{prefix}{id}.nxs"
    full_path = f"{folder_path}{file_name}"
    try:
        file = nxload(full_path)
        data_list, metadata_list = get_nexus_data(file)
        whatever_function_you_want_to_run(data_list,metadata_list)
    except:
        error_list.append(file_name)
        print(f"File {file_name} came out with an error")

error_path = f"{folder_path}/graphs"
write_list(error_list, error_path)

```
Normally you will have a file list that contains all the files you want to process. 
And you apply whatever_function_you_want_to_run() in this loop.
I think most scripts work somewhat this way.

Lets have a look at the first script, plot_xps.py 

# XPS Data Plotter (plot_xps.py)

## Overview
This script is designed for processing and visualizing X-ray Photoelectron Spectroscopy (XPS) data. It reads data from Nexus files, generates plots for each data entry, and saves these plots as images. The script also logs any files that encounter errors during processing.

## Dependencies
- numpy
- matplotlib
- seaborn
- os
- importnexus (custom module)
- nexusformat.nexus

## Functions

### `plot_xps_data(data_list, metadata_list)`
Generates and saves plots for XPS data.

#### Parameters:
- `data_list`: List of dictionaries containing XPS data.
- `metadata_list`: List of dictionaries containing metadata for each XPS data entry.

#### Description:
For each entry in the `data_list` and `metadata_list`, this function creates a line and scatter plot using matplotlib and seaborn. The plot includes relevant metadata as annotations and is saved as a PNG file.

### `write_list(list, folder_path)`
Writes a list of strings to a file.

#### Parameters:
- `list`: A list of strings to be written to the file.
- `folder_path`: Path of the folder where the file will be saved.

#### Description:
Saves the content of `list` to a file named `files_with_errors.txt` in the specified `folder_path`.

## Usage

### Setting Inputs
1. Define the path to the Nexus file:
```python
   prefix = "i09-"
   session_code = "si31574-3"
   folder_path = f"/home/{user}/i09/{session_code}/"
```

2. Specify the detector entry if necessary (normally it should be entry1)

3. Set the plotting flag and file list:
   
```python  
   plot_all_flag = True  # Set to False to plot only specified files
   file_list = [254656]  # List of file IDs to plot if plot_all_flag is False
```

4. Configure plot stuff (style, fonts etc):
```python  
figure_size = (16, 10)
font_size = 36
font_size_label = 20
font_size_text = 16
marker_size = 30
edge_width = 0.8
marker_transparency = 0.15
save_path = f"/home/{user}/i09/{session_code}/graphs/"
```

## Runing the script

To run the script simply do python xps_plot.py
