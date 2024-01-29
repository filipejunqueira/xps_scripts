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




