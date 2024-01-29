from importnexus import get_nexus_data
from nexusformat.nexus import nxload
import os

user = os.environ['USER']
entry_string = "entry1"
prefix = f"i09-"
entry_string = "entry1"
detector = "ew4000"
id = 254735
session = "si31574-3"
# if plot_all_flag is True, this will be ignored, if True only these will be plotted.
folder_path = f"/home/{user}/i09/{session}/"
file_name = f"{prefix}{id}.nxs"
full_path = f"{folder_path}{file_name}"
file = nxload(full_path)
data_list, metadata_list = get_nexus_data(file,detector=detector)
image_data = data_list[0]['image_data']




print("Hello")