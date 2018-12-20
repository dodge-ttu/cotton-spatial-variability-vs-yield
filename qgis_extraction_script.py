import processing
import os
from datetime import datetime

# Get date to tag output.

raw_time = datetime.now()
formatted_time = datetime.strftime(raw_time, format="%Y-%m-%d %H:%M:%S")

# Output directory and input layer names. In the case of staggared plantinge there will need to be 
# a series of input files. Rather than taking all sample from one layer we must use several layers
# because the date of maturity is staggared.

output_dir = "/home/will/Desktop"
input_layer = "2018-07-18_75_75_20_rainMatrix_odm_orthophoto_modified"

# Return the layer tree and isolate the group of interest to programmatically extract the individual 
# sample spaces.

my_layer_tree = QgsProject.instance().layerTreeRoot()
my_group = my_layer_tree.findGroup("spatial_analysis_p6_aoms")

# Generete a list of items in the group of interest.

layer_list = my_group.children()

# Function to iteratively extract virtual sample spaces.

def make_samples(layer_list, input_layer_name):
    
    for i in layer_list:
        
        parameters = {
        'ALPHA_BAND': False,
        'CROP_TO_CUTLINE': True,
        'DATA_TYPE': 0,
        'INPUT': '{0}'.format(input_layer_name),
        'KEEP_RESOLUTION': True,
        'MASK': '{0}'.format(i.name()),
        'NODATA': None,
        'OPTIONS': '',
        'OUTPUT': os.path.join(output_dir, "{0}.tif".format(i.name())),
       }

        processing.run('gdal:cliprasterbymasklayer', parameters)
        
# Process the eraly sample spaces.

make_samples(layer_list=layer_list, input_layer_name=input_layer)

# Write a meta-data file with the deatials of this extraction for future referecne.

with open(os.path.join(output_dir, "sample_meta_data.txt"), "w") as tester:
    tester.write("""Sample Layer ID early: {0}\n
                    Salple Layer ID late: {1}\n
                    Number of Samples: {2}\n
                    Samples Generated On: {3}\n
                    """.format(input_layer_name_earlier, input_layer_name_later, len(layer_list), formatted_time))

tester.close()

