import os
import sys
from datetime import datetime
from qgis.core import QgsProject
from qgis.core import QgsApplication
import processing
from processing.core.Processing import Processing


# Function to iteratively extract virtual sample spaces.
def make_samples(layer_list=None, output_dir=None, input_layer_name=None):
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


if __name__ == '__main__':

    # Append path to QGIS.
    sys.path.append("/home/will/cotton spatial variability vs yield analysis/"
                    "cott_spat_interp/lib/python3/dist-packages")

    # Get date to tag output.
    raw_time = datetime.now()
    formatted_time = datetime.strftime(raw_time, "%Y-%m-%d %H:%M:%S")

    # Define path to output directory.
    an_output_dir = "/home/will/cotton spatial variability vs yield analysis/" \
        "2018-rain-matrix-p7-p6-extractions-and-data/2018_p7_p6_extractions/p6"

    # Define input layer.
    input_layer = "2018-06-21_75_75_20_rainMatrix_odm_orthophoto_modified"

    # Create a reference to the QGIS application.
    qgs = QgsApplication([], False)

    # Load providers.
    qgs.initQgis()

    # Create a project instance.
    project = QgsProject.instance()

    # Load a project.
    project.read('/home/will/MAHAN MAP 2018/MAHAN MAP 2018.qgs')

    # Initialize processing.
    Processing.initialize()

    # Create bridge between loaded project and canvas
    bridge = project.layerTreeRegistryBridge()

    # Return the layer tree and isolate the group of interest to programmatically extract the individual
    my_layer_tree = QgsProject.instance().layerTreeRoot()
    my_group = my_layer_tree.findGroup("spatial_analysis_p6_aoms")

    # Generate a list of items in the group of interest.
    a_layer_list = my_group.children()

    # Details
    planting = 'p6'
    what = 'aoms'

    # Create an out directory.
    directory_path = os.path.join(an_output_dir, "{0}-{1}-extracted".format(planting, what))
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # Process the eraly sample spaces.
    params = {'output_dir': directory_path,
              'layer_list': a_layer_list,
              'input_layer_name': input_layer}

    make_samples(**params)

    # Write a meta-data file with the details of this extraction for future referecne.
    with open(os.path.join(directory_path, "sample_meta_data.txt"), "w") as tester:
        tester.write("""Sample Layer ID: {0}\n
                        Number of Samples: {1}\n
                        Samples Generated On: {2}\n
                        """.format(input_layer, len(a_layer_list), formatted_time))

    tester.close()

    # Close project.
    qgs.exitQgis()
