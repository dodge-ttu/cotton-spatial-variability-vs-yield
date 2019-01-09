import re
import os
import sys

import pandas as pd

from datetime import datetime
from qgis.core import QgsProject
from qgis.core import QgsApplication
from qgis.core import QgsVectorLayer
from qgis.core import QgsCoordinateReferenceSystem


# Get extent of layers layers in a QGIS project instance:
def get_extent(layer_list=None):

    extent_tuples_and_names = []
    for layer in layer_list:
        extent_rectangle = layer.extent()
        layer_name = layer.name()

        x_min = extent_rectangle.xMinimum()
        y_min = extent_rectangle.yMinimum()
        x_max = extent_rectangle.xMaximum()
        y_max = extent_rectangle.yMaximum()

        extent_tuples_and_names.append((x_min,x_max,y_min,y_max,layer_name))

    return extent_tuples_and_names

if __name__ == '__main__':

    # Details.
    planting = 'p6'
    what = 'aom'
    
    # Append QGIS to path.
    sys.path.append("/home/will/cotton spatial variability vs yield analysis/" \
                    "cott_spat_interp/lib/python3/dist-packages")

    # Get date to tag output.
    raw_time = datetime.now()
    formatted_time = datetime.strftime(raw_time, "%Y-%m-%d %H:%M:%S")

    # Define path to output directory.
    output_dir = "/home/will/cotton spatial variability vs yield analysis/" \
                 "/2018-p7-p6-analysis/"

    # Create a reference to the QGIS application.
    qgs = QgsApplication([], False)

    # Load providers.
    qgs.initQgis()

    # Create a project instance.
    project = QgsProject.instance()

    # Load a project.
    project.read('/home/will/MAHAN MAP 2018/MAHAN MAP 2018.qgs')

    # Get map layers.
    map_layers = project.mapLayers()

    # Read layers into project.


    # Filter for desired aoms.
    aom_layers = []
    for (k,v) in map_layers.items():
        if re.findall(r'spatial_{0}_aom_(..)'.format(planting), k) and 'points' not in k:
            aom_layers.append(v)





    # Get extent.
    params = {
        'layer_list': aom_layers,
    }

    extents = get_extent(**params)

    # Make df of extents.
    data = {
        "x_min": [i[0] for i in extents],
        "y_min": [i[2] for i in extents],
        "x_max": [i[1] for i in extents],
        "y_max": [i[3] for i in extents],
        "aomid": [i[4] for i in extents],
    }

    df = pd.DataFrame(data)

    print(df)

    # # Write a meta-data file with the details of this extraction for future reference.
    # with open(os.path.join(directory_path, "sample_meta_data.txt"), "w") as tester:
    #     tester.write("""planting: {0}\n
    #                     what was extracted: {1}\n
    #                     Samples Generated On: {2}\n
    #                     """.format(planting, what, formatted_time))
    #
    # tester.close()

    # Close project.
    qgs.exitQgis()
