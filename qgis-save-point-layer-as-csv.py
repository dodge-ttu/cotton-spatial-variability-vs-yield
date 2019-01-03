# Use: >>> processing.algorithmHelp('qgis:fieldcalculator')
# to display documentation about each algorithm.

import os
import sys
from datetime import datetime
from qgis.core import QgsProject
from qgis.core import QgsApplication
from qgis.core import QgsVectorFileWriter
from qgis.core import QgsCoordinateReferenceSystem


# Change layer CRS for a list of layers:
def change_projections(layer_list=None, output_dir=None):

    for layer in layer_list:

        parameters = {
            'layer': layer,
            'fileName': os.path.join(output_dir, "{0}_process".format(layer.name())),
            'fileEncoding': "utf-8",
            'destCRS': QgsCoordinateReferenceSystem('EPSG:4326'),
            'driverName': 'CSV',
            'layerOptions': ['GEOMETRY=AS_XY',
                             'CREATE_CSVT=NO',
                             'SEPARATOR=COMMA',
                             'WRITE_BOM=NO'],
        }

        # Use ** to unpack the dictionary as arguments for the function
        QgsVectorFileWriter.writeAsVectorFormat(**parameters)


if __name__ == '__main__':

    # Append path to QGIS.
    sys.path.append("/home/will/cotton spatial variability vs yield analysis/"
                    "cott_spat_interp/lib/python3/dist-packages")

    # Get date to tag output.
    raw_time = datetime.now()
    formatted_time = datetime.strftime(raw_time, "%Y-%m-%d %H:%M:%S")

    # Define path to output directory.
    output_dir = "/home/will/cotton spatial variability vs yield analysis/" \
                 "2018-rain-matrix-p7-p6-extractions-and-data/2018_p7_p6_extractions/p6"

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

    # Details.
    planting = 'p6'
    what = 'points'

    a_layer_of_points = [v for k, v in map_layers.items() if planting in k and what in k]

    # Create a directory to hold partially processed copies of shapefile layers as
    # processing runs are called on the layers.

    directory_path = os.path.join(output_dir, "{0}-{1}-csv-data".format(planting, what))
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # Run function
    params = {'output_dir': directory_path,
              'layer_list': a_layer_of_points}

    change_projections(**params)

    # Write a meta-data file with the details of this extraction for future referecne.
    with open(os.path.join(directory_path, "sample_meta_data.txt"), "w") as tester:
        tester.write("""planting: {0}\n
                        what was extracted: {1}\n
                        Samples Generated On: {2}\n
                        """.format(planting, what, formatted_time))

    tester.close()

    # Close project.
    qgs.exitQgis()
