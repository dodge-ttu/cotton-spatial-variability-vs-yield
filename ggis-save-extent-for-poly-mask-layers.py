import re
import os
import sys

import pandas as pd

from datetime import datetime
from qgis.core import QgsProject
from qgis.core import QgsApplication
from qgis.core import QgsVectorLayer
from qgis.core import QgsVectorFileWriter
from qgis.core import QgsCoordinateReferenceSystem


# Change layer CRS for a list of layers:
def save_shapefiles(layer_list=None, output_dir=None, crs=None):

    for layer in layer_list:

        parameters = {
            'layer': layer,
            'fileName': os.path.join(output_dir, "{0}_process".format(layer.name())),
            'fileEncoding': "utf-8",
            'destCRS': QgsCoordinateReferenceSystem(crs),
            'driverName': 'ESRI Shapefile',
            'layerOptions': [
            ],
        }

        QgsVectorFileWriter.writeAsVectorFormat(**parameters)


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
    plantings = ['p6', 'p7']
    what = 'extent'
    crs = 'EPSG:3670'

    # Append QGIS to path.
    sys.path.append("/home/will/cotton spatial variability vs yield analysis/" \
                    "cott_spat_interp/lib/python3/dist-packages")

    # Get date to tag output.
    raw_time = datetime.now()
    formatted_time = datetime.strftime(raw_time, "%Y-%m-%d %H:%M:%S")

    # Define path to output directory.
    output_dir = "/home/will/cotton spatial variability vs yield analysis" \
                 "/2018-p7-p6-analysis"

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

    # Process for given plantings.
    for planting in plantings:

        # Create an out directory.
        directory_path = os.path.join(output_dir, "{0}-{1}-csv-data".format(planting, what))
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        # Create a directory to hold the shapefiles.
        shapefile_path = os.path.join(directory_path, "{0}-{1}-shapefiles".format(planting, what))
        if not os.path.exists(shapefile_path):
            os.makedirs(shapefile_path)

        # Filter for desired aoms.
        aom_layers = []
        for (k,v) in map_layers.items():
            if re.findall(r'spatial_{0}_aom_(..)'.format(planting), k) and 'points' not in k:
                aom_layers.append(v)

        # Save shapefiles with the desired CRS.
        params = {
            'layer_list': aom_layers,
            'output_dir': shapefile_path,
            'crs': crs,
        }

        save_shapefiles(**params)

        # Get the names of the shapefiles we just created.
        shapefile_names = os.listdir(shapefile_path)
        shapefile_names = [i for i in shapefile_names if i.endswith('.shp')]

        # Read in the layers with the new CRS to get extent.
        shapefiles = []
        for a_name in shapefile_names:
            a_path = os.path.join(shapefile_path, a_name)
            a_layer = QgsVectorLayer(a_path, a_name, "ogr")
            shapefiles.append(a_layer)

        # Get extents.
        params = {
            'layer_list': shapefiles,
        }

        extents = get_extent(**params)

        # Process extents into a data_frame.
        x_mins = []
        y_mins = []
        x_maxs = []
        y_maxs = []
        layer_names = []
        layer_ids = []

        for (x_min,x_max,y_min,y_max,layer_name) in extents:
            x_mins.append(x_min)
            y_mins.append(y_min)
            x_maxs.append(x_max)
            y_maxs.append(y_max)
            layer_names.append(layer_name)

            # Great numeric re pattern from Stack Overflow.
            numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
            rx = re.compile(numeric_const_pattern, re.VERBOSE)

            numbers_in_filename = rx.findall(layer_name)
            layer_id = numbers_in_filename[1]

            layer_ids.append(layer_id)

        data = {
            'x_min': x_mins,
            'y_min': y_mins,
            'x_max': x_maxs,
            'y_max': y_maxs,
            'layer_name': layer_names,
            'layer_id': layer_ids,
        }

        df = pd.DataFrame(data)

        extent_df_path = os.path.join(directory_path, "{0}-{1}-all-aoms.csv".format(planting, what))
        df.to_csv(extent_df_path)

        # Write a meta-data file with the details of this extraction for future reference.
        with open(os.path.join(directory_path, "sample_meta_data.txt"), "w") as tester:
            tester.write("""planting: {0}\n
                            what is this data: {1}\n
                            Samples Generated On: {2}\n
                            """.format(planting, what, formatted_time))

    tester.close()

    # Close project.
    qgs.exitQgis()
