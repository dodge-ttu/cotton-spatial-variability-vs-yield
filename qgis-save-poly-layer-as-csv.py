import os
import re
import sys
from datetime import datetime

import pandas as pd

from qgis.core import QgsProject
from qgis.core import QgsApplication
from qgis.core import QgsVectorFileWriter
from qgis.core import QgsCoordinateReferenceSystem

# Change layer CRS for a list of layers:
def save_csv(layer_list=None, output_dir=None, crs=None):

    for layer in layer_list:

        parameters = {
            'layer': layer,
            'fileName': os.path.join(output_dir, "{0}_process".format(layer.name())),
            'fileEncoding': "utf-8",
            'destCRS': QgsCoordinateReferenceSystem(crs),
            'driverName': 'CSV',
            'layerOptions': [
                'GEOMETRY=AS_WKT',
                'CREATE_CSVT=NO',
                'SEPARATOR=COMMA',
                'WRITE_BOM=NO',
            ],
        }

        QgsVectorFileWriter.writeAsVectorFormat(**parameters)

if __name__ == '__main__':

    # Details.
    plantings = ['p6', 'p7']
    what = 'aoms'
    crs = 'EPSG:3670'

    # Append QGIS to path.
    sys.path.append("/home/will/cotton spatial variability vs yield analysis/"
                    "cott_spat_interp/lib/python3/dist-packages")

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

    # Process for given plantings.
    for planting in plantings:

        # Get date to tag output.
        raw_time = datetime.now()
        formatted_time = datetime.strftime(raw_time, "%Y-%m-%d %H:%M:%S")

        # Filter for desired aoms.
        aom_layers = []
        for (k, v) in map_layers.items():
            if re.findall(r'spatial_{0}_aom_(..)'.format(planting), k) and 'points' not in k:
                aom_layers.append(v)

        # Create a sub-directory.
        directory_path = os.path.join(output_dir, "{0}-{1}-csv-data".format(planting, what))
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        # Run function.
        params = {
            'output_dir': directory_path,
            'layer_list': aom_layers,
            'crs': crs,
        }

        save_csv(**params)

        # Re write csv files to clean them up and organize the vertices.
        csv_files = os.listdir(directory_path)
        csv_files = [i for i in csv_files if i.endswith('.csv')]

        data_frames = []

        for filename in csv_files:
            path = os.path.join(directory_path, filename)
            with open(path, 'r+') as my_file:
                data = my_file.readlines()

                # In this case there is only one polygon per layer, so we only need the line after the header.
                data = data[1]

                # Great numeric re pattern from Stack Overflow.
                # Needed because the original polygon geometry is saved in WKT format that is not pandas-friendly.
                # The vertices are basically saved as a giant oddly formatted tuple with a bunch of parenthetic cruft.
                numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
                rx = re.compile(numeric_const_pattern, re.VERBOSE)

                vertices = rx.findall(data)

                # The data from the High Plains of Texas will always be (7XXXXXX.XXXXXX, 9XXXXXX.XXXXXX)
                area = vertices[-1]
                y = [i for i in vertices if i.startswith('9') and len(i) > 10]
                x = [i for i in vertices if i.startswith('7') and len(i) > 10]

                y = [float(i) for i in y]
                x = [float(i) for i in x]

                # Covert to df and write the data back to a csv now that it's in a common format.
                df = pd.DataFrame({'X':x, 'Y':y})
                df.to_csv(path)

        # Write a meta-data file with the details of this extraction for future reference.
        with open(os.path.join(directory_path, "sample_meta_data.txt"), "w") as tester:
            tester.write("""planting: {0}\n
                            CRS: {1}\n
                            Samples Generated On: {2}\n
                            """.format(planting, crs, formatted_time))

        tester.close()

    # Close project.
    qgs.exitQgis()
