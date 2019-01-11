import os
import sys
from datetime import datetime
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
            'layerOptions': ['GEOMETRY=AS_XY',
                             'CREATE_CSVT=NO',
                             'SEPARATOR=COMMA',
                             'WRITE_BOM=NO'],
        }

        QgsVectorFileWriter.writeAsVectorFormat(**parameters)


if __name__ == '__main__':

    # Details.
    plantings = ['p6', 'p7']
    what = 'points'
    crs = 'EPSG:3670'

    # Append QGIS to path.
    sys.path.append("/home/will/cotton spatial variability vs yield analysis/"
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

    # Process for given plantings.
    for planting in plantings:

        point_layers = [v for k, v in map_layers.items() if planting in k and what in k]

        # Create a sub-directory.
        directory_path = os.path.join(output_dir, "{0}-{1}-csv-data".format(planting, what))
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        # Run function.
        params = {
            'output_dir': directory_path,
            'layer_list': point_layers,
            'crs': crs,
        }

        save_csv(**params)

        # Write a meta-data file with the details of this extraction for future reference.
        with open(os.path.join(directory_path, "sample_meta_data.txt"), "w") as tester:
            tester.write("""planting: {0}\n
                            what was extracted: {1}\n
                            Samples Generated On: {2}\n
                            """.format(planting, what, formatted_time))

        tester.close()

    # Close project.
    qgs.exitQgis()
