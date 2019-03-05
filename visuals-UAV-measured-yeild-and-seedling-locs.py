import os
import re
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

from scipy.stats import norm
from scipy.spatial import distance
from scipy.interpolate import griddata

from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
# from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import KernelDensity


# Details.
planting = 'p7'

# Define input directory.
input_directory = '/home/will/cotton spatial variability vs yield analysis/' \
                  '2018-p7-p6-analysis/{0}-points-csv-data'.format(planting)

# Define input directory for extracted samples.
image_in_dir = '/home/will/cotton spatial variability vs yield analysis/' \
               '2018-p7-p6-analysis/{0}-aoms-yield-estimates'.format(planting)

# Define output directory for plots.
visuals_directory = '/home/will/cotton spatial variability vs yield analysis/' \
            '2018-p7-p6-analysis/{0}-visuals'.format(planting)

# Create an out directory for visuals.
if not os.path.exists(visuals_directory):
    os.makedirs(visuals_directory)

# List files of interest in the input directory.
point_csv_files = [filename for filename in os.listdir(input_directory) if filename.endswith('.csv')]

# List image files of interest from the image_in_dir.
marked_image_filenames = [filename for filename in os.listdir(image_in_dir) if filename.endswith('.png')]

# Read csv data into pandas.
data_frames = []
for filename in point_csv_files:
    df = pd.read_csv(os.path.join(input_directory, filename))
    aom_number = re.findall(r'\d+', filename)[1]
    data_frames.append((df, aom_number))

# Read marked sample images so they can be used on plots.
marked_images = []
for filename in marked_image_filenames:
    matplotlib_img = mpimg.imread(os.path.join(image_in_dir, filename))
    aom_number = re.findall(r'\d+', filename)[1]
    marked_images.append((matplotlib_img, aom_number))

# Clean up data frames and insert id column.
for (df, aom_number) in data_frames:
    df.loc[:, 'id'] = list(range(len(df)))
    df.drop(columns=['Unnamed: 3'], inplace=True)

# Sort by aom_number.
data_frames = sorted(data_frames, key=lambda x: x[1])
marked_images = sorted(marked_images, key=lambda x: x[1])

# Force annotated axes on plots for data with large numbers.
mpl.rcParams['axes.formatter.useoffset'] = True
mpl.rcParams['axes.formatter.offset_threshold'] = 1

# UAV measured yield analysis.
yield_est_data = '/home/will/cotton spatial variability vs yield analysis/' \
                 '2018-p7-p6-analysis/{0}-aoms-yield-estimates/' \
                 'pix-counts-for-2018-11-15_65_75_35_rainMatrix_modified.csv'.format(planting)

yield_est_data = pd.read_csv(yield_est_data)

data = yield_est_data.loc[:, 'turnout_lb_per_ac_yield']
labels = [int(re.findall(r'\d+', i)[1]) for i in yield_est_data.loc[:, 'ID_tag']]

# Plot UAV measured yield values for a Planting.
fig, ax = plt.subplots(1,1, figsize=(12,12))
ax.bar(range(1,17), data, width=0.8, align='center', edgecolor='k', tick_label=labels)
ax.set_xlabel('Virtual Sample ID', fontdict={'fontsize':16})
ax.set_ylabel('Estimated Lint Yield (lb/ac)', fontdict={'fontsize':16})
ax.tick_params(axis='both', labelsize=14)
ax.set_title('UAV Lint Yield Estimates for Sample Spaces - Planting {0}'.format(planting[1]),
             fontdict={'fontsize':16})

anchored_text = AnchoredText('Turnout: 38%', loc=2, prop={'size':16})
ax.add_artist(anchored_text)

plt.savefig(os.path.join(visuals_directory, 'uav-yield-estimates-planting-{0}.png'.format(planting)))
plt.close()

# Interpolated yield maps.
# Define input directory.
cott_pix_input_directory = '/home/will/cotton spatial variability vs yield analysis/' \
                            '2018-p7-p6-analysis/{0}-white-pixel-locations'.format(planting)

cott_pix_filenames = [i for i in os.listdir(cott_pix_input_directory) if i.endswith('.csv')]

# Read csv data into pandas.
cott_pix_dfs = []
for filename in cott_pix_filenames:
    df = pd.read_csv(os.path.join(cott_pix_input_directory, filename))
    aom_number = re.findall(r'\d+', filename)[1]
    cott_pix_dfs.append((df, aom_number))

# Sort dfs.
cott_pix_dfs = sorted(cott_pix_dfs, key=lambda x: x[1])

# UAV measured yield mask multiples:
fig, axs = plt.subplots(4, 4, figsize=(12, 12))

for (ax, (df, aom_number)) in zip(axs.ravel(), cott_pix_dfs):
    ax.plot(df.x, df.y, 'ro', markersize=.6)

    anchored_text = AnchoredText(aom_number, loc='upper right', prop={'size': 10})
    ax.add_artist(anchored_text)

fig.suptitle('UAV measured yield Planting {0}'.format(planting[1]), fontsize=16, fontweight='bold')
fig.tight_layout(pad=2.0, w_pad=1.0, h_pad=0.0)
plt.subplots_adjust(top=0.94)
plt.savefig(os.path.join(visuals_directory, 'uav-measured-yield-multiples.png'))
plt.close()

# DBSCAN over UAV estimated yield.
fig, axs = plt.subplots(4, 4, figsize=(12, 12))

# Get extracted aom extents to scale image data to GPS seedling point data.
extent_data_path = '/home/will/cotton spatial variability vs yield analysis/' \
                   '2018-p7-p6-analysis/{0}-extent-csv-data/' \
                   '{0}-extent-all-aoms.csv'.format(planting)

df_extents = pd.read_csv(extent_data_path)

print(df_extents)

# Multiples of DBSCAN clustered seedling point locations and the UAV measured seeded cotton pixel locations.
for (ax, (cott_df, cott_aom_number), (df, aom_number)) in zip(axs.ravel(), cott_pix_dfs, data_frames):

    # UAV measured yield data.
    cott_x = cott_df.loc[:, 'x'].values
    cott_y = cott_df.loc[:, 'y'].values
    cott_h = cott_df.loc[:, 'h'].values
    cott_w = cott_df.loc[:, 'w'].values

    # Modify y-axis data because it's image data and has origin at "upper left".
    cott_y = cott_h - cott_y

    # GPS seedling location data.
    X = df.loc[:, 'X'].values
    Y = df.loc[:, 'Y'].values

    x_min = X.min()
    y_min = Y.min()
    x_max = X.max()
    y_max = Y.max()

    # GPS extent data from csv file.
    aom_extents = df_extents[(df_extents.layer_id == int(aom_number))]

    x_min_extent = aom_extents.loc[:, 'x_min'].values[0]
    y_min_extent = aom_extents.loc[:, 'y_min'].values[0]
    x_max_extent = aom_extents.loc[:, 'x_max'].values[0]
    y_max_extent = aom_extents.loc[:, 'y_max'].values[0]

    # Find scalers to relate the x,y pixel coords to EPSG:3670 Coordinate Reference System.
    cott_x_scaler = (x_max_extent - x_min_extent) / (cott_w)
    cott_y_scaler = (y_max_extent - y_min_extent) / (cott_h)

    # Scale pixel coords.
    cott_x = x_min_extent + (cott_x * cott_x_scaler)
    cott_y = y_min_extent + (cott_y * cott_y_scaler)

    coords = []
    for (x, y) in zip(X, Y):
        coords.append((x, y))

    pred_y = DBSCAN(eps=3, min_samples=10).fit_predict(coords)

    ax.plot(cott_x, cott_y, 'ro', markersize=0.1)
    ax.scatter(X, Y, c=pred_y, cmap='tab10', s=4)

    anchored_text = AnchoredText(aom_number, loc='upper right', prop={'size': 10})
    ax.add_artist(anchored_text)

fig.suptitle('DBSCAN Clusters over UAV Yield Planting {0}'.format(planting[1]), fontsize=16, fontweight='bold')
fig.tight_layout(pad=2.0, w_pad=1.0, h_pad=0.0)
plt.subplots_adjust(top=0.94)
plt.savefig(os.path.join(visuals_directory, 'dbscan-uav-measured-yield-multiples.png'))
plt.close()

# Multiples of DBSCAN generated polygons and interpolated UAV measured seeded cotton yield map.
for (ax, (cott_df, cott_aom_number), (df, aom_number)) in zip(axs.ravel(), cott_pix_dfs, data_frames):

    # UAV measured yield data.
    cott_x = cott_df.loc[:, 'x'].values
    cott_y = cott_df.loc[:, 'y'].values
    cott_h = cott_df.loc[:, 'h'].values
    cott_w = cott_df.loc[:, 'w'].values

    # Modify y-axis data because it's image data and has origin at "upper left".
    cott_y = cott_h - cott_y

    # GPS seedling location data.
    X = df.loc[:, 'X'].values
    Y = df.loc[:, 'Y'].values

    x_min = X.min()
    y_min = Y.min()
    x_max = X.max()
    y_max = Y.max()

    # GPS extent data from csv file.
    aom_extents = df_extents[(df_extents.layer_id == int(aom_number))]

    x_min_extent = aom_extents.loc[:, 'x_min'].values[0]
    y_min_extent = aom_extents.loc[:, 'y_min'].values[0]
    x_max_extent = aom_extents.loc[:, 'x_max'].values[0]
    y_max_extent = aom_extents.loc[:, 'y_max'].values[0]

    # Find scalers to relate the x,y pixel coords to EPSG:3670 Coordinate Reference System.
    cott_x_scaler = (x_max_extent - x_min_extent) / (cott_w)
    cott_y_scaler = (y_max_extent - y_min_extent) / (cott_h)

    # Scale pixel coords.
    cott_x = x_min_extent + (cott_x * cott_x_scaler)
    cott_y = y_min_extent + (cott_y * cott_y_scaler)

    coords = []
    for (x, y) in zip(X, Y):
        coords.append((x, y))

    pred_y = DBSCAN(eps=3, min_samples=10).fit_predict(coords)
    a_fit = DBSCAN(eps=3, min_samples=10).fit(coords)

    # Interpolate clusters.
    xx, yy = np.mgrid[x_min_extent:x_max_extent:300j,y_min_extent:y_max_extent:300j]

    plt.scatter(xx,yy)


    grid_z0 = griddata((X,Y), a_fit.labels, (xx,yy))

    plt.imshow(grid_z0, a_fit.labels_, interpolation=None)

    plt.plot(X, Y, '.')








#
#     anchored_text = AnchoredText(aom_number, loc='upper right', prop={'size': 10})
#     ax.add_artist(anchored_text)
#
# fig.suptitle('DBSCAN Clusters over UAV Yield Planting {0}'.format(planting[1]), fontsize=16, fontweight='bold')
# fig.tight_layout(pad=2.0, w_pad=1.0, h_pad=0.0)
# plt.subplots_adjust(top=0.94)
# plt.savefig(os.path.join(visuals_directory, 'dbscan-uav-measured-yield-multiples.png'))
# plt.close()
#
