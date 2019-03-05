import os
import re
import pandas as pd


import numpy as np

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics

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

# Force annotated axes on plots for data with large numbers.
mpl.rcParams['axes.formatter.useoffset'] = True
mpl.rcParams['axes.formatter.offset_threshold'] = 1

# # Seedling points only.
# fig, axs = plt.subplots(4,4, figsize=(12,12))
#
# for (ax, (df, aom_number)) in zip(axs.ravel(), data_frames):
#     ax.plot(df.X, df.Y, 'o', markersize = 2)
#     ax.tick_params(axis='both', labelsize=6)
#
#     anchored_text = AnchoredText(aom_number, loc='upper right', prop={'size': 10})
#     ax.add_artist(anchored_text)
#
# fig.suptitle('GPS Location of Seedlings for AOMS in Planting {0}'.format(planting[1]), fontsize=16, fontweight='bold')
# fig.tight_layout(pad=2.0, w_pad=1.0, h_pad=0.0)
# plt.subplots_adjust(top=0.94)
# plt.savefig(os.path.join(visuals_directory, 'points-multiples.png'))
# plt.close()


# # Points with extracted images.
# fig, axs = plt.subplots(4,4, figsize=(12,12))
#
# for (ax, (df, df_aom_number), (img, img_aom_number)) in zip(axs.ravel(), data_frames, marked_images):
#
#     x_min = df.X.min() - 1
#     x_max = df.X.max() + 1
#     y_min = df.Y.min() - 1
#     y_max = df.Y.max() + 1
#
#     extent = (x_min, x_max, y_min, y_max)
#
#     im = ax.imshow(img, origin="upper", extent=extent)
#     ax.plot(df.X, df.Y, "go", markersize=2)
#     ax.tick_params(axis='both', labelsize=6)
#
#     anchored_text = AnchoredText(img_aom_number, loc='upper right', prop={'size': 10})
#     ax.add_artist(anchored_text)
#
# fig.suptitle('GPS Location of Seedlings for AOMS in Planting {0}'.format(planting[1]), fontsize=16, fontweight='bold')
# fig.tight_layout(pad=2.0, w_pad=1.0, h_pad=0.0)
# plt.subplots_adjust(top=0.94)
# plt.savefig(os.path.join(visuals_directory, 'image-points-multiples.png'))
# plt.close()

# # KMeans clustering with scikit-learn.
# fig, axs = plt.subplots(4, 4, figsize=(12, 12))
#
# for (ax, (df, aom_number)) in zip(axs.ravel(), data_frames):
#     X = df.loc[:, 'X'].values
#     Y = df.loc[:, 'Y'].values
#
#     coords = []
#     for (x, y) in zip(X, Y):
#         coords.append((x, y))
#
#     #coords = np.array(coords)
#     pred_y = KMeans(n_clusters=5).fit_predict(coords)
#
#     ax.scatter(X,Y, c=pred_y, s = 4)
#
#     anchored_text = AnchoredText(aom_number, loc='upper right', prop={'size': 10})
#     ax.add_artist(anchored_text)
#
# fig.suptitle('KMeans Clustering Planting {0}'.format(planting[1]), fontsize=16, fontweight='bold')
# fig.tight_layout(pad=2.0, w_pad=1.0, h_pad=0.0)
# plt.subplots_adjust(top=0.94)
# plt.savefig(os.path.join(visuals_directory, 'cluster-multiples.png'))
# plt.close()
#
# # DBSCAN clustering with scikit-learn.
# fig, axs = plt.subplots(4, 4, figsize=(12, 12))
#
# for (ax, (df, aom_number)) in zip(axs.ravel(), data_frames):
#     X = df.loc[:, 'X'].values
#     Y = df.loc[:, 'Y'].values
#
#     coords = []
#     for (x, y) in zip(X, Y):
#         coords.append((x, y))
#
#     #coords = np.array(coords)
#     pred_y = DBSCAN(eps=3, min_samples=10).fit_predict(coords)
#
#     ax.scatter(X,Y, c=pred_y, cmap='RdYlGn', s = 4)
#
#     anchored_text = AnchoredText(aom_number, loc='upper right', prop={'size': 10})
#     ax.add_artist(anchored_text)
#
# fig.suptitle('DBSCAN Clustering Planting {0}'.format(planting[1]), fontsize=16, fontweight='bold')
# fig.tight_layout(pad=2.0, w_pad=1.0, h_pad=0.0)
# plt.subplots_adjust(top=0.94)
# plt.savefig(os.path.join(visuals_directory, 'DBSCAN-cluster-multiples.png'))
# plt.close()


# import numpy as np
#
# from sklearn.cluster import DBSCAN
# from sklearn import metrics
# from sklearn.datasets.samples_generator import make_blobs
# from sklearn.preprocessing import StandardScaler
#
# # Generate paired coordinates for the plant locations.
# X = data_frames[0][0].loc[:, 'X']
# Y = data_frames[0][0].loc[:, 'Y']
#
# seedling_centers = [[i,j] for (i,j) in zip(X,Y)]
#
# X, labels_true = make_blobs(n_samples=750, centers=seedling_centers, cluster_std=0.4,
#                             random_state=0)
#
# X = StandardScaler().fit_transform(X)
#
# # #############################################################################
# # Compute DBSCAN
#
# db = DBSCAN(eps=0.3, min_samples=10).fit(X)
# core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# core_samples_mask[db.core_sample_indices_] = True
# labels = db.labels_
#
# # Number of clusters in labels, ignoring noise if present.
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# n_noise_ = list(labels).count(-1)
#
# print('Estimated number of clusters: %d' % n_clusters_)
# print('Estimated number of noise points: %d' % n_noise_)
# print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
# print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
# print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
# print("Adjusted Rand Index: %0.3f"
#       % metrics.adjusted_rand_score(labels_true, labels))
# print("Adjusted Mutual Information: %0.3f"
#       % metrics.adjusted_mutual_info_score(labels_true, labels))
# print("Silhouette Coefficient: %0.3f"
#       % metrics.silhouette_score(X, labels))
#
# # Black removed and is used for noise instead.
# unique_labels = set(labels)
# colors = [plt.cm.Spectral(each)
#           for each in np.linspace(0, 1, len(unique_labels))]
# for k, col in zip(unique_labels, colors):
#     if k == -1:
#         # Black used for noise.
#         col = [0, 0, 0, 1]
#
#     class_member_mask = (labels == k)
#
#     xy = X[class_member_mask & core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#              markeredgecolor='k', markersize=14)
#
#     xy = X[class_member_mask & ~core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#              markeredgecolor='k', markersize=6)
#
# plt.title('Estimated number of clusters: %d' % n_clusters_)
# plt.show()

# # DBSCAN over UAV estimated yield.
# fig, axs = plt.subplots(4, 4, figsize=(12, 12))
#
# # Get extracted aom extents to scale image data to GPS seedling point data.
# extent_data_path = '/home/will/cotton spatial variability vs yield analysis/' \
#                    '2018-p7-p6-analysis/{0}-extent-csv-data/' \
#                    '{0}-extent-all-aoms.csv'.format(planting)
#
# df_extents = pd.read_csv(extent_data_path)
#
# print(df_extents)
#
# for (ax, (cott_df, cott_aom_number), (df, aom_number)) in zip(axs.ravel(), cott_pix_dfs, data_frames):
#
#     # UAV measured yield data.
#     cott_x = cott_df.loc[:, 'x'].values
#     cott_y = cott_df.loc[:, 'y'].values
#     cott_h = cott_df.loc[:, 'h'].values
#     cott_w = cott_df.loc[:, 'w'].values
#
#     # Modify y-axis data because it's image data and has origin at "upper left".
#     cott_y = cott_h - cott_y
#
#     # GPS seedling location data.
#     X = df.loc[:, 'X'].values
#     Y = df.loc[:, 'Y'].values
#
#     x_min = X.min()
#     y_min = Y.min()
#     x_max = X.max()
#     y_max = Y.max()
#
#     # GPS extent data from csv file.
#     aom_extents = df_extents[(df_extents.layer_id == int(aom_number))]
#
#     x_min_extent = aom_extents.loc[:, 'x_min'].values[0]
#     y_min_extent = aom_extents.loc[:, 'y_min'].values[0]
#     x_max_extent = aom_extents.loc[:, 'x_max'].values[0]
#     y_max_extent = aom_extents.loc[:, 'y_max'].values[0]
#
#     # Find scalers to relate the x,y pixel coords to EPSG:3670 Coordinate Reference System.
#     cott_x_scaler = (x_max_extent - x_min_extent) / (cott_w)
#     cott_y_scaler = (y_max_extent - y_min_extent) / (cott_h)
#
#     # Scale pixel coords.
#     cott_x = x_min_extent + (cott_x * cott_x_scaler)
#     cott_y = y_min_extent + (cott_y * cott_y_scaler)
#
#     coords = []
#     for (x, y) in zip(X, Y):
#         coords.append((x, y))
#
#     pred_y = DBSCAN(eps=3, min_samples=10).fit_predict(coords)
#
#     ax.plot(cott_x, cott_y, 'ro', markersize=0.1)
#     ax.scatter(X, Y, c=pred_y, cmap='RdYlGn', s=4)
#
#     anchored_text = AnchoredText(aom_number, loc='upper right', prop={'size': 10})
#     ax.add_artist(anchored_text)
#
# fig.suptitle('DBSCAN Clusters over UAV Yield Planting {0}'.format(planting[1]), fontsize=16, fontweight='bold')
# fig.tight_layout(pad=2.0, w_pad=1.0, h_pad=0.0)
# plt.subplots_adjust(top=0.94)
# plt.savefig(os.path.join(visuals_directory, 'dbscan-uav-measured-yield-multiples.png'))
# plt.close()

# DBSCAN over UAV estimated yield.
# fig, axs = plt.subplots(4, 4, figsize=(12, 12))

# Get extracted aom extents to scale image data to GPS seedling point data.
extent_data_path = '/home/will/cotton spatial variability vs yield analysis/' \
                   '2018-p7-p6-analysis/{0}-extent-csv-data/' \
                   '{0}-extent-all-aoms.csv'.format(planting)

df_extents = pd.read_csv(extent_data_path)

all_label_class_info_dfs = []

# for (ax, (cott_df, cott_aom_number), (df, aom_number)) in zip(axs.ravel(), cott_pix_dfs, data_frames):
for ((cott_df, cott_aom_number), (df, aom_number)) in zip(cott_pix_dfs, data_frames):

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

    # Seedling
    coords = []
    for (x, y) in zip(X, Y):
        coords.append((x, y))

    # UAS measured end of season yield pixels.
    cott_yld_coords = []
    for (cx, cy) in zip(cott_x, cott_y):
        cott_yld_coords.append((cx,cy))

    pred_y = DBSCAN(eps=3, min_samples=10).fit_predict(coords)

    # slcl: Seedling locations and class labels.
    slcl = [(coordinates, label) for (coordinates, label) in zip(coords, pred_y)]

    # cgsl: Class groups for seedling locations.
    cgsl = []
    for a_label in set(pred_y):
        group = [i for i in slcl if i[1] == a_label]
        cgsl.append(group)

    # Get all of the UAS estimated cotton pixels that fall within a given distance of the DBSCAN.
    pix_yld_by_class = []
    for locs in cgsl:
        for (seedling_loc, label) in locs:
            lon, lat = seedling_loc
            # The CRS is in feet. Isolate yield coords around a plant for per plant yield calculation.
            upper_left = (lon-1, lat+1)
            lower_left = (lon-1, lat-1)
            upper_right = (lon+1, lat-1)
            lower_right = (lon+1, lat+1)

            ulx, uly = upper_left
            llx, lly = lower_left
            urx, ury = upper_right
            lrx, lry = lower_right

            cott_on_plant = []
            for (cx, cy) in cott_yld_coords:
                if (llx < cx < lrx) & (lly < cy < uly):
                    cott_on_plant.append((cx,cy))

            on_plant = len(cott_on_plant)
            print(on_plant)
            pix_yld_by_class.append((label, on_plant))

    df_label_info = pd.DataFrame()
    df_label_info['label'] = [i[0] for i in pix_yld_by_class]
    df_label_info['pix_yield'] = [i[1] for i in pix_yld_by_class]
    df_label_info['planting'] = planting
    df_label_info['aom'] = aom_number
    df_label_info['cott_pix_aom'] = cott_aom_number

    all_label_class_info_dfs.append(df_label_info)

all_df = pd.concat(all_label_class_info_dfs)

all_df.to_csv('/home/will/cotton spatial variability vs yield analysis/2018-p7-p6-analysis/p6-and-p7-per-plant-uas-estimated-yield-by-class-csv-data/{0}-per-plant-yield-by-class-DBSCAN.csv'.format(planting), index=False)




