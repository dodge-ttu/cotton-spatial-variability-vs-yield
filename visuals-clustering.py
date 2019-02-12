import os
import re
import pandas as pd

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


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

# Seedling points only.
fig, axs = plt.subplots(4,4, figsize=(12,12))

for (ax, (df, aom_number)) in zip(axs.ravel(), data_frames):
    ax.plot(df.X, df.Y, 'o', markersize = 2)
    ax.tick_params(axis='both', labelsize=6)

    anchored_text = AnchoredText(aom_number, loc='upper right', prop={'size': 10})
    ax.add_artist(anchored_text)

fig.suptitle('GPS Location of Seedlings for AOMS in Planting {0}'.format(planting[1]), fontsize=16, fontweight='bold')
fig.tight_layout(pad=2.0, w_pad=1.0, h_pad=0.0)
plt.subplots_adjust(top=0.94)
plt.savefig(os.path.join(visuals_directory, 'points-multiples.png'))
plt.close()

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

# KMeans clustering with scikit-learn.
fig, axs = plt.subplots(4, 4, figsize=(12, 12))

for (ax, (df, aom_number)) in zip(axs.ravel(), data_frames):
    X = df.loc[:, 'X'].values
    Y = df.loc[:, 'Y'].values

    coords = []
    for (x, y) in zip(X, Y):
        coords.append((x, y))

    #coords = np.array(coords)
    pred_y = KMeans(n_clusters=5).fit_predict(coords)

    ax.scatter(X,Y, c=pred_y, s = 4)

    anchored_text = AnchoredText(aom_number, loc='upper right', prop={'size': 10})
    ax.add_artist(anchored_text)

fig.suptitle('KMeans Clustering Planting {0}'.format(planting[1]), fontsize=16, fontweight='bold')
fig.tight_layout(pad=2.0, w_pad=1.0, h_pad=0.0)
plt.subplots_adjust(top=0.94)
plt.savefig(os.path.join(visuals_directory, 'cluster-multiples.png'))
plt.close()

# DBSCAN clustering with scikit-learn.
fig, axs = plt.subplots(4, 4, figsize=(12, 12))

for (ax, (df, aom_number)) in zip(axs.ravel(), data_frames):
    X = df.loc[:, 'X'].values
    Y = df.loc[:, 'Y'].values

    coords = []
    for (x, y) in zip(X, Y):
        coords.append((x, y))

    #coords = np.array(coords)
    pred_y = DBSCAN(eps=3, min_samples=10).fit_predict(coords)

    ax.scatter(X,Y, c=pred_y, cmap='RdYlGn', s = 4)

    anchored_text = AnchoredText(aom_number, loc='upper right', prop={'size': 10})
    ax.add_artist(anchored_text)

fig.suptitle('DBSCAN Clustering Planting {0}'.format(planting[1]), fontsize=16, fontweight='bold')
fig.tight_layout(pad=2.0, w_pad=1.0, h_pad=0.0)
plt.subplots_adjust(top=0.94)
plt.savefig(os.path.join(visuals_directory, 'DBSCAN-cluster-multiples.png'))
plt.close()
