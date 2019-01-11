import os
import re
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

from scipy.spatial import distance
from sklearn.neighbors import KernelDensity


# Details.
planting = 'p6'

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

# Organize point coordinates to be used in scipy.spatial.
# Calculate distance within aoms.
within_aoms_distance = []
for (df, aom_number) in data_frames:
    x = df.loc[:, 'X'].values
    y = df.loc[:, 'Y'].values

    coords = []
    for (x,y) in zip(x,y):
        coords.append((x,y))

    coords = np.array(coords)

    distance_matrix = distance.cdist(coords, coords, 'euclidean')
    within_aoms_distance.append((distance_matrix, aom_number))

# >>>len(distance_matrix) == len(coords) == len(distance_matrix[0])
# True

# Calculate mean distances for all.
mean_distances_all_aoms = []
for (distance_matrix, aom_number) in within_aoms_distance:
    mean_distances = []
    for dist_one2all in distance_matrix:
        # Mean of each plant's distance to all other plants.
        a_mean = np.mean(dist_one2all)
        # Convert State Plane NAD83 unit of survey feet to meters.
        a_mean_meters = a_mean * 0.3048
        mean_distances.append(a_mean_meters)

    mean_distances_all_aoms.append(mean_distances)

# Merge mean distances back to each point.
for (mean_distances, (df, aom_number)) in zip(mean_distances_all_aoms, data_frames):
    df.loc[:, 'mean_distance'] = mean_distances

# Sort by aom_number.
data_frames = sorted(data_frames, key=lambda x: x[1])
marked_images = sorted(marked_images, key=lambda x: x[1])

# Force annotated axes on plots for data with large numbers.
mpl.rcParams['axes.formatter.useoffset'] = True
mpl.rcParams['axes.formatter.offset_threshold'] = 1

# Make multiples of histograms.
fig, axs = plt.subplots(4, 4, figsize=(12, 12), sharey=True, sharex=True)

for (ax, (df, aom_number)) in zip(axs.ravel(), data_frames):
    data = df.loc[:, 'mean_distance'].values
    bins = np.linspace(0, 10, 21)
    ax.hist(data, color='#0FC25B', edgecolor='k', bins=bins, rwidth=0.70)

    ax.set_xlim(0, 10)

    anchored_text = AnchoredText(aom_number, loc='upper right', prop={'size': 10})
    ax.add_artist(anchored_text)

fig.suptitle('Distribution Based on Mean Distance Planting {0}'.format(planting[1]), fontsize=16, fontweight='bold')
fig.text(0.5, 0.01, 'Binned Mean Euclidean Distance (m)', ha='center', fontsize=12, fontweight='bold')
fig.text(0.01, 0.5, 'Bin Count', va='center', rotation='vertical', fontsize=12, fontweight='bold')
fig.tight_layout(pad=2.0, w_pad=1.0, h_pad=0.0)
plt.subplots_adjust(top=0.94)
plt.savefig(os.path.join(visuals_directory, 'distance-hist-multiples.png'))
plt.close()

# Gaussian probability density curves.
fig, axs = plt.subplots(4, 4, figsize=(12, 12), sharey=True, sharex=True)

for (ax, (df, aom_number)) in zip(axs.ravel(), data_frames):
    data = df.loc[:, 'mean_distance'].values
    bins = np.linspace(0, 10, 21)
    ax.hist(data, color='#0FC25B', edgecolor='k', bins=bins, rwidth=0.70, density=True, alpha=0.5)

    # Reshape data for scikit-learn.
    data = data[:, np.newaxis]
    bins = bins[:, np.newaxis]

    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(data)
    log_dens = kde.score_samples(bins)

    ax.plot(bins[:, 0], np.exp(log_dens), 'r-')

fig.suptitle('Probability Density Curve by Planting {0}'.format(planting[1]), fontsize=16, fontweight='bold')
fig.text(0.5, 0.01, 'Binned Mean Euclidean Distance (m)', ha='center', fontsize=12, fontweight='bold')
fig.text(0.01, 0.5, 'Density', va='center', rotation='vertical', fontsize=12, fontweight='bold')
fig.tight_layout(pad=2.0, w_pad=1.0, h_pad=0.0)
plt.subplots_adjust(top=0.94)
plt.savefig(os.path.join(visuals_directory, 'probability-density-curve-multiples.png'))
plt.close()

# Fancy probability density function plots.
for (df, aom_number) in (data_frames):
    fig, ax = plt.subplots(1,1, figsize=(12, 12))
    data = df.loc[:, 'mean_distance'].values
    bins = np.linspace(0, 10, 21)
    ax.hist(data, color='#0FC25B', edgecolor='k', bins=bins, rwidth=0.80, density=True, alpha=0.5)

    # Reshape data for scikit-learn.
    data = data[:, np.newaxis]
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(data)

    # Resize bins for probability density function.
    bins = np.linspace(0, 10, 101)[:, np.newaxis]
    log_dens = kde.score_samples(bins)

    ax.plot(bins[:, 0], np.exp(log_dens), 'r-')
    ax.plot(data[:, 0], -0.008 - 0.04 * np.random.random(data.shape[0]), 'kd')

    #
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.06, 0.90)
    ax.set_title('Probability Density Curve: AOM {0} Planting {1}'.format(aom_number, planting[1]),
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Binned Mean Euclidean Distance (m)', ha='center', fontsize=12, fontweight='bold',
                  labelpad=20)
    ax.set_ylabel('Density', va='center', rotation='vertical', fontsize=12, fontweight='bold',
                  labelpad=20)
    # fig.tight_layout()
    plt.savefig(os.path.join(visuals_directory, 'probability-density-curve-aom-{0}-planting-{0}.png'.format(aom_number)))
    plt.close()
