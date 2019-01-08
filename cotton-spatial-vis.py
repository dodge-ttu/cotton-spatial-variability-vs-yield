import os
import re
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

# Force annotated axes for large numbers
mpl.rcParams['axes.formatter.useoffset'] = True
mpl.rcParams['axes.formatter.offset_threshold'] = 1

from scipy.stats import norm
from scipy.spatial import distance
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
# from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import KernelDensity


def clean_poly_eq(coefficients, dec_dig):
    n = len(coefficients)
    degs = [i for i in range(n)]
    coefficients = [round(i, dec_dig) for i in coefficients]
    coefficients.reverse()
    pieces = []
    for (cof, deg) in zip(coefficients, degs):
        if deg == 0:
            a = ' + {0}'.format(cof)
            pieces.append(a)
        else:
            a = '{0} x^{1} '.format(cof, deg)
            pieces.append(a)

    equation = 'y = ' + ''.join(pieces[::-1])

    return equation


def get_poly_hat(x_values, y_values, poly_degree):
    coeffs = np.polyfit(x_values, y_values, poly_degree)
    poly_eqn = np.poly1d(coeffs)

    y_bar = np.sum(y_values) / len(y_values)
    ssreg = np.sum((poly_eqn(x_values) - y_bar) ** 2)
    sstot = np.sum((y_values - y_bar) ** 2)
    r_square = ssreg / sstot

    return (coeffs, poly_eqn, r_square)

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

# Create some visuals.
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

# Points with extracted images.
fig, axs = plt.subplots(4,4, figsize=(12,12))

for (ax, (df, df_aom_number), (img, img_aom_number)) in zip(axs.ravel(), data_frames, marked_images):
    ax.imshow(img)
    # ax.plot(df.X, df.Y, 'o', markersize = 2)
    ax.tick_params(axis='both', labelsize=6)

    anchored_text = AnchoredText(img_aom_number, loc='upper right', prop={'size': 10})
    ax.add_artist(anchored_text)

fig.suptitle('GPS Location of Seedlings for AOMS in Planting {0}'.format(planting[1]), fontsize=16, fontweight='bold')
fig.tight_layout(pad=2.0, w_pad=1.0, h_pad=0.0)
plt.subplots_adjust(top=0.94)
plt.savefig(os.path.join(visuals_directory, 'image-points-multiples.png'))
plt.close()

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

    ax.scatter(X,Y, c=pred_y, s = 4)

    anchored_text = AnchoredText(aom_number, loc='upper right', prop={'size': 10})
    ax.add_artist(anchored_text)

fig.suptitle('DBSCAN Clustering Planting {0}'.format(planting[1]), fontsize=16, fontweight='bold')
fig.tight_layout(pad=2.0, w_pad=1.0, h_pad=0.0)
plt.subplots_adjust(top=0.94)
plt.savefig(os.path.join(visuals_directory, 'DBSCAN-cluster-multiples.png'))
plt.close()

#region Silhouette analysis
#
# Silhouette analysis code from scikit-learn documentation here:
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
#
# Excerpt:
#
# ===============================================================================
# Selecting the number of clusters with silhouette analysis on KMeans clustering
# ===============================================================================
#
# Silhouette analysis can be used to study the separation distance between the
# resulting clusters. The silhouette plot displays a measure of how close each
# point in one cluster is to points in the neighboring clusters and thus provides
# a way to assess parameters like number of clusters visually. This measure has a
# range of [-1, 1].
#
# Silhouette coefficients (as these values are referred to as) near +1 indicate
# that the sample is far away from the neighboring clusters. A value of 0
# indicates that the sample is on or very close to the decision boundary between
# two neighboring clusters and negative values indicate that those samples might
# have been assigned to the wrong cluster.
#
# In this example the silhouette analysis is used to choose an optimal value for
# ``n_clusters``. The silhouette plot shows that the ``n_clusters`` value of 3, 5
# and 6 are a bad pick for the given data due to the presence of clusters with
# below average silhouette scores and also due to wide fluctuations in the size
# of the silhouette plots. Silhouette analysis is more ambivalent in deciding
# between 2 and 4.
#
# Also from the thickness of the silhouette plot the cluster size can be
# visualized. The silhouette plot for cluster 0 when ``n_clusters`` is equal to
# 2, is bigger in size owing to the grouping of the 3 sub clusters into one big
# cluster. However when the ``n_clusters`` is equal to 4, all the plots are more
# or less of similar thickness and hence are of similar sizes as can be also
# verified from the labelled scatter plot on the right.
#
#
#
# X = coords
#
# range_n_clusters = list(range(2,3))
# for n_clusters in range_n_clusters:
#     # Create a subplot with 1 row and 2 columns
#     fig, (ax1, ax2) = plt.subplots(1, 2)
#     fig.set_size_inches(18, 7)
#
#     # The 1st subplot is the silhouette plot
#     # The silhouette coefficient can range from -1, 1 but in this example all
#     # lie within [-0.1, 1]
#     ax1.set_xlim([-0.1, 1])
#     # The (n_clusters+1)*10 is for inserting blank space between silhouette
#     # plots of individual clusters, to demarcate them clearly.
#     ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
#
#     # Initialize the clusterer with n_clusters value and a random generator
#     # seed of 10 for reproducibility.
#     clusterer = KMeans(n_clusters=n_clusters, random_state=10)
#     cluster_labels = clusterer.fit_predict(X)
#
#     # The silhouette_score gives the average value for all the samples.
#     # This gives a perspective into the density and separation of the formed
#     # clusters
#     silhouette_avg = silhouette_score(X, cluster_labels)
#     print("For n_clusters =", n_clusters,
#           "The average silhouette_score is :", silhouette_avg)
#
#     # Compute the silhouette scores for each sample
#     sample_silhouette_values = silhouette_samples(X, cluster_labels)
#
#     y_lower = 10
#     for i in range(n_clusters):
#         # Aggregate the silhouette scores for samples belonging to
#         # cluster i, and sort them
#         ith_cluster_silhouette_values = \
#             sample_silhouette_values[cluster_labels == i]
#
#         ith_cluster_silhouette_values.sort()
#
#         size_cluster_i = ith_cluster_silhouette_values.shape[0]
#         y_upper = y_lower + size_cluster_i
#
#         color = cm.nipy_spectral(float(i) / n_clusters)
#         ax1.fill_betweenx(np.arange(y_lower, y_upper),
#                           0, ith_cluster_silhouette_values,
#                           facecolor=color, edgecolor=color, alpha=0.7)
#
#         # Label the silhouette plots with their cluster numbers at the middle
#         ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
#
#         # Compute the new y_lower for next plot
#         y_lower = y_upper + 10  # 10 for the 0 samples
#
#     ax1.set_title("The silhouette plot for the various clusters.")
#     ax1.set_xlabel("The silhouette coefficient values")
#     ax1.set_ylabel("Cluster label")
#
#     # The vertical line for average silhouette score of all the values
#     ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
#
#     ax1.set_yticks([])  # Clear the yaxis labels / ticks
#     ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
#
#     # 2nd Plot showing the actual clusters formed
#     colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
#
#     x = [i[0] for i in coords]
#     y = [i[1] for i in coords]
#
#     ax2.scatter(x,y, marker='.', s=30, lw=0, alpha=0.7,
#                 c=colors, edgecolor='k')
#
#     # Labeling the clusters
#     centers = clusterer.cluster_centers_
#     # Draw white circles at cluster centers
#     ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
#                 c="white", alpha=1, s=200, edgecolor='k')
#
#     for i, c in enumerate(centers):
#         ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
#                     s=50, edgecolor='k')
#
#     ax2.set_title("The visualization of the clustered data.")
#     ax2.set_xlabel("Feature space for the 1st feature")
#     ax2.set_ylabel("Feature space for the 2nd feature")
#
#     plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
#                   "with n_clusters = %d" % n_clusters),
#                  fontsize=14, fontweight='bold')
#
# plt.show()
#
#endregion

# UAV measured yield analysis.

yield_est_data = '/home/will/cotton spatial variability vs yield analysis/' \
                 '2018-p7-p6-analysis/{0}-aoms-yield-estimates/' \
                 'pix-counts-for-2018-11-15_65_75_35_rainMatrix_modified.csv'.format(planting)

yield_est_data = pd.read_csv(yield_est_data)

data = yield_est_data.loc[:, 'turnout_lb_per_ac_yield']
labels = [int(re.findall(r'\d+', i)[1]) for i in yield_est_data.loc[:, 'ID_tag']]

# Plot UAV measured yield values for each AOM.
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
