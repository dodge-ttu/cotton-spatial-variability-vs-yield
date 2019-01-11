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

from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
# from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import KernelDensity

# Details.
planting = 'p6'

# Define input directory for extracted samples.
image_in_dir = '/home/will/cotton spatial variability vs yield analysis/' \
               '2018-p7-p6-analysis/{0}-aoms-yield-estimates'.format(planting)

# Define output directory for plots.
visuals_directory = '/home/will/cotton spatial variability vs yield analysis/' \
            '2018-p7-p6-analysis/{0}-visuals'.format(planting)

# Create an out directory for visuals.
if not os.path.exists(visuals_directory):
    os.makedirs(visuals_directory)

# UAV measured yield analysis.
yield_est_data = '/home/will/cotton spatial variability vs yield analysis/' \
                 '2018-p7-p6-analysis/{0}-aoms-yield-estimates/' \
                 'pix-counts-for-2018-11-15_65_75_35_rainMatrix_modified.csv'.format(planting)

yield_est_data = pd.read_csv(yield_est_data)

data = yield_est_data.loc[:, 'turnout_lb_per_ac_yield']
labels = [int(re.findall(r'\d+', i)[1]) for i in yield_est_data.loc[:, 'ID_tag']]

# Plot UAV measured yield values for each Planting.
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

fig.suptitle('UAV Measured Yield Pixel Locations Planting {0}'.format(planting[1]), fontsize=16, fontweight='bold')
fig.tight_layout(pad=2.0, w_pad=1.0, h_pad=0.0)
plt.subplots_adjust(top=0.94)
plt.savefig(os.path.join(visuals_directory, 'uav-measured-yield-multiples.png'))
plt.close()
