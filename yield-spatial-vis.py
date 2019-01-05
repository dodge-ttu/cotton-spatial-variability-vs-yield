import os
import numpy as np
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

mpl.rcParams['axes.formatter.useoffset'] = True
mpl.rcParams['axes.formatter.offset_threshold'] = 1

import pandas as pd
from scipy.spatial import distance
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

planting = 'p6'

yield_est_data = '/home/will/cotton spatial variability vs yield analysis/' \
                 '2018-rain-matrix-p7-p6-extractions-and-data/2018_p7_p6_extractions/' \
                 '{0}-aoms-yield-estimates/' \
                 'pix-counts-for-2018-11-15_65_75_35_rainMatrix_modified.csv'.format(planting)

yield_est_data = pd.read_csv(yield_est_data)

# Plot
data = yield_est_data.loc[:, 'turnout_lb_per_ac_yield']
labels = [int(re.findall(r'\d+', i)[1]) for i in yield_est_data.loc[:, 'ID_tag']]

fig, ax = plt.subplots(1,1, figsize=(12,12))
ax.bar(range(1,17), data, width=0.8, align='center', edgecolor='k', tick_label=labels)
ax.set_xlabel('Virtual Sample ID', fontdict={'fontsize':16})
ax.set_ylabel('Estimated Lint Yield (lb/ac)', fontdict={'fontsize':16})
ax.tick_params(axis='both', labelsize=14)
ax.set_title('UAV Lint Yield Estimates for Sample Spaces - Planting {0}'.format(planting[1]),
             fontdict={'fontsize':16})

anchored_text = AnchoredText('Turnout: 38%', loc=2, prop={'size':16})
ax.add_artist(anchored_text)

plt.savefig('/home/will/cotton spatial variability vs yield analysis/' \
            '2018-rain-matrix-p7-p6-extractions-and-data/2018_p7_p6_extractions/' \
            '{0}-aoms-yield-estimates/uav-yield-estimates-planting-{0}.png'.format(planting))
