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

yield_est_data = '/home/will/cotton spatial variability vs yield analysis/' \
                 '2018-rain-matrix-p7-p6-extractions-and-data/2018_p7_p6_extractions/' \
                 'p6-yield-aoms-yield-estimates/pix-counts-for-2018-11-15_65_75_35_rainMatrix_modified.csv'

virtual_sample_spaces_in_meters = '/home/will/cotton spatial variability vs yield analysis/' \
                                  '2018-rain-matrix-p7-p6-extractions-and-data/2018_p7_p6_extractions/' \
                                  'p6-yield-aoms-yield-estimates/virtual_aom_areas.csv'

planting = 6

df = pd.read_csv(yield_est_data)

# Get area data for each virtual region of interest.
df_area = pd.read_csv(virtual_sample_spaces_in_meters)

# spatial_p6_aom15.tif
df_area.loc[:, 'ID_tag'] = ['spatial_p6_aom{0}.tif'.format(x) for x in df_area.loc[:, 'aom_id'].values]

# Merge data
df_both = df.merge(df_area, left_on='ID_tag', right_on='ID_tag', how='inner')

# Yield model y = 0.658 * x - 35.691
df_both.loc[:, 'est_yield'] = df_both.loc[:, '2D_yield_area'] * 0.658

# Per square meter yield,
df_both.loc[:, 'g_per_sq_meter_yield'] = df_both.loc[:, 'est_yield'] / df_both.loc[:, 'area']

# Sort values.
df_both.sort_values(by=['g_per_sq_meter_yield'], inplace=True)

# Grams per meter to pounds per acre 1:8.92179.
df_both.loc[:, 'lb_per_ac_yield'] = df_both.loc[:, 'g_per_sq_meter_yield'] * 8.92179

# Lint Yield, turnout.
df_both.loc[:, 'turnout_lb_per_ac_yield'] = df_both.loc[:, 'lb_per_ac_yield'] * .38

# Plot
data = df_both.loc[:, 'turnout_lb_per_ac_yield']
labels = [int(re.findall(r'\d+', i)[1]) for i in df_both.loc[:, 'ID_tag']]

fig, ax = plt.subplots(1,1, figsize=(12,12))
ax.bar(range(1,17), data, width=0.8, align='center', edgecolor='k', tick_label=labels)
ax.set_xlabel('Virtual Sample ID', fontdict={'fontsize':16})
ax.set_ylabel('Estimated Lint Yield (lb/ac)', fontdict={'fontsize':16})
ax.tick_params(axis='both', labelsize=14)
ax.set_title('UAV Lint Yield Estimates for Sample Spaces - Planting {0}'.format(planting),
             fontdict={'fontsize':16})

anchored_text = AnchoredText('Turnout: 38%', loc=2, prop={'size':16})
ax.add_artist(anchored_text)

plt.savefig('/home/will/cotton spatial variability vs yield analysis/' \
            '2018-rain-matrix-p7-p6-extractions-and-data/2018_p7_p6_extractions/' \
            'p6-yield-aoms-yield-estimates/uav-yield-estimates-planting-p{0}.png'.format(planting))
