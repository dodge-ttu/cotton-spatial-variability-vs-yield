import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

filepath = "/home/will/MAHAN MAP 2018/spatial_p6_aom01_points_WGS_84.csv"

df = pd.read_csv(filepath)

x = df.loc[:, "X"].values
y = df.loc[:, "Y"].values

fig, ax = plt.subplots(1,1, figsize=(10,10))
ax.plot(x,y, "o")