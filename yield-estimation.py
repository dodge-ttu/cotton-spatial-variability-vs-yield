import os
import cv2
import pandas as pd
import numpy as np


def count_white_pix(sample_images=None, thresh_value=None):

    font = cv2.FONT_HERSHEY_SIMPLEX

    images_counted_marked = []
    pixel_counts = []

    for image, ID_tag in sample_images:
        h,w,c = image.shape
        b,g,r = cv2.split(image)
        image_copy = image.copy()
        img_gray = b
        #img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
        (T, thresh) = cv2.threshold(img_gray, thresh_value, 255, cv2.THRESH_BINARY)

        mask = np.where(thresh)

        x, y = mask
        x = x.tolist()
        y = y.tolist()
        marks = [(x, y) for (x, y) in zip(x, y)]

        pixel_counts.append((len(marks), ID_tag))

        for i in marks:
            # cv2.circle(img, (i[1], i[0]), 1, (255,255,255), 1)
            image_copy[i] = (0, 0, 255)

        images_counted_marked.append((image_copy, ID_tag))

    return images_counted_marked, pixel_counts


# # convert all values to centimeters
# def calc_GSD(base_image=None, sensor_h=None, sens_w=None, focal_length=None, altitude=None):
#     sens_h = 0.8 # 8mm / 10 = 0.8 cm
#     sens_w = 1.32 # 13.2mm / 10 = 1.32 cm
#     focal_length = 0.88 # 8.8mm / 10 = 0.88 cm
#     altitude = 3000 # 30m * 100 = 3000 cm
#
#     GSD_h = (altitude * sens_h) / (focal_length * img_h)
#     GSD_w = (altitude * sens_w) / (focal_length * img_w)
#
#     if (GSD_h > GSD_w):
#         GSD = GSD_h
#     else:
#         GSD = GSD_w
#
#     return GSD


# Define path to extracted samples.
input_dir = '/home/will/cotton spatial variability vs yield analysis/2018-rain-matrix-p7-p6-extractions-and-data/' \
            '2018_p7_p6_extractions/p7-yield-aoms-extracted'

# Get extracted samples filenames.
files_in_dir = [i for i in os.listdir(input_dir) if i.endswith(".tif")]

# Create a list of aom images.
some_sample_images = []
for image_name in files_in_dir:
    a_path = os.path.join(input_dir, image_name)
    an_image = cv2.imread(a_path)
    some_sample_images.append((an_image, image_name))

# Define path to output directory.
an_output_dir = "/home/will/cotton spatial variability vs yield analysis/" \
                "2018-rain-matrix-p7-p6-extractions-and-data/2018_p7_p6_extractions/"

# Provide an ID for the analysis.
analysis_id = "2018-11-15_65_75_35_rainMatrix_modified"

# Details.
planting = 'p7'
what = 'yield-aoms'

# Create an out sub-directory.
directory_path = os.path.join(an_output_dir, "{0}-{1}-yield-estimates".format(planting, what))
if not os.path.exists(directory_path):
    os.makedirs(directory_path)

# Count pixels.
params = {
    "sample_images": some_sample_images,
    "thresh_value": 225,
}

images_counted_and_marked, pixel_counts = count_white_pix(**params)

# Generate csv data.
df = pd.DataFrame(pixel_counts)
df.columns = ["pix_counts", "ID_tag"]

# GSD value should be retrieved from the metadata file associated with a given composite.
# GSD for "2018-11-15_65_75_35_rainMatrix_modified" processing run: 90.9090909091 pix/meter
# GSD for "2018-11-15_65_75_35_rainMatrix_modified" processing run: 1.1000011 cm/pix
GSD = 1.1

# Calculate 2D yield area.
df.loc[:, "2D_yield_area"] = df.loc[:, "pix_counts"] * 1.1

# Write pix count data.
df.to_csv(os.path.join(directory_path, "pix-counts-for-{0}.csv".format(analysis_id)))

# Write marked sample images for inspection.
for (image,image_name) in images_counted_and_marked:
    cv2.imwrite(os.path.join(directory_path, '{0}-marked.png'.format(image_name)), image)
