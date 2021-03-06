import os
import cv2
import pandas as pd
import numpy as np


def count_white_pix(sample_images=None, thresh_value=None):

    font = cv2.FONT_HERSHEY_SIMPLEX

    images_counted_marked = []
    pixel_counts = []
    masks = []

    for image, ID_tag in sample_images:
        image_copy = image.copy()
        h,w,c = image.shape
        b,g,r = cv2.split(image)

        img_gray = b
        #img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
        (T, thresh) = cv2.threshold(img_gray, thresh_value, 255, cv2.THRESH_BINARY)

        mask_data = np.nonzero(thresh)

        x, y = mask_data
        x = x.tolist()
        y = y.tolist()
        marks = [(x, y) for (x, y) in zip(x, y)]

        pixel_counts.append((len(marks), ID_tag))

        for i in marks:
            # cv2.circle(img, (i[1], i[0]), 1, (255,255,255), 1)
            image_copy[i] = (0, 0, 255)

        images_counted_marked.append((image_copy, ID_tag))
        masks.append((mask_data, ID_tag, (h,w)))

    return images_counted_marked, pixel_counts, masks

if __name__ == "__main__":

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

    # Details.
    plantings = ['p6', 'p7']
    what = 'aoms'

    # Define path to output directory.
    output_dir = "/home/will/cotton spatial variability vs yield analysis/" \
                    "2018-p7-p6-analysis/"

    # Provide an ID for the analysis.
    analysis_id = "2018-11-15_65_75_35_rainMatrix_modified"

    # Process desired plantings.
    for planting in plantings:

        # Define path to read in extracted samples.
        input_dir = '/home/will/cotton spatial variability vs yield analysis/' \
                    '2018-p7-p6-analysis/{0}-yield-aoms-extracted'.format(planting)

        # Get extracted sample file names.
        files_in_dir = [i for i in os.listdir(input_dir) if i.endswith(".tif")]

        # Create a list of aom images.
        some_sample_images = []
        for image_name in files_in_dir:
            a_path = os.path.join(input_dir, image_name)
            an_image = cv2.imread(a_path)
            some_sample_images.append((an_image, image_name))

        # Create an out sub-directory.
        directory_path = os.path.join(output_dir, "{0}-{1}-yield-estimates".format(planting, what))
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        # Count pixels.
        params = {
            "sample_images": some_sample_images,
            "thresh_value": 225,
        }

        images_counted_and_marked, pixel_counts, yield_masks = count_white_pix(**params)

        # Generate CSV data.
        df = pd.DataFrame(pixel_counts)
        df.columns = ["pix_counts", "ID_tag"]

        # GSD value should be retrieved from the metadata file associated with a given composite.
        # GSD for "2018-11-15_65_75_35_rainMatrix_modified" processing run: 90.9090909091 pix/meter
        # GSD for "2018-11-15_65_75_35_rainMatrix_modified" processing run: 1.1000011 cm/pix
        GSD = 1.1 # pixels / cm

        # Calculate 2D yield area.
        df.loc[:, "PCCA"] = df.loc[:, "pix_counts"] * 1.1 # pixels / cm

        # Get per AOM area, manually exported from QGIS at the moment.
        virtual_sample_spaces_in_meters = '/home/will/cotton spatial variability vs yield analysis/' \
                                          '2018-p6-p7-data/' \
                                          'virtual_aom_areas-{0}.csv'.format(planting)

        # Get area data for each virtual region of interest.
        df_area = pd.read_csv(virtual_sample_spaces_in_meters)

        # Generate an ID for the join column from the filename written as: spatial_p6_aom_15.tif
        df_area.loc[:, 'ID_tag'] = ['spatial_{0}_aom_{1}.tif'.format(planting, str(x).zfill(2)) for x in df_area.loc[:, 'aom_id'].values]

        # Merge data.
        df_both = df.merge(df_area, left_on='ID_tag', right_on='ID_tag', how='outer')

        # Yield model y = 2.28 * x for a 35 meter flight based on current findings.
        # slope = 5.885 - .1348 * (altitude) + 9.077*(10**-4) * (altitude**2)
        # PCCA = pixels * GSD
        # seeded_cotton_weight = slope * PCCA

        df_both.loc[:, 'pred_yield_grams_per_cm2'] = df_both.loc[:, 'PCCA'] * 2.28

        # Per square meter yield.
        df_both.loc[:, 'pred_yield_grams_per_m2'] = df_both.loc[:, 'pred_yield_grams_per_cm2'] / df_both.loc[:, 'area']

        # Sort values.
        df_both.sort_values(by=['g_per_sq_meter_yield'], inplace=True)

        # 1 gram per meter is 8.92179 pounds per acre.
        df_both.loc[:, 'lb_per_ac_yield'] = df_both.loc[:, 'g_per_sq_meter_yield'] * 8.92179

        # Lint Yield, turnout.
        df_both.loc[:, 'turnout_lb_per_ac_yield'] = df_both.loc[:, 'lb_per_ac_yield'] * .38

        # Write pix count data.
        df_both.to_csv(os.path.join(directory_path, "pix-counts-for-{0}.csv".format(analysis_id)))

        # Write marked sample images for inspection.
        for (image,image_name) in images_counted_and_marked:
            cv2.imwrite(os.path.join(directory_path, '{0}-marked.png'.format(image_name)), image)

        # Make directory for pixel location data.
        yield_pixel_location_csv_dir = os.path.join(output_dir, "{0}-white-pixel-locations".format(planting))
        if not os.path.exists(yield_pixel_location_csv_dir):
            os.makedirs(yield_pixel_location_csv_dir)

        # Write pixel locations of measured seed-cotton.
        for ((y, x), ID_tag, (h,w)) in yield_masks:

            # Virtual sample space (image) height and width is recorded in df.
            df = pd.DataFrame({'x':x,'y':y,'h':h,'w':w})
            df.to_csv(os.path.join(yield_pixel_location_csv_dir, ID_tag.split('.')[0] + '.csv'))

