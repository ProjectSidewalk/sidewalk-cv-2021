"""
** Crop Extractor for Project Sidewalk **

Given label metadata from the Project Sidewalk database, this script will
extract JPEG crops of the features that have been labeled. The required metadata
may be obtained by running the SQL query in "samples/getFullLabelList.sql" on the
Sidewalk database, and exporting the results in CSV format. You must supply the
path to the CSV file containing this data below. You can find an example of what
this file should look like in "samples/labeldata.csv".

Additionally, you should have downloaded original panorama
images from Street View using DownloadRunner.py. You will need to supply the
path to the folder containing these files.

"""

import csv
import logging
import multiprocessing as mp
from itertools import islice
from time import perf_counter
from PIL import Image, ImageDraw
import os

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# *****************************************
# Update paths below                      *
# *****************************************

# Path to CSV data from database - Place in 'metadata'
csv_export_path = "metadata/gathered_panos.csv"
# Path to panoramas downloaded using DownloadRunner.py. Reference correct directory
gsv_pano_path = "../pano-downloads"
# Path to location for saving the crops
destination_path = "crops"
# Name of csv containing info about crops
csv_crop_info = "crop_info.csv"

# Mark the center of the crop?
mark_center = True

logging.basicConfig(filename='crop.log', level=logging.DEBUG)

def predict_crop_size(sv_image_y):
    """
    # Calculate distance from point to image center
    dist_to_center = math.sqrt((x-im_width/2)**2 + (y-im_height/2)**2)
    # Calculate distance from point to center of left edge
    dist_to_left_edge = math.sqrt((x-0)**2 + (y-im_height/2)**2)
    # Calculate distance from point to center of right edge
    dist_to_right_edge = math.sqrt((x - im_width) ** 2 + (y - im_height/2) ** 2)

    min_dist = min([dist_to_center, dist_to_left_edge, dist_to_right_edge])

    crop_size = (4.0/15.0)*min_dist + 200

    print("Min dist was "+str(min_dist))
    """
    crop_size = 0
    distance = max(0, 19.80546390 + 0.01523952 * sv_image_y)

    if distance > 0:
        crop_size = 8725.6 * (distance ** -1.192)
    if crop_size > 1500 or distance == 0:
        crop_size = 1500
    if crop_size < 50:
        crop_size = 50

    return crop_size

def make_single_crop(pano_img_path, sv_image_x, sv_image_y, pano_yaw_deg, crop_destination, draw_mark=False):
    """
    Makes a crop around the object of interest
    :param path_to_image: where the GSV pano is stored
    :param sv_image_x: position
    :param sv_image_y: position
    :param PanoYawDeg: heading
    :param output_filename: name of file for saving
    :param draw_mark: if a dot should be drawn in the centre of the object/image
    :return: none
    """
    try: 
        im = Image.open(pano_img_path)
        # draw = ImageDraw.Draw(im)

        im_width = im.size[0]
        im_height = im.size[1]
        # print(im_width, im_height)

        predicted_crop_size = predict_crop_size(sv_image_y)
        crop_width = predicted_crop_size
        crop_height = predicted_crop_size

        # Work out scaling factor based on image dimensions
        scaling_factor = im_width / 13312
        sv_image_x *= scaling_factor
        sv_image_y *= scaling_factor

        x = ((float(pano_yaw_deg) / 360) * im_width + sv_image_x) % im_width
        y = im_height / 2 - sv_image_y

        r = 10
        # if draw_mark:
        #     draw.ellipse((x - r, y - r, x + r, y + r), fill=128)

        # print("Plotting at " + str(x) + "," + str(y) + " using yaw " + str(pano_yaw_deg))

        # print(x, y)
        top_left_x = x - crop_width / 2
        top_left_y = y - crop_height / 2
        cropped_square = im.crop((top_left_x, top_left_y, top_left_x + crop_width, top_left_y + crop_height))
        cropped_square.save(crop_destination)
        im.close()
    except Exception as e:
        print(e)
        print("Error for {}".format(pano_img_path))

    return

def bulk_extract_crops(path_to_db_export, path_to_gsv_scrapes, destination_dir, mark_label=False):
    t_start = perf_counter()
    # create reader to read input csv with pano info
    csv_file = open(path_to_db_export)
    csv_f = csv.reader(csv_file)
    label_list = list(csv_f)
    row_count = len(label_list)

    with mp.Manager() as manager:
        # get cpu core count
        cpu_count = mp.cpu_count()
        print("cpu_count: {}".format(cpu_count))

        # Create interprocess list to store output csv rows.
        output_rows = manager.list()

        # split label csv into chunks for multiprocessing
        # 1-index to ignore header row
        i = 1
        processes = []
        while i < row_count:
            chunk_size = (row_count - i) // cpu_count
            print("chunk size: {}".format(chunk_size))
            labels = list(islice(label_list, i, i + chunk_size))
            process = mp.Process(target=crop_label_subset, args=(labels, output_rows, path_to_gsv_scrapes, destination_dir))
            processes.append(process)
            cpu_count -= 1
            i += chunk_size

        # start processes
        for p in processes:
            p.start()

        # join processes once finished
        for p in processes:
            p.join()

        # create writer to write output csv with crop info
        # TODO: for now, we will just have image_name point to a cropped jpg as model input 
        # and label_type as the output
        fields = ['image_name', 'label_type']
        csv_out = open(csv_crop_info, 'w')
        csv_w = csv.writer(csv_out)
        csv_w.writerow(fields)
        successful_crop_count = len(output_rows)
        # no_metadata_fail = 0
        # don't count header row as a failed crop
        no_pano_fail = row_count - successful_crop_count - 1

        for row in output_rows:
            csv_w.writerow(row)

        print("Finished.")
        print(str(successful_crop_count) + " successful crop extractions")
        print(str(no_pano_fail) + " extractions failed because panorama image was not found.")
        t_stop = perf_counter()
        print("Elapsed time during bulk cropping in seconds for {} labels:".format(row_count - 1),
                                            t_stop-t_start)

def crop_label_subset(input_rows, output_rows, path_to_gsv_scrapes, destination_dir):
    counter = 0
    process_pid = os.getpid()
    for row in input_rows:
        counter += 1
        pano_id = row[0]
        sv_image_x = float(row[1])
        sv_image_y = float(row[2])
        label_type = int(row[3])
        photographer_heading = float(row[4])

        pano_img_path = os.path.join(path_to_gsv_scrapes, pano_id + ".jpg")

        pano_yaw_deg = 180 - photographer_heading

        # Extract the crop
        if os.path.exists(pano_img_path):
            destination_folder = os.path.join(destination_dir)
            if not os.path.isdir(destination_folder):
                os.makedirs(destination_folder)

            if not label_type == 0:
                label_id = int(row[7])
                crop_name = str(label_id) + ".jpg"  
            else:
                # In order to uniquely identify null crops, we concatenate the pid of process they
                # were generated on and the counter within the process to the name of the null crop.
                crop_name = "null_" + str(process_pid) + "_" +  str(counter) + ".jpg"

            crop_destination = os.path.join(destination_dir, crop_name)

            if not os.path.exists(crop_destination):
                make_single_crop(pano_img_path, sv_image_x, sv_image_y, pano_yaw_deg, crop_destination, False)
                print("Successfully extracted crop to " + crop_name)
                logging.info(crop_name + " " + pano_id + " " + str(sv_image_x)
                             + " " + str(sv_image_y) + " " + str(pano_yaw_deg))
                logging.info("---------------------------------------------------")

            output_rows.append([crop_name, label_type])
        else:
            print("Panorama image not found.")
            try:
                logging.warning("Skipped label id " + str(label_id) + " due to missing image.")
            except NameError:
                logging.warning("Skipped null crop " + str(process_pid) + " " + str(counter) + " due to missing image.")