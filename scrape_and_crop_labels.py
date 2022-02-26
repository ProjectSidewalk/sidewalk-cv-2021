import csv
import math
import multiprocessing as mp
import os
import pandas as pd
import re

from CropRunner import bulk_extract_crops
from PanoScraper import bulk_scrape_panos, clean_panos
from time import perf_counter

# current city we are gathering data for
CITY = "spgg"

# the raw label data
PATH_TO_LABELDATA_CSV = 'rawdata/test.csv' #f'rawdata/labels-cv-2-9-2022-{CITY}.csv'

# the local directory panos will be downloaded to
LOCAL_DIR = 'pano-downloads/'

# the remote directory panos will be scraped from
REMOTE_DIR = f'sidewalk_panos/Panoramas/scrapes_dump_{CITY}'

# destination folder for crops
CROP_DESTINATION_PATH = 'crops'

# name of csv containing preliminary (unilabel) info about crops
CSV_CROP_INFO = "crop_info.csv"

# finalized crop info csv
FINAL_CROP_CSV = "final_crop_info.csv"

def get_nearest_label_types(crop_info, panos, threshold=750):
    # remove the suffix we append to get image name by splitting
    # at first occurence of non-digit character
    # TODO: Note this will likely only work for crops prefixed with the label_id.
    #       We may consider a different strategy when acquiring null crops
    res = re.search(r'\D+', crop_info[0]).start()
    label_id = int(crop_info[0][:res])
    print(label_id)
    current_label_type = crop_info[1]
    pano_id = crop_info[2]
    curr_pano = panos[pano_id]
    current_label = curr_pano.feats[label_id]

    # set to hold our labels for this crop
    label_set = set()

    # add current label
    label_set.add(current_label_type)

    # check to see which features are in range of the dominant label of the crop
    # to account for the label in the label set of the crop
    for _, label in curr_pano.feats.items():
        curr_label_point = current_label.point()
        other_label_point = label.point()

        absolute_x_dist = abs(curr_label_point.x - other_label_point.x)
        complement_x_dist = curr_pano.width - absolute_x_dist
        y_dist = curr_label_point.y - other_label_point.y

        if math.sqrt(absolute_x_dist**2 + y_dist**2) < threshold or math.sqrt(complement_x_dist**2 + y_dist**2) < threshold:
            label_set.add(label.label_type)
    
    final_list = list(label_set)
    return final_list

if __name__ ==  '__main__':
    print("CPU count: ", mp.cpu_count())

    # local directory to write to (relative to shell root)
    if not os.path.isdir(LOCAL_DIR):
        os.makedirs(LOCAL_DIR)

    # A datastructure containing panorama and associated label data
    panos = {}

    # TODO: probably want to not do this
    # create intermediary output dataset csv
    with open(CSV_CROP_INFO, 'w', newline='') as csv_out:
        fields = ['image_name', 'label_set', 'pano_id']
        csv_w = csv.writer(csv_out)
        csv_w.writerow(fields)

    total_successful_extractions = 0
    total_failed_extractions = 0

    t_start = perf_counter()
    for chunk in pd.read_csv(PATH_TO_LABELDATA_CSV, chunksize=10000):
        # gather panos for current data batch then scrape panos from SFTP server
        pano_set_size, scraper_exec_time = bulk_scrape_panos(chunk, panos, LOCAL_DIR, REMOTE_DIR)

        # clean panos
        clean_time = clean_panos(LOCAL_DIR)

        # make crops for current batch
        metrics = bulk_extract_crops(chunk, LOCAL_DIR, CROP_DESTINATION_PATH, CSV_CROP_INFO, panos)

        # output execution metrics
        print("====================================================================================================")
        print("Pano Scraping metrics:")
        print("Elapsed time scraping {} panos for {} labels in seconds:".format(pano_set_size, len(chunk)),
                                                scraper_exec_time)
        print()
        print("Pano Cleaning metrics:")
        print("Elapsed time cleaning {} panos in seconds:".format(pano_set_size),
                                                clean_time)
        print()
        print("Label Cropping metrics:")
        print(str(metrics[1]) + " successful crop extractions")
        print(str(metrics[2]) + " extractions failed because panorama image was not found.")
        print("Elapsed time during bulk cropping in seconds for {} labels:".format(metrics[0]),
                                                metrics[3])
        print()

        total_successful_extractions += metrics[1]
        total_failed_extractions += metrics[2]

        # delete pano downloads from current batch
        for file in os.scandir(LOCAL_DIR):
            os.remove(file.path)

    t_stop = perf_counter()
    total_execution_time = t_stop - t_start

    # TODO: might want to remove
    # sanity checks
    total_count = 0
    for _, pano in panos.items():
        if pano.width is None and pano.height is None:
            print(f'somethign spooky with pano size of {pano.pano_id}')
        total_count += len(pano)
        for _, label in pano.feats.items():
            if label.final_sv_image_x is None and label.final_sv_image_y is None:
                print(f"something spooky with final sv positoon of {label.label_id}")

    # make sure crops have label sets rather than single labels
    crop_df = pd.read_csv(CSV_CROP_INFO)
    crop_df['label_set'] = crop_df.apply(lambda x: get_nearest_label_types(x, panos), axis=1)
    crop_df.to_csv(FINAL_CROP_CSV, index=False)

    
    print(total_count)
    print()
    print("====================================================================================================")
    print(f'Total successful crop extractions: {total_successful_extractions}')
    print(f'Total failed extractions: {total_failed_extractions}')
    print(f'Total execution time in seconds: {total_execution_time}')
