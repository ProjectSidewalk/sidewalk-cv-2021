#!/usr/bin/env python3
import argparse
import math
import multiprocessing as mp
import os
import pandas as pd
import re

from CropRunner import bulk_extract_crops
from PanoScraper import bulk_scrape_panos
from time import perf_counter

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

    # Work with the assumption that current label will have finalized sv positions
    current_label = curr_pano.feats[label_id]
    curr_label_point = current_label.point()

    # set to hold our labels for this crop
    label_set = set()

    # add current label
    label_set.add(current_label_type)

    # check to see which features are in range of the dominant label of the crop
    # to account for the label in the label set of the crop
    for _, label in curr_pano.feats.items():
        other_label_point = label.point()
        if other_label_point is not None:
            absolute_x_dist = abs(curr_label_point.x - other_label_point.x)
            complement_x_dist = curr_pano.width - absolute_x_dist
            y_dist = curr_label_point.y - other_label_point.y

            if math.sqrt(absolute_x_dist**2 + y_dist**2) < threshold or math.sqrt(complement_x_dist**2 + y_dist**2) < threshold:
                label_set.add(label.label_type)
    
    final_list = list(label_set)
    return final_list

if __name__ ==  '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('local_dir', help='local_dir - the local directory panos will be downloaded to, i.e. pano-downloads/')
    parser.add_argument('crops', help='crops - destination folder for crops')
    parser.add_argument('city', help='city - city from which we want to gather pano data from')
    args = parser.parse_args()

    local_dir = args.local_dir
    base_crops_path = args.crops
    city = args.city

    # the raw label data
    path_to_labeldata_csv = f'rawdata/labels-cv-4-20-2022-{city}.csv'

    # the remote directory panos will be scraped from
    remote_dir = f'sidewalk_panos/Panoramas/scrapes_dump_{city}'

    # destination folder for crops
    crop_destination_path = f'{base_crops_path}/{city}'

    # finalized crop info csv
    final_crop_csv = f'{city}_final_crop_info.csv'  

    print("CPU count: ", mp.cpu_count())

    # local directory to write to (relative to shell root)
    if not os.path.isdir(local_dir):
        os.makedirs(local_dir)

    # A datastructure containing panorama and associated label data
    panos = {}

    # TODO: probably want to not do this
    # create intermediary output dataset csv
    # with open(CSV_CROP_INFO, 'w', newline='') as csv_out:
    #     fields = ['image_name', 'label_set', 'pano_id', 'agree_count', 'disagree_count', 'notsure_count']
    #     csv_w = csv.writer(csv_out)
    #     csv_w.writerow(fields)
    crop_info = []

    total_successful_extractions = 0
    total_failed_extractions = 0

    t_start = perf_counter()
    for chunk in pd.read_csv(path_to_labeldata_csv, chunksize=10000):
        # filter out deleted or tutorial labels from data chunk
        chunk = chunk.loc[(chunk['deleted'] == 'f') & (chunk['tutorial'] == 'f')]

        # filter out labels from panos with missing pano metadata
        has_image_size_filter = pd.notnull(chunk["image_width"]) 
        chunk = chunk[has_image_size_filter]

        # gather panos for current data batch then scrape panos from SFTP server
        pano_set_size, scraper_exec_time = bulk_scrape_panos(chunk, panos, local_dir, remote_dir)

        # clean panos
        # clean_time = clean_panos(LOCAL_DIR)

        # make crops for current batch
        metrics = bulk_extract_crops(chunk, local_dir, crop_destination_path, crop_info, panos)

        # output execution metrics
        print("====================================================================================================")
        print("Pano Scraping metrics:")
        print("Elapsed time scraping {} panos for {} labels in seconds:".format(pano_set_size, len(chunk)),
                                                scraper_exec_time)
        # print()
        # print("Pano Cleaning metrics:")
        # print("Elapsed time cleaning {} panos in seconds:".format(pano_set_size),
        #                                         clean_time)
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
        for file in os.scandir(local_dir):
            os.remove(file.path)

    t_stop = perf_counter()
    total_execution_time = t_stop - t_start

    # TODO: might want to remove
    # sanity checks
    # total_count = 0
    # for _, pano in panos.items():
    #     if pano.width is None and pano.height is None:
    #         print(f'somethign spooky with pano size of {pano.pano_id}')
    #     total_count += len(pano)
    #     for _, label in pano.feats.items():
    #         if label.final_sv_image_x is None and label.final_sv_image_y is None:
    #             print(f"something spooky with final sv positoon of {label.label_id}")

    # make sure crops have label sets rather than single labels
    crop_df = pd.DataFrame.from_records(crop_info)
    crop_df['label_set'] = crop_df.apply(lambda x: get_nearest_label_types(x, panos), axis=1)
    crop_df.to_csv(final_crop_csv, index=False)

    
    # print(total_count)
    print()
    print("====================================================================================================")
    print(f'Total successful crop extractions: {total_successful_extractions}')
    print(f'Total failed extractions: {total_failed_extractions}')
    print(f'Total execution time in seconds: {total_execution_time}')
