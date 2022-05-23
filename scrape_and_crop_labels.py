#!/usr/bin/env python3
import argparse
import datetime
import http.client
import json
import logging
import math
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import re

from CropRunner import bulk_extract_crops
from PanoScraper import bulk_scrape_panos
from time import perf_counter

CROP_LOGS_FOLDER = "crop_logs"

def label_metadata_from_csv(metadata_csv_path):
    df_meta = pd.read_csv(metadata_csv_path, true_values=['t'], false_values=['f'])
    return df_meta
    
def label_metadata_from_api(sidewalk_server_fqdn):
    conn = http.client.HTTPSConnection(sidewalk_server_fqdn)
    conn.request("GET", "/adminapi/labels/cvMetadata")
    r1 = conn.getresponse()
    data = r1.read()
    pano_info = json.loads(data)

    # Structure of JSON data
    # [
    #     {
    #         "label_id":47614,
    #         "gsv_panorama_id":"sHMY67LdNX48BFwpbGMD3A",
    #         "label_type_id":2,
    #         "agree_count":1,
    #         "disagree_count":0,
    #         "notsure_count":0,
    #         "image_width":16384,
    #         "image_height":8192,
    #         "sv_image_x":6538,
    #         "sv_image_y":-731,
    #         "canvas_width":720,
    #         "canvas_height":480,
    #         "canvas_x":275,
    #         "canvas_y":152,
    #         "zoom":1,
    #         "heading":190.25,
    #         "pitch":-34.4375,
    #         "photographer_heading":292.4190368652344,
    #         "photographer_pitch":-3.3052749633789062
    #     },
    #     ...
    # ]
    return pd.DataFrame.from_records(pano_info)

def get_nearest_label_types(crop_info, panos, city, threshold=750):
    # remove the suffix we append to get image name by splitting
    # at first occurence of non-digit character.
    # TODO: Note this will likely only work for crops prefixed with the label_id.
    #       We may consider a different strategy when acquiring null crops
    # get only the image_name by removing city prefix
    crop_base_name = crop_info[0][len(city) + 1:]
    res = re.search(r'\D+', crop_base_name).start()
    label_id = int(crop_base_name[:res])
    # print(label_id)
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
    parser.add_argument('-c', nargs='?', default=None, help='csv_path - location of csv from which to read label metadata')
    parser.add_argument('-d', nargs='?', default=None, help='sidewalk_server_domain - FDQN of SidewalkWebpage server to fetch pano list from, i.e. sidewalk-sea.cs.washington.edu')
    args = parser.parse_args()

    local_dir = args.local_dir
    base_crops_path = args.crops
    city = args.city
    label_metadata_csv = args.c
    sidewalk_server_fqdn = args.d

    if not os.path.isdir(CROP_LOGS_FOLDER):
        os.makedirs(CROP_LOGS_FOLDER)

    logging.basicConfig(filename=f'{CROP_LOGS_FOLDER}/{city}_crop_failure.log', level=logging.DEBUG)
    logging.info(f'CROP SESSION TIMESTAMP: {datetime.datetime.now().strftime("%d %b %Y %H:%M:%S")}')

    # the raw label data
    # path_to_labeldata_csv = f'rawdata/test-seattle.csv' #f'rawdata/labels-cv-4-20-2022-{city}.csv'
    if label_metadata_csv is not None:
        label_metadata = label_metadata_from_csv(label_metadata_csv)
    elif sidewalk_server_fqdn is not None:
        label_metadata = label_metadata_from_api(sidewalk_server_fqdn)
    else:
        # no option to read data
        print("No options from which to read data")
        os._exit(0)

    # label_metadata = label_metadata.head(40)
    total_metadata_size = len(label_metadata)
    print(f'Total metadata size: {total_metadata_size}')
    print()

    # the remote directory panos will be scraped from
    remote_dir = f'sidewalk_panos/Panoramas/scrapes_dump_{city}'

    # destination folder for crops
    crop_destination_path = f'{base_crops_path}/{city}'

    # finalized crop info csv
    final_crop_csv = f'{base_crops_path}/{city}_crop_info.csv'  

    print("CPU count: ", mp.cpu_count())

    # local directory to write to (relative to shell root)
    if not os.path.isdir(local_dir):
        os.makedirs(local_dir)

    # get set of label ids we already have
    existing_crops = None
    existing_label_ids = set()
    if os.path.exists(final_crop_csv):
        existing_crops = pd.read_csv(final_crop_csv)
        existing_label_ids = set(existing_crops['image_name'].str[len(city) + 1:-4].astype(int))  # remove city prefix and .jpg extension

        # TODO: update validation counts here

    # filter out labels that already have crops for them
    label_metadata = label_metadata[~label_metadata['label_id'].isin(existing_label_ids)]

    crop_already_exists_count = total_metadata_size - len(label_metadata)

    # filter out deleted or tutorial labels from data chunk
    # label_metadata = label_metadata[(~label_metadata['deleted']) & (~label_metadata['tutorial'])]

    # deleted_or_tutorial_count = total_metadata_size - crop_already_exists_count - len(label_metadata)

    # filter out labels from panos with missing pano metadata
    has_image_size_filter = pd.notnull(label_metadata['image_width']) 
    label_metadata = label_metadata[has_image_size_filter]

    missing_pano_metadata_count = total_metadata_size - crop_already_exists_count -len(label_metadata) #- deleted_or_tutorial_count - len(label_metadata)

    # A datastructure containing panorama and associated label data
    panos = {}

    # stores intermediary metadata info about crops
    crop_info = []

    assert crop_already_exists_count + missing_pano_metadata_count == total_metadata_size - len(label_metadata) # deleted_or_tutorial_count + missing_pano_metadata_count == total_metadata_size - len(label_metadata)
    total_prefiltered_labels = total_metadata_size - len(label_metadata)
    total_successful_extractions = 0
    total_failed_extractions = 0

    t_start = perf_counter()
    batch_size = 1000
    for i, chunk in label_metadata.groupby(np.arange(len(label_metadata)) // batch_size):
        print("====================================================================================================")
        print(f'Iteration {i + 1}/{math.ceil(len(label_metadata) / batch_size)}')

        # gather panos for current data batch then scrape panos from SFTP server
        pano_set_size, scraper_exec_time = bulk_scrape_panos(chunk, panos, local_dir, remote_dir)

        # make crops for current batch
        metrics = bulk_extract_crops(chunk, local_dir, crop_destination_path, crop_info, panos)

        # output execution metrics
        print()
        print("Pano Scraping metrics:")
        print("Elapsed time scraping {} panos for {} labels in seconds:".format(pano_set_size, len(chunk)),
                                                scraper_exec_time)

        print()
        print("Label Cropping metrics:")
        print(str(metrics[1]) + " successful crop extractions")
        print(str(metrics[2]) + " extractions failed.")
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

    # make sure crops have label sets rather than single labels
    crop_df = pd.DataFrame.from_records(crop_info)
    if 'label_set' in crop_df.columns:
        crop_df['label_set'] = crop_df.apply(lambda x: get_nearest_label_types(x, panos, city), axis=1)
    if existing_crops is not None:
        crop_df = pd.concat([existing_crops, crop_df])
    crop_df.to_csv(final_crop_csv, index=False)

    print()
    print("====================================================================================================")
    print(f'Total prefiltered labels: {total_prefiltered_labels}')
    print("Prefilter counts:")
    print(f'Crop already exists: {crop_already_exists_count}')
    # print(f'Deleted or tutorial: {deleted_or_tutorial_count}')
    print(f'Missing pano metadata: {missing_pano_metadata_count}')
    print()
    print(f'Total successful crop extractions: {total_successful_extractions}')
    print(f'Total failed extractions: {total_failed_extractions}')
    print(f'Total execution time in seconds: {total_execution_time}')
