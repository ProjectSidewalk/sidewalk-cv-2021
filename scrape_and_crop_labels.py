#!/usr/bin/env python3
import argparse
import http
import json
import math
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import re

from CropRunner import bulk_extract_crops
from PanoScraper import bulk_scrape_panos
from time import perf_counter

def label_metadata_from_csv(metadata_csv_path):
    df_meta = pd.read_csv(metadata_csv_path)
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
    #         "label_id": 173572,
    #         "gsv_panorama_id": "qos0p3mBdh7Pynq2N-1SWw",
    #         "label_type_id": 4,
    #         "deleted": false,
    #         "tutorial": false,
    #         "agree_count": 0,
    #         "disagree_count": 0,
    #         "notsure_count": 1,
    #         "sv_image_x": 9927,
    #         "sv_image_y": -624,
    #         "canvas_width": 720,
    #         "canvas_height": 480,
    #         "canvas_x": 384,
    #         "canvas_y": 212,
    #         "zoom": 3,
    #         "heading": 267.84820556640625,
    #         "pitch": -18.546875,
    #         "photographer_heading": 180.2176055908203,
    #         "photographer_pitch": 0.13916778564453125
    #     },
    #     ...
    # ]
    return pd.DataFrame.from_records(pano_info)

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
    parser.add_argument('-c', nargs='?', default=None, help='csv_path - location of csv from which to read label metadata')
    parser.add_argument('-d', nargs='?', default=None, help='sidewalk_server_domain - FDQN of SidewalkWebpage server to fetch pano list from, i.e. sidewalk-sea.cs.washington.edu')
    args = parser.parse_args()

    local_dir = args.local_dir
    base_crops_path = args.crops
    city = args.city
    label_metadata_csv = args.c
    sidewalk_server_fqdn = args.d

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

    # stores intermediary metadata info about crops
    crop_info = []

    total_successful_extractions = 0
    total_failed_extractions = 0

    t_start = perf_counter()
    batch_size = 10000
    for _, chunk in label_metadata.groupby(np.arange(len(label_metadata)) // batch_size):
        # filter out deleted or tutorial labels from data chunk
        chunk = chunk.loc[(chunk['deleted'] == 'f') & (chunk['tutorial'] == 'f')]

        # filter out labels from panos with missing pano metadata
        has_image_size_filter = pd.notnull(chunk["image_width"]) 
        chunk = chunk[has_image_size_filter]

        # gather panos for current data batch then scrape panos from SFTP server
        pano_set_size, scraper_exec_time = bulk_scrape_panos(chunk, panos, local_dir, remote_dir)

        # make crops for current batch
        metrics = bulk_extract_crops(chunk, local_dir, crop_destination_path, crop_info, panos)

        # output execution metrics
        print("====================================================================================================")
        print("Pano Scraping metrics:")
        print("Elapsed time scraping {} panos for {} labels in seconds:".format(pano_set_size, len(chunk)),
                                                scraper_exec_time)

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

    # make sure crops have label sets rather than single labels
    crop_df = pd.DataFrame.from_records(crop_info)
    crop_df['label_set'] = crop_df.apply(lambda x: get_nearest_label_types(x, panos), axis=1)
    crop_df.to_csv(final_crop_csv, index=False)

    print()
    print("====================================================================================================")
    print(f'Total successful crop extractions: {total_successful_extractions}')
    print(f'Total failed extractions: {total_failed_extractions}')
    print(f'Total execution time in seconds: {total_execution_time}')
