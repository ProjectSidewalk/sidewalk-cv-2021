from PanoScraper import bulk_scrape_panos, clean_panos
from CropRunner import bulk_extract_crops

import multiprocessing as mp
import os

if __name__ ==  '__main__':
    # scrape panos from SFTP server
    n = 20
    start_row = 1
    path_to_labeldata_csv = "rawdata/seattle-labels-cv-10-29-2021.csv"

    # local directory to write to (relative to shell root)
    local_dir = 'pano-downloads/'
    if not os.path.isdir(local_dir):
        os.makedirs(local_dir)

    # remote scrapes directory to acquire from
    remote_dir = 'sidewalk_panos/Panoramas/scrapes_dump_seattle'

    output_csv_name = 'gathered_panos.csv'
    pano_set_size, scraper_exec_time = bulk_scrape_panos(n, start_row, path_to_labeldata_csv, local_dir, remote_dir, output_csv_name)

    # clean panos
    gsv_pano_path = 'pano-downloads'
    clean_time = clean_panos(gsv_pano_path)

    # crop labels with scrapped panos
    csv_export_path = 'pano-downloads/gathered_panos.csv'
    destination_path = 'crops'
    metrics = bulk_extract_crops(csv_export_path, gsv_pano_path, destination_path, mark_label=False)

    # output execution metrics
    print("====================================================================================================")
    print("Pano Scraping metrics:")
    print("Elapsed time scraping {} panos for {} labels in seconds:".format(pano_set_size, n),
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
    print("CPU count: ", mp.cpu_count())
