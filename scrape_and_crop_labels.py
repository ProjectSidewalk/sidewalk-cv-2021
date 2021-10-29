from PanoScrapper import bulk_scrape_panos
from CropRunner import bulk_extract_crops

import os

# Scrape panos from SFTP server
n = 10
path_to_labeldata_csv = "rawdata/seattle-labels-cv-10-21-2021.csv"

# local directory to write to (relative to shell root)
local_dir = '/home/micdun/sidewalk-cv-2021/pano-downloads/'
if not os.path.isdir(local_dir):
    os.makedirs(local_dir)

# remote scrapes directory to acquire from
remote_dir = 'sidewalk_panos/Panoramas/scrapes_dump_seattle'

output_csv_name = 'gathered_panos.csv'
bulk_scrape_panos(n, path_to_labeldata_csv, local_dir, remote_dir, output_csv_name)

# Crop labels with scrapped panos
csv_export_path = 'pano-downloads/gathered_panos.csv'
gsv_pano_path = 'pano-downloads'
destination_path = 'crops'
bulk_extract_crops(csv_export_path, gsv_pano_path, destination_path, mark_label=False)