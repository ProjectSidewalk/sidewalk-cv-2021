# sidewalk-cv-2021

## Setup
1. Acquire sftp server credentials from Mikey and place them at the root of the project folder as `alphie-sftp`.
2. Run `pip3 install -r requirements.txt`.
3. Create a folder called `rawdata` in the project root and add a metadata csv (from Mikey) and update `path_to_labeldata_csv` in `scrape_and_crop_labels.py`.
4. Make sure to remove `batch.txt`, `crop_info.csv`, `crop.log`, `/pano-downloads`, and `/crops`.
5. Run `python3 scrape_and_crop_labels.py`.
6. Crops will be stored in the generated `crops` folder.
