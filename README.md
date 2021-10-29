# sidewalk-cv-2021

## Setup
1. Acquire sftp server credentials from Mikey and place them at the root of the project folder as `alphie-sftp`
2. Run `pip3 install -r requirements.txt`
3. Create a folder called `rawdata` in the project root and add a metadata csv (from Mikey)
4. Run `python3 scrape_and_crop_labels.py`
5. Crops will be stored in the generated `crops` folder