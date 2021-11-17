# sidewalk-cv-2021

## Setup
1. Acquire sftp server credentials from Mikey and place them at the root of the project folder as `alphie-sftp`.
2. Run `pip3 install -r requirements.txt`.
3. Create a folder called `rawdata` in the project root and add a metadata csv (from Mikey) and update `path_to_labeldata_csv` in `scrape_and_crop_labels.py`.
4. Make sure to remove `batch.txt`, `crop_info.csv`, `crop.log`, `pano-downloads/`, and `crops/`.
5. Run `python3 scrape_and_crop_labels.py`.
6. Crops will be stored in the generated `crops` folder.

## Training
### Training on UW Instructional GPU machine
1. SSH into machine
2. Run this command to create a new nested bash session: `tmux new -s my_session`
3. Detach the bash session so that it persists after the SSH connection is dropped via `ctrl+b` then just `d`
4. Reattach/re-enter the session via `tmux attach-session -t my_session`
5. Kill the bash session by stopping running processes and typing `exit` in the terminal
 