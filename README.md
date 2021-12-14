# Classification of Sidewalk Accessibility Features

Problem statement here

## Abstract
Recent work has applied machine learning methods to automatically find and/or assess pedestrian infrastructure in online map imagery (e.g., satellite photos, streetscape panoramas). While promising, these methods have been limited by two interrelated issues: small training sets and the choice of machine learning model. In this project, supplemented by the ever-growing Project Sidewalk dataset of 300,000+ image-based sidewalk accessibility labels, we aim to improve on the results/findings from the [2019 iteration](https://github.com/ProjectSidewalk/sidewalk-cv-assets19) of the project, which provided a novel examination of deep learning as a means to assess sidewalk quality in Google Street View (GSV) panoramas. To do so, we focus on one application area, the automatic validation of crowdsourced labels. Our goal is to introduce improvements in two regards: data quality (particularly, the types of image crops we are gathering) and utilizing modern deep learning techniques to highlight spatial/background context in supplementing the classification of the actual accessibility feature. In tackling the issue of data quality, we investigate strategies such as multi-size cropping as well as delving deeper and adjusting the pipeline used to map labels placed in the Project Sidewalk audit interface to the GSV panorama image being cropped from. In regards to model training, we compare strategies such as ensembling various models trained on different sized crops, utilizing attention mechanisms such as those found in the state-of-the-art [CoAtNet](https://arxiv.org/abs/2106.04803), as well as comparing model quality given a simplified problem space through the binary classification of individual label types. In evaluating the success of our strategies, we provide an analysis on dataset-wide metrics such as accuracy and loss, while also considering label-type-specific metrics such as precision and recall per label type.

## Video Here

## Related Work

## Methodology
### Data Acquisition
As mentioned before, our datasets consisted of crowdsourced labels from the Project Sidewalk database. In order to ensure the quality of the human labels, we only chose validated labels satisfying the following conditions:

`disagree validations <= 2` and `disagree validations < 2 * agree validations`

In addition, we aimed to have a relatively balanced dataset. Given that our explorations for this project were limited to the Seattle Project Sidewalk label dataset, below are the individual label counts for the four accessibility features/problems we aimed to classify:

| Label Type | Count |
| ---------- | ----- |
| Curb Ramp |    69k    |
| Missing Curb Ramp |   34k    |
| Obstacle |   10k     |
| Surface Problem |   25k    |

This are the counts without filtering the labels that don't satisfy the validation conditions above, so in trying to build a balanced dataset of 10k labels per label type, our final counts were 9998 curb ramps, 10000 missing curb ramps, 8930 obstacles, 10000 surface problems. One thing to note is that labels might have been discarded due to the panorama imagery they were cropped from not existing or the crop size going out of the vertical bounds of the GSV imagery, which will be discussed later. This explains why there are less than 10000 curb ramps despite the overwhelming majority of curb ramps in the Seattle database. The smaller quantity of obstacles is due to the smaller quantity of obstacles in the Seattle database overall.

### Model Training

## Results/Analysis

## Next Steps

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
 
