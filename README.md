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

### Initial Training Attempts
Our initial objective was to pick out some promising network architectures from the [torchvision.models](https://pytorch.org/vision/stable/models.html) package. Our first dataset had a huge imbalance of null crops and was causing models to just predict null for almost every image, so we did these initial training runs with no null crops while waiting on a more balanced dataset in order to get a rough idea of performance on the non-null classes. We trained several models for 50 epochs using SGD with default hyperparameters (learning rate, momentum, etc.) and no weight decay. We plotted per-class precision and recall, as well as overall accuracy and loss, as a function of epoch. We tried several mid-size architectures including [efficientnet](https://pytorch.org/hub/nvidia_deeplearningexamples_efficientnet/), [densenet](https://pytorch.org/hub/pytorch_vision_densenet/), and [regnet](https://pytorch.org/vision/master/_modules/torchvision/models/regnet.html). Note that each of these models comes in varying sizes, so we selected the largest ones that would fit in the RAM available to us and take a reasonable amount of time to train. We found that all of these models achieved similar performace on the metrics we tracked, but efficientnet and regnet trained significantly faster than densenet, so we focused on these architectures moving forward. This initial round of training revealed some issues such as overfitting and noisy updates. These plots, aquired from efficientnet training runs, are representative of some issues we faced:

<img src="./writeup_images/overfit_loss.png" width=400 /><img src="./writeup_images/overfit_accuracy.png" width=400 />

<center> <figcaption>Increasing loss and decreasing accuracy on validation set, indicative of overfitting</figcaption> </center>

<img src="./writeup_images/spiky_recall.png" width=400 /> <img src="./writeup_images/spiky_precision.png" width=400 />

<center><figcaption>Spiky recall and precision curves, indicating noisy updates</figcaption></center>

### Improving Training Hyperparameters
To help resolve these issues, we implemented learning rate scheduling and added weight decay to our loss calculations. The best scheduling strategy we found was to decrease learning rate by a factor of .3 every 10 epochs, starting from .01, and the best weight decay we found was around 1e-6. This improvements gave us plots such as the following, training on the same dataset: <br>
<img src="./writeup_images/better_loss.png" width=400></img>
<img src="./writeup_images/better_accuracy.png" width=400></img>

<center><figcaption>Less overfitting, though still some</figcaption></center>

<img src="./writeup_images/less_spiky_recall.png" width=400></img>
<img src="./writeup_images/less_spiky_precision.png" width=400></img>

<center><figcaption>Precision and recall curves begin to converge</figcaption></center>

### Ensembling Architectures
In addition to aquiring a better dataset, our next step was coming up with more novel architectures to try and improve performance. A friend of ours pointed us to some interesting documentation on [ensembling](https://ensemble-pytorch.readthedocs.io/en/latest/introduction.html), which is essentially training multiple models and somehow combining their results to give a better overall performance.

Ensembling of Binary Classifiers: One approach we tried that we found interesting involved independently training a binary classifier for each of the 4 non-null classes, and then combining them into a model that passes the combined output of each through an additional fully connected layer to get a multi-class prediction. Our idea here was that it may be easier for a model to learn which images are positive examples of a given class if we just label every other image as a negative example rather than with a variety of other labels. To do this training, we made a new set of labels for each class where all images that aren't a positive example of the class in question are labeled as negative, and then trained with the same dataset. We'll discuss the results of this approach more in the next section, but overall it didn't really work out.

Ensembling of Models Trained on Different Size Crops: The more promising ensembling strategy we used involved taking two image crops for each labeled sidewalk feature. We were able to do this because the original data is street view panoramas with coordinates of sidewalk features, so it's up to us how large to make the crops around each feature. We trained one model on the crops we had in our current dataset (we called these "small"), and trained another model on zoomed out versions of the same crops (we called these "large"). We then combined the models into a model that takes as input both the small and large crop for a given sidewalk feature. For a forward pass, the model passed each image to the corresponding model and combined the output of each to obtain a final prediction vector. We tried simply concatenating the output of each model and passing the result through a single fully-connected layer, as well as more complex strategies such as removing the last layer of each model and passing the larger concatenated feature map through several fully-connected layers with activation functions between each. We'll discuss the results of this approach more in the next section.


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
 
