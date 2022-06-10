# Script for Experiment 3.2
echo "Starting Experiment 3.2"

experiment="3_2"
# session name to uniquely identify a run
session_name="session_name"
# city names
cities=("city1" "city2")
# city to leave out and test on
excluded_city="city"
# label types
labels=("curb_ramp" "missing_curb_ramp" "obstacle" "surface_problem")
# path to train/test CSV data
csv_base_path="../datasets/"
# train set CSV filename
train_set_csv="train_set.csv"
# test set CSV filename
test_set_csv="test_set.csv"
# path to train/test image data
image_base_path="/mnt/disks/shared_disk/crops/"
# name of model architecture
model_name="hrnet"
# save path for model weights
model_save_folder="../models/"
# save path the visualizations
visualizations_path="../visualizations/"
# number of epochs for training
num_epochs="10"
# crop size
crop_size="1000"
# number of plots for mistake visualization
num_plots="5"

echo "initializing..."
# make binarized train sets for each city
for city in ${cities[@]}; do
  mkdir -p $csv_base_path/tmp/$city
  for label in {1..4}; do
    python ../utils/dataset_creator.py binarize $csv_base_path/$city/${city}_$train_set_csv $label $csv_base_path/tmp/$city/train_set_${labels[$label - 1]}.csv
  done
done

# binarize test set for exluded city
for label in {1..4}; do
  python ../utils/dataset_creator.py binarize $csv_base_path/$excluded_city/${excluded_city}_$test_set_csv $label $csv_base_path/tmp/$excluded_city/test_set_${labels[$label - 1]}.csv
done

# make relevant directories
mkdir -p $csv_base_path/tmp/exclude_$excluded_city/
mkdir -p $model_save_folder/$experiment/exclude_$excluded_city/
mkdir -p $visualizations_path/$experiment/$excluded_city/

for label in {1..4}; do
  echo "training label ${labels[$label - 1]} classifier on all cities except $excluded_city..."
  # combine train sets for all cities except the excluded city
  arguments=""
  for city in ${cities[@]}; do
    if [ $city != $excluded_city ]; then
      arguments+="$csv_base_path/tmp/$city/train_set_${labels[$label - 1]}.csv "
    fi
  done
  python ../utils/dataset_creator.py combine $arguments "$csv_base_path/tmp/exclude_$excluded_city/train_set_${labels[$label - 1]}.csv"

  # train model on combined train set
  python ../train.py ${experiment}_${session_name}_${labels[$label - 1]} $image_base_path/ $csv_base_path/tmp/exclude_$excluded_city/train_set_${labels[$label - 1]}.csv $model_name $model_save_folder/$experiment/exclude_$excluded_city/ $num_epochs $crop_size

  echo "testing label ${labels[$label - 1]} classifier on $excluded_city..."
  # evaluate model on excluded city
  python ../eval.py ${experiment}_${session_name}_${excluded_city} ${experiment}_${session_name}_${labels[$label - 1]} $image_base_path/ $csv_base_path/tmp/$excluded_city/test_set_${labels[$label - 1]}.csv $model_name $model_save_folder/$experiment/exclude_$excluded_city/ $visualizations_path/$experiment/$excluded_city/ $crop_size
  # analyze results
  python ../visualization_utils/analyze_results.py ${experiment}_${session_name}_${labels[$label - 1]} $model_save_folder/$experiment/exclude_$excluded_city/ $visualizations_path/$experiment/$excluded_city/
  # visualize mistakes
  python ../visualization_utils/visualize_mistakes.py ${experiment}_${session_name}_${labels[$label - 1]} $image_base_path/ $visualizations_path/$experiment/$excluded_city/ $crop_size $num_plots
done

python ../visualization_utils/plot_pr_roc.py ${experiment}_${session_name}_${excluded_city} $visualizations_path/$experiment/$excluded_city

echo "Finished Experiment 3.2!"
