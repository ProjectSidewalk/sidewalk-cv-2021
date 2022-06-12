# Script for Experiment 2.1
echo "Starting Experiment 2.1"

experiment="2_1"
# the paper this experiment is being run for
paper="assets-2022"
# session name to uniquely identify an experiment run
session_name="session_name"
# city name
city="spgg"
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
num_epochs="7"
# crop size
crop_size="1000"
# number of plots for mistake visualization
num_plots="5"

echo "initializing..."
# make binarized train and test sets for each city
mkdir -p "$csv_base_path/tmp/$city"
for label in {1..4}; do
  # binarize train set
  python ../utils/dataset_creator.py "binarize" "$csv_base_path/$paper/$city/${city}_$train_set_csv" "$label" "$csv_base_path/tmp/$city/train_set_${labels[$label - 1]}.csv"
  # validate and filter on pv for train set
  python ../utils/dataset_creator.py "validate" "$csv_base_path/tmp/$city/train_set_${labels[$label - 1]}.csv" "$csv_base_path/tmp/$city/train_set_${labels[$label - 1]}.csv"
  python ../utils/dataset_creator.py "filter" "$csv_base_path/tmp/$city/train_set_${labels[$label - 1]}.csv" "$csv_base_path/tmp/$city/train_set_${labels[$label - 1]}.csv" "pv"

  # binarize test set
  python ../utils/dataset_creator.py "binarize" "$csv_base_path/$paper/$city/${city}_$test_set_csv" "$label" "$csv_base_path/tmp/$city/test_set_${labels[$label - 1]}.csv"
done

# make relevant directories
mkdir -p "$model_save_folder/$experiment/$city"
mkdir -p "$visualizations_path/$experiment/$city"

for label in {1..4}; do
  echo "training label ${labels[$label - 1]} classifier on all cities..."
  # compose list of train set csvs to combine

  # train model on combined train set
  python ../train.py "${experiment}_${session_name}_${labels[$label - 1]}" "$image_base_path" "$csv_base_path/tmp/$city/train_set_${labels[$label - 1]}.csv" "$model_name" "$model_save_folder/$experiment/$city" "$num_epochs" "$crop_size"
  echo "testing label ${labels[$label - 1]} classifier on $city..."
  # evaluate model on each city
  python ../eval.py ${experiment}_${session_name}_${city} ${experiment}_${session_name}_${labels[$label - 1]} $image_base_path $csv_base_path/"tmp/"$city/"test_set_"${labels[$label - 1]}".csv" $model_name $model_save_folder/$experiment/$city $visualizations_path/$experiment/$city $crop_size
  # analyze results
  python ../visualization_utils/analyze_results.py "${experiment}_${session_name}_${labels[$label - 1]}" "$model_save_folder/$experiment/$city" "$visualizations_path/$experiment/$city"
  # visualize mistakes
  python ../visualization_utils/visualize_mistakes.py "${experiment}_${session_name}_${labels[$label - 1]}" "$image_base_path" "$visualizations_path/$experiment/$city" "$crop_size" "$num_plots"
done

python ../visualization_utils/plot_pr_roc.py ${experiment}_${session_name}_${city} $visualizations_path/$experiment/$city


echo "Finished Experiment 2.1!"