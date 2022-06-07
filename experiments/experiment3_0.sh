# Script for Experiment 3.1
echo "Starting Experiment 3.1"

experiment="3_0"
# city names
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
image_base_path="/mnt/disks/shared-disk/crops/"
# name of model architecture
model_name="hrnet"
# save path for model weights
model_save_folder="../models/"
# save path the visualizations
visualizations_path="../visualizations/"
# number of epochs for training
num_epochs="2"
# crop size
crop_size="1000"
# number of plots for mistake visualization
num_plots="5"

echo "initializing..."
# make binarized train and test sets for each city
mkdir -p "$csv_base_path/tmp/$city"
for label in {1..4}; do
  python ../utils/dataset_creator.py "binarize" "$csv_base_path/$city/${city}_$train_set_csv" "$label" "$csv_base_path/tmp/$city/train_set_${labels[$label - 1]}.csv"
  python ../utils/dataset_creator.py "binarize" "$csv_base_path/$city/${city}_$test_set_csv" "$label" "$csv_base_path/tmp/$city/test_set_${labels[$label - 1]}.csv"
done

# make relevant directories
mkdir -p "$csv_base_path/tmp/all_cities/"
mkdir -p "$model_save_folder/$experiment"
mkdir -p "$visualizations_path/$experiment/$city"

for label in {1..4}; do
  echo "training label ${labels[$label - 1]} classifier on all cities..."
  # compose list of train set csvs to combine

  # train model on combined train set
  python ../train.py "$experiment_$model_name_${labels[$label - 1]}" "$image_base_path" "$csv_base_path/tmp/$city/train_set_${labels[$label - 1]}.csv" "$model_name" "$model_save_folder/$experiment" "$num_epochs" "$crop_size"
  echo "testing label ${labels[$label - 1]} classifier on $city..."
    # evaluate model on each city
  python ../eval.py ${experiment}_${model_name}_$city ${experiment}_${model_name}_${labels[$label - 1]} $image_base_path $csv_base_path/"tmp/"$city/"test_set_"${labels[$label - 1]}".csv" $model_name $model_save_folder/$experiment $visualizations_path/$experiment/$city $crop_size
    # analyze results
  python ../visualization_utils/analyze_results.py "${experiment}_${model_name}_${labels[$label - 1]}" "$model_save_folder/$experiment" "$visualizations_path/$experiment/$city"
    # visualize mistakes
  python ../visualization_utils/visualize_mistakes.py "${experiment}_${model_name}_${labels[$label - 1]}" "$image_base_path" "$visualizations_path/$experiment/$city" "$crop_size" "$num_plots"
done

for city in ${cities[@]}; do
  python ../visualization_utils/plot_pr_roc.py ${experiment}_${model_name}_${city} $visualizations_path/$experiment/$city
done

echo "Finished Experiment 3.0!"