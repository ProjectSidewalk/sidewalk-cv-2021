# Test Experiment Script
echo "Running Experiment: Draft"

# city names
cities=("city1" "city2" "city3" "city4" "city5")
# path to train/test CSV data
csv_base_path="../datasets/"
# train set CSV filename
train_set_csv="train_set.csv"
# test set CSV filename
test_set_csv="test_set.csv"
# path to train/test image data
image_base_path="/tmp/crops/"
# name of model architecture
model_name="efficientnet"
# save path for model weights
model_save_folder="../models/"
# save path the visualizations
visualizations_path="../visualizations/"
# number of epochs for training
num_epochs="1"
# crop size
crop_size="1500"
# number of plots for mistake visualization
num_plots="1"

echo "initializing..."
mkdir -p $csv_base_path/"tmp/"

# make tmporary binarized train and test sets for each city
for city in ${cities[@]}; do
  mkdir -p $csv_base_path/"tmp"/$city
  for label in {1..4}; do
    python ../utils/csv_binarizer.py $csv_base_path/$city/$train_set_csv $csv_base_path/"tmp/"$city/"train_set"$label".csv" $label
    python ../utils/csv_binarizer.py $csv_base_path/$city/$test_set_csv $csv_base_path/"tmp/"$city/"test_set"$label".csv" $label
  done
done

echo "beginning experiment 3.1"
experiment="3.1"

mkdir -p $csv_base_path/"tmp/all_cities/"
mkdir -p $model_save_folder/$experiment
mkdir -p $visualizations_path/$experiment
for label in {1..4}; do
  # combine train sets for all cities
  head -n1 $csv_base_path/"tmp/"${cities[1]}/"train_set"$label".csv" > $csv_base_path/"tmp/all_cities/train_set"$label".csv"
  for city in ${cities[@]}; do
    tail -n+2 $csv_base_path/"tmp/"$city/"train_set"$label".csv" >> $csv_base_path/"tmp/all_cities/train_set"$label".csv"
  done
  wc $csv_base_path/"tmp/all_cities/train_set"$label".csv"

  # train model on combined train set
  python ../train.py $model_name$label $image_base_path $csv_base_path/"tmp/all_cities/train_set"$label".csv" $model_name $model_save_folder/$experiment $num_epochs $crop_size

  # test and evaluate model on each city's individual test set
  for city in ${cities[@]}; do
    mkdir -p $visualizations_path/$experiment/$city
    # evaluate model
    python ../eval.py $model_name$label $image_base_path $csv_base_path/"tmp/"$city/"test_set"$label".csv" $model_name $model_save_folder/$experiment $visualizations_path/$experiment/$city $crop_size
    # analyze results
    python ../visualization_utils/analyze_results.py $model_name$label $model_save_folder/$experiment $visualizations_path/$experiment/$city
    # visualize mistakes
    python ../visualization_utils/visualize_mistakes.py $model_name$label $image_base_path $visualizations_path/$experiment/$city $crop_size $num_plots
  done
done