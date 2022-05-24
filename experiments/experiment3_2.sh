# Script for Experiment 3.2
echo "Starting Experiment 3.2"

experiment="3.2"
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
model_name="hrnet"
# save path for model weights
model_save_folder="../models/"
# save path the visualizations
visualizations_path="../visualizations/"
# number of epochs for training
num_epochs="15"
# crop size
crop_size="1500"
# number of plots for mistake visualization
num_plots="5"

echo "initializing..."
# make binarized train and test sets for each city
for city in ${cities[@]}; do
  mkdir -p $csv_base_path/"tmp/"$city
  for label in {1..4}; do
    python ../utils/dataset_creator.py "binarize" $csv_base_path/$city/$train_set_csv $label $csv_base_path/"tmp/"$city/"train_set"$label".csv"
    python ../utils/dataset_creator.py "binarize" $csv_base_path/$city/$test_set_csv $label $csv_base_path/"tmp/"$city/"test_set"$label".csv"
  done
done

for excluded_city in ${cities[@]}; do
  # make relevant directories
  mkdir -p $csv_base_path/"tmp/exclude_"$excluded_city/
  mkdir -p $model_save_folder/$experiment/"exclude_"$excluded_city/
  mkdir -p $visualizations_path/$experiment/$excluded_city/
  
  for label in {1..4}; do
    echo "training label "$label" classifier on all cities except "$excluded_city"..."
    # combine train sets for all cities except the excluded city
    head -n1 $csv_base_path/"tmp/"${cities[1]}/"train_set"$label".csv" > $csv_base_path/"tmp/exclude_"$excluded_city/"train_set"$label".csv"
    for city in ${cities[@]}; do
      if [ $city != $excluded_city ]; then
        python ../utils/dataset_creator.py "combine" $csv_base_path/"tmp/exclude_"$excluded_city/"train_set"$label".csv" $csv_base_path/"tmp/"$city/"train_set"$label".csv" $csv_base_path/"tmp/exclude_"$excluded_city/"train_set"$label".csv"
      fi
    done
    wc $csv_base_path/"tmp/exclude_"$excluded_city/"train_set"$label".csv"

    # train model on combined train set
    python ../train.py $model_name$label $image_base_path/ $csv_base_path/"tmp/exclude_"$excluded_city/"train_set"$label".csv" $model_name $model_save_folder/$experiment/"exclude_"$excluded_city/ $num_epochs $crop_size

    echo "testing label "$label" classifier on "$excluded_city"..."
    # evaluate model on excluded city
    python ../eval.py $model_name$label $image_base_path/ $csv_base_path/"tmp/"$excluded_city/"test_set"$label".csv" $model_name $model_save_folder/$experiment/"exclude_"$excluded_city/ $visualizations_path/$experiment/$excluded_city/ $crop_size
    # analyze results
    python ../visualization_utils/analyze_results.py $model_name$label $model_save_folder/$experiment/"exclude_"$excluded_city/ $visualizations_path/$experiment/$excluded_city/
    # visualize mistakes
    python ../visualization_utils/visualize_mistakes.py $model_name$label $image_base_path/ $visualizations_path/$experiment/$excluded_city/ $crop_size $num_plots
  done
done

echo "Finished Experiment 3.2!"
