# Test Experiment Script
echo "Running Experiment: Draft"

# path to train/test image data
image_base_path="/tmp/crops/"
# path to train/test CSV data
csv_base_path="../datasets/"
# train set CSV filename
train_set_csv="final_crop_info.csv"
# test set CSV filename
test_set_csv="seattle_true_test.csv"
# name of model architecture
model_name="efficientnet"
# save path for model weights
model_save_folder="../models/"
# name of train session
session_name="draft"
# crop size
crop_size="1500"
# save path the visualizations
visualizations_path="../visualizations/"
# number of plots for mistake visualization
num_plots="5"

# create temporary binarized train sets
python ../utils/binary_relabeler.py $csv_base_path $train_set_csv "train_set_1.csv" "1"
python ../utils/binary_relabeler.py $csv_base_path $train_set_csv "train_set_2.csv" "2"
python ../utils/binary_relabeler.py $csv_base_path $train_set_csv "train_set_3.csv" "3"
python ../utils/binary_relabeler.py $csv_base_path $train_set_csv "train_set_4.csv" "4"
# create temporary binarized test sets
python ../utils/binary_relabeler.py $csv_base_path $test_set_csv "test_set_1.csv" "1"
python ../utils/binary_relabeler.py $csv_base_path $test_set_csv "test_set_2.csv" "2"
python ../utils/binary_relabeler.py $csv_base_path $test_set_csv "test_set_3.csv" "3"
python ../utils/binary_relabeler.py $csv_base_path $test_set_csv "test_set_4.csv" "4"

# train model
python ../train.py $session_name"_1" $image_base_path $csv_base_path "train_set_1.csv" $model_name $model_save_folder $crop_size
python ../train.py $session_name"_2" $image_base_path $csv_base_path "train_set_2.csv" $model_name $model_save_folder $crop_size
python ../train.py $session_name"_3" $image_base_path $csv_base_path "train_set_3.csv" $model_name $model_save_folder $crop_size
python ../train.py $session_name"_4" $image_base_path $csv_base_path "train_set_4.csv" $model_name $model_save_folder $crop_size
# evaluate model
python ../eval.py $session_name"_1" $image_base_path $csv_base_path "test_set_1.csv" $model_name $model_save_folder $crop_size $visualizations_path
python ../eval.py $session_name"_2" $image_base_path $csv_base_path "test_set_2.csv" $model_name $model_save_folder $crop_size $visualizations_path
python ../eval.py $session_name"_3" $image_base_path $csv_base_path "test_set_3.csv" $model_name $model_save_folder $crop_size $visualizations_path
python ../eval.py $session_name"_4" $image_base_path $csv_base_path "test_set_4.csv" $model_name $model_save_folder $crop_size $visualizations_path
# analyze results
python ../visualization_utils/analyze_results.py $session_name"_1" $model_save_folder $visualizations_path
python ../visualization_utils/analyze_results.py $session_name"_2" $model_save_folder $visualizations_path
python ../visualization_utils/analyze_results.py $session_name"_3" $model_save_folder $visualizations_path
python ../visualization_utils/analyze_results.py $session_name"_4" $model_save_folder $visualizations_path
# visualize mistakes
python ../visualization_utils/visualize_mistakes.py $session_name"_1" $image_base_path $crop_size $visualizations_path $num_plots
python ../visualization_utils/visualize_mistakes.py $session_name"_2" $image_base_path $crop_size $visualizations_path $num_plots
python ../visualization_utils/visualize_mistakes.py $session_name"_3" $image_base_path $crop_size $visualizations_path $num_plots
python ../visualization_utils/visualize_mistakes.py $session_name"_4" $image_base_path $crop_size $visualizations_path $num_plots

# remove temporary binarized train sets
rm $csv_base_path"train_set_1.csv"
rm $csv_base_path"train_set_2.csv"
rm $csv_base_path"train_set_3.csv"
rm $csv_base_path"train_set_4.csv"
# remove temporary binarized test sets
rm $csv_base_path"test_set_1.csv"
rm $csv_base_path"test_set_2.csv"
rm $csv_base_path"test_set_3.csv"
rm $csv_base_path"test_set_4.csv"
