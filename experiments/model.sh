# Model Comparisons Experiment Script.
echo "Running Experiment: Model Comparisons"

# path to training/test image data
image_base_path="/tmp/crops/"
# path to training/test CSV data
csv_base_path="../datasets/"
# training set CSV filename
training_set_csv="crop_info.csv"
# training set CSV filename
test_set_csv="test_sho.csv"
# name of model architecture
model_name="efficientnet"
# save path for model weights
model_save_folder="../models/"
# name of training session
session_name="ModelComparisonsTest"
# crop size
crop_size="1500"
# save path the visualizations
visualizations_path="../visualizations/"

# create temporary binarized training set for each label
python3 ../utils/binary_relabeler.py $csv_base_path $training_set_csv crop_info_0.csv 0
python3 ../utils/binary_relabeler.py $csv_base_path $training_set_csv crop_info_1.csv 1
python3 ../utils/binary_relabeler.py $csv_base_path $training_set_csv crop_info_2.csv 2
python3 ../utils/binary_relabeler.py $csv_base_path $training_set_csv crop_info_3.csv 3
python3 ../utils/binary_relabeler.py $csv_base_path $training_set_csv crop_info_4.csv 4

# train model
python3 ../train.py $image_base_path $csv_base_path crop_info_1.csv $model_name $model_save_folder $session_name $crop_size
# evaluate model
python3 ../eval.py $image_base_path $csv_base_path crop_info_1.csv $model_name $model_save_folder $session_name $crop_size $visualizations_path
# analyze/visualize results
python3 ../visualization_utils/analyze_results.py $model_save_folder $session_name $visualizations_path
python3 ../visualization_utils/visualize_mistakes.py $image_base_path $session_name $crop_size $visualizations_path 5

# remove each temporary training set
rm $csv_base_path/crop_info_0.csv
rm $csv_base_path/crop_info_1.csv
rm $csv_base_path/crop_info_2.csv
rm $csv_base_path/crop_info_3.csv
rm $csv_base_path/crop_info_4.csv
