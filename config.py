# Training Parameters

# path to training/test image data
IMAGE_BASE_PATH = "/tmp/datasets/crops/"
# path to training/test CSV data
CSV_BASE_PATH = "./datasets/"
# training set CSV filename
TRAINING_SET_CSV = "crop_info.csv"
# training set CSV filename
TEST_SET_CSV = "test_sho.csv"
# name of model architecture
MODEL_NAME = "efficientnet"
# number of output classes
NUM_CLASSES = 2
# save path for model weights
MODEL_SAVE_FOLDER = "./models/"
# name of training session
SESSION_NAME = "test"
# crop size
CROP_SIZE = 1500
# save path the visualizations
VISUALIZATIONS_PATH = "./visualizations/"
# the actual classes
CLASSES = ["null", "curb ramp", "missing ramp", "obstruction", "sfc problem"]
