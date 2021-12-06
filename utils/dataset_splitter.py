import pandas as pd

COMPLETE_DATASET_CSV_PATH = "../datasets/train_crop_info.csv"
NON_NULL_CROP_DATASET_CSV_PATH = "../datasets/train_non_null_crop_info.csv"
SUBSET_CSV_PATH = "../datasets/train_subset_crop_info.csv"
LARGE_CROPS_CSV_PATH = "../datasets/train_large_crop_info.csv"
SMALL_CROPS_CSV_PATH = "../datasets/train_small_crop_info.csv"

GET_SUBSET = False
GET_LARGE_SMALL_SPLIT = True

# read CSV into a Pandas Dataframe
complete_dataset_df = pd.read_csv(COMPLETE_DATASET_CSV_PATH)
print(complete_dataset_df.groupby('label_type').count())

# create CSV with just null crops
# creates a subset df of all rows with 'label_type' not 0
non_null_crops = complete_dataset_df.loc[complete_dataset_df['label_type'] != 0]

# writes non-null crop df to a csv at specified path without the indexing column
non_null_crops.to_csv(NON_NULL_CROP_DATASET_CSV_PATH, index=False)

# create null crop dataframe
# creates a subset df of all rows with 'label_type' 0
null_crops = complete_dataset_df.loc[complete_dataset_df['label_type'] == 0]
print(null_crops.head(30))

# get null crop count
n = len(null_crops.index)
print(n)

if GET_SUBSET:
    # create Subset CSV of all non-null crops and the first m of the null crops
    m = 10000
    m_null_crops = null_crops.head(m)

    subset_df = pd.concat([non_null_crops, m_null_crops])
    print(subset_df.head(30))

    subset_df.to_csv(SUBSET_CSV_PATH, index=False)

if GET_LARGE_SMALL_SPLIT:
    # create large crops dataframe from all non-null crops
    large_crop_suffix = "_1.jpg"
    large_crop = non_null_crops.loc[non_null_crops['image_name'].str.endswith(large_crop_suffix)]
    print(large_crop.head(30))

    # create small crops dataframe from all non-null crops
    small_crop_suffix = "_0.jpg"
    small_crop = non_null_crops.loc[non_null_crops['image_name'].str.endswith(small_crop_suffix)]
    print(small_crop.head(30))

    # split null crops in half so we can give both the large and small crops some null crops
    null_crop_first_half = null_crops.iloc[:n//2]
    null_crop_second_half = null_crops.iloc[n//2:]

    large_crop = pd.concat([large_crop, null_crop_first_half])
    small_crop = pd.concat([small_crop, null_crop_second_half])

    large_crop.to_csv(LARGE_CROPS_CSV_PATH, index=False)
    small_crop.to_csv(SMALL_CROPS_CSV_PATH, index=False)
