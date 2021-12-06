import pandas as pd

COMPLETE_DATASET_CSV_PATH = "../datasets/crop_info.csv"
NON_NULL_CROP_DATASET_CSV_PATH = "../datasets/train_non_null_crop_info.csv"
SUBSET_CSV_PATH = "../datasets/train_subset_crop_info.csv"

# read CSV into a Pandas Dataframe
complete_dataset_df = pd.read_csv(COMPLETE_DATASET_CSV_PATH)
print(complete_dataset_df.head(30))
print(complete_dataset_df.groupby('label_type').count())

# create CSV with just null crops
# creates a subset df of all rows with 'label_type' not 0
non_null_crops = complete_dataset_df.loc[complete_dataset_df['label_type'] != 0]
print(non_null_crops.head(30))

# writes non-null crop df to a csv at specified path without the indexing column
non_null_crops.to_csv(NON_NULL_CROP_DATASET_CSV_PATH, index=False)

# create null crop dataframe
# creates a subset df of all rows with 'label_type' 0
null_crops = complete_dataset_df.loc[complete_dataset_df['label_type'] == 0]
print(null_crops.head(30))

# get null crop count
n = len(null_crops.index)
print(n)

# create Subset CSV of all non-null crops and the first m of the null crops
m = 10000
m_null_crops = null_crops.head(m)

subset_df = pd.concat([non_null_crops, m_null_crops])
print(subset_df.head(30))

subset_df.to_csv(SUBSET_CSV_PATH, index=False)
