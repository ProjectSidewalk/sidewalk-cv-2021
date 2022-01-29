import pandas as pd
DATASET_CSV_PATH = "../datasets/train_crop_info.csv"
PAIRED_DATASET_CSV_PATH = "../datasets/paired_train_crop_info.csv"

# read CSV into a Pandas Dataframe
dataset_df = pd.read_csv(DATASET_CSV_PATH)
print(dataset_df.groupby('label_type').count())

# creates a subset df of all non-null crops
non_null_crops = dataset_df.loc[dataset_df['label_type'] != 0]
assert len(non_null_crops) % 2 == 0

# creates a subset df of all null crops
null_crops = dataset_df.loc[dataset_df['label_type'] == 0]

# since every large crop is paired with a small crop, we create a new csv
# containing image names with just the prefix (no size indicating suffix)
large_crop_suffix = "_1.jpg"

# first get df of all large image names, then simplify image names
paired_crops = non_null_crops.loc[non_null_crops['image_name'].str.endswith(large_crop_suffix)]
for i in paired_crops.index:
    paired_crops.at[i, 'image_name'] = paired_crops.at[i, 'image_name'][:-len(large_crop_suffix)]

paired_crops = pd.concat([paired_crops, null_crops])
paired_crops.to_csv(PAIRED_DATASET_CSV_PATH, index=False)
print(paired_crops.groupby('label_type').count())
