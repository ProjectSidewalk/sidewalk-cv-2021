import pandas as pd

LABEL = 1
BINARY_LABELS_CSV_PATH = f"../datasets/{LABEL}_crop_labels.csv"
DATASET_CSV_PATH = "../datasets/train_crop_info.csv"

# read CSV into a Pandas Dataframe
dataset_df = pd.read_csv(DATASET_CSV_PATH)
print(dataset_df.head(30))
print(dataset_df.groupby('label_type').count())

dataset_df.loc[dataset_df['label_type'] != LABEL, 'label_type'] = 0
dataset_df.loc[dataset_df['label_type'] == LABEL, 'label_type'] = 1

print(dataset_df.groupby('label_type').count())
dataset_df.to_csv(BINARY_LABELS_CSV_PATH, index=False)
