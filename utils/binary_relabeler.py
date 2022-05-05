import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('csv_base_path', type=str)
parser.add_argument('training_set_csv', type=str)
parser.add_argument('output_csv', type=str)
parser.add_argument('label_type', type=int)
args = parser.parse_args()

# read CSV into a Pandas Dataframe
dataset_df = pd.read_csv(os.path.join(args.csv_base_path, args.training_set_csv))
dataset_df.loc[dataset_df['label_type'] != args.label_type, 'label_type'] = -1
dataset_df.loc[dataset_df['label_type'] == args.label_type, 'label_type'] = 1
dataset_df.loc[dataset_df['label_type'] == -1, 'label_type'] = 0
dataset_df.to_csv(os.path.join(args.csv_base_path, args.output_csv), index=False)
