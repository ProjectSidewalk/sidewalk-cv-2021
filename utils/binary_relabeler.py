import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('csv_base_path', type=str)
parser.add_argument('train_set_csv', type=str)
parser.add_argument('output_csv', type=str)
parser.add_argument('label_type', type=int)
args = parser.parse_args()

# read CSV into a Pandas dataframe
dataframe = pd.read_csv(os.path.join(args.csv_base_path, args.train_set_csv))
dataframe['label_type'] = dataframe['label_set'].apply(lambda labels: int(str(args.label_type) in labels))
dataframe.to_csv(os.path.join(args.csv_base_path, args.output_csv), index=False)
