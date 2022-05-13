import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('input_csv_path', type=str)
parser.add_argument('output_csv_path', type=str)
parser.add_argument('label_type', type=int)
args = parser.parse_args()

# read CSV into a Pandas dataframe
dataframe = pd.read_csv(args.input_csv_path)
dataframe['label_type'] = dataframe['label_set'].apply(lambda labels: int(str(args.label_type) in labels))
dataframe.to_csv(args.output_csv_path, index=False)
