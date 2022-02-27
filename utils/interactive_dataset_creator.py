import argparse
from enum import Enum
from ast import literal_eval
import glob
import pandas as pd
import random
import json

DEFAULT_CSV_FOLDER = "../datasets/"

class Operations(str, Enum):
    LOAD = "load"
    COMBINE = "combine"
    BINARIZE = "binarize"
    BALANCE = "balance"
    SUBSET = "subset"
    FILTER = "filter"
    LABEL_CITY = "label_city"
    QUIT = "quit"
    OUTPUT = "output"

def receive_operation():
    print("Provide an operation")
    operation = input()
    operation_components = operation.split()
    if len(operation_components):
        return operation_components[0], operation_components[1:] if len(operation_components) > 1 else None
    else:
        return None, None

def combine(dataset_dfs):
    return pd.concat(dataset_dfs)

def binarize(dataframe, positive_class, is_label_set=True):
    if is_label_set:
        dataframe['label_type'] = dataframe['label_set'].apply(lambda labels: int(positive_class in labels))
    else:
        dataframe['label_type'] = (dataframe['label_type'] == positive_class).astype(int)

def balance(dataframe, fraction_positive=.5):
    positives = dataframe.loc[dataframe['label_type'] == 1]
    negatives = dataframe.loc[dataframe['label_type'] == 0]


    if len(negatives) > len(positives):
        num_negatives = int(len(positives) * (1 - fraction_positive) / fraction_positive)
        if num_negatives > len(negatives):
            num_negatives = len(negatives)
            num_positive = int(len(negatives) * fraction_positive / (1 - fraction_positive))
            positives = positives = positives.sample(n=num_positive)

        negatives = negatives.sample(n=num_negatives)
    else:
        num_positive = int(len(negatives) * fraction_positive / (1 - fraction_positive))
        positives = positives.sample(n=num_positive)
    return combine([negatives, positives])

def subset(dataframe, subset_size):
    return dataframe.sample(n=subset_size)

def filter(dataframe, label_type):
    # This function is incomplete
    return dataframe.loc[dataframe['label_type'] == label_type]

def label_city(dataframe, city):
    dataframe['image_name'] = dataframe['image_name'].apply(lambda x: f"{city}/{x}")

def output(dataframe, output_path):
    dataframe.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_folder", default=DEFAULT_CSV_FOLDER)
    args = parser.parse_args()

    dataset_csv_folder = args.csv_folder

    # get a list of dataset csvs
    csv_list = glob.glob(dataset_csv_folder + "*.csv")
    print("The following CSVs are available:")
    for i in range(len(csv_list)):
        print(f'{i + 1}: {csv_list[i]}')
    
    print()

    # give list of options
    print("Options:")
    for operation in Operations:
        print(operation.value)

    print()

    output_df = None
    
    while True:
        command, arguments = receive_operation()
        print()

        print(command)
        print(arguments)


        if command == Operations.QUIT:
            break
        elif command == Operations.LOAD:
            csv_idx = int(arguments[0]) - 1
            output_df = pd.read_csv(csv_list[csv_idx], converters={'label_set': eval})
            print("loaded")
        elif command == Operations.COMBINE:
            dataframes = [output_df] if output_df is not None else []
            for i in arguments:
                dataset_df = pd.read_csv(csv_list[int(i) - 1], converters={'label_set': eval})
                dataframes.append(dataset_df)
            combined_df = combine(dataframes)
            output_df = combined_df
            print("combined")
        elif command == Operations.BINARIZE:
            positive_class = int(arguments[0])
            is_label_set = int(arguments[1]) if len(arguments) > 1 else True
            if output_df is not None:
                binarize(output_df, positive_class, is_label_set)
                print("binarized")
        elif command == Operations.BALANCE:
            fraction_positive = float(arguments[0]) if arguments else .5
            if output_df is not None:
                output_df = balance(output_df, fraction_positive)
                print("balanced")
        elif command == Operations.SUBSET:
            # right now just grabs a random subset of specified size
            subset_size = int(arguments[0])
            if output_df is not None:
                output_df = subset(output_df, subset_size)
                print("subsetted lol")
        elif command == Operations.FILTER:
            label_type = int(arguments[0])
            if output_df is not None:
                output_df = filter(output_df, label_type)
                print("filtered")
        elif command == Operations.LABEL_CITY:
            city = arguments[0]
            if output_df is not None:
                label_city(output_df, city)
                print("labeled")
        elif command == Operations.OUTPUT:
            output_path = dataset_csv_folder + arguments[0]
            if output_df is not None:
                output(output_df, output_path)
                print("saved")
                output_df = None
        else:
            print("Unrecognized operation")

        

