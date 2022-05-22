import argparse
from enum import Enum
import glob
import pandas as pd

DEFAULT_CSV_FOLDER = "../datasets/"
SEMANTIC_LABEL_TYPES = {
    1: "Curb Ramp",
    2: "Missing Curb Ramp",
    3: "Obstacle",
    4: "Surface Problem"
}

class Operations(str, Enum):
    LOAD = "load"
    CONTAINS = "contains"
    COMBINE = "combine"
    BINARIZE = "binarize"
    BALANCE = "balance"
    SUBSET = "subset"
    FILTER = "filter"
    SETDIFF = "setdiff"
    LABEL_CITY = "label_city"
    METRICS = "metrics"
    REFRESH = "refresh"
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

def contains(df, subset_df):
    return len(subset_df.merge(df, on='image_name')) == len(subset_df)

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

def filter(dataframe, label_type, is_label_set=True):
    if is_label_set:
        return dataframe.loc[dataframe['label_set'].apply(lambda labels: label_type in labels)]
    return dataframe.loc[dataframe['label_type'] == label_type]

def setdiff(df1, df2):
    return pd.concat([df1, df2, df2]).drop_duplicates(keep=False)

def label_city(dataframe, city):
    dataframe['image_name'] = dataframe['image_name'].apply(lambda x: f"{city}/{x}")

def metrics(dataset_dfs, output_path):
    metrics = []
    for df_tuple in dataset_dfs:
        # csv name
        csv_name = df_tuple[0]
        df = df_tuple[1]
        
        # Total crops in city
        total_crops = len(df)

        # Crops per label type
        label_type_counts = [len(df.loc[df['label_set'].apply(lambda labels: str(label_type) in labels)]) for label_type in [1, 2, 3, 4]]

        # validated count
        validated_labels = df.loc[df['agree_count'] + df['disagree_count'] + df['notsure_count'] > 0]
        validated_count = len(validated_labels)

        # validated counts per label type
        label_type_validated_counts = [len(validated_labels.loc[validated_labels['label_set'].apply(lambda labels: str(label_type) in labels)]) for label_type in [1, 2, 3, 4]]

        # agree count > disagree count
        pv_labels = df.loc[df['agree_count'] > df['disagree_count']]
        pv_crop_count = len(pv_labels)

        # positively validated counts per label type
        label_type_pv_counts = [len(pv_labels.loc[pv_labels['label_set'].apply(lambda labels: str(label_type) in labels)]) for label_type in [1, 2, 3, 4]]

        metrics.append({
            "name": csv_name,
            "total_crops": total_crops,
            "curb_ramp_count": label_type_counts[0],
            "missing_curb_ramp_count": label_type_counts[1],
            "obstacle_count": label_type_counts[2],
            "surface_problem_count": label_type_counts[3],
            "validated_count": validated_count,
            "curb_ramp_validated_count": label_type_validated_counts[0],
            "missing_curb_ramp_validated_count": label_type_validated_counts[1],
            "obstacle_validated_count": label_type_validated_counts[2],
            "surface_problem_validated_count": label_type_validated_counts[3],
            "positively_validated_count": pv_crop_count,
            "curb_ramp_pv_count": label_type_pv_counts[0],
            "missing_curb_ramp_pv_count": label_type_pv_counts[1],
            "obstacle_pv_count": label_type_pv_counts[2],
            "surface_problem_pv_count": label_type_pv_counts[3]
        })

    metrics_df = pd.DataFrame.from_records(metrics)
    metrics_df.to_csv(output_path, index=False)

def refresh(dataset_csv_folder):
    csv_list = glob.glob(dataset_csv_folder + "*.csv")
    csv_list.sort()
    print("The following CSVs are available:")
    for i in range(len(csv_list)):
        print(f'{i + 1}: {csv_list[i]}')

    return csv_list

def output(dataframe, output_path):
    dataframe.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_folder', default=DEFAULT_CSV_FOLDER, help='csv_folder - directory from which to list csvs from')
    parser.add_argument('-i', action='store_true', help='interactive_mode - allows the dataset creator to be used interactively')
    args = parser.parse_args()

    dataset_csv_folder = args.csv_folder
    interactive_mode = args.i

    if interactive_mode:
        # get a list of dataset csvs
        csv_list = refresh(dataset_csv_folder)
        
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
                output_df = pd.read_csv(csv_list[csv_idx])
                print("loaded")
            elif command == Operations.COMBINE:
                dataframes = [output_df] if output_df is not None else []
                for i in arguments:
                    dataset_df = pd.read_csv(csv_list[int(i) - 1])
                    dataframes.append(dataset_df)
                combined_df = combine(dataframes)
                output_df = combined_df
                print("combined")
            elif command == Operations.CONTAINS:
                csv_idx = int(arguments[0]) - 1
                subset_df = pd.read_csv(csv_list[csv_idx])
                if output_df is not None:
                    print(contains(output_df, subset_df))
            elif command == Operations.BINARIZE:
                positive_class = arguments[0]
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
                    print("subsetted")
            elif command == Operations.FILTER:
                label_type = arguments[0]
                is_label_set = int(arguments[1]) if len(arguments) > 1 else True
                if output_df is not None:
                    output_df = filter(output_df, label_type, is_label_set)
                    print("filtered")
            elif command == Operations.SETDIFF:
                csv_idx = int(arguments[0]) - 1
                df = pd.read_csv(csv_list[csv_idx])
                if output_df is not None:
                    output_df = setdiff(output_df, df)
                    print("setdiffed")
            elif command == Operations.LABEL_CITY:
                city = arguments[0]
                if output_df is not None:
                    label_city(output_df, city)
                    print("labeled")
            elif command == Operations.METRICS:
                output_path = dataset_csv_folder + arguments[0]
                dataframes = []
                for i in arguments[1:]:
                    csv_path = csv_list[int(i) - 1]
                    dataset_df = pd.read_csv(csv_path)
                    dataframes.append((csv_path, dataset_df))
                metrics(dataframes, output_path)
                print("outputted metrics")
            elif command == Operations.REFRESH:
                csv_list = refresh(dataset_csv_folder)
                print()
            elif command == Operations.OUTPUT:
                output_path = dataset_csv_folder + arguments[0]
                if output_df is not None:
                    output(output_df, output_path)
                    print("saved")
                    output_df = None
            else:
                print("Unrecognized operation")
