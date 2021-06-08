import argparse
import csv
import json
import pickle

from sklearn import metrics, svm
from sklearn.model_selection import train_test_split


def load_data(X_path, y_path):
    with open(X_path) as input_file:
        csv_reader = csv.reader(input_file, quoting=csv.QUOTE_NONNUMERIC)
        X = list(csv_reader)

    with open(y_path) as input_file:
        csv_reader = csv.reader(input_file, quoting=csv.QUOTE_NONNUMERIC)
        y = [row[0] for row in csv_reader]

    return X, y


def process_data(X, y):
    # Split data into 50% train and 50% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, shuffle=False
    )
    return X_train, X_test, y_train, y_test


def export_processed_data(data, output_path):
    with open(output_path, "wb") as output_file:
        pickle.dump(data, output_file)


def load_processed_data(data_path):
    with open(data_path, "rb") as data_file:
        X, y = pickle.load(data_file)
        return X, y


def train_model(X_train, y_train):
    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=0.001)
    # Learn the digits on the train subset
    return clf.fit(X_train, y_train)


def export_model(model, output_path="output/model.pkl"):
    with open(output_path, "wb") as model_output_file:
        pickle.dump(model, model_output_file)


def load_model(model, model_path="output/model.pkl"):
    with open(model_path, "rb") as model_file:
        return pickle.load(model_file)


def output_metrics(y_test, predicted_y, output_filename: str = "output/metrics.json"):
    with open(output_filename, "w") as metrics_output_file:
        result_metrics = {
            "accuracy_score": metrics.accuracy_score(y_test, predicted_y),
            "weighted_precision": metrics.precision_score(
                y_test, predicted_y, average="weighted"
            ),
            "weighted_recall": metrics.recall_score(
                y_test, predicted_y, average="weighted"
            ),
            "weighted_f1_score": metrics.f1_score(
                y_test, predicted_y, average="weighted"
            ),
        }
        json.dump(
            result_metrics,
            metrics_output_file,
            indent=4,
        )


def output_test_data_results(
    y_test, predicted_y, output_filename: str = "output/test_data_results.csv"
):
    with open(output_filename, "w") as classes_output_file:
        writer = csv.writer(classes_output_file)
        writer.writerow(["actual", "predicted"])
        writer.writerows([(x, y) for x, y in zip(y_test, predicted_y)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help="Supported commands: process-data,train,report")
    args = parser.parse_args()

    if args.command == "process-data":
        X, y = load_data("data/digit_data.csv", "data/digit_target.csv")
        X_train, X_test, y_train, y_test = process_data(X, y)
        export_processed_data((X_train, y_train), "output/training_data.pkl")
        export_processed_data((X_test, y_test), "output/testing_data.pkl")
    elif args.command == "train":
        X_train, y_train = load_processed_data("output/training_data.pkl")
        model = train_model(X_train, y_train)
        export_model(model, "output/model.pkl")
    elif args.command == "report":
        X_test, y_test = load_processed_data("output/testing_data.pkl")
        model = load_model("output/model.pkl")
        predicted_y = model.predict(X_test)
        output_test_data_results(y_test, predicted_y)
        output_metrics(y_test, predicted_y)


if __name__ == "__main__":
    main()
