import csv
import json

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


def train_model(X_train, y_train):
    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=0.001)
    # Learn the digits on the train subset
    return clf.fit(X_train, y_train)


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
    X, y = load_data("data/digit_data.csv", "data/digit_target.csv")
    X_train, X_test, y_train, y_test = process_data(X, y)
    model = train_model(X_train, y_train)
    predicted_y = model.predict(X_test)
    output_test_data_results(y_test, predicted_y)
    output_metrics(y_test, predicted_y)


if __name__ == "__main__":
    main()
