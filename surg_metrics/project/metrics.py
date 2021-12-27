import argparse
import csv
from pathlib import Path
import yaml
from sklearn.metrics import f1_score, accuracy_score


class MetricsClass:
    def f1_score(labels, preds):
        return f1_score(labels, preds, average='macro')
    
    def accuracy(labels, preds):
        return accuracy_score(labels, preds)


class Evaluation:
    def __init__(self, preds_path, parameters_file, output_file):
        with open(parameters_file, "r") as f:
            self.params = yaml.full_load(f)

        self.available_metrics = {
            "f1-score": MetricsClass.f1_score,
            "accuracy": MetricsClass.accuracy,
        }

        self.output_file = output_file
        self.preds_path = Path(preds_path)
    
    def run(self):
        labels = list()
        preds = list()

        preds_files = self.preds_path.glob("*.csv")
        for file in preds_files:
            with open(file) as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i == 0:
                        continue
                    labels.append(int(row[1]))
                    preds.append(int(row[2]))

        
        results = {}
        for metric_name in self.params["metrics"]:
            metric = self.available_metrics[metric_name]
            scores = metric(labels, preds)
            results[metric_name] = float(scores)

        with open(self.output_file, "w") as f:
            yaml.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preds_path",
        "--preds-path",
        type=str,
        required=True,
        help="folder containing the labels and preds",
    )

    parser.add_argument(
        "--output_file",
        "--output-file",
        type=str,
        required=True,
        help="file to store metrics results as YAML",
    )
    parser.add_argument(
        "--parameters_file",
        "--parameters-file",
        type=str,
        required=True,
        help="File containing parameters for evaluation",
    )
    args = parser.parse_args()


    preprocessor = Evaluation(args.preds_path,
                                args.parameters_file,
                                args.output_file)
                                
    preprocessor.run()