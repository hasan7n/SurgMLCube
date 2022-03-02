"""MLCube handler file"""

import typer
import subprocess


app = typer.Typer()


class EvaluateTask(object):
    """Runs evaluation metrics given the predictions and label files

    Args:
        object ([type]): [description]
    """

    @staticmethod
    def run(
        preds_path: str, labels: str, parameters_file: str, output_file: str
    ) -> None:
        cmd = f"python3 metrics.py --preds_path={preds_path} --parameters_file={parameters_file} --output_file={output_file}"
        splitted_cmd = cmd.split()

        process = subprocess.Popen(splitted_cmd, cwd=".")
        process.wait()


@app.command("evaluate")
def evaluate(
    preds_path: str = typer.Option(..., "--predictions"),
    labels: str = typer.Option(..., "--labels"),
    parameters_file: str = typer.Option(..., "--parameters_file"),
    output_path: str = typer.Option(..., "--output_path"),
):
    EvaluateTask.run(preds_path, labels, parameters_file, output_path)


@app.command("test")
def test():
    pass


if __name__ == "__main__":
    app()
