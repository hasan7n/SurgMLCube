"""MLCube handler file"""

import typer
import subprocess


app = typer.Typer()


def exec_python(cmd: str) -> None:
    splitted_cmd = cmd.split()

    process = subprocess.Popen(splitted_cmd, cwd=".")
    process.wait()


class InferenceTask(object):
    """
    Task for preparing the data

    Arguments:
    - data_path: data location.
    - labels_path: labels location
    - params_file: yaml file with additional parameters
    - output_path: location to store prepared data
    """

    @staticmethod
    def run(
        data_root: str, feature_extraction_weights_path: str, mstcn_weights_path: str, transformer_weights_path: str, params_file: str, output_path: str
    ) -> None:
        cmd = f"python3 inference.py --data_path={data_root} --feature_extraction_weights_path={feature_extraction_weights_path} --mstcn_weights_path={mstcn_weights_path} --transformer_weights_path={transformer_weights_path} --params_file={params_file} --output_path={output_path}"
        exec_python(cmd)



@app.command("infer")
def prepare(
    data_path: str = typer.Option(..., "--data_path"),
    feature_extractor_weights: str = typer.Option(..., "--feature_extraction_weights"),
    mstcn_weights: str = typer.Option(..., "--mstcn_weights"),
    transformer_weights: str = typer.Option(..., "--transformer_weights"),
    parameters_file: str = typer.Option(..., "--parameters_file"),
    output_path: str = typer.Option(..., "--output_path"),
):
    InferenceTask.run(data_path, feature_extractor_weights, mstcn_weights, transformer_weights, parameters_file, output_path)

@app.command("dummy")
def dummy():
    print("This is added just to avoid 'typer' throwing an error when having only one task available")



if __name__ == "__main__":
    app()
