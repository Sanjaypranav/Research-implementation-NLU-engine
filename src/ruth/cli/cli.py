import os
from pathlib import Path
from typing import Text
from urllib import request

import click
import matplotlib.pyplot as plt
import uvicorn as uvicorn
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from progressbar import progressbar
from rich import print as rprint
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table
from ruth import VERSION
from ruth.cli.constants import (
    BOLD_GREEN,
    BOLD_GREEN_CLOSE,
    BOLD_RED,
    BOLD_RED_CLOSE,
    BOLD_YELLOW,
    BOLD_YELLOW_CLOSE,
    FOLDER,
    ROCKET,
    TARGET,
)
from ruth.cli.utills import (
    Item,
    build_pipeline_from_metadata,
    check_model_path,
    get_config,
    get_interpreter_from_model_path,
    get_metadata_from_model,
    local_example_path,
    local_pipeline_path,
)
from ruth.constants import INTENT, INTENT_RANKING, TEXT
from ruth.nlu.model import Interpreter
from ruth.nlu.train import train_pipeline
from ruth.shared.constants import DATA_PATH, PIPELINE_PATH, RAW_GITHUB_URL
from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.ruth_config import RuthConfig
from sklearn.metrics import confusion_matrix
from starlette.responses import JSONResponse

console = Console()


def get_logo():
    logo_path = (Path(os.path.realpath(__file__))).parent / "data" / "banner.txt"
    return f"{logo_path.read_text()}"


def add_heading_to_description_table(table: Table) -> Table:
    table.add_column("Command", style="#c47900")
    table.add_column("Arguments ", style="#c47900")
    table.add_column("Description", style="#c47900")
    return table


def print_logo_and_description():
    console.print(f"[bold purple]{get_logo()}[/bold purple]", style="#6E1CF3")
    console.print(
        "[bold magenta]Website: [/bold magenta][link]https://puretalk.ai[/link]"
    )
    console.print("[bold magenta]Commands: [/bold magenta]")
    table = Table(show_header=True, header_style="bold #c47900", show_lines=True)
    table = add_heading_to_description_table(table)
    table.add_row(
        "[bold]train[/bold]",
        "-p [bold red]Pipeline_file[/bold red], -d [bold red]Data_file[/bold red]",
        "[green]Train a model with the given pipeline and data[/green]",
    )
    table.add_row(
        "[bold]parse[/bold]",
        "-m [bold red]Model_file[/bold red], -t [bold red]Text[/bold red]",
        "[green]Classify intend for a sentence from a trained model[/green]",
    )
    table.add_row(
        "[bold]evaluate[/bold]",
        "-m [bold red]Model_file[/bold red], -d [bold red]Data_file[/bold red]",
        "[green]Evaluates trained model for a test dataset[/green]",
    )
    table.add_row(
        "[bold]deploy[/bold]",
        "-m [bold red]Model_file[/bold red], -p [bold red]Port[/bold red], -h [bold red]Host[/bold red]",
        "[green]Trained models can be served using deploy command[/green]",
    )
    console.print(table)


class RichGroup(click.Group):
    def format_help(self, ctx, formatter):
        print_logo_and_description()
        # TODO: Want to write the help description whenever the user call the ruth --help


@click.group(cls=RichGroup)
@click.version_option(VERSION)
def entrypoint():
    pass


@entrypoint.command(name="train")
@click.option(
    "-d",
    "--data",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Data for training as json",
)
@click.option(
    "-p",
    "--pipeline",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="pipeline for training as yaml",
)
def train(data: Path, pipeline: Path):
    config = get_config(pipeline)
    training_data = TrainData.build(data)

    config = RuthConfig(config)
    model_absolute_dir = train_pipeline(config, training_data)
    console.print(
        f"Training completed {ROCKET}..."
        f"\nModel is stored at {FOLDER} {BOLD_YELLOW} {model_absolute_dir} {BOLD_YELLOW_CLOSE} \n",
        f"\nTo evaluate model:{BOLD_GREEN} ruth parse --help{BOLD_GREEN_CLOSE}",
    )


@entrypoint.command(name="parse")
@click.option(
    "-t",
    "--text",
    type=click.STRING,
    required=True,
    help="Data that need to be get parsed",
)
@click.option(
    "-m",
    "--model_path",
    type=click.STRING,
    required=False,
    help="Directory where the model is stored",
)
def parse(text: Text, model_path: Text):
    model_file = check_model_path(model_path)
    console.print(f"Latest Model found {FOLDER}  {model_file}")
    metadata = get_metadata_from_model(model_file.absolute())
    pipeline = build_pipeline_from_metadata(metadata=metadata, model_dir=model_file)
    interpreter = Interpreter(pipeline)
    output = interpreter.parse(text)
    console.print(
        f"{TARGET} Predicted intent is {output.get(INTENT)} \n",
        f"\nTo deploy your model run: {BOLD_GREEN}ruth deploy --help{BOLD_GREEN_CLOSE}",
    )


@entrypoint.command(name="evaluate")
@click.option(
    "-d",
    "--data",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Data for testing as json",
)
@click.option(
    "-m",
    "--model_path",
    type=click.STRING,
    required=False,
    help="Directory where the model is stored",
)
@click.option(
    "-o",
    "--output_folder",
    type=click.Path(),
    default=Path("results"),
    help="Directory where the results is stored",
)
def evaluate(data: Path, model_path: Text, output_folder: Text):
    model_file = check_model_path(model_path)
    console.print(f"Latest Model found {FOLDER} {model_file}")
    metadata = get_metadata_from_model(model_file.absolute())
    pipeline = build_pipeline_from_metadata(metadata=metadata, model_dir=model_file)
    interpreter = Interpreter(pipeline)
    training_data = TrainData.build(data)

    correct_predictions = 0
    y_pred = []
    y_actual = []
    for example in training_data.training_examples:
        output = interpreter.parse(example.get("text"))

        y_pred.append(output.get(INTENT).get("name"))
        y_actual.append(example.get("intent"))

        if output.get(INTENT).get("name") == example.get("intent"):
            correct_predictions += 1

    accuracy = correct_predictions / len(training_data)
    conf_matrix = confusion_matrix(y_true=y_actual, y_pred=y_pred)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(
                x=j, y=i, s=conf_matrix[i, j], va="center", ha="center", size="xx-large"
            )

    plt.xlabel("Predictions", fontsize=18)
    plt.ylabel("Actual", fontsize=18)
    plt.title("Confusion Matrix", fontsize=18)

    if output_folder:
        result_path = Path(output_folder).absolute()
    else:
        result_path = Path().absolute() / output_folder
    result_path.mkdir(exist_ok=True)
    directories = os.listdir(str(result_path))
    indexes = []
    model_name = str(model_file).split("/")[-1]
    for result in directories:
        if model_name in result:
            indexes.append(int(result.split("@")[-1]))
    if indexes:
        index = max(indexes) + 1
    else:
        index = 0

    folder_for_the_result = result_path / f"{model_name}@{index}"
    folder_for_the_result.mkdir(exist_ok=True)
    final_file_path = folder_for_the_result / "confusion_matrix.png"
    plt.savefig(final_file_path)

    rprint(f"{TARGET} accuracy: ", accuracy)
    rprint(f"{BOLD_GREEN} confusion matrix is created.{BOLD_GREEN_CLOSE}")
    rprint(" results are stored here: ", folder_for_the_result)
    rprint(
        f" To deploy your model run: {BOLD_GREEN}ruth deploy --help{BOLD_GREEN_CLOSE}"
    )


@entrypoint.command(name="deploy")
@click.option(
    "-m",
    "--model_path",
    type=click.STRING,
    required=False,
    help="Directory where the model is stored",
)
@click.option(
    "-p",
    "--port",
    type=click.INT,
    default=5500,
    help="Port where the application should run",
)
@click.option(
    "-h",
    "--host",
    type=click.STRING,
    default="localhost",
    help="host where the application should run",
)
def deploy(model_path: Text, port: int, host: str):
    app = FastAPI()

    app.interpreter = get_interpreter_from_model_path(model_path)

    @app.get("/parse")
    async def parse(item: Item):
        output = app.interpreter.parse(item.text)
        output = {
            key: output[key] for key in output.keys() & {INTENT_RANKING, TEXT, INTENT}
        }
        json_compatible_item_data = jsonable_encoder(output)
        return JSONResponse(content=json_compatible_item_data)

    uvicorn.run(app, host=host, port=port)


pbar = None


def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None


@entrypoint.command(name="init")
@click.option(
    "-o",
    "--output-path",
    type=click.STRING,
    required=False,
    help="Directory where the model is stored",
)
def init(output_path: Text):
    global pbar
    pipeline_path = f"{RAW_GITHUB_URL}/{PIPELINE_PATH}"
    data_path = f"{RAW_GITHUB_URL}/{DATA_PATH}"

    files_in_dir = 0
    for _ in Path().absolute().iterdir():
        files_in_dir += 1

    if files_in_dir:
        override_changes = Confirm.ask(
            f"{BOLD_RED}You already have project in the current directory. "
            f"Do you still want to create new project?{BOLD_RED_CLOSE}"
        )
        if not override_changes:
            return None
    rprint(f"{BOLD_GREEN}Downloading pipeline.yml {BOLD_GREEN_CLOSE}")
    request.urlretrieve(
        str(pipeline_path), str(local_pipeline_path(output_path)), show_progress
    )
    rprint(f"{BOLD_GREEN}Downloading data.yml{BOLD_GREEN_CLOSE}")
    request.urlretrieve(
        str(data_path), str(local_example_path(output_path)), show_progress
    )
    rprint(f"{BOLD_GREEN}Project is Successfully build{ROCKET}{BOLD_GREEN_CLOSE}")
    rprint(f" To train your model run: {BOLD_GREEN}ruth train --help{BOLD_GREEN_CLOSE}")
